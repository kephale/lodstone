"""Lifecycle regressions against Blender itself; optional in the Python matrix."""

from __future__ import annotations

import asyncio
import json
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

bpy = pytest.importorskip("bpy")

from lodstone.adapters.blender import (
    BlenderController,
    MainThreadQueue,
    blender_camera_snapshot,
)
from lodstone.sources import ArrayPyramidSource

ROOT = Path(__file__).resolve().parents[1]


class SlowSource(ArrayPyramidSource):
    def __init__(self, *, delay=0.0, fail=False):
        super().__init__(
            [np.ones((16, 16, 16), np.uint16)],
            axes=("z", "y", "x"),
            chunks=[(4, 4, 4)],
        )
        self.delay = delay
        self.fail = fail

    async def read(self, level, region):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise OSError("injected failed source read")
        return await super().read(level, region)


@pytest.fixture
def scene():
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("Blender regression")
    bpy.context.window.scene = scene
    camera = bpy.data.objects.new("Regression camera", bpy.data.cameras.new("camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 24
    camera.location = (8, 8, 50)
    scene.render.resolution_x = scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    yield scene
    bpy.context.window.scene = previous
    for obj in list(scene.objects):
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data is not None and data.users == 0:
            if isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
            elif isinstance(data, bpy.types.Camera):
                bpy.data.cameras.remove(data)
    bpy.data.scenes.remove(scene)


@pytest.fixture
def make_controller(scene):
    controllers = []

    def make(source=None, **options):
        controller = BlenderController(
            source or SlowSource(),
            clims=(0, 1),
            memory_limit=1024,
            render_memory_limit=8192,
            block_shape=(4, 4, 4),
            **options,
        )
        controllers.append(controller)
        return controller

    yield make
    for controller in controllers:
        controller.close()


def view_of(controller, scene):
    return blender_camera_snapshot(
        scene,
        controller.root,
        controller.blender_canvas.displayed_axes,
        controller.blender_canvas.index,
    )


def wait_complete(controller):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        controller.queue.drain(budget_seconds=0.05)
        status = controller.stream.status
        if status.state == "complete":
            return
        assert status.state != "failed", status.error
        time.sleep(0.005)
    pytest.fail("stream did not recover")


@pytest.mark.parametrize("in_flight", [False, True])
def test_render_cannot_be_superseded_by_pending_camera_worker(
    scene,
    make_controller,
    monkeypatch,
    in_flight,
):
    controller = make_controller(
        SlowSource(delay=0.05),
        camera_debounce_ms=0 if in_flight else 180,
    )
    render_view = view_of(controller, scene)
    controller.update(render_view)
    outside = render_view.world_to_clip.copy()
    outside[0, 3] += 100
    monkeypatch.setattr(
        controller.blender_canvas,
        "camera_state",
        lambda: (render_view.viewport, outside),
    )
    entered, release = threading.Event(), threading.Event()
    submit = controller.stream.submit

    def delayed_submit(view, plan):
        if threading.current_thread() is controller._camera_thread:
            entered.set()
            assert release.wait(3)
        return submit(view, plan)

    if in_flight:
        monkeypatch.setattr(controller.stream, "submit", delayed_submit)
    controller._camera_changed()
    if in_flight:
        assert entered.wait(3)
        timer = threading.Timer(0.03, release.set)
        timer.start()
    try:
        plan = controller.prepare_render(timeout=5)
        actual = controller.stream.performance
        assert actual.status.state == "complete"
        assert actual.plan.desired_tiles == len(plan.desired or plan.wanted) > 0
        np.testing.assert_array_equal(
            actual.view.world_to_clip, render_view.world_to_clip
        )
        assert controller.last_publication is not None
        assert controller._camera_updates_paused
    finally:
        release.set()
        if in_flight:
            timer.join(3)


@pytest.mark.parametrize("fault", ["timeout", "read", "publication"])
def test_failed_preparation_restores_state_and_navigation(
    scene,
    make_controller,
    monkeypatch,
    fault,
):
    source = SlowSource(delay=0.3 if fault == "timeout" else 0, fail=fault == "read")
    controller = make_controller(source)
    interactive = view_of(controller, scene)
    controller.update(interactive)
    monkeypatch.setattr(controller.blender_canvas, "view", lambda: interactive)
    publish = controller.blender_canvas.add_volume

    def failed_publication(*args, **kwargs):
        raise RuntimeError("injected failed publication")

    if fault == "publication":
        monkeypatch.setattr(controller.blender_canvas, "add_volume", failed_publication)
    error = TimeoutError if fault == "timeout" else RuntimeError
    with pytest.raises(error):
        controller.prepare_render(timeout=0.01 if fault == "timeout" else 5)
    assert controller.blender_canvas.render_view is None
    assert controller.target.memory_limit == 1024
    assert not controller._camera_updates_paused
    assert controller._render_restore is None
    source.delay, source.fail = 0, False
    monkeypatch.setattr(controller.blender_canvas, "add_volume", publish)
    controller.update(interactive)
    wait_complete(controller)
    np.testing.assert_array_equal(
        controller.stream.performance.view.world_to_clip,
        interactive.world_to_clip,
    )


def test_cancelled_render_generation_does_not_count_as_ready(make_controller):
    controller = make_controller()
    controller.queue(controller.stream.cancel)
    with pytest.raises(RuntimeError, match="superseded or cancelled"):
        controller.prepare_render(timeout=1)
    assert controller.blender_canvas.render_view is None
    assert controller.target.memory_limit == 1024
    assert not controller._camera_updates_paused


@pytest.mark.parametrize("cancelled", [False, True])
def test_render_completion_or_cancel_restores_custom_interactive_budget(
    scene,
    make_controller,
    monkeypatch,
    cancelled,
):
    controller = make_controller()
    interactive = view_of(controller, scene)
    monkeypatch.setattr(controller.blender_canvas, "view", lambda: interactive)
    controller.set_focus_policy(memory_limit=2048, replan=False)
    controller.start()
    controller.prepare_render(timeout=5)
    assert controller.target.memory_limit == 8192
    assert controller._camera_updates_paused
    if cancelled:
        callbacks = {id(group): callback for group, callback in controller._handlers}
        callbacks[id(bpy.app.handlers.render_init)](scene)
        assert controller.queue.paused and controller.rendering
        callbacks[id(bpy.app.handlers.render_cancel)](scene)
    else:
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 2
        bpy.ops.render.render()
    assert not controller.rendering
    assert not controller.queue.paused
    controller.finish_render()
    assert controller.target.memory_limit == 2048
    assert controller.blender_canvas.render_view is None
    assert not controller._camera_updates_paused
    wait_complete(controller)
    controller.finish_render()  # repeated restoration is harmless


def test_timer_stop_and_restart_use_the_exact_registered_callback(monkeypatch):
    registered = []
    register = bpy.app.timers.register

    def capture(callback, **kwargs):
        registered.append(callback)
        return register(callback, **kwargs)

    monkeypatch.setattr(bpy.app.timers, "register", capture)
    queue = MainThreadQueue()
    try:
        queue.start()
        queue.start()
        assert len(registered) == 1
        callback = registered[0]
        queue.stop()
        assert not bpy.app.timers.is_registered(callback)
        queue.start()
        assert registered[1] is callback
        assert bpy.app.timers.is_registered(callback)
    finally:
        queue.stop()
        for callback in registered:
            if bpy.app.timers.is_registered(callback):
                bpy.app.timers.unregister(callback)


def test_controller_start_is_idempotent_and_close_removes_timers(
    make_controller,
    monkeypatch,
):
    controller = make_controller()
    monkeypatch.setattr(controller.blender_canvas, "view", lambda: None)
    registered = []
    register = bpy.app.timers.register

    def capture(callback, **kwargs):
        registered.append(callback)
        return register(callback, **kwargs)

    monkeypatch.setattr(bpy.app.timers, "register", capture)
    groups = (
        bpy.app.handlers.render_init,
        bpy.app.handlers.render_complete,
        bpy.app.handlers.render_cancel,
        bpy.app.handlers.load_pre,
    )
    before = [len(group) for group in groups]
    try:
        controller.start()
        controller.start()
        assert len(registered) == 2
        assert [len(group) for group in groups] == [n + 1 for n in before]
        controller.close()
        assert not any(bpy.app.timers.is_registered(cb) for cb in registered)
        assert [len(group) for group in groups] == before
        with pytest.raises(RuntimeError, match="closed"):
            controller.start()
    finally:
        controller.close()
        for callback in registered:
            if bpy.app.timers.is_registered(callback):
                bpy.app.timers.unregister(callback)


@pytest.mark.parametrize("outcome", ["exception", "cancelled"])
def test_render_operator_restores_state_if_invocation_fails(
    make_controller,
    outcome,
    monkeypatch,
):
    demo = runpy.run_path(str(ROOT / "examples/blender_ome_zarr.py"))
    controller = make_controller()
    controller.info = {}

    def invoke(*args):
        if outcome == "exception":
            raise RuntimeError("injected render invocation failure")
        return {"CANCELLED"}

    fake = SimpleNamespace(
        app=SimpleNamespace(driver_namespace={demo["KEY"]: controller}),
        ops=SimpleNamespace(render=SimpleNamespace(render=invoke)),
    )
    operator = demo["LODSTONE_OT_render_finest"].execute
    monkeypatch.setitem(operator.__globals__, "bpy", fake)
    result = operator(SimpleNamespace(report=lambda *args: None), None)
    assert result == {"CANCELLED"}
    assert controller.blender_canvas.render_view is None
    assert controller.target.memory_limit == 1024
    assert not controller._camera_updates_paused


def test_native_verification_exits_nonzero_for_incorrect_grid(tmp_path):
    # Execute the entire actual check with a deliberately corrupted small-grid
    # upload; avoid a test that merely mirrors its final boolean expression.
    script = """
import runpy, sys
import numpy as np
from lodstone.adapters import blender as adapter
original = adapter._GridHandle.set_data
def corrupt(self, data):
    if data.shape == (5, 7, 11):
        data = data + np.uint16(1)
    return original(self, data)
adapter._GridHandle.set_data = corrupt
sys.argv = ['examples/blender_native_grid_check.py', '--', sys.argv[1],
            '--engine', 'CYCLES', '--threads', '2']
runpy.run_path('examples/blender_native_grid_check.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((tmp_path / "check.json").read_text())
    assert report["ok"] is False
    assert report["index_order_max_abs_error"] > 1e-6
