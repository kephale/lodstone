"""Matched VisPy camera replay: actual frames, quality, cost and controls.

EGL_PLATFORM=surfaceless python examples/volume_replay.py --output /tmp/replay
Use --zebrahub --roi z0,y0,x0,z1,y1,x1 for a bounded real-data comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import psutil
from vispy import app, scene
from vispy.util.quaternion import Quaternion
from vispy.visuals.transforms import STTransform

from lodstone import Level, Pyramid, Region, View
from lodstone.adapters.dense import DenseTarget
from lodstone.planner import _map_region
from lodstone.replay import image_error, pareto_front
from lodstone.sources import ArrayPyramidSource, OMEZarrSource
from lodstone.stream import Stream

URL = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr"


class CroppedSource:
    def __init__(self, source, region):
        self.source = source
        self.regions = tuple(
            _map_region(region, source.pyramid.levels[0], level)
            for level in source.pyramid.levels
        )
        levels = []
        for level, crop in zip(source.pyramid.levels, self.regions, strict=True):
            shift = np.eye(4)
            shift[:3, 3] = crop.start
            grid = []
            for axis in range(3):
                lengths = []
                offset = 0
                for length in level.chunk_sizes(axis):
                    overlap = min(offset + length, crop.stop[axis]) - max(
                        offset, crop.start[axis]
                    )
                    if overlap > 0:
                        lengths.append(overlap)
                    offset += length
                grid.append(tuple(lengths))
            levels.append(
                Level(
                    crop.shape,
                    level.dtype,
                    level.chunks,
                    level.voxel_to_world @ shift,
                    chunk_grid=tuple(grid),
                )
            )
        self.pyramid = Pyramid(source.pyramid.axes, tuple(levels))

    async def read(self, level, region):
        offset = self.regions[level].start
        return await self.source.read(
            level,
            Region(
                tuple(a + b for a, b in zip(region.start, offset)),
                tuple(a + b for a, b in zip(region.stop, offset)),
            ),
        )


class Handle:
    def __init__(self, canvas, data):
        self.canvas = canvas
        self.visual = scene.visuals.Volume(
            data,
            parent=canvas.view.scene,
            method="mip",
            interpolation="nearest",
            clim=canvas.clims,
            texture_format="auto",
        )
        # Match ndv's current VisPy multi-volume composition.
        self.visual.set_gl_state("additive", depth_test=False)

    def set_data(self, data):
        self.visual.set_data(data)

    def set_world_transform(self, scales, origins):
        self.visual.transform = STTransform(
            scale=tuple(reversed(scales)), translate=tuple(reversed(origins))
        )

    def set_clims(self, clims):
        self.visual.clim = self.canvas.clims  # fixed across every level and policy

    def set_visible(self, visible):
        self.visual.visible = visible

    def remove(self):
        self.visual.parent = None


class Canvas:
    def __init__(self, size, clims):
        self.clims = clims
        self.scene = scene.SceneCanvas(size=(size, size), show=False, bgcolor="black")
        self.view = self.scene.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=0)

    def set_ndim(self, ndim):
        if ndim != 3:
            raise ValueError("3-D only")

    def add_volume(self, data, **kwargs):
        return Handle(self, data)

    def set_range(self):
        pass  # the replay, not publication, owns camera state

    def refresh(self):
        self.scene.update()

    def frame(self):
        return self.scene.render(alpha=True)

    def pose(self, pose):
        pose = dict(pose)
        if "_quaternion" in pose:
            if not isinstance(self.view.camera, scene.cameras.ArcballCamera):
                self.view.camera = scene.cameras.ArcballCamera(fov=0)
            pose["_quaternion"] = Quaternion(*pose["_quaternion"]["quaternion"])
        self.view.camera.set_state(pose)
        self.frame()  # update transforms before reading the camera
        points = np.zeros((4, 4))
        points[:, 3] = 1
        for axis in range(3):
            points[axis + 1, 2 - axis] = 1
        mapped = np.asarray(
            self.view.scene.node_transform(self.scene.scene).map(points)
        )
        matrix = np.eye(4)
        matrix[:, 3] = mapped[0]
        for axis in range(3):
            matrix[:, axis] = mapped[axis + 1] - mapped[0]
        width, height = self.scene.size
        screen = np.eye(4)
        screen[0] = (2 / width, 0, 0, -1)
        screen[1] = (0, -2 / height, 0, 1)
        return View((0, 1, 2), (None,) * 3, (width, height), screen @ matrix)


def synthetic():
    z, y, x = np.indices((64, 64, 64), dtype=float)
    data = np.zeros(z.shape, np.float32)
    for angle in np.linspace(-2, 2, 18):
        center = (
            32 + 12 * np.sin(angle * 2),
            32 + 22 * np.sin(angle),
            32 + 22 * np.cos(angle),
        )
        r2 = sum((axis - c) ** 2 for axis, c in zip((z, y, x), center))
        data += np.exp(-r2 / 3)
    arrays = [data]
    for _ in range(3):
        a = arrays[-1]
        arrays.append(
            a.reshape(*(v for n in a.shape for v in (n // 2, 2))).mean((1, 3, 5))
        )
    return ArrayPyramidSource(
        arrays,
        axes=("z", "y", "x"),
        chunks=[(8, 8, 8)] * 4,
        transforms=[np.diag([2**i] * 3 + [1]) for i in range(4)],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zebrahub", action="store_true")
    parser.add_argument(
        "--roi", help="finest-level z0,y0,x0,z1,y1,x1; shared by all runs"
    )
    parser.add_argument("--time", type=int, default=400)
    parser.add_argument(
        "--path", type=Path, help="JSON list of VisPy Turntable camera states"
    )
    parser.add_argument(
        "--stride", type=int, default=1, help="Subsample supplied camera path"
    )
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--poses", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--reference-mib", type=int, default=512)
    parser.add_argument("--clim", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--budgets", type=float, nargs="+", default=(0.0625, 0.25))
    parser.add_argument("--app", default="egl", help="VisPy application backend")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a new output directory to avoid mixing runs")
    app.use_app(args.app)
    source = (
        OMEZarrSource.open(URL, fixed_index={"t": args.time, "c": 0})
        if args.zebrahub
        else synthetic()
    )
    if args.roi:
        values = tuple(map(int, args.roi.split(",")))
        if len(values) != 6:
            raise ValueError("ROI needs six integers")
        roi = Region(values[:3], values[3:])
        if any(
            b > n or b <= a
            for a, b, n in zip(roi.start, roi.stop, source.pyramid.levels[0].shape)
        ):
            raise ValueError("ROI must be nonempty and inside the finest level")
        source = CroppedSource(source, roi)
    level = source.pyramid.levels[0]
    full_bytes = int(np.prod(level.shape)) * level.dtype.itemsize
    if full_bytes > args.reference_mib * 1024**2:
        raise ValueError(
            f"Fine reference requires {full_bytes} decoded bytes. Supply an explicit ROI or increase --reference-mib; no coarse fallback."
        )
    matrix = level.voxel_to_world
    center = (matrix @ np.array([*(np.asarray(level.shape) / 2), 1]))[:3][::-1]
    extent = float(np.max(np.abs(matrix[:3, :3]) @ level.shape))
    path = (
        json.loads(args.path.read_text())
        if args.path
        else [
            {
                "center": (center + [0, (i % 3 - 1) * extent * 0.12, 0]).tolist(),
                "scale_factor": extent * (1.3, 0.7, 0.35)[i % 3],
                "azimuth": i * 360 / args.poses,
                "elevation": 25,
            }
            for i in range(args.poses)
        ]
    )
    if args.stride < 1:
        raise ValueError("stride must be positive")
    path = path[:: args.stride]
    if not path:
        raise ValueError("camera path is empty")
    args.output.mkdir(parents=True)
    (args.output / "camera-path.json").write_text(json.dumps(path, indent=2))
    configurations = [
        ("reference", None, 0),
        ("coarse", None, 0),
        ("blank", None, 0),
        ("stationary", None, 0),
    ]
    configurations += [
        (f"adaptive-{budget:g}-{weight:g}", budget, weight)
        for budget in args.budgets
        for weight in (0, 1)
    ]
    manifest = {
        "source": URL if args.zebrahub else "synthetic-curved-nuclei",
        "time": args.time,
        "roi": args.roi,
        "reference_decoded_bytes": full_bytes,
        "clim": args.clim,
        "renderer": "standalone VisPy MIP/additive, ndv-style dense target; not ndv GUI",
        "cache_protocol": "fresh stream per configuration, warm across poses",
        "source_hashes": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "examples/volume_replay.py",
                "src/lodstone/replay.py",
                "src/lodstone/adapters/dense.py",
                "src/lodstone/planner.py",
            )
        },
        "lodstone_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cost_scope": "source_bytes are decoded payload, not encoded network bandwidth; RSS is sampled process total; GPU bytes unmeasured",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    references = []
    rows = []
    process = psutil.Process()
    for name, budget, weight in configurations:
        directory = args.output / name
        directory.mkdir()
        canvas = Canvas(args.size, tuple(args.clim))
        callbacks = queue.Queue()
        frames = []
        current_pose = [0]
        clock = [time.monotonic()]

        def capture(
            kind,
            canvas=canvas,
            current_pose=current_pose,
            frames=frames,
            directory=directory,
            clock=clock,
        ):
            frame = canvas.frame()
            filename = f"{current_pose[0]:03d}-{len(frames):05d}-{kind}.png"
            iio.imwrite(directory / filename, frame)
            frames.append(
                {
                    "pose": current_pose[0],
                    "kind": kind,
                    "file": filename,
                    "seconds": time.monotonic() - clock[0],
                }
            )
            return frame

        target = DenseTarget(
            canvas,
            source.pyramid,
            memory_limit=int((budget or args.reference_mib) * 1024**2),
            block_shape=(8, 16, 16),
            on_presented=lambda p: capture(f"phase-L{p.level}"),
        )
        target.content_weight = weight
        stream = Stream(source, target, dispatch=callbacks.put)
        # Status counters reset per generation: retain maxima separately before summing.
        generations = {}

        def sample(status, stream=stream, generations=generations):
            d = asdict(stream.diagnostics)
            generations[d["generation"]] = d

        disconnect = stream.on_status_changed(sample)
        peak_rss = 0
        peak_resident = 0
        errors = []
        statuses = []
        try:
            for index, pose in enumerate(path):
                current_pose[0] = index
                clock[0] = time.monotonic()
                view = canvas.pose(path[0] if name == "stationary" else pose)
                peak_rss = max(peak_rss, process.memory_info().rss)
                if name != "blank":
                    if budget is None:
                        fixed_level = (
                            0 if name == "reference" else len(source.pyramid.levels) - 1
                        )
                        plan = stream.planner.plan_overview(
                            source.pyramid,
                            view,
                            level_index=fixed_level,
                            memory_limit=args.reference_mib * 1024**2,
                            available=stream.available,
                        )
                        if not plan.desired:
                            raise RuntimeError(
                                "reference/control cannot fit; refusing empty baseline"
                            )
                        stream.submit(view, plan)
                    else:
                        stream.update(view)
                    refined = False
                    last_capture = time.monotonic()
                    while True:
                        try:
                            callbacks.get(timeout=0.01)()
                        except queue.Empty:
                            pass
                        peak_rss = max(peak_rss, process.memory_info().rss)
                        peak_resident = max(peak_resident, target.resident.nbytes)
                        if time.monotonic() - last_capture > 0.25:
                            capture("loading")
                            last_capture = time.monotonic()
                        state = stream.status.state
                        if stream.status.error is not None:
                            raise RuntimeError(str(stream.status.error))
                        if state == "complete":
                            if weight and not refined:
                                refined = True
                                stream.update(
                                    view
                                )  # same pose, now guided by loaded overview
                                continue
                            break
                        if time.monotonic() - clock[0] > args.timeout:
                            raise TimeoutError(f"{name} pose {index}")
                settled = capture("settled")
                time.sleep(0.03)
                delayed = capture("delayed")
                if name == "reference":
                    if image_error(settled, settled)["foreground_fraction"] == 0:
                        raise ValueError(
                            f"Reference pose {index} has no measurable foreground: choose a signal-bearing ROI or fixed contrast; do not optimize this case"
                        )
                    references.append(settled)
                elif name not in ("stationary",):
                    errors.append(image_error(references[index], settled))
                statuses.append(
                    {
                        "pose": index,
                        "status": asdict(stream.status),
                        "desired_level": None
                        if name == "blank"
                        else plan.target_level
                        if budget is None
                        else stream._plan.target_level,
                        "handles": {
                            str(k): bool(h.visual.visible)
                            for k, h in target.handles.items()
                        },
                        "delayed_error": image_error(settled, delayed),
                    }
                )
            if name not in ("reference", "stationary"):
                for frame in frames:
                    frame["error"] = image_error(
                        references[frame["pose"]], iio.imread(directory / frame["file"])
                    )
            regressions = []
            for index in range(len(path)):
                measured = [f for f in frames if f["pose"] == index and "error" in f]
                if measured:
                    best = min(f["error"]["foreground_rmse"] for f in measured)
                    final = measured[-1]["error"]["foreground_rmse"]
                    if final > best + 0.01:
                        regressions.append(
                            {"pose": index, "best": best, "final": final}
                        )
            row = {
                "id": name,
                "source_bytes": sum(d["source_bytes"] for d in generations.values()),
                "source_reads": sum(d["source_reads"] for d in generations.values()),
                "resident_bytes": peak_resident,
                "rss_bytes": peak_rss,
                "foreground_rmse": float(
                    np.mean([e["foreground_rmse"] for e in errors])
                )
                if errors
                else 0.0,
                "rmse": float(np.mean([e["rmse"] for e in errors])) if errors else 0.0,
            }
            rows.append(row)
            (directory / "run.json").write_text(
                json.dumps(
                    {
                        "summary": row,
                        "frames": frames,
                        "errors": errors,
                        "statuses": statuses,
                        "generations": generations,
                        "quality_regressions": regressions,
                    },
                    indent=2,
                )
            )
        finally:
            disconnect()
            stream.close()
            target.close()
            canvas.scene.close()
    candidates = [row for row in rows if row["id"] not in ("blank", "stationary")]
    report = {
        "rows": rows,
        "pareto": pareto_front(
            candidates, ("foreground_rmse", "source_bytes", "resident_bytes")
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
