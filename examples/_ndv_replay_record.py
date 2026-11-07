"""Opt-in capture of the actual ndv VisPy canvas and camera state."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from qtpy.QtCore import QTimer


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if all(hasattr(value, key) for key in ("w", "x", "y", "z")):
        return {"quaternion": [value.w, value.x, value.y, value.z]}
    if isinstance(value, BaseException):
        return repr(value)
    raise TypeError(type(value).__name__)


class Capture:
    def __init__(self, viewer, controller, directory, settings, *, max_frames=1200):
        self.canvas = viewer.canvas
        if type(self.canvas).__name__ != "VispyArrayCanvas":
            raise ValueError("Live recording currently requires ndv VisPy")
        self.controller = controller
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.settings = settings
        self.max_frames = max_frames
        self.index = 0
        self.started = time.monotonic()
        self.path = []
        self.timer = QTimer(viewer.widget())
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.capture)
        self.timer.start()

    def capture(self, kind="timer"):
        if self.index >= self.max_frames:
            self.timer.stop()
            return
        target = self.controller.target
        canvas = self.canvas
        pixels = canvas._canvas.render(alpha=True)
        filename = f"{self.index:05d}.npy"
        np.save(self.directory / filename, pixels, allow_pickle=False)
        viewport, matrix = canvas.camera_state()
        native = canvas._camera.get_state()
        self.path.append(native)
        row = {
            "frame": filename,
            "seconds": time.monotonic() - self.started,
            "kind": kind,
            "viewport": viewport,
            "world_to_clip": matrix,
            "native_camera": native,
            "settings": self.settings(),
            "performance": asdict(self.controller.stream.performance),
            "displayed_focus": target._displayed_focus_level,
            "handles": {
                str(level): {
                    "visible": handle.visible(),
                    "shape": handle.data().shape,
                    "clims": handle.clims(),
                }
                for level, handle in target.handles.items()
            },
        }
        with (self.directory / "frames.jsonl").open("a") as out:
            out.write(json.dumps(row, default=serial) + "\n")
        (self.directory / "camera-path.json").write_text(
            json.dumps(self.path, default=serial)
        )
        self.index += 1

    def close(self):
        self.timer.stop()
