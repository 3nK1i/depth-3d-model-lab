#!/usr/bin/env python3
"""Fixed-view preview for I2.5.1 extremum-driving survivor diagnostics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402


_VIEW_ANGLES = {"oblique": (32.0, 24.0), "front": (0.0, 0.0), "side": (90.0, 0.0), "top": (0.0, 89.0)}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2-5-1-dir", type=Path, default=Path("/tmp/depth3d_i2_5_1_spatial_extremum_survivor_diagnosis"))
    parser.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    parser.add_argument("--color-mode", choices=("extremum", "rejection-count", "rgb"), default="extremum")
    parser.add_argument("--fit", choices=("full", "robust"), default="full")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _set_view(canvas: PointCloudCanvas, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _fit_robust(canvas: PointCloudCanvas, lower: np.ndarray, upper: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    center = (lower + upper) * 0.5
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.5)
    camera.center = tuple(float(v) for v in center)
    camera.distance = max(diagonal * 1.35, 1.5)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _rejection_colors(count: np.ndarray) -> np.ndarray:
    count = np.asarray(count, dtype=np.float64)
    maximum = max(float(np.max(count, initial=0.0)), 1.0)
    t = np.clip(count / maximum, 0.0, 1.0)
    return np.clip(np.rint(np.column_stack([0.25 + 0.75*t, 0.75*(1.0-t), 0.95 - 0.70*t]) * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    args = _args()
    directory = args.i2_5_1_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_5_1_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I2.5.1 preview requires completed diagnosis.")
    path = Path(str(summary["artifacts"]["extremum_survivor_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        drivers = np.asarray(payload["is_extremum_driver"], dtype=bool)
        stubborn = np.asarray(payload["is_stubborn_extremum_survivor"], dtype=bool)
        far = np.asarray(payload["is_far_peripheral"], dtype=bool)
        rejection_count = np.asarray(payload["i2_5_rejection_count"], dtype=np.int16)
        lower = np.asarray(payload["robust_lower"], dtype=np.float64)
        upper = np.asarray(payload["robust_upper"], dtype=np.float64)

    if args.color_mode == "rgb":
        colors = rgb
        legend = "RGB"
    elif args.color_mode == "rejection-count":
        colors = _rejection_colors(rejection_count)
        legend = "blue/green=few I2.5 policies reject | red=many policies reject"
    else:
        colors = np.clip(np.rint(rgb.astype(np.float64) * 0.28), 0, 255).astype(np.uint8)
        colors[drivers] = [235, 155, 45]
        colors[stubborn] = [235, 65, 65]
        colors[stubborn & far] = [225, 65, 215]
        legend = "dim RGB=ordinary | orange=extremum driver | red=survives all I2.5 | magenta=far + survives all"

    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — I2.5.1 Spatial Extremum Survivor Review")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I2.5.1 | {len(points):,} rendered | color={args.color_mode} | view={args.view} | fit={args.fit} | {legend} | diagnostic only"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    if args.fit == "full":
        canvas.fit_scene()
    else:
        _fit_robust(canvas, lower, upper)
    _set_view(canvas, args.view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents(); time.sleep(0.02)
    output = args.output.expanduser().resolve() if args.output else directory / f"i2_5_1_{args.color_mode}_{args.view}_{args.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.5.1 preview: {output}")
    print(f"I2.5.1 extremum-survivor preview: {output}")
    print(f"rendered_points={len(points)} color_mode={args.color_mode} view={args.view} fit={args.fit}")
    print("I2.5.1 diagnostics only: no geometry/filter policy changed; metric/measurement/mesh authority remains unpromoted.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
