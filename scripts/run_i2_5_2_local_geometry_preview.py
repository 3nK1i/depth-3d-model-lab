#!/usr/bin/env python3
"""Fixed-view preview for I2.5.2 local-geometry / manifold diagnostics."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i2-5-2-dir", type=Path, default=Path("/tmp/depth3d_i2_5_2_extremum_local_geometry_manifold_diagnosis"))
    p.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    p.add_argument("--color-mode", choices=("manifold", "linearity", "ray", "rgb"), default="manifold")
    p.add_argument("--fit", choices=("full", "robust"), default="full")
    p.add_argument("--point-budget", type=int, default=250_000)
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _gradient(value: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
    return np.clip(np.rint(np.column_stack([t, 0.25 + 0.65*(1.0-t), 1.0-t]) * 255.0), 0, 255).astype(np.uint8)


def _fit_robust(canvas: PointCloudCanvas, lower: np.ndarray, upper: np.ndarray) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    center = (lower + upper) * .5
    diagonal = max(float(np.linalg.norm(upper - lower)), .5)
    camera.center = tuple(float(v) for v in center)
    camera.distance = max(diagonal * 1.35, 1.5)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    args = _args()
    directory = args.i2_5_2_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_5_2_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I2.5.2 preview requires completed diagnosis.")
    path = Path(str(summary["artifacts"]["local_geometry_manifold_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as p:
        points = np.asarray(p["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(p["colors_rgb"], dtype=np.uint8)
        drivers = np.asarray(p["is_extremum_driver"], dtype=bool)
        stubborn = np.asarray(p["is_stubborn_extremum_survivor"], dtype=bool)
        signature = np.asarray(p["has_combined_extremum_signature"], dtype=bool)
        anchors = np.asarray(p["is_strong_manifold_anchor"], dtype=bool)
        linearity = np.asarray(p["local_linearity"], dtype=np.float64)
        ray = np.asarray(p["supporting_ray_alignment"], dtype=np.float64)
        lower = np.asarray(p["robust_lower"], dtype=np.float64)
        upper = np.asarray(p["robust_upper"], dtype=np.float64)

    if args.color_mode == "rgb":
        colors, legend = rgb, "RGB"
    elif args.color_mode == "linearity":
        colors, legend = _gradient(linearity), "blue=surface-like/low linearity | red=line-like"
    elif args.color_mode == "ray":
        colors, legend = _gradient(ray), "blue=low local-axis/ray alignment | red=ray-aligned"
    else:
        colors = np.clip(np.rint(rgb.astype(np.float64) * .25), 0, 255).astype(np.uint8)
        colors[anchors] = [80, 180, 115]
        colors[drivers] = [235, 155, 45]
        colors[stubborn] = [235, 65, 65]
        colors[stubborn & signature] = [225, 65, 215]
        legend = "green=strong manifold | orange=driver | red=stubborn | magenta=stubborn + local-geometry signature"

    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I2.5.2 Local Geometry / Manifold Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(f"I2.5.2 | {len(points):,} rendered | {args.color_mode} | {args.view} | {args.fit} | {legend} | diagnostic only"))
    canvas = PointCloudCanvas(window); canvas.set_point_size(float(args.point_size)); canvas.set_grid_visible(True); canvas.update_points(points, colors)
    if args.fit == "full": canvas.fit_scene()
    else: _fit_robust(canvas, lower, upper)
    camera = getattr(canvas, "_camera", None)
    if camera is not None:
        camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[args.view])
        changed = getattr(camera, "view_changed", None)
        if callable(changed): changed()
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14): app.processEvents(); time.sleep(.02)
    output = args.output.expanduser().resolve() if args.output else directory / f"i2_5_2_{args.color_mode}_{args.view}_{args.fit}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.5.2 preview: {output}")
    print(f"I2.5.2 local-geometry preview: {output}")
    print(f"rendered_points={len(points)} color_mode={args.color_mode} view={args.view} fit={args.fit}")
    print("I2.5.2 diagnostics only: no geometry/filter policy changed; metric/measurement/mesh authority remains unpromoted.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
