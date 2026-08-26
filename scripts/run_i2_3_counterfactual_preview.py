#!/usr/bin/env python3
"""Fixed-view preview for one I2.3 balanced exact counterfactual."""
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
    parser.add_argument("--i2-3-dir", type=Path, default=Path("/tmp/eia_i2_3_balanced_cross_segment_counterfactuals"))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--representation", choices=("extended", "core"), default="extended")
    parser.add_argument("--color-mode", choices=("rgb", "region"), default="rgb")
    parser.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _set_view(canvas: PointCloudCanvas, name: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.azimuth, camera.elevation = _VIEW_ANGLES[name]
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _region_colors(peripheral: np.ndarray, far: np.ndarray, weak: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[80, 205, 110]], dtype=np.uint8), (len(peripheral), 1))
    colors[peripheral] = [235, 190, 55]
    colors[weak] = [235, 120, 55]
    colors[far] = [225, 70, 70]
    return colors


def main() -> int:
    args = _args()
    root = args.i2_3_dir.expanduser().resolve()
    summary = json.loads((root / "i2_3_summary.json").read_text(encoding="utf-8"))
    matches = [item for item in summary["scenarios"] if str(item["name"]) == args.scenario]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown I2.3 scenario: {args.scenario}")
    record = matches[0]
    exact = record["exact_i2_2_result"]
    diagnostic = Path(str(exact["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    with np.load(diagnostic, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        is_core = np.asarray(payload["is_core"], dtype=bool)
        peripheral = np.asarray(payload["is_peripheral"], dtype=bool)
        far = np.asarray(payload["is_far_peripheral"], dtype=bool)
        weak = np.asarray(payload["is_weak_peripheral"], dtype=bool)
    mask = np.ones(len(points), dtype=bool) if args.representation == "extended" else is_core
    points, rgb = points[mask], rgb[mask]
    peripheral, far, weak = peripheral[mask], far[mask], weak[mask]
    colors = rgb if args.color_mode == "rgb" else _region_colors(peripheral, far, weak)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)

    ev = record["balanced_evaluation"]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow(); window.setWindowTitle("EIA TECH — I2.3 Balanced Cross-Segment Review")
    central = QWidget(); layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I2.3 | {args.scenario} | {args.representation} | {points.shape[0]:,} rendered | color={args.color_mode} | "
        f"view={args.view} | core_loss={float(ev['combined_core_loss_fraction']):.1%} | "
        f"far_reduction={float(ev['combined_far_peripheral_reduction_fraction']):.1%} | "
        f"full/robust={float(ev['combined_full_to_robust_ratio']):.3f} | diagnostic only"
    ))
    canvas = PointCloudCanvas(window); canvas.set_point_size(float(args.point_size)); canvas.set_grid_visible(True)
    canvas.update_points(points, colors); canvas.fit_scene(); _set_view(canvas, args.view)
    layout.addWidget(canvas.native, 1); window.setCentralWidget(central); window.resize(1366, 768); window.show()
    for _ in range(14):
        app.processEvents(); time.sleep(0.02)
    output = args.output.expanduser().resolve() if args.output is not None else root / f"i2_3_{args.scenario}_{args.representation}_{args.color_mode}_{args.view}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save preview: {output}")
    print(f"I2.3 counterfactual preview: {output}")
    print(f"scenario={args.scenario} representation={args.representation} color_mode={args.color_mode} view={args.view}")
    print("Balanced exact counterfactual only; source persistent scene and authority remain unchanged.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
