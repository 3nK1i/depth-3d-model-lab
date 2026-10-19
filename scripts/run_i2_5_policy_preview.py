#!/usr/bin/env python3
"""Fixed-view preview for I2.5 diversity-aware filter counterfactuals."""
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


_VIEW_ANGLES = {
    "oblique": (32.0, 24.0),
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "top": (0.0, 89.0),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i2-5-dir", type=Path, default=Path("/tmp/depth3d_i2_5_conservative_diversity_filter_counterfactual"))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--representation", choices=("extended", "core"), default="extended")
    parser.add_argument("--color-mode", choices=("rgb", "decision", "region", "diversity"), default="rgb")
    parser.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    parser.add_argument("--include-rejected", action="store_true", help="Show rejected points for decision/diversity review.")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0.0:
        parser.error("point budget and point size must be positive")
    return args


def _set_view(canvas: PointCloudCanvas, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _decision_colors(rgb: np.ndarray, reject: np.ndarray, core: np.ndarray) -> np.ndarray:
    colors = np.asarray(rgb, dtype=np.uint8).copy()
    colors = np.clip(np.rint(colors.astype(np.float64) * 0.55), 0, 255).astype(np.uint8)
    colors[core & ~reject] = [85, 205, 115]
    colors[reject & ~core] = [240, 85, 65]
    colors[reject & core] = [225, 65, 210]
    return colors


def _region_colors(peripheral: np.ndarray, far: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[70, 205, 105]], dtype=np.uint8), (len(peripheral), 1))
    colors[peripheral] = [235, 165, 55]
    colors[far] = [235, 70, 70]
    return colors


def _diversity_colors(low: np.ndarray, reject: np.ndarray, peripheral: np.ndarray, far: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[70, 195, 105]], dtype=np.uint8), (len(low), 1))
    colors[low] = [70, 120, 225]
    colors[peripheral & low] = [235, 75, 65]
    colors[far & low] = [225, 65, 215]
    colors[reject] = [255, 220, 60]
    return colors


def main() -> int:
    args = _args()
    directory = args.i2_5_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_5_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "counterfactuals_ready":
        raise RuntimeError("I2.5 preview requires completed counterfactual evidence.")
    records = {str(item["name"]): item for item in summary.get("scenarios", [])}
    if args.scenario not in records:
        raise RuntimeError(f"Unknown I2.5 scenario {args.scenario!r}; available: {','.join(sorted(records))}")
    record = records[args.scenario]
    path = Path(str(record["artifacts"]["diagnostics_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        reject = np.asarray(payload["is_rejected_by_policy"], dtype=bool)
        kept = np.asarray(payload["is_kept"], dtype=bool)
        core_before = np.asarray(payload["is_core_before"], dtype=bool)
        core_after = np.asarray(payload["is_core_after"], dtype=bool)
        periph_before = np.asarray(payload["is_peripheral_before"], dtype=bool)
        far_before = np.asarray(payload["is_far_before"], dtype=bool)
        low = np.asarray(payload["is_low_diversity"], dtype=bool)
        cand_periph = np.asarray(payload["is_candidate_peripheral"], dtype=bool)
        cand_far = np.asarray(payload["is_candidate_far"], dtype=bool)

    if args.representation == "core":
        representation = core_before if args.include_rejected else core_after
    else:
        representation = np.ones(len(points), dtype=bool) if args.include_rejected else kept

    if args.color_mode == "rgb":
        colors = rgb
        legend = "candidate RGB"
    elif args.color_mode == "decision":
        colors = _decision_colors(rgb, reject, core_before)
        legend = "green=kept Core | red=rejected Extended-only | magenta=rejected Core"
    elif args.color_mode == "region":
        colors = _region_colors(cand_periph, cand_far)
        legend = "green=central kept | orange=peripheral kept | red=far kept"
    else:
        colors = _diversity_colors(low, reject, periph_before, far_before)
        legend = "blue=low-div | red=peripheral low-div | magenta=far low-div | yellow=policy reject"

    points = points[representation]
    colors = colors[representation]
    if len(points) == 0:
        raise RuntimeError("I2.5 selected preview representation is empty.")
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)

    comparison = record["comparison"]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — I2.5 Diversity-Aware Filter Counterfactual Review")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I2.5 | {args.scenario} | {args.representation} | {len(points):,} rendered | color={args.color_mode} | "
        f"view={args.view} | core_loss={100.0*float(comparison['core_loss_fraction']):.1f}% | "
        f"far_reduction={100.0*float(comparison['far_peripheral_reduction_fraction']):.1f}% | {legend} | diagnostic only"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    canvas.fit_scene()
    _set_view(canvas, args.view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents(); time.sleep(0.02)

    suffix = "all" if args.include_rejected else "candidate"
    output = args.output.expanduser().resolve() if args.output else directory / (
        f"i2_5_{args.scenario}_{args.representation}_{args.color_mode}_{args.view}_{suffix}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.5 preview: {output}")
    print(f"I2.5 counterfactual preview: {output}")
    print(
        f"scenario={args.scenario} representation={args.representation} color_mode={args.color_mode} "
        f"view={args.view} include_rejected={args.include_rejected}"
    )
    print("I2.5 diagnostic counterfactual only: source F/G geometry and all authority remain unchanged.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
