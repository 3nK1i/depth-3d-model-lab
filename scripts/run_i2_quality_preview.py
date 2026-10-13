#!/usr/bin/env python3
"""Fixed-view preview for I2 full-video Dense RGB quality diagnostics."""
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
    parser.add_argument("--i2-dir", type=Path, default=Path("/tmp/eia_i2_full_video_quality"))
    parser.add_argument("--representation", choices=("extended", "core"), default="extended")
    parser.add_argument(
        "--color-mode",
        choices=("rgb", "region", "direction", "support", "confidence", "dispersion", "component"),
        default="rgb",
    )
    parser.add_argument("--view", choices=tuple(_VIEW_ANGLES), default="oblique")
    parser.add_argument("--fit", choices=("full", "robust"), default="full")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0.0:
        parser.error("point budget and point size must be positive")
    return args


def _scalar_colors(values: np.ndarray, *, invert: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    normalized = np.zeros(len(data), dtype=np.float64)
    if np.any(finite):
        lo = float(np.quantile(data[finite], 0.05))
        hi = float(np.quantile(data[finite], 0.95))
        if hi <= lo:
            hi = lo + 1.0
        normalized[finite] = np.clip((data[finite] - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    r = np.clip(2.0 * normalized - 0.2, 0.0, 1.0)
    g = np.clip(1.7 * normalized, 0.0, 1.0)
    b = np.clip(1.2 - 1.4 * normalized, 0.0, 1.0)
    colors = np.column_stack([r, g, b])
    colors[~finite] = np.asarray([0.4, 0.4, 0.4])
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def _region_colors(peripheral: np.ndarray, far: np.ndarray, weak: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[80, 205, 110]], dtype=np.uint8), (len(peripheral), 1))
    colors[peripheral] = [235, 190, 55]
    colors[weak] = [235, 120, 55]
    colors[far] = [225, 70, 70]
    return colors


def _direction_colors(sector: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [110, 110, 110],
            [235, 80, 80],
            [160, 70, 220],
            [70, 210, 100],
            [45, 150, 220],
            [235, 190, 55],
            [225, 105, 45],
        ],
        dtype=np.uint8,
    )
    return palette[np.clip(np.asarray(sector, dtype=np.int64), 0, 6)]


def _component_colors(dominant: np.ndarray) -> np.ndarray:
    colors = np.tile(np.asarray([[205, 95, 215]], dtype=np.uint8), (len(dominant), 1))
    colors[dominant] = [80, 215, 105]
    return colors


def _set_fixed_view(canvas: PointCloudCanvas, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    azimuth, elevation = _VIEW_ANGLES[view]
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _fit_robust(canvas: PointCloudCanvas, summary: dict) -> None:
    envelope = summary["spatial"]["core_anchor_envelope"]
    lower = np.asarray(envelope["lower"], dtype=np.float64)
    upper = np.asarray(envelope["upper"], dtype=np.float64)
    center = (lower + upper) * 0.5
    diagonal = max(float(np.linalg.norm(upper - lower)), 0.5)
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.center = tuple(float(v) for v in center)
    camera.distance = max(diagonal * 1.35, 1.5)
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def main() -> int:
    args = _args()
    directory = args.i2_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_summary.json").read_text(encoding="utf-8"))
    automated_pass = bool(summary.get("automated_pass"))
    manual_review_required = bool(summary.get("manual_visual_review_required"))

    if not automated_pass and not manual_review_required:
        raise RuntimeError(
            "I2 preview requires passing automated quality gates "
            "or an explicit manual visual review requirement."
        )

    if not automated_pass:
        print(
            "WARNING: I2 automated quality gates are DEGRADED. "
            "Showing diagnostic preview for manual visual review only."
        )
    diagnostic_path = Path(str(summary["artifacts"]["spatial_diagnostics_npz"])).expanduser().resolve()
    with np.load(diagnostic_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        support = np.asarray(payload["view_observation_counts"], dtype=np.float64)
        confidence = np.asarray(payload["filter_confidence"], dtype=np.float64)
        dispersion = np.asarray(payload["p90_dispersion_voxels"], dtype=np.float64)
        is_core = np.asarray(payload["is_core"], dtype=bool)
        dominant = np.asarray(payload["is_dominant_component"], dtype=bool)
        peripheral = np.asarray(payload["is_peripheral"], dtype=bool)
        far = np.asarray(payload["is_far_peripheral"], dtype=bool)
        weak = np.asarray(payload["is_weak_peripheral"], dtype=bool)
        sector = np.asarray(payload["directional_sector_code"], dtype=np.uint8)

    mask = np.ones(len(points), dtype=bool) if args.representation == "extended" else is_core
    points = points[mask]
    rgb = rgb[mask]
    support = support[mask]
    confidence = confidence[mask]
    dispersion = dispersion[mask]
    dominant = dominant[mask]
    peripheral = peripheral[mask]
    far = far[mask]
    weak = weak[mask]
    sector = sector[mask]
    if len(points) == 0:
        raise RuntimeError(f"I2 {args.representation} preview is empty.")

    if args.color_mode == "rgb":
        colors = rgb
    elif args.color_mode == "region":
        colors = _region_colors(peripheral, far, weak)
    elif args.color_mode == "direction":
        colors = _direction_colors(sector)
    elif args.color_mode == "support":
        colors = _scalar_colors(support)
    elif args.color_mode == "confidence":
        colors = _scalar_colors(confidence)
    elif args.color_mode == "dispersion":
        colors = _scalar_colors(dispersion, invert=True)
    else:
        colors = _component_colors(dominant)

    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — I2 Full-Video Dense RGB Quality Review")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(
        QLabel(
            f"I2 — full-video quality review | {args.representation} | {points.shape[0]:,} rendered | "
            f"color={args.color_mode} | view={args.view} | fit={args.fit} | arbitrary SfM gauge | metric scale NOT claimed"
        )
    )
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    if args.fit == "full":
        canvas.fit_scene()
    else:
        _fit_robust(canvas, summary)
    _set_fixed_view(canvas, args.view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents()
        time.sleep(0.02)

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else directory / f"i2_{args.representation}_{args.color_mode}_{args.view}_{args.fit}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save I2 preview: {output}")
    print(f"I2 fixed-view preview: {output}")
    print(
        f"representation={args.representation} rendered_points={points.shape[0]} color_mode={args.color_mode} "
        f"view={args.view} fit={args.fit}"
    )
    print("I2 diagnostics only: no geometry changed; metric/measurement/mesh authority remains unpromoted.")
    if args.show:
        code = app.exec()
        canvas.close()
        window.close()
        return int(code)
    canvas.close()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
