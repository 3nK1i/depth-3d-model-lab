#!/usr/bin/env python3
"""Fixed-view preview for I2.4 support-diversity / parallax diagnostics."""
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
    parser.add_argument("--i2-4-dir", type=Path, default=Path("/tmp/eia_i2_4_support_diversity_parallax_diagnosis"))
    parser.add_argument("--representation", choices=("extended", "core"), default="extended")
    parser.add_argument(
        "--color-mode",
        choices=("diversity", "parallax", "baseline-range", "rank-span", "support", "rgb"),
        default="diversity",
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


def _scalar_colors(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(data)
    normalized = np.zeros(len(data), dtype=np.float64)
    if np.any(finite):
        lo = float(np.quantile(data[finite], 0.05))
        hi = float(np.quantile(data[finite], 0.95))
        if hi <= lo:
            hi = lo + 1.0
        normalized[finite] = np.clip((data[finite] - lo) / (hi - lo), 0.0, 1.0)
    r = np.clip(1.3 - 1.5 * normalized, 0.0, 1.0)
    g = np.clip(1.6 * normalized, 0.0, 1.0)
    b = np.clip(0.4 + 0.8 * normalized, 0.0, 1.0)
    colors = np.column_stack([r, g, b])
    colors[~finite] = [0.45, 0.45, 0.45]
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def _diversity_colors(
    *,
    peripheral: np.ndarray,
    far: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    colors = np.tile(np.asarray([[75, 200, 105]], dtype=np.uint8), (len(peripheral), 1))
    colors[low & ~peripheral] = [75, 125, 230]
    colors[peripheral & ~low] = [235, 165, 55]
    colors[peripheral & low] = [235, 70, 70]
    colors[far & low] = [225, 65, 215]
    return colors


def _set_view(canvas: PointCloudCanvas, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        return
    camera.azimuth, camera.elevation = map(float, _VIEW_ANGLES[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _fit_robust(canvas: PointCloudCanvas, i2_summary: dict) -> None:
    envelope = i2_summary["spatial"]["core_anchor_envelope"]
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
    directory = args.i2_4_dir.expanduser().resolve()
    summary = json.loads((directory / "i2_4_summary.json").read_text(encoding="utf-8"))
    if str(summary.get("status")) != "diagnosis_ready":
        raise RuntimeError("I2.4 preview requires completed diagnosis.")
    path = Path(str(summary["artifacts"]["support_diversity_npz"])).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        support = np.asarray(payload["baseline_support"], dtype=np.float64)
        is_core = np.asarray(payload["is_core"], dtype=bool)
        peripheral = np.asarray(payload["is_peripheral"], dtype=bool)
        far = np.asarray(payload["is_far_peripheral"], dtype=bool)
        low = np.asarray(payload["is_low_support_diversity"], dtype=bool)
        angle = np.asarray(payload["maximum_pair_parallax_degrees"], dtype=np.float64)
        baseline_ratio = np.asarray(payload["maximum_baseline_to_range_ratio"], dtype=np.float64)
        rank_span = np.asarray(payload["supporting_rank_span"], dtype=np.float64)

    mask = np.ones(len(points), dtype=bool) if args.representation == "extended" else is_core
    points = points[mask]
    rgb = rgb[mask]
    support = support[mask]
    peripheral = peripheral[mask]
    far = far[mask]
    low = low[mask]
    angle = angle[mask]
    baseline_ratio = baseline_ratio[mask]
    rank_span = rank_span[mask]
    if len(points) == 0:
        raise RuntimeError("I2.4 selected representation is empty.")

    if args.color_mode == "rgb":
        colors = rgb
        legend = "RGB"
    elif args.color_mode == "diversity":
        colors = _diversity_colors(peripheral=peripheral, far=far, low=low)
        legend = "green=adequate central | blue=low-div central | orange=peripheral diverse | red=peripheral low-div | magenta=far low-div"
    elif args.color_mode == "parallax":
        colors = _scalar_colors(angle)
        legend = "red=lower parallax | green/blue=higher parallax"
    elif args.color_mode == "baseline-range":
        colors = _scalar_colors(baseline_ratio)
        legend = "red=lower baseline/range | green/blue=higher baseline/range"
    elif args.color_mode == "rank-span":
        colors = _scalar_colors(rank_span)
        legend = "red=narrow selected-view span | green/blue=wider span"
    else:
        colors = _scalar_colors(support)
        legend = "red=lower support | green/blue=higher support"

    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("EIA TECH — I2.4 Support Diversity / Parallax Review")
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(QLabel(
        f"I2.4 | {args.representation} | {len(points):,} rendered | color={args.color_mode} | view={args.view} | "
        f"fit={args.fit} | {legend} | diagnostic only"
    ))
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    if args.fit == "full":
        canvas.fit_scene()
    else:
        i2_summary = json.loads((Path(summary["source_i2_dir"]) / "i2_summary.json").read_text(encoding="utf-8"))
        _fit_robust(canvas, i2_summary)
    _set_view(canvas, args.view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents(); time.sleep(0.02)

    output = args.output.expanduser().resolve() if args.output else directory / (
        f"i2_4_{args.representation}_{args.color_mode}_{args.view}_{args.fit}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); raise RuntimeError(f"Unable to save I2.4 preview: {output}")
    print(f"I2.4 support-diversity preview: {output}")
    print(
        f"representation={args.representation} rendered_points={len(points)} color_mode={args.color_mode} "
        f"view={args.view} fit={args.fit}"
    )
    print("I2.4 diagnostics only: no geometry/filter policy changed; metric/measurement/mesh authority remains unpromoted.")
    if args.show:
        code = app.exec(); canvas.close(); window.close(); return int(code)
    canvas.close(); window.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
