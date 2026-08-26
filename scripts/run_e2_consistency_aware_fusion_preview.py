#!/usr/bin/env python3
"""Preview E2 core/extended/coverage/rejected consistency-aware clouds."""
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


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-dir", type=Path, default=Path("/tmp/eia_e2_consistency_aware_fusion_refinement"))
    parser.add_argument("--cloud", choices=("core", "extended", "coverage", "rejected", "fringe"), default="extended")
    parser.add_argument("--color-mode", choices=("rgb", "confidence", "dispersion", "support", "class"), default="rgb")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--overlay-sparse", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0:
        parser.error("point budget and point size must be positive")
    return args


def _support_colors(counts: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [105, 105, 105], [230, 190, 55], [95, 205, 90], [65, 205, 185],
            [70, 160, 235], [105, 115, 245], [165, 100, 240], [220, 95, 205],
            [245, 110, 135], [250, 235, 220],
        ], dtype=np.uint8,
    )
    values = np.clip(np.asarray(counts, dtype=np.int64), 1, 10)
    return palette[values - 1]


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
    # dark blue -> cyan/green -> yellow/white, written directly to avoid extra plotting deps
    r = np.clip(2.0 * normalized - 0.2, 0.0, 1.0)
    g = np.clip(1.7 * normalized, 0.0, 1.0)
    b = np.clip(1.2 - 1.4 * normalized, 0.0, 1.0)
    colors = np.column_stack([r, g, b])
    colors[~finite] = np.asarray([0.4, 0.4, 0.4])
    return np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)


def _class_colors(code: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [210, 70, 70],   # rejected supported
            [130, 130, 130], # single-view fringe
            [235, 190, 55],  # extended
            [80, 215, 105],  # core
        ], dtype=np.uint8,
    )
    values = np.clip(np.asarray(code, dtype=np.int64), 0, 3)
    return palette[values]


def main() -> int:
    args = _args()
    directory = args.e2_dir.expanduser().resolve()
    summary = json.loads((directory / "e2_summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise RuntimeError("E2 preview requires a passing refinement result.")
    artifact_key = {
        "core": "core_npz",
        "extended": "extended_npz",
        "coverage": "coverage_candidate_npz",
        "rejected": "rejected_supported_npz",
        "fringe": "attached_single_view_fringe_npz",
    }[args.cloud]
    cloud_path = Path(summary["artifacts"][artifact_key]).expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        confidence = np.asarray(payload["refinement_confidence"], dtype=np.float32)
        dispersion = np.asarray(payload["p90_dispersion_voxels"], dtype=np.float32)
        class_code = np.asarray(payload["class_code"], dtype=np.uint8)
        sparse = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
    if len(points) == 0:
        raise RuntimeError(f"E2 {args.cloud} cloud is empty.")
    if args.color_mode == "rgb":
        colors = rgb
    elif args.color_mode == "confidence":
        colors = _scalar_colors(confidence)
    elif args.color_mode == "dispersion":
        colors = _scalar_colors(dispersion, invert=True)
    elif args.color_mode == "support":
        colors = _support_colors(counts)
    else:
        colors = _class_colors(class_code)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        points = np.vstack([points, sparse])
        colors = np.vstack([colors, np.tile(np.asarray([[245, 245, 245]], dtype=np.uint8), (sparse.shape[0], 1))])

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("EIA TECH — E2 Consistency-Aware Fusion Refinement")
    central = QWidget()
    layout = QVBoxLayout(central)
    voxel = float(summary["voxel_size_reconstruction_units"])
    layout.addWidget(
        QLabel(
            f"E2 — consistency-aware refinement | cloud={args.cloud} | {points.shape[0]:,} rendered | "
            f"color={args.color_mode} | voxel={voxel:.4f} | arbitrary SfM gauge | metric scale NOT claimed"
        )
    )
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(args.point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    canvas.fit_scene()
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(14):
        app.processEvents()
        time.sleep(0.02)

    output = args.output.expanduser().resolve() if args.output is not None else directory / f"e2_{args.cloud}_{args.color_mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save E2 preview: {output}")
    print(f"E2 existing-PointCloudCanvas preview: {output}")
    print(
        f"orders={'->'.join(map(str, orders.tolist()))} cloud={args.cloud} "
        f"rendered_points={points.shape[0]} color_mode={args.color_mode} overlay_sparse={bool(args.overlay_sparse)}"
    )
    print("Consistency-aware refinement only; persistence/metric/measurement/mesh authority remains unpromoted.")
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
