#!/usr/bin/env python3
"""Interactive preview of E1 voxel-fused qualified ten-view geometry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget  # noqa: E402
from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e1-dir", type=Path, default=Path("/tmp/eia_e1_qualified_ten_view_voxel_fusion"))
    parser.add_argument("--cloud", choices=("supported", "all", "high"), default="supported")
    parser.add_argument("--color-mode", choices=("rgb", "support"), default="rgb")
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
            [105, 105, 105],
            [230, 190, 55],
            [95, 205, 90],
            [65, 205, 185],
            [70, 160, 235],
            [105, 115, 245],
            [165, 100, 240],
            [220, 95, 205],
            [245, 110, 135],
            [250, 235, 220],
        ],
        dtype=np.uint8,
    )
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 1) or np.any(values > 10):
        raise ValueError("E1 support counts must contain values in [1, 10].")
    return palette[values - 1]


def main() -> int:
    args = _args()
    directory = args.e1_dir.expanduser().resolve()
    summary = json.loads((directory / "e1_summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise RuntimeError("E1 preview requires a passing fusion result.")
    key = {"all": "all_npz", "supported": "supported_npz", "high": "high_support_npz"}[args.cloud]
    cloud_path = Path(summary["artifacts"][key]).expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        sparse = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
    if len(points) == 0:
        raise RuntimeError(f"E1 {args.cloud} cloud is empty.")
    colors = rgb if args.color_mode == "rgb" else _support_colors(counts)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        points = np.vstack([points, sparse])
        colors = np.vstack(
            [colors, np.tile(np.asarray([[245, 245, 245]], dtype=np.uint8), (sparse.shape[0], 1))]
        )

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("EIA TECH — E1 Qualified Ten-View Voxel Fusion")
    central = QWidget()
    layout = QVBoxLayout(central)
    voxel = float(summary["policy"]["voxel_size_reconstruction_units"])
    support_gate = int(summary["policy"]["minimum_supported_views_for_primary_output"])
    layout.addWidget(
        QLabel(
            f"E1 — 10 qualified views | VOXEL FUSION | cloud={args.cloud} | "
            f"{points.shape[0]:,} rendered | color={args.color_mode} | voxel={voxel:.4f} | "
            f"support≥{support_gate} | arbitrary SfM gauge | metric scale NOT claimed"
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

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else directory / f"e1_{args.cloud}_{args.color_mode}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save E1 preview: {output}")
    print(f"E1 existing-PointCloudCanvas preview: {output}")
    print(
        f"orders={'->'.join(map(str, orders.tolist()))} cloud={args.cloud} "
        f"rendered_points={points.shape[0]} color_mode={args.color_mode} overlay_sparse={bool(args.overlay_sparse)}"
    )
    print("Real voxel fusion + support filtering applied; persistence/metric/measurement authority remains unpromoted.")
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
