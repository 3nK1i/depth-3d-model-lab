#!/usr/bin/env python3
"""Preview E3 expanded fused clouds with the existing PointCloudCanvas."""
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
    parser.add_argument("--e3-dir", type=Path, default=Path("/tmp/eia_e3_qualified_scene_fusion_expansion"))
    parser.add_argument("--cloud", choices=("all", "supported", "high"), default="supported")
    parser.add_argument("--color-mode", choices=("rgb", "support"), default="rgb")
    parser.add_argument("--point-budget", type=int, default=300_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--overlay-sparse", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0:
        parser.error("point budget and point size must be positive")
    return args


def _support_colors(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    if values.ndim != 1:
        raise RuntimeError("E3 support counts must be one-dimensional.")
    maximum = max(int(values.max()) if values.size else 1, 2)
    normalized = np.clip((values.astype(np.float64) - 1.0) / max(maximum - 1.0, 1.0), 0.0, 1.0)
    r = np.clip(1.8 * normalized, 0.0, 1.0)
    g = np.clip(0.25 + 1.4 * normalized, 0.0, 1.0)
    b = np.clip(1.0 - 1.1 * normalized, 0.0, 1.0)
    return np.clip(np.rint(np.column_stack([r, g, b]) * 255.0), 0, 255).astype(np.uint8)


def main() -> int:
    args = _args()
    directory = args.e3_dir.expanduser().resolve()
    summary = json.loads((directory / "e3_summary.json").read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise RuntimeError("E3 preview requires a passing scene-expansion result.")
    artifact_key = {"all": "all_npz", "supported": "supported_npz", "high": "high_support_npz"}[args.cloud]
    cloud_path = Path(summary["artifacts"][artifact_key]).expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        counts = np.asarray(payload["view_observation_counts"], dtype=np.uint32)
        sparse = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)
        orders = np.asarray(payload["selected_workspace_orders"], dtype=np.int32)
    if len(points) == 0:
        raise RuntimeError(f"E3 {args.cloud} cloud is empty.")
    colors = rgb if args.color_mode == "rgb" else _support_colors(counts)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        points = np.vstack([points, sparse])
        colors = np.vstack([colors, np.tile(np.asarray([[245, 245, 245]], dtype=np.uint8), (sparse.shape[0], 1))])

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — E3 Qualified Scene Fusion Expansion")
    central = QWidget()
    layout = QVBoxLayout(central)
    voxel = float(summary["fusion"]["voxel_size_reconstruction_units"])
    layout.addWidget(
        QLabel(
            f"E3 — {summary['selected_view_count']} qualified views | cloud={args.cloud} | "
            f"{points.shape[0]:,} rendered | color={args.color_mode} | voxel={voxel:.4f} | "
            "arbitrary SfM gauge | metric scale NOT claimed"
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

    output = args.output.expanduser().resolve() if args.output is not None else directory / f"e3_{args.cloud}_{args.color_mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save E3 preview: {output}")
    print(f"E3 existing-PointCloudCanvas preview: {output}")
    print(
        f"orders={'->'.join(map(str, orders.tolist()))} cloud={args.cloud} "
        f"rendered_points={points.shape[0]} color_mode={args.color_mode} overlay_sparse={bool(args.overlay_sparse)}"
    )
    print("Qualified scene expansion only; formal confidence filtering/persistence/UI/metric/mesh authority remain unpromoted.")
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
