#!/usr/bin/env python3
"""Interactive preview of D3 raw five-view canonical accumulation.

Uses the existing PointCloudCanvas renderer. RGB mode shows real colors; source
mode assigns one diagnostic color to each contributing view so overlap and
possible duplicated surfaces remain visible before any fusion is introduced.
"""

from __future__ import annotations

import argparse
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cloud",
        type=Path,
        default=Path("/tmp/eia_d3_five_view_accumulation/d3_five_view_raw_accumulation.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/eia_d3_five_view_accumulation/d3_five_view_preview.png"),
    )
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--color-mode", choices=("rgb", "source"), default="rgb")
    parser.add_argument("--overlay-sparse", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0:
        parser.error("--point-budget must be positive.")
    if args.point_size <= 0:
        parser.error("--point-size must be positive.")
    return args


def _source_colors(source_view_index: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [255, 85, 85],
            [255, 180, 65],
            [80, 235, 120],
            [80, 180, 255],
            [190, 110, 255],
        ],
        dtype=np.uint8,
    )
    values = np.asarray(source_view_index)
    if values.ndim != 1 or np.any(values < 0) or np.any(values >= palette.shape[0]):
        raise ValueError("D3 source_view_index must be a 1D array containing only 0..4.")
    return palette[values.astype(np.int64)]


def main() -> int:
    args = _parse_args()
    cloud_path = args.cloud.expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        rgb = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        source_view_index = np.asarray(payload["source_view_index"], dtype=np.uint8)
        sparse = np.asarray(
            payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
        )
    colors = rgb if args.color_mode == "rgb" else _source_colors(source_view_index)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        sparse_colors = np.tile(np.asarray([[255, 255, 255]], dtype=np.uint8), (sparse.shape[0], 1))
        points = np.vstack([points, sparse])
        colors = np.vstack([colors, sparse_colors])

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("EIA TECH — D3 Five-View Canonical Accumulation")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"D3 — 5 overlap-aware raw registered views | {points.shape[0]:,} rendered points | "
        f"color={args.color_mode} | NO fusion | arbitrary SfM gauge"
    )
    layout.addWidget(label)
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

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save D3 preview screenshot: {output}")
    print(f"D3 existing-PointCloudCanvas preview: {output}")
    print(
        f"rendered_points={points.shape[0]} color_mode={args.color_mode} "
        f"overlay_sparse={bool(args.overlay_sparse)}"
    )
    print("Raw five-view accumulation only: no voxel fusion, no persistence, no authority promotion.")

    if args.show:
        exit_code = app.exec()
        canvas.close()
        window.close()
        return int(exit_code)
    canvas.close()
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
