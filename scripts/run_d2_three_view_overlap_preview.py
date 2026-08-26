#!/usr/bin/env python3
"""Interactive D2 pairwise overlap diagnostic using the existing PointCloudCanvas.

Residual mode colors reprojected points by geometric interpretation:
  green  = consistent with the other view
  red    = source surface lies significantly in front of the other view
  amber  = moderate back-surface disagreement
  gray   = source point is behind a nearer target surface (occluded)

Source mode colors the two directed point sets separately.  This is diagnostic
visualization only; it does not alter, fuse, persist, or promote geometry.
"""

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d2-dir", type=Path, default=Path("/tmp/eia_d2_overlap_diagnostics")
    )
    parser.add_argument("--pair", choices=("0-1", "1-2", "0-2"), default="0-1")
    parser.add_argument("--color-mode", choices=("residual", "source"), default="residual")
    parser.add_argument("--direction", choices=("both", "a-to-b", "b-to-a"), default="both")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--overlay-sparse", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0:
        parser.error("D2 preview point budget/size must be positive.")
    return args


def _residual_colors(classes: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [80, 235, 115],   # consistent
            [255, 80, 80],    # front conflict
            [255, 185, 65],   # back conflict
            [145, 145, 150],  # occluded
        ],
        dtype=np.uint8,
    )
    values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values > 3):
        raise ValueError("D2 diagnostic classifications must be integers in [0,3].")
    return palette[values]


def _source_colors(count: int, which: int) -> np.ndarray:
    palette = np.asarray([[255, 90, 90], [90, 160, 255]], dtype=np.uint8)
    return np.tile(palette[which], (count, 1))


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _read_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("D2 summary must be a JSON object.")
    return payload


def main() -> int:
    args = _parse_args()
    d2_dir = args.d2_dir.expanduser().resolve()
    pair_numbers = tuple(int(value) for value in args.pair.split("-"))
    diagnostic_path = d2_dir / f"d2_pair_{pair_numbers[0]}{pair_numbers[1]}_diagnostic.npz"
    with np.load(diagnostic_path, allow_pickle=False) as payload:
        a_points = np.asarray(payload["a_to_b_points_reconstruction"], dtype=np.float64)
        a_classes = np.asarray(payload["a_to_b_classification"], dtype=np.uint8)
        b_points = np.asarray(payload["b_to_a_points_reconstruction"], dtype=np.float64)
        b_classes = np.asarray(payload["b_to_a_classification"], dtype=np.uint8)
        sparse = np.asarray(
            payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64
        )

    points_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    if args.direction in ("both", "a-to-b"):
        points_parts.append(a_points)
        color_parts.append(_residual_colors(a_classes) if args.color_mode == "residual" else _source_colors(a_points.shape[0], 0))
    if args.direction in ("both", "b-to-a"):
        points_parts.append(b_points)
        color_parts.append(_residual_colors(b_classes) if args.color_mode == "residual" else _source_colors(b_points.shape[0], 1))
    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        sparse_colors = np.tile(np.asarray([[245, 245, 245]], dtype=np.uint8), (sparse.shape[0], 1))
        points = np.vstack([points, sparse])
        colors = np.vstack([colors, sparse_colors])

    summary = _read_summary(d2_dir / "d2_summary.json")
    pair_summary = next(item for item in summary["pairs"] if item["pair_label"] == args.pair)
    a_to_b = pair_summary["a_to_b"]
    b_to_a = pair_summary["b_to_a"]

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("EIA TECH — D2 Multi-View Overlap Diagnostics")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"D2 pair {args.pair} | {pair_summary['status'].upper()} | color={args.color_mode} | "
        f"{pair_numbers[0]}→{pair_numbers[1]} consistent={a_to_b['consistent_ratio_of_comparable']:.1%} "
        f"p90={_fmt(a_to_b['p90_absolute_relative_depth_residual'])} | "
        f"{pair_numbers[1]}→{pair_numbers[0]} consistent={b_to_a['consistent_ratio_of_comparable']:.1%} "
        f"p90={_fmt(b_to_a['p90_absolute_relative_depth_residual'])} | NO fusion"
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

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else d2_dir / f"d2_pair_{pair_numbers[0]}{pair_numbers[1]}_{args.color_mode}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save D2 preview screenshot: {output}")
    print(f"D2 existing-PointCloudCanvas preview: {output}")
    print(
        f"pair={args.pair} status={pair_summary['status']} rendered_points={points.shape[0]} "
        f"color_mode={args.color_mode} direction={args.direction} overlay_sparse={bool(args.overlay_sparse)}"
    )
    print("Diagnostic only: no fusion, no filtering mutation, no persistence, no authority promotion.")

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
