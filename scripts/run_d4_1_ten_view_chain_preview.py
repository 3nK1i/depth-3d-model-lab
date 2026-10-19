#!/usr/bin/env python3
"""Interactive D4.1 adjacent-pair residual preview using PointCloudCanvas."""

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
    parser.add_argument("--d4-1-dir", type=Path, default=Path("/tmp/depth3d_d4_1_ten_view_qualification"))
    parser.add_argument("--pair", choices=tuple(f"{index}-{index + 1}" for index in range(9)), default="0-1")
    parser.add_argument("--color-mode", choices=("residual", "source"), default="residual")
    parser.add_argument("--direction", choices=("both", "a-to-b", "b-to-a"), default="both")
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--overlay-sparse", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0:
        parser.error("D4.1 preview point budget/size must be positive.")
    return args


def _residual_colors(classes: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [[80, 235, 115], [255, 80, 80], [255, 185, 65], [145, 145, 150]],
        dtype=np.uint8,
    )
    values = np.asarray(classes, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0) or np.any(values > 3):
        raise ValueError("D4.1 classifications must be integers in [0,3].")
    return palette[values]


def _source_colors(count: int, which: int) -> np.ndarray:
    palette = np.asarray([[255, 90, 90], [90, 160, 255]], dtype=np.uint8)
    return np.tile(palette[which], (count, 1))


def _fmt(value: object, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def main() -> int:
    args = _parse_args()
    output_dir = args.d4_1_dir.expanduser().resolve()
    a_index, b_index = (int(value) for value in args.pair.split("-"))
    diagnostic_path = output_dir / f"d4_1_pair_{a_index}{b_index}_diagnostic.npz"
    with np.load(diagnostic_path, allow_pickle=False) as payload:
        a_points = np.asarray(payload["a_to_b_points_reconstruction"], dtype=np.float64)
        a_classes = np.asarray(payload["a_to_b_classification"], dtype=np.uint8)
        b_points = np.asarray(payload["b_to_a_points_reconstruction"], dtype=np.float64)
        b_classes = np.asarray(payload["b_to_a_classification"], dtype=np.uint8)
        sparse = np.asarray(payload.get("sparse_landmarks_reconstruction", np.empty((0, 3))), dtype=np.float64)

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

    summary = json.loads((output_dir / "d4_1_summary.json").read_text(encoding="utf-8"))
    pair_summary = next(item for item in summary["adjacent_pairs"] if item["pair_label"] == args.pair)
    a_to_b = pair_summary["a_to_b"]
    b_to_a = pair_summary["b_to_a"]
    orders = pair_summary["workspace_orders"]

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — D4.1 Ten-View Chain Qualification")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"D4.1 pair {args.pair} orders {orders[0]}↔{orders[1]} | {pair_summary['status'].upper()} | "
        f"color={args.color_mode} | {a_index}→{b_index} consistent={a_to_b['consistent_ratio_of_comparable']:.1%} "
        f"p90={_fmt(a_to_b['p90_absolute_relative_depth_residual'])} | "
        f"{b_index}→{a_index} consistent={b_to_a['consistent_ratio_of_comparable']:.1%} "
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

    output = args.output.expanduser().resolve() if args.output is not None else output_dir / f"d4_1_pair_{a_index}{b_index}_{args.color_mode}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        raise RuntimeError(f"Unable to save D4.1 preview screenshot: {output}")
    print(f"D4.1 existing-PointCloudCanvas preview: {output}")
    print(
        f"pair={args.pair} orders={orders[0]}<->{orders[1]} status={pair_summary['status']} "
        f"rendered_points={points.shape[0]} color_mode={args.color_mode} direction={args.direction} "
        f"overlay_sparse={bool(args.overlay_sparse)}"
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
