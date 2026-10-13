#!/usr/bin/env python3
"""Real-VisPy preview of the Phase C single-frame canonical XYZRGB cloud.

This validation uses the existing PointCloudCanvas used by Live 3D / Snapshot /
Stable Video. It does not integrate the cloud into the Reconstruction page yet;
that first-class UI integration remains Phase H.
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
        default=Path("/tmp/eia_c_single_frame_projection/c_canonical_single_frame_cloud.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/eia_c_single_frame_projection/c_canonical_preview.png"),
    )
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--overlay-sparse",
        action="store_true",
        help="Append selected-keyframe sparse SfM landmarks in bright green for visual context.",
    )
    parser.add_argument("--show", action="store_true", help="Keep the interactive preview open.")
    args = parser.parse_args()
    if args.point_budget <= 0:
        parser.error("--point-budget must be positive.")
    if args.point_size <= 0:
        parser.error("--point-size must be positive.")
    return args


def main() -> int:
    args = _parse_args()
    cloud_path = args.cloud.expanduser().resolve()
    with np.load(cloud_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        colors = np.asarray(payload["colors_rgb"], dtype=np.uint8)
        sparse = np.asarray(
            payload.get("visible_sparse_landmarks_reconstruction", np.empty((0, 3))),
            dtype=np.float64,
        )
    points, colors = enforce_point_budget(points, colors, int(args.point_budget), seed=0)
    if args.overlay_sparse and sparse.size:
        sparse_colors = np.tile(np.asarray([[60, 255, 110]], dtype=np.uint8), (sparse.shape[0], 1))
        points = np.vstack([points, sparse])
        colors = np.vstack([colors, sparse_colors])

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = QMainWindow()
    window.setWindowTitle("Depth & 3D Model Lab — Phase C Canonical Dense RGB Validation")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"Phase C — single registered keyframe | {points.shape[0]:,} rendered points | "
        "arbitrary SfM gauge | metric scale NOT claimed"
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
        raise RuntimeError(f"Unable to save Phase C preview screenshot: {output}")
    print(f"Phase C existing-PointCloudCanvas preview: {output}")
    print(f"rendered_points={points.shape[0]} overlay_sparse={bool(args.overlay_sparse)}")
    print("Validation-only single-frame geometry; canonical persistence/authority not promoted.")

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
