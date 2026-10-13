#!/usr/bin/env python3
"""I4.1.9A A3: render a controlled fixed-view DA3-vs-MVS House visual A/B."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_visual_ab import (  # noqa: E402
    MVS_VISUAL_AB_DEFAULT_POINT_BUDGET,
    MVS_VISUAL_AB_VIEWS,
    build_visual_ab_summary,
    descriptive_cloud_stats,
    load_da3_cloud,
    load_visual_ab_sources,
    read_colmap_fused_ply,
    require_clean_output,
    sha256_file,
    write_json,
)

_DEFAULT_DA3 = Path("/tmp/eia_i4_1_9_house_visual_acceptance_02_cap3m")
_DEFAULT_A2 = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a2_mvs_attempt01")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_1_9a_house_mvs_positive_control/a3_visual_ab")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--da3-dir", type=Path, default=_DEFAULT_DA3)
    parser.add_argument("--a2-dir", type=Path, default=_DEFAULT_A2)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--point-budget", type=int, default=MVS_VISUAL_AB_DEFAULT_POINT_BUDGET)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument(
        "--views",
        default="oblique,front,side,top",
        help="Comma-separated fixed views from: " + ",".join(MVS_VISUAL_AB_VIEWS),
    )
    parser.add_argument("--show", action="store_true", help="Keep the final A/B preview window open.")
    args = parser.parse_args()
    if args.point_budget <= 0 or args.point_size <= 0.0:
        parser.error("point budget and point size must be positive")
    views = [value.strip().lower() for value in str(args.views).split(",") if value.strip()]
    if not views or len(set(views)) != len(views) or any(value not in MVS_VISUAL_AB_VIEWS for value in views):
        parser.error("--views must contain unique names from: " + ",".join(MVS_VISUAL_AB_VIEWS))
    args.views = views
    return args


def _apply_shared_camera(canvas: PointCloudCanvas, *, center_viewer: tuple[float, float, float], distance: float, view: str) -> None:
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        raise RuntimeError("A3 requires PointCloudCanvas camera access for matched framing.")
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = float(distance)
    camera.fov = 45.0
    camera.azimuth, camera.elevation = map(float, MVS_VISUAL_AB_VIEWS[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()


def _render_one(
    *,
    app: QApplication,
    points: np.ndarray,
    colors: np.ndarray,
    method: str,
    view: str,
    output: Path,
    center_viewer: tuple[float, float, float],
    distance: float,
    point_size: float,
    source_count: int,
) -> None:
    window = QMainWindow()
    window.setWindowTitle(f"Depth & 3D Model Lab — I4.1.9A A3 {method} {view}")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"A3 CONTROLLED VISUAL A/B — {method} | view={view} | "
        f"rendered={len(points):,}/{source_count:,} | SAME SfM FRAME + SAME CAMERA | arbitrary scale | review only"
    )
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    _apply_shared_camera(canvas, center_viewer=center_viewer, distance=distance, view=view)
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(18):
        app.processEvents()
        time.sleep(0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close()
        window.close()
        raise RuntimeError(f"Unable to save A3 preview: {output}")
    canvas.close()
    window.close()
    app.processEvents()


def _combine(left: Path, right: Path, output: Path) -> None:
    left_image = QImage(str(left))
    right_image = QImage(str(right))
    if left_image.isNull() or right_image.isNull():
        raise RuntimeError("Unable to load A3 rendered images for contact sheet.")
    if left_image.height() != right_image.height():
        raise RuntimeError("A3 matched renders must have identical heights.")
    combined = QImage(
        left_image.width() + right_image.width(),
        left_image.height(),
        QImage.Format.Format_RGB32,
    )
    combined.fill(Qt.GlobalColor.black)
    painter = QPainter(combined)
    painter.drawImage(0, 0, left_image)
    painter.drawImage(left_image.width(), 0, right_image)
    painter.end()
    if not combined.save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save A3 A/B contact sheet: {output}")


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    try:
        require_clean_output(output)
        sources = load_visual_ab_sources(args.da3_dir, args.a2_dir)
        da3_points, da3_colors = load_da3_cloud(sources.da3_diagnostics_npz)
        mvs_points, mvs_colors = read_colmap_fused_ply(sources.mvs_fused_ply)
    except ReconstructionValidationError as exc:
        print(f"I4.1.9A A3 Visual A/B: FAIL — {exc}", file=sys.stderr)
        return 3

    da3_stats = descriptive_cloud_stats(da3_points, sources.shared_frame)
    mvs_stats = descriptive_cloud_stats(mvs_points, sources.shared_frame)
    da3_draw, da3_draw_colors = enforce_point_budget(da3_points, da3_colors, int(args.point_budget), seed=0)
    mvs_draw, mvs_draw_colors = enforce_point_budget(mvs_points, mvs_colors, int(args.point_budget), seed=0)

    output.mkdir(parents=True, exist_ok=True)
    renders = output / "renders"
    renders.mkdir()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    artifacts: dict[str, object] = {"renders": {}}
    final_window: QMainWindow | None = None
    for view in args.views:
        da3_path = renders / f"a3_{view}_da3.png"
        mvs_path = renders / f"a3_{view}_mvs.png"
        ab_path = renders / f"a3_{view}_ab.png"
        _render_one(
            app=app,
            points=da3_draw,
            colors=da3_draw_colors,
            method="LEFT: DA3 I4.1.9",
            view=view,
            output=da3_path,
            center_viewer=sources.shared_frame.center_viewer,
            distance=sources.shared_frame.camera_distance,
            point_size=args.point_size,
            source_count=len(da3_points),
        )
        _render_one(
            app=app,
            points=mvs_draw,
            colors=mvs_draw_colors,
            method="RIGHT: COLMAP MVS A2",
            view=view,
            output=mvs_path,
            center_viewer=sources.shared_frame.center_viewer,
            distance=sources.shared_frame.camera_distance,
            point_size=args.point_size,
            source_count=len(mvs_points),
        )
        _combine(da3_path, mvs_path, ab_path)
        artifacts["renders"][view] = {
            "da3": str(da3_path),
            "mvs": str(mvs_path),
            "ab": str(ab_path),
            "sha256": {
                "da3": sha256_file(da3_path),
                "mvs": sha256_file(mvs_path),
                "ab": sha256_file(ab_path),
            },
        }
        print(f"A3 {view}: {ab_path}")

    summary = build_visual_ab_summary(
        sources=sources,
        da3_stats=da3_stats,
        mvs_stats=mvs_stats,
        rendered_views=list(args.views),
        artifacts=artifacts,
        point_budget=int(args.point_budget),
    )
    summary_path = write_json(output / "i4_1_9a_a3_summary.json", summary)
    print("I4.1.9A A3 Controlled Visual A/B: READY FOR MANUAL REVIEW")
    print(f"DA3_points={len(da3_points)} MVS_points={len(mvs_points)} rendered_per_side<={args.point_budget}")
    print("shared_camera=True same_authoritative_sfm_frame=True independent_auto_fit=False")
    print("automatic_winner_selected=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"summary={summary_path}")

    if args.show:
        # Re-open the primary oblique contact sheet at a screen-friendly scale for review.
        primary = Path(artifacts["renders"][args.views[0]]["ab"])
        image = QImage(str(primary))
        final_window = QMainWindow()
        final_window.setWindowTitle("Depth & 3D Model Lab — I4.1.9A A3 Controlled DA3 vs MVS")
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        from PyQt6.QtGui import QPixmap
        label.setPixmap(QPixmap.fromImage(image).scaled(1320, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        final_window.setCentralWidget(label)
        final_window.resize(1366, 768)
        final_window.show()
        return int(app.exec())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
