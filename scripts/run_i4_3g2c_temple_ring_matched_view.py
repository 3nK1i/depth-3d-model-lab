#!/usr/bin/env python3
"""Render I4.3G2C blinded fixed-camera TempleRing MVS-vs-DA3 review artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)

from app.widgets.viewport.vispy_canvas import PointCloudCanvas  # noqa: E402
from core.geometry.sampling import enforce_point_budget  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.mvs_visual_ab import read_colmap_fused_ply  # noqa: E402
from core.reconstruction.mvs_da3_matched_view import (  # noqa: E402
    MATCHED_VIEW_FRAMINGS,
    MATCHED_VIEW_VIEWS,
    load_matched_view_sources,
    sha256_file,
)
from core.reconstruction.mvs_da3_g2c_review import (  # noqa: E402
    G2C_PHASE,
    G2C_RENDER_HEIGHT,
    G2C_RENDER_WIDTH,
    blinded_render_label,
    build_blind_plan,
    build_manual_review_template,
    canonical_render_policy,
    resolve_equal_render_point_count,
    verify_g2b_closure_for_g2c,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g2b-closure", type=Path, required=True)
    parser.add_argument("--g2b-closure-sha256", required=True)
    parser.add_argument("--production-snapshot", type=Path, required=True)
    parser.add_argument("--production-sha256", required=True)
    parser.add_argument("--production-verdict", type=Path, required=True)
    parser.add_argument("--verdict-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-budget", type=int, default=250_000)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--views", default="oblique,front,side,top")
    parser.add_argument("--framings", default="detail,context")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    args.views = [v.strip() for v in args.views.split(",") if v.strip()]
    args.framings = [v.strip() for v in args.framings.split(",") if v.strip()]
    if not args.views or any(v not in MATCHED_VIEW_VIEWS for v in args.views):
        parser.error("--views contains an unsupported fixed view")
    if not args.framings or any(v not in MATCHED_VIEW_FRAMINGS for v in args.framings):
        parser.error("--framings contains an unsupported shared framing")
    if len(set(args.views)) != len(args.views) or len(set(args.framings)) != len(args.framings):
        parser.error("views/framings must be unique")
    if args.point_budget <= 0 or args.point_size <= 0:
        parser.error("point budget and point size must be positive")
    return args


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _render(
    *, app: QApplication, points: np.ndarray, colors: np.ndarray, label_text: str,
    view: str, framing: str, output: Path, center_viewer: tuple[float, float, float],
    distance: float, point_size: float,
) -> None:
    window = QMainWindow()
    window.setWindowTitle(f"EIA TECH — I4.3G2C matched view {label_text} {view} {framing}")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        blinded_render_label(
            blind_label=label_text,
            view=view,
            framing=framing,
        )
    )
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        raise RuntimeError("G2C matched-view rendering requires PointCloudCanvas camera access.")
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = float(distance)
    camera.fov = 45.0
    camera.azimuth, camera.elevation = map(float, MATCHED_VIEW_VIEWS[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    # QWidget/QMainWindow resize() is only a request.  Repeated native VisPy
    # windows can occasionally be granted slightly different client extents by
    # the window manager.  That leaked into _combine() as unequal image heights.
    # Freeze the logical window size and normalize every successful capture to
    # the same reviewer-facing raster before constructing an A/B pair.
    window.setFixedSize(G2C_RENDER_WIDTH, G2C_RENDER_HEIGHT)
    window.show()
    for _ in range(18):
        app.processEvents()
        time.sleep(0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    captured = window.grab().toImage()
    if captured.isNull() or captured.width() <= 0 or captured.height() <= 0:
        canvas.close(); window.close()
        raise RuntimeError(
            f"Unable to capture G2C matched-view render: {output}"
        )
    if (captured.width(), captured.height()) != (G2C_RENDER_WIDTH, G2C_RENDER_HEIGHT):
        captured = captured.scaled(
            G2C_RENDER_WIDTH,
            G2C_RENDER_HEIGHT,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    if (captured.width(), captured.height()) != (G2C_RENDER_WIDTH, G2C_RENDER_HEIGHT):
        canvas.close(); window.close()
        raise RuntimeError(
            "Unable to normalize G2C matched-view render to canonical extent: "
            f"got={captured.width()}x{captured.height()} "
            f"expected={G2C_RENDER_WIDTH}x{G2C_RENDER_HEIGHT}"
        )
    if not captured.save(str(output), "PNG"):
        canvas.close(); window.close()
        raise RuntimeError(f"Unable to save G2C matched-view render: {output}")
    canvas.close(); window.close(); app.processEvents()


def _combine(left: Path, right: Path, output: Path) -> None:
    a = QImage(str(left)); b = QImage(str(right))
    if a.isNull() or b.isNull():
        raise RuntimeError(
            "G2C matched-view rendered image pair is unreadable: "
            f"cloud_a_null={a.isNull()} cloud_b_null={b.isNull()}"
        )
    expected = (G2C_RENDER_WIDTH, G2C_RENDER_HEIGHT)
    a_size = (a.width(), a.height())
    b_size = (b.width(), b.height())
    if a_size != expected or b_size != expected:
        raise RuntimeError(
            "G2C matched-view rendered image pair extent mismatch: "
            f"cloud_a={a_size[0]}x{a_size[1]} "
            f"cloud_b={b_size[0]}x{b_size[1]} "
            f"expected={expected[0]}x{expected[1]}"
        )
    combined = QImage(a.width() + b.width(), a.height(), QImage.Format.Format_RGB32)
    combined.fill(Qt.GlobalColor.black)
    painter = QPainter(combined)
    painter.drawImage(0, 0, a); painter.drawImage(a.width(), 0, b); painter.end()
    if not combined.save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save G2C matched-view contact sheet: {output}")


def main() -> int:
    args = _args()
    out = args.output_dir.expanduser().resolve()
    try:
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"G2C matched-view output must be empty or absent: {out}")

        verify_g2b_closure_for_g2c(
            args.g2b_closure,
            expected_closure_sha256=args.g2b_closure_sha256,
            expected_temple_ring_verdict_sha256=args.verdict_sha256,
        )
        sources = load_matched_view_sources(
            args.production_snapshot,
            args.production_verdict,
            expected_production_sha256=args.production_sha256,
            expected_verdict_sha256=args.verdict_sha256,
        )

        mvs_points, mvs_colors = read_colmap_fused_ply(sources.mvs_ply)
        da3_points, da3_colors = read_colmap_fused_ply(sources.da3_ply)
        equal_draw_count = resolve_equal_render_point_count(
            mvs_source_count=len(mvs_points),
            da3_source_count=len(da3_points),
            requested_point_budget=args.point_budget,
        )
        mvs_draw, mvs_draw_colors = enforce_point_budget(
            mvs_points, mvs_colors, equal_draw_count, seed=0
        )
        da3_draw, da3_draw_colors = enforce_point_budget(
            da3_points, da3_colors, equal_draw_count, seed=0
        )
        if len(mvs_draw) != equal_draw_count or len(da3_draw) != equal_draw_count:
            raise ReconstructionValidationError(
                "G2C matched-view equal-density render contract was not satisfied."
            )

        blind_plan = build_blind_plan(
            sources.production_snapshot_sha256,
            scenario_id=sources.scenario_id,
        )
        provider_clouds = {
            "colmap_mvs": (mvs_draw, mvs_draw_colors, len(mvs_points)),
            "da3": (da3_draw, da3_draw_colors, len(da3_points)),
        }

        out.mkdir(parents=True, exist_ok=False)
        renders = out / "renders"
        renders.mkdir()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        artifacts: dict[str, Any] = {}

        for framing in args.framings:
            for view in args.views:
                pair: dict[str, Path] = {}
                for blind_label in ("A", "B"):
                    provider = blind_plan.assignment[blind_label]
                    pts, rgb, _source_count = provider_clouds[provider]
                    target = renders / f"{framing}_{view}_cloud_{blind_label.lower()}.png"
                    _render(
                        app=app,
                        points=pts,
                        colors=rgb,
                        label_text=blind_label,
                        view=view,
                        framing=framing,
                        output=target,
                        center_viewer=sources.shared_frame.center_viewer,
                        distance=sources.shared_frame.camera_distance(framing),
                        point_size=args.point_size,
                    )
                    pair[blind_label] = target

                ab = renders / f"{framing}_{view}_ab.png"
                _combine(pair["A"], pair["B"], ab)
                artifacts[f"{framing}:{view}"] = {
                    "cloud_a": {"path": str(pair["A"]), "sha256": sha256_file(pair["A"])},
                    "cloud_b": {"path": str(pair["B"]), "sha256": sha256_file(pair["B"])},
                    "ab": {"path": str(ab), "sha256": sha256_file(ab)},
                }
                print(f"g2c_matched_view {framing}/{view}: {ab}")

        commitment_path = _write_json(
            out / "blind_assignment_commitment.json",
            blind_plan.public_dict(),
        )
        review_path = _write_json(
            out / "manual_review.json",
            build_manual_review_template(
                scenario_id=sources.scenario_id,
                views=args.views,
                framings=args.framings,
                assignment_commitment_sha256=blind_plan.commitment_sha256,
            ),
        )

        summary = {
            "schema_version": 1,
            "phase": G2C_PHASE,
            "scenario_id": sources.scenario_id,
            "comparison_mode": "matched_view_diagnostic",
            "status": "ready_for_blinded_manual_review",
            "pass": True,
            "g2b_closure": {
                "path": str(args.g2b_closure.expanduser().resolve()),
                "sha256": str(args.g2b_closure_sha256).lower(),
                "g2c_authorized": True,
            },
            "production_snapshot": {
                "path": str(sources.production_snapshot),
                "sha256": sources.production_snapshot_sha256,
            },
            "production_verdict": {
                "path": str(sources.production_verdict),
                "sha256": sources.production_verdict_sha256,
            },
            "authoritative_sfm": {
                "path": str(sources.sfm_snapshot),
                "sha256": sources.sfm_snapshot_sha256,
                "frame_id": sources.frame_id,
            },
            "provider_artifacts": {
                "colmap_mvs_ply": {
                    "path": str(sources.mvs_ply),
                    "sha256": sources.mvs_ply_sha256,
                    "source_points": len(mvs_points),
                },
                "da3_ply": {
                    "path": str(sources.da3_ply),
                    "sha256": sources.da3_ply_sha256,
                    "source_points": len(da3_points),
                },
            },
            "render_policy": {
                "shared_visual_frame": sources.shared_frame.as_dict(),
                "views": list(args.views),
                "framings": list(args.framings),
                "requested_point_budget_per_cloud": int(args.point_budget),
                "equal_rendered_point_count_per_cloud": int(equal_draw_count),
                "equal_render_density_enforced": True,
                "reviewer_labels_expose_source_point_counts": False,
                "point_size": float(args.point_size),
                "independent_auto_fit": False,
                "blinded_labels": True,
                "canonical_render_raster": canonical_render_policy(),
            },
            "artifacts": artifacts,
            "blind_assignment": {
                "mapping_written_to_review_package": False,
                "commitment_path": str(commitment_path),
                "commitment_sha256": sha256_file(commitment_path),
                "assignment_commitment_sha256": blind_plan.commitment_sha256,
                "reveal_only_after_completed_manual_review": True,
            },
            "manual_review": {
                "path": str(review_path),
                "sha256": sha256_file(review_path),
                "required": True,
                "completed": False,
                "blind_key_revealed": False,
            },
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "same_authoritative_sfm_shared": True,
                "authoritative_sfm_modified": False,
                "provider_results_modified": False,
                "provider_selected": False,
                "provider_promoted": False,
                "auto_policy_implemented": False,
                "hybrid_authorized": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "permanent_global_view_cap": None,
            },
        }
        summary_path = _write_json(
            out / "i4_3g2c_temple_ring_matched_view_summary.json",
            summary,
        )

        print("I4.3G2C TempleRing matched-view: READY FOR BLINDED MANUAL REVIEW")
        print("g2b_closure_verified=True same_authoritative_sfm=True independent_auto_fit=False")
        print(
            f"equal_render_density=True equal_draw_count={equal_draw_count} "
            "reviewer_count_hints_exposed=False"
        )
        print("blind_mapping_written=False provider_selected=False metric_scale_claimed=False")
        print(f"summary={summary_path}")
        print(f"manual_review={review_path}")

        if args.show:
            primary = Path(artifacts[f"{args.framings[0]}:{args.views[0]}"]["ab"]["path"])
            image = QImage(str(primary))
            window = QMainWindow()
            window.setWindowTitle("EIA TECH — I4.3G2C TempleRing blinded matched view")
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(
                QPixmap.fromImage(image).scaled(
                    1320,
                    700,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            window.setCentralWidget(label)
            window.resize(1366, 768)
            window.show()
            return int(app.exec())
        return 0
    except Exception as exc:
        print(f"I4.3G2C matched-view ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
