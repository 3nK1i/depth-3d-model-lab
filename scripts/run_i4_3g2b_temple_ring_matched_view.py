#!/usr/bin/env python3
"""Render blinded, fixed-camera TempleRing MVS-vs-DA3 G2B matched views."""
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
    blind_assignment,
    load_matched_view_sources,
    review_template,
    sha256_file,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    distance: float, point_size: float, source_count: int,
) -> None:
    window = QMainWindow()
    window.setWindowTitle(f"Depth & 3D Model Lab — I4.3G2B matched view {label_text} {view} {framing}")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(
        f"I4.3G2B MATCHED VIEW — CLOUD {label_text} | {view} | {framing} | "
        f"rendered={len(points):,}/{source_count:,} | SAME SfM FRAME + SAME CAMERA | blinded review"
    )
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    canvas = PointCloudCanvas(window)
    canvas.set_point_size(float(point_size))
    canvas.set_grid_visible(True)
    canvas.update_points(points, colors)
    camera = getattr(canvas, "_camera", None)
    if camera is None:
        raise RuntimeError("Matched-view rendering requires PointCloudCanvas camera access.")
    camera.center = tuple(float(v) for v in center_viewer)
    camera.distance = float(distance)
    camera.fov = 45.0
    camera.azimuth, camera.elevation = map(float, MATCHED_VIEW_VIEWS[view])
    changed = getattr(camera, "view_changed", None)
    if callable(changed):
        changed()
    layout.addWidget(canvas.native, 1)
    window.setCentralWidget(central)
    window.resize(1366, 768)
    window.show()
    for _ in range(18):
        app.processEvents()
        time.sleep(0.02)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(output), "PNG"):
        canvas.close(); window.close()
        raise RuntimeError(f"Unable to save matched-view render: {output}")
    canvas.close(); window.close(); app.processEvents()


def _combine(left: Path, right: Path, output: Path) -> None:
    a = QImage(str(left)); b = QImage(str(right))
    if a.isNull() or b.isNull() or a.height() != b.height():
        raise RuntimeError("Matched-view rendered image pair is invalid.")
    combined = QImage(a.width() + b.width(), a.height(), QImage.Format.Format_RGB32)
    combined.fill(Qt.GlobalColor.black)
    painter = QPainter(combined)
    painter.drawImage(0, 0, a); painter.drawImage(a.width(), 0, b); painter.end()
    if not combined.save(str(output), "PNG"):
        raise RuntimeError(f"Unable to save matched-view contact sheet: {output}")


def main() -> int:
    args = _args()
    out = args.output_dir.expanduser().resolve()
    try:
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"Matched-view output must be empty or absent: {out}")
        sources = load_matched_view_sources(
            args.production_snapshot, args.production_verdict,
            expected_production_sha256=args.production_sha256,
            expected_verdict_sha256=args.verdict_sha256,
        )
        mvs_points, mvs_colors = read_colmap_fused_ply(sources.mvs_ply)
        da3_points, da3_colors = read_colmap_fused_ply(sources.da3_ply)
        mvs_draw, mvs_draw_colors = enforce_point_budget(mvs_points, mvs_colors, args.point_budget, seed=0)
        da3_draw, da3_draw_colors = enforce_point_budget(da3_points, da3_colors, args.point_budget, seed=0)
        assignment = blind_assignment(sources.production_snapshot_sha256)
        provider_clouds = {
            "colmap_mvs": (mvs_draw, mvs_draw_colors, len(mvs_points)),
            "da3": (da3_draw, da3_draw_colors, len(da3_points)),
        }
        out.mkdir(parents=True, exist_ok=False)
        renders = out / "renders"; renders.mkdir()
        app = QApplication.instance() or QApplication(sys.argv[:1])
        artifacts: dict[str, Any] = {}
        for framing in args.framings:
            for view in args.views:
                pair: dict[str, Path] = {}
                for blind_label in ("A", "B"):
                    provider = assignment[blind_label]
                    pts, rgb, source_count = provider_clouds[provider]
                    target = renders / f"{framing}_{view}_cloud_{blind_label.lower()}.png"
                    _render(
                        app=app, points=pts, colors=rgb, label_text=blind_label,
                        view=view, framing=framing, output=target,
                        center_viewer=sources.shared_frame.center_viewer,
                        distance=sources.shared_frame.camera_distance(framing),
                        point_size=args.point_size, source_count=source_count,
                    )
                    pair[blind_label] = target
                ab = renders / f"{framing}_{view}_ab.png"
                _combine(pair["A"], pair["B"], ab)
                artifacts[f"{framing}:{view}"] = {
                    "cloud_a": {"path": str(pair["A"]), "sha256": sha256_file(pair["A"])},
                    "cloud_b": {"path": str(pair["B"]), "sha256": sha256_file(pair["B"])},
                    "ab": {"path": str(ab), "sha256": sha256_file(ab)},
                }
                print(f"matched_view {framing}/{view}: {ab}")

        key_path = _write_json(out / "blind_key.json", {
            "schema_version": 1,
            "phase": "I4.3G2B",
            "scenario_id": sources.scenario_id,
            "mapping": assignment,
            "reveal_only_after_manual_review": True,
            "provider_selection_performed": False,
        })
        review_path = _write_json(out / "manual_review_template.json", review_template(views=args.views, framings=args.framings))
        summary = {
            "schema_version": 1,
            "phase": "I4.3G2B",
            "scenario_id": sources.scenario_id,
            "comparison_mode": "matched_view_diagnostic",
            "status": "ready_for_blinded_manual_review",
            "pass": True,
            "production_snapshot": {"path": str(sources.production_snapshot), "sha256": sources.production_snapshot_sha256},
            "production_verdict": {"path": str(sources.production_verdict), "sha256": sources.production_verdict_sha256},
            "authoritative_sfm": {"path": str(sources.sfm_snapshot), "sha256": sources.sfm_snapshot_sha256, "frame_id": sources.frame_id},
            "provider_artifacts": {
                "colmap_mvs_ply": {"path": str(sources.mvs_ply), "sha256": sources.mvs_ply_sha256, "source_points": len(mvs_points)},
                "da3_ply": {"path": str(sources.da3_ply), "sha256": sources.da3_ply_sha256, "source_points": len(da3_points)},
            },
            "render_policy": {
                "shared_visual_frame": sources.shared_frame.as_dict(),
                "views": list(args.views),
                "framings": list(args.framings),
                "point_budget_per_cloud": int(args.point_budget),
                "point_size": float(args.point_size),
                "independent_auto_fit": False,
                "blinded_labels": True,
            },
            "artifacts": artifacts,
            "blind_key": {"path": str(key_path), "sha256": sha256_file(key_path)},
            "manual_review_template": {"path": str(review_path), "sha256": sha256_file(review_path)},
            "manual_review": {"required": True, "completed": False, "blind_key_revealed": False},
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
        summary_path = _write_json(out / "i4_3g2b_temple_ring_matched_view_summary.json", summary)
        print("I4.3G2B TempleRing matched-view diagnostic: READY FOR BLINDED MANUAL REVIEW")
        print(f"same_authoritative_sfm=True independent_auto_fit=False blinded=True renders={len(artifacts)}")
        print("provider_selected=False universal_winner_assumed=False metric_scale_claimed=False")
        print(f"summary={summary_path}")

        if args.show:
            primary = Path(artifacts[f"{args.framings[0]}:{args.views[0]}"]["ab"]["path"])
            image = QImage(str(primary))
            window = QMainWindow(); window.setWindowTitle("Depth & 3D Model Lab — I4.3G2B TempleRing blinded matched view")
            label = QLabel(); label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(QPixmap.fromImage(image).scaled(1320, 700, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            window.setCentralWidget(label); window.resize(1366, 768); window.show()
            return int(app.exec())
        return 0
    except Exception as exc:
        print(f"I4.3G2B matched-view ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
