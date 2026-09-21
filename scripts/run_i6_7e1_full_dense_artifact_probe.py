#!/usr/bin/env python3
"""Read-only full-vertex preflight of a frozen I6.7B COLMAP-MVS artifact.

This checks geometry integrity, NOT I6.7E production admission or full GPU
rendering. Only scan existing files; do not rerun PatchMatch/Stereo Fusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_dense_artifact_preflight import inspect_full_dense_ply


def _require(ok: bool, detail: str) -> None:
    if not ok:
        raise ReconstructionValidationError("I6.7E1 real source preflight: " + detail)


def verify_report(report: dict, report_path: Path) -> tuple[Path, int, str, Path]:
    """Bind a full geometry preflight to one explicit, passing source report."""
    _require(report.get("pass") is True and report.get("status") == "pass", "source report must pass")
    _require(report.get("provider", {}).get("selection_mode") == "explicit_provider", "explicit provider required")
    provider = report["provider"]
    result = provider.get("result", {})
    _require(result.get("status") == "completed", "degraded/failing result requires separate review")
    _require(provider.get("implementation") == "colmap_mvs" and
             result.get("implementation") == "colmap_mvs", "this first real probe supports COLMAP MVS only")
    _require(provider.get("provider_id") == result.get("provider_id") == "colmap-mvs-external",
             "provider identity mismatch")
    _require(result.get("selected_view_count") == report.get("source_sfm", {}).get("registered_views"),
             "real source view count mismatch")
    _require(result.get("canonical_frame_id") == report.get("source_sfm", {}).get("reconstruction_frame_id"),
             "PyCOLMAP SfM frame mismatch")
    _require(result.get("authority", {}).get("camera_pose_authority_owner") == "pycolmap_sfm", "SfM camera authority mismatch")
    _require(result.get("authority", {}).get("single_authoritative_sfm_frame_preserved") is True,
             "canonical SfM frame not preserved")
    _require(result.get("authority", {}).get("canonical_geometry_authoritative") is True,
             "provider did not qualify completed geometry")
    forbidden = ("final_dense_result_admitted_by_progressive_path", "persistent_dense_scene_created",
                 "production_auto_activated", "metric_scale_claimed", "measurement_authoritative",
                 "certified_metrology_claimed", "surface_mesh_generated")
    _require(all(report.get("authority", {}).get(field) is False for field in forbidden),
             "source report asserts disallowed progressive authority")
    artifact = report.get("artifacts", {})
    raw_path = Path(artifact.get("canonical_geometry", ""))
    source = raw_path.resolve(strict=True)
    _require(str(raw_path) == str(source) == result.get("canonical_geometry_ref"),
             "complete artifact reference mismatch")
    _require(source.suffix.lower() == ".ply" and source.parent.name == "dense", "expected fused PLY under dense output")
    _require(report.get("progressive", {}).get("current_dense_points", 0) > 4096,
             "real dense source does not exceed provisional preview population")
    _require(report.get("progressive", {}).get("preview_point_count", 0) <= 4096,
             "provisional preview exceeds frozen bound")
    # The report's own directory anchors the replay; do not write beside it.
    _require(source.is_relative_to(report_path.parent), "artifact is outside report's source evidence root")
    return source, report["progressive"]["current_dense_points"], artifact["canonical_geometry_sha256"], source.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-report", type=Path, required=True)
    p.add_argument("--expected-source-report-sha256", required=True)
    args = p.parse_args()
    report_path = args.source_report.expanduser().resolve(strict=True)
    actual_report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    _require(actual_report_sha == args.expected_source_report_sha256,
             "source report SHA256 does not match supplied identity")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifact, declared_points, artifact_sha, output_root = verify_report(report, report_path)
    verified = inspect_full_dense_ply(artifact, expected_sha256=artifact_sha,
                                       expected_root=output_root, expected_vertex_count=declared_points)
    print("I6_7E1_FULL_DENSE_ARTIFACT_PREFLIGHT=PASS")
    print("ARTIFACT_PATH=" + verified.path)
    print("ARTIFACT_SHA256=" + verified.sha256)
    print("FULL_SCANNED_VERTICES=" + str(verified.scanned_vertex_count))
    print("HAS_RGB=" + ("YES" if verified.has_rgb else "NO"))
    print("FORMAT=" + verified.ply_format)
    print("FINAL_DENSE_RESULT_ADMITTED=NO")
    print("PRODUCTION_ONE_START_DENSE_WIRING=NO")
    print("FULL_DENSITY_GPU_VIEWER_ENABLED=NO")
    print("METRIC_SCALE_CLAIMED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
