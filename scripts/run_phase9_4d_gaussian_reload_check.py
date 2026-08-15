#!/usr/bin/env python3
"""Fresh-process restart/reload validation for a Phase 9.4D persistent Gaussian scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_gaussian_visualization import build_gaussian_visualization_presentation
from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    reload_persistent_gaussian_representation,
)

DEFAULT_SUMMARY = Path("/tmp/phase9_4d_persistent_gaussian_scene/phase9_4d_persistence_summary.json")


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Phase 9.4D persistence summary is missing: {summary_path}")
    source = _json(summary_path)
    if source.get("status") != "PASS":
        raise SystemExit("Phase 9.4D reload refuses a non-PASS persistence summary.")
    loaded = reload_persistent_gaussian_representation(source["scene_manifest_path"])
    mesh = load_canonical_viewer_mesh(loaded.scene)
    presentation = build_gaussian_visualization_presentation(loaded.scene)
    payload = {
        "phase": "9.4D",
        "status": "PASS",
        "persistent_scene_id": loaded.scene.scene_id,
        "parent_scene_id": loaded.parent_scene_id,
        "scene_fingerprint_matches_summary": loaded.scene.scene_fingerprint_sha256 == source.get("persistent_scene_fingerprint_sha256"),
        "restart_reload_verified": loaded.restart_reload_verified,
        "canonical_geometry_available": mesh is not None,
        "canonical_vertex_count": 0 if mesh is None else mesh.vertex_count,
        "canonical_face_count": 0 if mesh is None else mesh.face_count,
        "gaussian_artifact_available": loaded.scene.gaussian_artifact is not None,
        "gaussian_scene_path": loaded.artifact.scene_path,
        "gaussian_scene_exists": Path(loaded.artifact.scene_path).is_file(),
        "gaussian_static_presentation_ready": bool(presentation and presentation.ready),
        "gaussian_preview_path": "" if presentation is None else presentation.preview_path,
        "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
        "gaussian_canonical_geometry_authoritative": loaded.gaussian_canonical_geometry_authoritative,
        "metric_scale_authoritative": loaded.metric_scale_authoritative,
        "measurement_authoritative": loaded.measurement_authoritative,
        "qc_authoritative": loaded.qc_authoritative,
        "certified_metrology": loaded.certified_metrology,
        "backend_started": loaded.backend_started,
        "production_heavy_modules": list(current_process_gaussian_heavy_modules()),
    }
    required = (
        payload["scene_fingerprint_matches_summary"] is True
        and payload["restart_reload_verified"] is True
        and payload["gaussian_artifact_available"] is True
        and payload["gaussian_scene_exists"] is True
        and payload["gaussian_static_presentation_ready"] is True
        and payload["canonical_geometry_authoritative"] is True
        and payload["gaussian_canonical_geometry_authoritative"] is False
        and payload["certified_metrology"] is False
        and payload["backend_started"] is False
        and payload["production_heavy_modules"] == []
    )
    if not required:
        payload["status"] = "FAIL"
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Phase 9.4D restart/reload check: " + ("PASS" if payload["status"] == "PASS" else "FAIL"))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
