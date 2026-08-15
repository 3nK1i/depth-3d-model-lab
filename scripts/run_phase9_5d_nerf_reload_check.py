#!/usr/bin/env python3
"""Fresh-process backend-free reload check for Phase 9.5D persistent NeRF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_nerf_visualization import build_nerf_visualization_presentation
from core.reconstruction import current_process_nerf_heavy_modules, reload_persistent_nerf_representation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("/tmp/phase9_5d_persistent_nerf_scene/phase9_5d_persistence_summary.json"))
    args = parser.parse_args()
    source = json.loads(args.summary.expanduser().resolve().read_text(encoding="utf-8"))
    loaded = reload_persistent_nerf_representation(source["scene_manifest_path"])
    mesh = load_canonical_viewer_mesh(loaded.scene)
    presentation = build_nerf_visualization_presentation(loaded.scene)
    payload = {
        "phase": "9.5D",
        "status": "PASS",
        "persistent_scene_id": loaded.scene.scene_id,
        "parent_scene_id": loaded.parent_scene_id,
        "scene_fingerprint_matches_summary": loaded.scene.scene_fingerprint_sha256 == source.get("persistent_scene_fingerprint_sha256"),
        "restart_reload_verified": loaded.restart_reload_verified,
        "canonical_geometry_available": mesh is not None,
        "canonical_vertex_count": 0 if mesh is None else mesh.vertex_count,
        "canonical_face_count": 0 if mesh is None else mesh.face_count,
        "nerf_artifact_available": loaded.scene.nerf_artifact is not None,
        "nerf_checkpoint_exists": Path(loaded.artifact.checkpoint_path).is_file(),
        "nerf_config_exists": Path(loaded.artifact.config_path).is_file(),
        "nerf_dataset_exists": Path(loaded.dataset_root, "transforms.json").is_file(),
        "nerf_static_presentation_ready": bool(presentation and presentation.ready),
        "nerf_preview_path": "" if presentation is None else presentation.preview_path,
        "canonical_geometry_authoritative": loaded.canonical_geometry_authoritative,
        "nerf_canonical_geometry_authoritative": loaded.nerf_canonical_geometry_authoritative,
        "metric_scale_authoritative": loaded.metric_scale_authoritative,
        "measurement_authoritative": loaded.measurement_authoritative,
        "qc_authoritative": loaded.qc_authoritative,
        "certified_metrology": loaded.certified_metrology,
        "backend_started": loaded.backend_started,
        "rendering_started": loaded.rendering_started,
        "production_heavy_modules": list(current_process_nerf_heavy_modules()),
    }
    required = (
        payload["scene_fingerprint_matches_summary"] is True
        and payload["restart_reload_verified"] is True
        and payload["canonical_geometry_available"] is True
        and payload["nerf_artifact_available"] is True
        and payload["nerf_checkpoint_exists"] is True
        and payload["nerf_config_exists"] is True
        and payload["nerf_dataset_exists"] is True
        and payload["nerf_static_presentation_ready"] is True
        and payload["canonical_geometry_authoritative"] is True
        and payload["nerf_canonical_geometry_authoritative"] is False
        and payload["certified_metrology"] is False
        and payload["backend_started"] is False
        and payload["rendering_started"] is False
        and payload["production_heavy_modules"] == []
    )
    if not required:
        payload["status"] = "FAIL"
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Phase 9.5D restart/reload check: " + ("PASS" if payload["status"] == "PASS" else "FAIL"))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
