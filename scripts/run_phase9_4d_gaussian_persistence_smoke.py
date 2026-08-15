#!/usr/bin/env python3
"""Deterministic backend-free smoke for Phase 9.4D Gaussian persistence."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_visualization import build_gaussian_visualization_presentation
from core.reconstruction import (
    current_process_gaussian_heavy_modules,
    persist_gaussian_representation,
    reload_persistent_gaussian_representation,
)
from tests.phase9_4d_helpers import make_phase9_4d_source


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_4d_smoke_") as tmp:
        root = Path(tmp)
        base, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(root)
        output = (root / "persistent").resolve()
        result = persist_gaussian_representation(
            base_scene=base,
            gaussian_artifact=artifact,
            preview_path=preview,
            qualification_metrics_path=metrics,
            source_artifact_manifest_path=manifest,
            source_optimization_summary_path=summary,
            expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
            output_dir=output,
        )
        reloaded = reload_persistent_gaussian_representation(result.scene_manifest_path)
        presentation = build_gaussian_visualization_presentation(reloaded.scene)
        payload = {
            "status": "PASS",
            "phase": "9.4D",
            "parent_scene_id": result.parent_scene_id,
            "persistent_scene_id": result.scene.scene_id,
            "scene_revision_created": result.scene.scene_id != result.parent_scene_id,
            "canonical_geometry_artifact_preserved": (
                result.scene.canonical_geometry_artifact_id == base.canonical_geometry_artifact_id
            ),
            "gaussian_artifact_persisted": result.scene.gaussian_artifact is not None,
            "gaussian_scene_role": result.artifact.scene_role.value,
            "gaussian_file_reference_count": len(result.gaussian_file_references),
            "restart_reload_verified": reloaded.restart_reload_verified,
            "existing_static_gaussian_presentation_ready": bool(presentation and presentation.ready),
            "canonical_geometry_authoritative": result.canonical_geometry_authoritative,
            "gaussian_canonical_geometry_authoritative": result.gaussian_canonical_geometry_authoritative,
            "metric_scale_authoritative": result.metric_scale_authoritative,
            "measurement_authoritative": result.measurement_authoritative,
            "qc_authoritative": result.qc_authoritative,
            "certified_metrology": result.certified_metrology,
            "backend_started": result.backend_started,
            "production_heavy_modules": list(current_process_gaussian_heavy_modules()),
        }
        required = (
            payload["scene_revision_created"] is True
            and payload["canonical_geometry_artifact_preserved"] is True
            and payload["gaussian_artifact_persisted"] is True
            and payload["gaussian_scene_role"] == "appearance_only"
            and payload["restart_reload_verified"] is True
            and payload["existing_static_gaussian_presentation_ready"] is True
            and payload["canonical_geometry_authoritative"] is True
            and payload["gaussian_canonical_geometry_authoritative"] is False
            and payload["metric_scale_authoritative"] is False
            and payload["measurement_authoritative"] is False
            and payload["qc_authoritative"] is False
            and payload["certified_metrology"] is False
            and payload["backend_started"] is False
            and payload["production_heavy_modules"] == []
        )
        if not required:
            payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
