#!/usr/bin/env python3
"""Dependency-light non-UI smoke for Phase 9.3A canonical scene presentation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    read_optional_persistent_scene_authority_snapshot,
    read_persistent_reconstruction_scene,
    reference_target_metric_evidence,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
    write_persistent_scene_authority_snapshot,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_3a_smoke_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(
            root / "run", request_id="phase9-3a-smoke"
        )
        surface = (root / "canonical_surface.obj").resolve()
        surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
        snapshots = dict(snapshots)
        snapshots["surface_mesh"] = {
            "surface_obj_path": str(surface),
            "metric_geometry_status": "arbitrary",
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        }
        scene = persistent_scene_from_pipeline_result(
            result,
            payload_snapshots_by_kind=snapshots,
        )
        persistent_root = (root / "persistent").resolve()
        manifest = write_persistent_reconstruction_scene(scene, persistent_root)
        loaded_scene = read_persistent_reconstruction_scene(manifest)
        verify_persistent_scene_file_references(loaded_scene)

        legacy = build_canonical_scene_presentation(loaded_scene)
        assert legacy.canonical_geometry_authoritative
        assert legacy.viewer_resource_ready
        assert legacy.scale_state.value == "not_qualified"
        assert legacy.certified_metrology is False

        profile = create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=520.0,
            fy=521.0,
            cx=320.0,
            cy=240.0,
            profile_id="phase9-3a-smoke-profile",
            calibration_record_id="phase9-3a-smoke-record",
            camera_id="phase9-3a-smoke-camera",
            provenance="phase9_3a_smoke",
        )
        evidence = reference_target_metric_evidence(
            evidence_id="phase9-3a-smoke-target",
            reconstruction_frame_id="reconstruction-world",
            reference_id="phase9-3a-smoke-target-ref",
            target_length_m=0.25,
            measured_length_reconstruction_units=5.0,
            target_relative_uncertainty=0.005,
            measurement_relative_uncertainty=0.02,
            provenance="phase9_3a_smoke",
            validation_state=MetricEvidenceValidationState.VALIDATED,
            validation_record_id="phase9-3a-smoke-validation",
        )
        authority = build_persistent_scene_authority_snapshot(
            loaded_scene,
            camera_calibration_profile=profile,
            metric_qualification_evidence=evidence,
        )
        write_persistent_scene_authority_snapshot(authority, persistent_root)
        loaded_authority, authority_path = read_optional_persistent_scene_authority_snapshot(
            manifest,
            expected_scene=loaded_scene,
        )
        assert loaded_authority is not None
        assert authority_path is not None
        qualified = build_canonical_scene_presentation(loaded_scene, loaded_authority)
        assert qualified.scale_state.value == "qualified"
        assert qualified.scale_relative_uncertainty is not None
        assert qualified.viewer_scale_banner_text == "VIEWER Z-UP • METRIC SCALE QUALIFIED"
        assert qualified.certified_metrology is False

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scene_id": qualified.scene_id,
                    "scene_state": qualified.scene_state,
                    "canonical_artifact_id": qualified.artifact_id,
                    "canonical_artifact_kind": qualified.artifact_kind,
                    "canonical_provider_id": qualified.provider_id,
                    "canonical_geometry_authoritative": qualified.canonical_geometry_authoritative,
                    "canonical_surface_resource": qualified.resource_state.value,
                    "viewer_resource_ready_for_9_3b": qualified.viewer_resource_ready,
                    "legacy_scale_state": legacy.scale_state.value,
                    "qualified_scale_state": qualified.scale_state.value,
                    "qualified_relative_scale_uncertainty": qualified.scale_relative_uncertainty,
                    "calibration_available": qualified.calibration_available,
                    "certified_metrology": qualified.certified_metrology,
                    "mesh_parsed_by_9_3a": False,
                    "backend_started_by_9_3a": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
