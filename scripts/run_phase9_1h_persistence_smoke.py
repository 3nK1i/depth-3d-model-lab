#!/usr/bin/env python3
"""Non-UI Phase 9.1H calibration/metric persistence reproducibility smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    MetricEvidenceValidationState,
    build_persistent_scene_authority_snapshot,
    persistent_scene_from_pipeline_result,
    read_optional_persistent_scene_authority_snapshot,
    read_persistent_reconstruction_scene,
    reference_target_metric_evidence,
    write_persistent_reconstruction_scene,
    write_persistent_scene_authority_snapshot,
)
from core.reconstruction.calibration_management import create_manual_camera_calibration_profile
from tests.reconstruction_f5_helpers import run_f5_fixture


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_1h_smoke_") as tmp:
        root = Path(tmp).resolve()
        _fixture, _providers, result, snapshots = run_f5_fixture(root / "run")
        scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
        profile = create_manual_camera_calibration_profile(
            width=640,
            height=480,
            fx=520.0,
            fy=521.0,
            cx=320.0,
            cy=240.0,
            profile_id="phase9-1h-smoke-profile",
            calibration_record_id="phase9-1h-smoke-record",
            camera_id="phase9-1h-smoke-camera",
            provenance="phase9_1h_smoke",
        )
        evidence = reference_target_metric_evidence(
            evidence_id="phase9-1h-smoke-target",
            reconstruction_frame_id="reconstruction-world",
            reference_id="smoke-target",
            target_length_m=0.25,
            measured_length_reconstruction_units=5.0,
            target_relative_uncertainty=0.005,
            measurement_relative_uncertainty=0.02,
            provenance="phase9_1h_smoke",
            validation_state=MetricEvidenceValidationState.VALIDATED,
            validation_record_id="phase9-1h-smoke-validation",
        )
        snapshot = build_persistent_scene_authority_snapshot(
            scene,
            camera_calibration_profile=profile,
            metric_qualification_evidence=evidence,
        )
        persistent_root = (root / "persistent").resolve()
        scene_manifest = write_persistent_reconstruction_scene(scene, persistent_root)
        write_persistent_scene_authority_snapshot(snapshot, persistent_root)

        loaded_scene = read_persistent_reconstruction_scene(scene_manifest)
        loaded_authority, authority_manifest = read_optional_persistent_scene_authority_snapshot(
            scene_manifest,
            expected_scene=loaded_scene,
        )
        assert loaded_authority is not None
        assert authority_manifest is not None
        assert loaded_authority.camera_calibration_profile is not None
        assert loaded_authority.camera_calibration_profile.as_dict() == profile.as_dict()
        assert loaded_authority.metric_qualification_evidence is not None
        assert loaded_authority.metric_qualification_evidence.as_dict() == evidence.as_dict()
        assert loaded_authority.metric_qualification_assessment is not None
        assert loaded_authority.metric_qualification_assessment.authorizes_calibrated_metric_scale
        assert loaded_authority.certified_metrology is False
        assert loaded_scene.schema_version == 1

        legacy_root = (root / "legacy").resolve()
        legacy_manifest = write_persistent_reconstruction_scene(scene, legacy_root)
        legacy_authority, legacy_authority_manifest = read_optional_persistent_scene_authority_snapshot(
            legacy_manifest,
            expected_scene=scene,
        )
        assert legacy_authority is None
        assert legacy_authority_manifest is None

        print(
            json.dumps(
                {
                    "status": "PASS",
                    "scene_schema_version_preserved": loaded_scene.schema_version,
                    "authority_sidecar": authority_manifest.name,
                    "calibration_profile_id": loaded_authority.camera_calibration_profile.profile_id,
                    "metric_qualification_state": loaded_authority.metric_qualification_assessment.state.value,
                    "metric_coordinates_allowed": loaded_authority.metric_coordinates_allowed,
                    "calibrated_metric_scale": loaded_authority.calibrated_metric_scale,
                    "relative_scale_uncertainty": loaded_authority.metric_qualification_assessment.relative_scale_uncertainty,
                    "certified_metrology": loaded_authority.certified_metrology,
                    "legacy_scene_without_sidecar": legacy_authority is None,
                    "external_profile_file_required": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
