from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from core.reconstruction.da3 import DA3_PROVIDER_ID
from core.reconstruction.da3_canonical_scene import integrate_da3_result_to_canonical_scene
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.orchestration import ReconstructionPipelineRunStatus
from tests.phase9_3h_helpers import make_phase9_3h_da3_fixture, rgb_frames_for_manifest


def test_3h_da3_supporting_evidence_becomes_canonical_without_metric_promotion(tmp_path: Path) -> None:
    input_path, _result_path, manifest = make_phase9_3h_da3_fixture(tmp_path)
    result = integrate_da3_result_to_canonical_scene(
        manifest,
        session_id=str(uuid4()),
        request_id="phase9-3h-core",
        rgb_frames=rgb_frames_for_manifest(input_path, manifest),
        pixel_stride=2,
    )

    assert result.pipeline_result.status is ReconstructionPipelineRunStatus.DEGRADED
    assert result.pipeline_result.canonical_geometry_authoritative is True
    assert result.depth.source_provider_id == DA3_PROVIDER_ID
    assert result.trajectory.source_provider_id == DA3_PROVIDER_ID
    assert result.surface.mesh.vertex_count > 0
    assert result.surface.mesh.face_count > 0
    assert result.surface.metric_geometry_status.value == "arbitrary"
    assert result.surface.unit_to_metre is None
    assert result.certified_metrology is False
    assert result.semantic_context_available is False
    assert result.inspection_qc_available is False


def test_3h_calibration_is_optional_and_does_not_create_unit_to_metre(tmp_path: Path) -> None:
    input_path, _result_path, manifest = make_phase9_3h_da3_fixture(tmp_path)
    result = integrate_da3_result_to_canonical_scene(
        manifest,
        session_id=str(uuid4()),
        request_id="phase9-3h-uncalibrated",
        rgb_frames=rgb_frames_for_manifest(input_path, manifest),
        pixel_stride=2,
    )

    assert result.ingestion.camera_profile.camera.intrinsics.calibration_status.value == "not_calibrated"
    assert result.ingestion.scale_profile.mode.value == "arbitrary"
    assert result.surface.metric_geometry_status.value == "arbitrary"
    assert result.surface.unit_to_metre is None


def test_3h_rejects_rgb_evidence_that_does_not_match_da3_result(tmp_path: Path) -> None:
    input_path, _result_path, manifest = make_phase9_3h_da3_fixture(tmp_path)
    frames = rgb_frames_for_manifest(input_path, manifest)
    with pytest.raises(ReconstructionValidationError, match="RGB frame count"):
        integrate_da3_result_to_canonical_scene(
            manifest,
            session_id=str(uuid4()),
            request_id="phase9-3h-bad-rgb",
            rgb_frames=frames[:-1],
        )


def test_3h_rejects_invalid_pixel_stride_without_fabricating_geometry(tmp_path: Path) -> None:
    input_path, _result_path, manifest = make_phase9_3h_da3_fixture(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="pixel_stride"):
        integrate_da3_result_to_canonical_scene(
            manifest,
            session_id=str(uuid4()),
            request_id="phase9-3h-bad-stride",
            rgb_frames=rgb_frames_for_manifest(input_path, manifest),
            pixel_stride=0,
        )
