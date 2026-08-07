from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    ReconstructionGauge,
    ReconstructionQualityProfile,
    ReconstructionValidationError,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmLandmarkEstimate,
    SfmMatchingStrategy,
    SfmObservation,
    SfmResumePolicy,
    SfmScaleStatus,
    SfmStageStatistics,
    SfmWorldAlignmentStatus,
    materialize_sfm_workspace,
)
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def test_arbitrary_gauge_never_claims_metres() -> None:
    gauge = ReconstructionGauge()
    assert gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert gauge.unit_to_metre is None
    assert not gauge.is_metric
    with pytest.raises(ReconstructionValidationError):
        ReconstructionGauge(unit_to_metre=1.0)


def test_metric_gauge_requires_scale_source_and_positive_conversion() -> None:
    with pytest.raises(ReconstructionValidationError):
        ReconstructionGauge(scale_status=SfmScaleStatus.ESTIMATED_METRIC)
    with pytest.raises(ReconstructionValidationError):
        ReconstructionGauge(
            scale_status=SfmScaleStatus.ESTIMATED_METRIC,
            unit_to_metre=-1.0,
            scale_source="depth",
        )
    gauge = ReconstructionGauge(
        scale_status=SfmScaleStatus.ESTIMATED_METRIC,
        alignment_status=SfmWorldAlignmentStatus.GRAVITY_ALIGNED,
        unit_to_metre=0.42,
        scale_source="metric_depth",
        relative_scale_uncertainty=0.08,
    )
    assert gauge.is_metric
    assert gauge.unit_to_metre == pytest.approx(0.42)


def test_sfm_pose_uses_reconstruction_units_not_metric_fields() -> None:
    matrix = np.eye(4)
    matrix[0, 3] = 12.5
    estimate = SfmCameraEstimate(
        estimate_id="pose-1",
        camera_id="camera-1",
        keyframe_id="kf-1",
        frame_id="frame-1",
        backend_image_id=1,
        T_reconstruction_from_camera=matrix,
    )
    assert np.allclose(estimate.camera_center_reconstruction, [12.5, 0.0, 0.0])
    payload = estimate.as_dict()
    assert "T_reconstruction_from_camera" in payload
    assert all("_m" not in key for key in payload)


def test_invalid_sfm_pose_rotation_is_rejected() -> None:
    matrix = np.eye(4)
    matrix[0, 0] = 2.0
    with pytest.raises(ReconstructionValidationError):
        SfmCameraEstimate(
            estimate_id="pose-1",
            camera_id="camera-1",
            keyframe_id="kf-1",
            frame_id="frame-1",
            backend_image_id=1,
            T_reconstruction_from_camera=matrix,
        )


def test_landmark_requires_two_distinct_keyframe_observations() -> None:
    observation = SfmObservation(
        observation_id="obs-1",
        landmark_id="lm-1",
        keyframe_id="kf-1",
        feature_xy=(10.0, 20.0),
    )
    with pytest.raises(ReconstructionValidationError):
        SfmLandmarkEstimate(
            landmark_id="lm-1",
            track_id="track-1",
            position_reconstruction=np.array([0.0, 0.0, 1.0]),
            observations=(observation,),
        )


def test_offline_request_matches_workspace_session(tmp_path) -> None:
    result = make_sfm_ingestion_result(tmp_path / "sequence")
    workspace = materialize_sfm_workspace(result, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.BALANCED,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=SfmMatchingStrategy.SEQUENTIAL,
        ),
    )
    assert request.workspace.images[0].order == 0
    assert request.configuration.matching_strategy is SfmMatchingStrategy.SEQUENTIAL


def test_stage_statistics_require_consistent_counts() -> None:
    with pytest.raises(ReconstructionValidationError):
        SfmStageStatistics(
            input_keyframes=2,
            registered_keyframes=3,
            landmarks=0,
            observations=0,
            elapsed_ns=0,
        )


def test_sfm_camera_model_estimate_preserves_backend_calibration_without_claiming_certification() -> None:
    camera = SfmCameraModelEstimate(
        camera_id="pycolmap-camera-7",
        backend_camera_id=7,
        backend_model_name="SIMPLE_RADIAL",
        image_size=ImageSize(width=640, height=480),
        parameters=(500.0, 320.0, 240.0, 0.01),
    )
    payload = camera.as_dict()
    assert payload["backend_model_name"] == "SIMPLE_RADIAL"
    assert payload["parameters"] == [500.0, 320.0, 240.0, 0.01]
    assert "calibrated" not in payload


def test_offline_sfm_configuration_exposes_resume_and_sequential_overlap() -> None:
    configuration = OfflineSfmConfiguration(
        matching_strategy=SfmMatchingStrategy.SEQUENTIAL,
        sequential_overlap=7,
        resume_policy=SfmResumePolicy.REUSE_COMPLETED_STAGES,
    )
    payload = configuration.as_dict()
    assert payload["sequential_overlap"] == 7
    assert payload["resume_policy"] == "reuse_completed_stages"
    with pytest.raises(ReconstructionValidationError):
        OfflineSfmConfiguration(sequential_overlap=0)
