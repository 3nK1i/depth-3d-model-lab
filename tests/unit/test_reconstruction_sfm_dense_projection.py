from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    CameraDistortionModel,
    DenseDepthAlignment,
    DenseDepthAlignmentMode,
    ReconstructionValidationError,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmDenseDepthAlignmentEvidence,
    SfmDenseDepthAlignmentRejectionCode,
    SfmDenseDepthAlignmentStatus,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    project_sfm_keyframe_dense_rgb,
    reconstruction_camera_from_sfm_model,
)

SESSION_ID = "00000000-0000-4000-8000-00000000c001"
KEYFRAME_ID = "kf-c"
FRAME_ID = "frame-c"


def _camera_estimate(*, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> SfmCameraEstimate:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = (tx, ty, tz)
    return SfmCameraEstimate(
        estimate_id="pose-c",
        camera_id="camera-1",
        keyframe_id=KEYFRAME_ID,
        frame_id=FRAME_ID,
        backend_image_id=1,
        T_reconstruction_from_camera=pose,
    )


def _camera_model(*, name: str = "SIMPLE_PINHOLE", params=None) -> SfmCameraModelEstimate:
    if params is None:
        params = (2.0, 1.5, 1.0)
    return SfmCameraModelEstimate(
        camera_id="camera-1",
        backend_camera_id=1,
        backend_model_name=name,
        image_size=ImageSize(4, 3),
        parameters=tuple(params),
    )


def _accepted_evidence(*, scale: float = 2.0) -> SfmDenseDepthAlignmentEvidence:
    alignment = DenseDepthAlignment(
        mode=DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE,
        scale_to_reconstruction_units=scale,
        relative_scale_uncertainty=0.05,
        provenance="C synthetic sparse-correspondence alignment",
    )
    return SfmDenseDepthAlignmentEvidence(
        session_id=SESSION_ID,
        keyframe_id=KEYFRAME_ID,
        frame_id=FRAME_ID,
        camera_id="camera-1",
        camera_model_name="SIMPLE_PINHOLE",
        depth_provenance="synthetic dense depth",
        status=SfmDenseDepthAlignmentStatus.ACCEPTED,
        sfm_scale_status=SfmScaleStatus.ARBITRARY,
        sfm_world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        depth_image_size=ImageSize(4, 3),
        minimum_correspondences=8,
        maximum_relative_residual=0.35,
        candidate_observations=8,
        valid_correspondences=8,
        inlier_correspondences=8,
        robust_relative_residual_limit=0.1,
        median_relative_residual=0.02,
        p90_relative_residual=0.05,
        alignment=alignment,
        message="accepted synthetic C alignment",
    )


def _rgb() -> np.ndarray:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    for y in range(3):
        for x in range(4):
            rgb[y, x] = (10 + x, 20 + y, 30 + x + y)
    return rgb


def test_c_converts_simple_radial_sfm_camera_without_claiming_calibration() -> None:
    model = _camera_model(name="SIMPLE_RADIAL", params=(100.0, 1.5, 1.0, 0.01))
    camera = reconstruction_camera_from_sfm_model(model, keyframe_id=KEYFRAME_ID)

    assert camera.camera_id == "camera-1"
    assert camera.distortion_model is CameraDistortionModel.BROWN_CONRADY
    assert camera.distortion_coefficients == pytest.approx((0.01, 0.0, 0.0, 0.0, 0.0))
    assert camera.intrinsics.matrix[0, 0] == pytest.approx(100.0)
    assert camera.intrinsics.matrix[1, 1] == pytest.approx(100.0)
    assert camera.intrinsics.calibration_status.value == "not_calibrated"
    assert camera.intrinsics.source.value == "estimated"


def test_c_rejects_unsupported_sfm_camera_model_instead_of_reinterpreting_it() -> None:
    model = _camera_model(name="FOV", params=(2.0, 2.0, 1.5, 1.0, 0.5))
    with pytest.raises(ReconstructionValidationError, match="unsupported SfM camera model"):
        reconstruction_camera_from_sfm_model(model, keyframe_id=KEYFRAME_ID)


def test_c_projects_scaled_dense_depth_into_reconstruction_coordinates() -> None:
    result = project_sfm_keyframe_dense_rgb(
        _camera_estimate(tx=10.0, ty=-3.0, tz=5.0),
        _camera_model(),
        timestamp_ns=123,
        rgb=_rgb(),
        dense_depth_source_units=np.ones((3, 4), dtype=np.float32),
        alignment_evidence=_accepted_evidence(scale=2.0),
    )

    # Pixel (u=0,v=0): z=2, x=(-1.5/2)*2=-1.5, y=(-1/2)*2=-1.
    assert result.points_reconstruction[0] == pytest.approx((8.5, -4.0, 7.0))
    assert np.array_equal(result.colors_rgb[0], np.array([10, 20, 30], dtype=np.uint8))
    assert result.projection.sampled_pixels == 12
    assert result.projection.valid_depth_samples == 12
    assert result.sfm_scale_status == "arbitrary"
    assert result.metric_scale_claimed is False
    assert result.canonical_geometry_authoritative is False
    assert result.certified_metrology is False


def test_c_pixel_stride_is_regular_and_preserves_rgb_pairing() -> None:
    result = project_sfm_keyframe_dense_rgb(
        _camera_estimate(),
        _camera_model(),
        timestamp_ns=1,
        rgb=_rgb(),
        dense_depth_source_units=np.ones((3, 4), dtype=np.float32),
        alignment_evidence=_accepted_evidence(scale=1.0),
        pixel_stride=2,
    )
    # rows 0,2 and columns 0,2 => 4 samples
    assert result.projection.sampled_pixels == 4
    assert result.projection.valid_depth_samples == 4
    assert np.array_equal(
        result.colors_rgb,
        np.asarray([[10, 20, 30], [12, 20, 32], [10, 22, 32], [12, 22, 34]], dtype=np.uint8),
    )


def test_c_rejects_alignment_keyframe_or_pixel_grid_mismatch() -> None:
    evidence = _accepted_evidence()
    wrong_pose = SfmCameraEstimate(
        estimate_id="wrong",
        camera_id="camera-1",
        keyframe_id="kf-other",
        frame_id=FRAME_ID,
        backend_image_id=1,
        T_reconstruction_from_camera=np.eye(4),
    )
    with pytest.raises(ReconstructionValidationError, match="same keyframe"):
        project_sfm_keyframe_dense_rgb(
            wrong_pose,
            _camera_model(),
            timestamp_ns=1,
            rgb=_rgb(),
            dense_depth_source_units=np.ones((3, 4), dtype=np.float32),
            alignment_evidence=evidence,
        )

    with pytest.raises(ReconstructionValidationError, match="pixel grid"):
        project_sfm_keyframe_dense_rgb(
            _camera_estimate(),
            _camera_model(),
            timestamp_ns=1,
            rgb=_rgb(),
            dense_depth_source_units=np.ones((2, 4), dtype=np.float32),
            alignment_evidence=evidence,
        )


def test_c_rejects_rejected_b1_evidence_without_projecting_geometry() -> None:
    rejected = SfmDenseDepthAlignmentEvidence(
        session_id=SESSION_ID,
        keyframe_id=KEYFRAME_ID,
        frame_id=FRAME_ID,
        camera_id="camera-1",
        camera_model_name="SIMPLE_PINHOLE",
        depth_provenance="synthetic rejected depth",
        status=SfmDenseDepthAlignmentStatus.REJECTED,
        sfm_scale_status=SfmScaleStatus.ARBITRARY,
        sfm_world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        depth_image_size=ImageSize(4, 3),
        minimum_correspondences=8,
        maximum_relative_residual=0.35,
        candidate_observations=4,
        valid_correspondences=4,
        inlier_correspondences=0,
        rejection_code=SfmDenseDepthAlignmentRejectionCode.INSUFFICIENT_CORRESPONDENCES,
        message="not enough evidence",
    )
    with pytest.raises(ReconstructionValidationError, match="requires accepted"):
        project_sfm_keyframe_dense_rgb(
            _camera_estimate(),
            _camera_model(),
            timestamp_ns=1,
            rgb=_rgb(),
            dense_depth_source_units=np.ones((3, 4), dtype=np.float32),
            alignment_evidence=rejected,
        )


def test_c_result_summary_never_promotes_single_frame_to_canonical_authority() -> None:
    result = project_sfm_keyframe_dense_rgb(
        _camera_estimate(),
        _camera_model(),
        timestamp_ns=1,
        rgb=_rgb(),
        dense_depth_source_units=np.ones((3, 4), dtype=np.float32),
        alignment_evidence=_accepted_evidence(),
    )
    payload = result.as_dict()
    assert payload["projection"]["point_count"] == 12
    assert payload["metric_scale_claimed"] is False
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["measurement_authoritative"] is False
    assert payload["certified_metrology"] is False
