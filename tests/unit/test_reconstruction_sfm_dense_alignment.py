from __future__ import annotations

import numpy as np
import pytest

from core.contracts import ImageSize
from core.reconstruction import (
    DenseDepthAlignmentMode,
    InputSourceKind,
    InputSourceMetadata,
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmCameraModelEstimate,
    SfmDenseDepthAlignmentRejectionCode,
    SfmDenseDepthAlignmentStatus,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmScaleStatus,
    SfmStageStatistics,
    SfmWorkspaceImage,
    SfmWorkspaceManifest,
    TimestampOrigin,
    align_sfm_keyframe_dense_depth,
)

SESSION_ID = "00000000-0000-4000-8000-00000000b101"
WIDTH = 64
HEIGHT = 48
KEYFRAME_ID = "kf-a"
SCALE = 4.0


def _workspace() -> SfmWorkspaceManifest:
    image_size = ImageSize(WIDTH, HEIGHT)
    source = InputSourceMetadata(
        source_id="b1-sequence",
        kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="B1 synthetic sequence",
        source_uri="file:///synthetic-b1",
        expected_image_size=image_size,
        expected_frame_rate_hz=10.0,
        timestamp_origin=TimestampOrigin.SYNTHETIC,
    )
    images = tuple(
        SfmWorkspaceImage(
            order=index,
            keyframe_id=keyframe_id,
            frame_id=f"frame-{index}",
            source_sequence_index=index,
            timestamp_ns=index * 100_000_000,
            image_size=image_size,
            source_locator=f"synthetic://frame-{index}",
            staged_image_path=f"images/frame-{index}.png",
            staged_sha256=("0" if index == 0 else "1") * 64,
        )
        for index, keyframe_id in enumerate((KEYFRAME_ID, "kf-b"))
    )
    return SfmWorkspaceManifest(
        session_id=SESSION_ID,
        source=source,
        workspace_root="/synthetic/b1",
        images_directory="images",
        database_path="database.db",
        sparse_directory="sparse",
        checkpoints_directory="checkpoints",
        reports_directory="reports",
        logs_directory="logs",
        images=images,
    )


def _sfm_result(
    *,
    count: int = 10,
    include_camera_model: bool = True,
    behind_indices: set[int] | None = None,
) -> tuple[OfflineSfmResult, list[tuple[int, int, float]]]:
    behind_indices = behind_indices or set()
    workspace = _workspace()
    pose = np.eye(4, dtype=np.float64)
    pose[2, 3] = 1.0
    cameras = (
        SfmCameraEstimate(
            estimate_id="pose-a",
            camera_id="camera-1",
            keyframe_id=KEYFRAME_ID,
            frame_id="frame-0",
            backend_image_id=1,
            T_reconstruction_from_camera=pose,
        ),
        SfmCameraEstimate(
            estimate_id="pose-b",
            camera_id="camera-1",
            keyframe_id="kf-b",
            frame_id="frame-1",
            backend_image_id=2,
            T_reconstruction_from_camera=np.eye(4, dtype=np.float64),
        ),
    )
    samples: list[tuple[int, int, float]] = []
    landmarks = []
    for index in range(count):
        x = 5 + (index % 5) * 10
        y = 5 + (index // 5) * 12
        camera_z = 2.0 + index * 0.2
        reconstruction_z = 1.0 + camera_z
        if index in behind_indices:
            reconstruction_z = 0.5  # camera-space Z = -0.5 for pose-a
        landmark_id = f"lm-{index}"
        observations = (
            SfmObservation(
                observation_id=f"obs-a-{index}",
                landmark_id=landmark_id,
                keyframe_id=KEYFRAME_ID,
                feature_xy=(float(x), float(y)),
            ),
            SfmObservation(
                observation_id=f"obs-b-{index}",
                landmark_id=landmark_id,
                keyframe_id="kf-b",
                feature_xy=(float(x + 1), float(y)),
            ),
        )
        landmarks.append(
            SfmLandmarkEstimate(
                landmark_id=landmark_id,
                track_id=f"track-{index}",
                position_reconstruction=np.array([0.02 * index, -0.01 * index, reconstruction_z]),
                observations=observations,
            )
        )
        samples.append((x, y, camera_z))

    camera_models = (
        SfmCameraModelEstimate(
            camera_id="camera-1",
            backend_camera_id=1,
            backend_model_name="SIMPLE_PINHOLE",
            image_size=ImageSize(WIDTH, HEIGHT),
            parameters=(50.0, WIDTH / 2.0, HEIGHT / 2.0),
        ),
    ) if include_camera_model else ()
    result = OfflineSfmResult(
        session_id=SESSION_ID,
        backend_id="pycolmap-offline-sfm",
        backend_version="test",
        gauge=ReconstructionGauge(),
        cameras=cameras,
        camera_models=camera_models,
        landmarks=tuple(landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=2,
            registered_keyframes=2,
            landmarks=len(landmarks),
            observations=len(landmarks) * 2,
            elapsed_ns=1,
        ),
        workspace=workspace,
    )
    return result, samples


def _depth_for_samples(
    samples: list[tuple[int, int, float]], *, scale: float = SCALE
) -> np.ndarray:
    depth = np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32)
    for x, y, camera_z in samples:
        depth[y, x] = camera_z / scale
    return depth


def test_b1_recovers_known_scale_from_camera_space_sfm_depth() -> None:
    sfm, samples = _sfm_result()
    depth = _depth_for_samples(samples)

    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=depth,
        depth_provenance="synthetic-dense-depth",
    )

    assert evidence.status is SfmDenseDepthAlignmentStatus.ACCEPTED
    assert evidence.alignment is not None
    assert evidence.alignment.mode is DenseDepthAlignmentMode.SPARSE_CORRESPONDENCE_SCALE
    assert evidence.alignment.scale_to_reconstruction_units == pytest.approx(SCALE)
    assert evidence.valid_correspondences == 10
    assert evidence.inlier_correspondences == 10
    assert evidence.p90_relative_residual == pytest.approx(0.0, abs=1e-6)
    assert evidence.sfm_scale_status is SfmScaleStatus.ARBITRARY
    assert evidence.preserves_arbitrary_gauge is True
    assert evidence.as_dict()["metric_scale_claimed"] is False


def test_b1_robust_fit_rejects_one_bad_depth_correspondence_as_outlier() -> None:
    sfm, samples = _sfm_result()
    depth = _depth_for_samples(samples)
    x, y, _ = samples[-1]
    depth[y, x] = 30.0

    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=depth,
        depth_provenance="synthetic-with-outlier",
    )

    assert evidence.is_accepted
    assert evidence.alignment is not None
    assert evidence.alignment.scale_to_reconstruction_units == pytest.approx(SCALE)
    assert evidence.valid_correspondences == 10
    assert evidence.inlier_correspondences == 9


def test_b1_filters_behind_camera_and_invalid_depth_before_fit() -> None:
    sfm, samples = _sfm_result(count=12, behind_indices={0})
    depth = _depth_for_samples(samples)
    for index, value in ((1, 0.0), (2, -1.0), (3, np.nan)):
        x, y, _ = samples[index]
        depth[y, x] = value

    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=depth,
        depth_provenance="synthetic-filtering",
        minimum_correspondences=8,
    )

    assert evidence.is_accepted
    assert evidence.candidate_observations == 12
    assert evidence.behind_camera_observations == 1
    assert evidence.invalid_depth_observations == 3
    assert evidence.valid_correspondences == 8
    assert evidence.inlier_correspondences == 8


def test_b1_rejects_when_filtering_leaves_insufficient_evidence() -> None:
    sfm, samples = _sfm_result(count=10, behind_indices={0, 1})
    depth = _depth_for_samples(samples)
    x, y, _ = samples[2]
    depth[y, x] = 0.0

    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=depth,
        depth_provenance="synthetic-insufficient",
        minimum_correspondences=8,
    )

    assert evidence.status is SfmDenseDepthAlignmentStatus.REJECTED
    assert (
        evidence.rejection_code
        is SfmDenseDepthAlignmentRejectionCode.INSUFFICIENT_CORRESPONDENCES
    )
    assert evidence.alignment is None
    assert evidence.valid_correspondences == 7


def test_b1_rejects_missing_registered_keyframe_without_fabricating_alignment() -> None:
    sfm, _ = _sfm_result()
    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id="kf-missing",
        dense_depth_source_units=np.ones((HEIGHT, WIDTH), dtype=np.float32),
        depth_provenance="synthetic-missing-keyframe",
    )
    assert evidence.rejection_code is SfmDenseDepthAlignmentRejectionCode.KEYFRAME_NOT_REGISTERED
    assert evidence.alignment is None
    assert evidence.candidate_observations == 0


def test_b1_rejects_missing_sfm_camera_model_pixel_space_evidence() -> None:
    sfm, _ = _sfm_result(include_camera_model=False)
    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=np.ones((HEIGHT, WIDTH), dtype=np.float32),
        depth_provenance="synthetic-no-camera-model",
    )
    assert evidence.rejection_code is SfmDenseDepthAlignmentRejectionCode.CAMERA_MODEL_UNAVAILABLE
    assert evidence.alignment is None


def test_b1_rejects_dense_depth_pixel_space_mismatch() -> None:
    sfm, _ = _sfm_result()
    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=np.ones((HEIGHT // 2, WIDTH // 2), dtype=np.float32),
        depth_provenance="synthetic-wrong-size",
    )
    assert evidence.rejection_code is SfmDenseDepthAlignmentRejectionCode.DEPTH_PIXEL_SPACE_MISMATCH
    assert evidence.alignment is None


def test_b1_rejects_keyframe_when_sparse_depth_ratios_have_no_robust_consensus() -> None:
    sfm, samples = _sfm_result(count=10)
    depth = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    # Camera-space landmark depths span 2.0..3.8 while source depth is fixed at 1.0,
    # so no eight-correspondence multiplicative consensus exists inside the 10% bound.
    evidence = align_sfm_keyframe_dense_depth(
        sfm,
        keyframe_id=KEYFRAME_ID,
        dense_depth_source_units=depth,
        depth_provenance="synthetic-no-consensus",
        minimum_correspondences=8,
        maximum_relative_residual=0.10,
    )
    assert evidence.rejection_code is SfmDenseDepthAlignmentRejectionCode.ROBUST_SCALE_FIT_REJECTED
    assert evidence.alignment is None
    assert evidence.valid_correspondences == len(samples)
