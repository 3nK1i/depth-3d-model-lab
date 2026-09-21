from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_CAMERA_AUTHORITY_OWNER,
    I66A_POSE_CONVENTION,
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmCameraSample,
    ProgressiveSfmGeometryPreview,
    ProgressiveSfmLandmarkSample,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
)

SESSION_ID = "11111111-2222-4333-8444-555555555555"
REQUEST_SHA = "a" * 64


def _pose(x: float = 0.0) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix


def _camera(index: int = 1) -> ProgressiveSfmCameraSample:
    return ProgressiveSfmCameraSample(
        backend_image_id=index,
        backend_camera_id=0,
        source_image_name=f"image_{index:04d}.jpg",
        T_reconstruction_from_camera=_pose(float(index)),
    )


def _landmark(index: int = 1) -> ProgressiveSfmLandmarkSample:
    return ProgressiveSfmLandmarkSample(
        backend_point3d_id=index,
        position_reconstruction=(float(index), 0.5, -0.25),
        color_rgb=(10, 20, 30),
        track_length=3,
    )


def _snapshot(**overrides) -> ProgressiveSfmSnapshot:
    values = dict(
        session_id=SESSION_ID,
        revision=1,
        timestamp_utc="2026-09-17T12:00:00+00:00",
        elapsed_ns=5_000_000,
        stage=ProgressiveSfmStage.MAPPING,
        stage_detail="registered_image_callback",
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=12,
            registered_cameras=3,
            sparse_landmarks=1,
            observations=3,
            mean_reprojection_error_px=0.7,
        ),
        progress_fraction=0.72,
        candidate_model_id="candidate-0",
        candidate_model_count=1,
        geometry=ProgressiveSfmGeometryPreview(
            camera_samples=(_camera(1), _camera(2)),
            landmark_samples=(_landmark(1),),
            sampling_profile="bounded_preview_v1",
        ),
        warnings=("preview is provisional",),
    )
    values.update(overrides)
    return ProgressiveSfmSnapshot(**values)


def test_snapshot_serialization_keeps_provisional_authority_explicit() -> None:
    payload = _snapshot().as_dict()
    assert payload["authority"]["camera_pose_authority_owner"] == I66A_CAMERA_AUTHORITY_OWNER
    assert payload["authority"]["pose_convention"] == I66A_POSE_CONVENTION
    assert payload["authority"]["provisional"] is True
    assert payload["authority"]["final_result_admitted"] is False
    assert payload["authority"]["persistent_canonical_scene"] is False
    assert payload["authority"]["canonical_geometry_authoritative"] is False
    assert payload["authority"]["metric_scale_claimed"] is False
    assert payload["authority"]["certified_metrology"] is False


def test_snapshot_digest_is_deterministic() -> None:
    first = _snapshot()
    second = _snapshot()
    assert first.snapshot_sha256 == second.snapshot_sha256
    assert len(first.snapshot_sha256) == 64


def test_snapshot_digest_changes_with_revision() -> None:
    assert _snapshot().snapshot_sha256 != _snapshot(revision=2).snapshot_sha256


def test_camera_sample_requires_valid_rigid_pose() -> None:
    bad = np.eye(4)
    bad[3, 3] = 2.0
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmCameraSample(
            backend_image_id=1,
            backend_camera_id=0,
            source_image_name="image.jpg",
            T_reconstruction_from_camera=bad,
        )


def test_landmark_sample_validates_color_and_track_length() -> None:
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmLandmarkSample(
            backend_point3d_id=1,
            position_reconstruction=(0.0, 0.0, 0.0),
            color_rgb=(0, 256, 0),
            track_length=2,
        )
    with pytest.raises(ReconstructionValidationError):
        replace(_landmark(), track_length=1)


def test_geometry_preview_rejects_duplicate_camera_and_landmark_ids() -> None:
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmGeometryPreview(
            camera_samples=(_camera(1), _camera(1)),
            landmark_samples=(),
        )
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmGeometryPreview(
            camera_samples=(),
            landmark_samples=(_landmark(1), _landmark(1)),
        )


def test_statistics_reject_registered_count_above_input() -> None:
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmStatistics(
            input_views=3,
            registered_cameras=4,
            sparse_landmarks=0,
            observations=0,
        )


def test_statistics_reject_observations_without_landmarks() -> None:
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmStatistics(
            input_views=3,
            registered_cameras=2,
            sparse_landmarks=0,
            observations=1,
        )


def test_snapshot_rejects_non_pycolmap_backend() -> None:
    with pytest.raises(ReconstructionValidationError):
        _snapshot(backend_id="other-sfm")


def test_snapshot_rejects_completed_progress_fraction() -> None:
    with pytest.raises(ReconstructionValidationError):
        _snapshot(progress_fraction=1.0)


def test_snapshot_rejects_authority_upgrade() -> None:
    for override in (
        {"provisional": False},
        {"final_result_admitted": True},
        {"persistent_canonical_scene": True},
        {"canonical_geometry_authoritative": True},
        {"metric_scale_claimed": True},
        {"measurement_authoritative": True},
        {"certified_metrology": True},
    ):
        with pytest.raises(ReconstructionValidationError):
            _snapshot(**override)


def test_snapshot_rejects_preview_counts_above_statistics() -> None:
    stats = ProgressiveSfmStatistics(
        input_views=12,
        registered_cameras=1,
        sparse_landmarks=0,
        observations=0,
    )
    with pytest.raises(ReconstructionValidationError):
        _snapshot(
            statistics=stats,
            geometry=ProgressiveSfmGeometryPreview(camera_samples=(_camera(1), _camera(2))),
        )


def test_snapshot_payload_roundtrip_preserves_identity() -> None:
    original = _snapshot()
    restored = ProgressiveSfmSnapshot.from_payload(original.as_dict())
    assert restored.as_dict() == original.as_dict()
    assert restored.snapshot_sha256 == original.snapshot_sha256


def test_warning_values_must_be_unique_nonempty() -> None:
    with pytest.raises(ReconstructionValidationError):
        _snapshot(warnings=("warning", "warning"))
    with pytest.raises(ValueError):
        _snapshot(warnings=("",))


def test_payload_rejects_contradictory_or_nonboolean_authority() -> None:
    payload = _snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["authority"]["camera_pose_authority_owner"] = "Other"
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmSnapshot.from_payload(payload)

    payload = _snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["authority"]["provisional"] = "true"
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmSnapshot.from_payload(payload)


def test_geometry_payload_rejects_nonobject_samples_instead_of_dropping_them() -> None:
    payload = _snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["geometry"]["camera_samples"].append("not-an-object")
    with pytest.raises(ReconstructionValidationError):
        ProgressiveSfmSnapshot.from_payload(payload)
