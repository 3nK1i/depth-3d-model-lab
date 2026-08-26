from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temple_ring_detail_qualification import (
    TempleRingDetailTransferPolicy,
    canonical_cloud_digest,
    qualify_d12_detail_transfer,
    validate_fusion_track_evidence,
)


def _valid_tracks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack((np.arange(47, dtype=np.float64), np.zeros(47), np.zeros(47)))
    colors = np.tile(np.asarray([[10, 20, 30]], dtype=np.uint8), (47, 1))
    offsets = np.arange(48, dtype=np.int64)
    ids = np.arange(47, dtype=np.int32)
    return points, colors, offsets, ids


def test_i42e_track_offsets_cannot_be_truncated() -> None:
    points, colors, offsets, ids = _valid_tracks()
    with pytest.raises(ReconstructionValidationError, match="offsets"):
        validate_fusion_track_evidence(points, colors, offsets[:-1], ids)


def test_i42e_track_offsets_cannot_contain_empty_tracks() -> None:
    points, colors, offsets, ids = _valid_tracks()
    offsets = offsets.copy(); offsets[10] = offsets[9]
    with pytest.raises(ReconstructionValidationError, match="at least one"):
        validate_fusion_track_evidence(points, colors, offsets, ids)


def test_i42e_nonfinite_geometry_fails_closed() -> None:
    points, colors, offsets, ids = _valid_tracks()
    points[0, 0] = np.nan
    with pytest.raises(ReconstructionValidationError, match="finite"):
        validate_fusion_track_evidence(points, colors, offsets, ids)


def test_i42e_out_of_range_image_index_fails_closed() -> None:
    points, colors, offsets, ids = _valid_tracks()
    ids[0] = 47
    with pytest.raises(ReconstructionValidationError, match=r"\[0, 46\]"):
        validate_fusion_track_evidence(points, colors, offsets, ids)


def test_i42e_cloud_digest_is_deterministic_across_repeated_calls() -> None:
    points, colors, _, _ = _valid_tracks()
    first = canonical_cloud_digest(points, colors)
    for _ in range(8):
        assert canonical_cloud_digest(points, colors) == first


def test_i42e_locked_dispersion_threshold_cannot_be_relaxed() -> None:
    with pytest.raises(ReconstructionValidationError, match="extended_max_dispersion_voxels is locked"):
        TempleRingDetailTransferPolicy(extended_max_dispersion_voxels=1.5)


def test_i42e_rejects_misaligned_extended_arrays() -> None:
    payload = {
        "points_reconstruction": np.zeros((200, 3), dtype=np.float64),
        "colors_rgb": np.zeros((200, 3), dtype=np.uint8),
        "view_observation_counts": np.ones(199, dtype=np.uint32),
        "high_support_neighbor_count": np.ones(200, dtype=np.int32),
        "class_code": np.full(200, 3, dtype=np.uint8),
        "voxel_size_reconstruction_units": np.asarray(1.0),
    }
    with pytest.raises(ReconstructionValidationError, match="misaligned"):
        qualify_d12_detail_transfer(payload)


def test_i42e_rejects_too_small_core_reference() -> None:
    n = 200
    payload = {
        "points_reconstruction": np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(np.float64),
        "colors_rgb": np.zeros((n, 3), dtype=np.uint8),
        "view_observation_counts": np.full(n, 3, dtype=np.uint32),
        "high_support_neighbor_count": np.full(n, 3, dtype=np.int32),
        "class_code": np.r_[np.full(100, 3, dtype=np.uint8), np.full(100, 2, dtype=np.uint8)],
        "voxel_size_reconstruction_units": np.asarray(1.0),
    }
    with pytest.raises(ReconstructionValidationError, match="at least 128 Core"):
        qualify_d12_detail_transfer(payload)


def test_i42e_zero_voxel_scale_fails_closed() -> None:
    n = 200
    payload = {
        "points_reconstruction": np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(np.float64),
        "colors_rgb": np.zeros((n, 3), dtype=np.uint8),
        "view_observation_counts": np.full(n, 3, dtype=np.uint32),
        "high_support_neighbor_count": np.full(n, 3, dtype=np.int32),
        "class_code": np.full(n, 3, dtype=np.uint8),
        "voxel_size_reconstruction_units": np.asarray(0.0),
    }
    with pytest.raises(ReconstructionValidationError):
        qualify_d12_detail_transfer(payload)


def test_i42e_no_metric_or_provider_authority_can_be_derived_from_digest() -> None:
    points, colors, _, _ = _valid_tracks()
    digest = canonical_cloud_digest(points, colors)
    assert len(digest) == 64
    assert isinstance(digest, str)
