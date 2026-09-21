from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_progressive_sfm_viewer import (
    I66D_VIEWER_PROFILE,
    build_progressive_sfm_viewer_frame,
)
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    I66A_PYCOLMAP_BACKEND_ID,
    ProgressiveSfmSnapshot,
    ProgressiveSfmStage,
    ProgressiveSfmStatistics,
)
from tests.i6_6d_helpers import REQUEST_SHA, SESSION_ID, make_progressive_view_snapshot


def test_i6_6d_projects_bounded_geometry_without_authority_upgrade() -> None:
    snapshot = make_progressive_view_snapshot(revision=3)
    frame = build_progressive_sfm_viewer_frame(snapshot)
    assert frame.session_id == SESSION_ID
    assert frame.revision == 3
    assert frame.stage is ProgressiveSfmStage.MAPPING
    assert frame.viewer_profile == I66D_VIEWER_PROFILE
    assert frame.provisional is True
    assert frame.camera_matrices_reconstruction.shape == (3, 4, 4)
    assert frame.landmark_points_reconstruction.shape == (5, 3)
    assert frame.landmark_colors_rgb is not None
    assert frame.landmark_colors_rgb.shape == (5, 3)
    assert frame.registered_camera_total == 7
    assert frame.landmark_total == 101
    assert frame.camera_ids == ("frame-000.jpg", "frame-001.jpg", "frame-002.jpg")


def test_i6_6d_viewer_arrays_are_read_only_and_independent() -> None:
    frame = build_progressive_sfm_viewer_frame(make_progressive_view_snapshot())
    assert frame.camera_matrices_reconstruction.flags.c_contiguous
    assert frame.landmark_points_reconstruction.flags.c_contiguous
    assert frame.camera_matrices_reconstruction.flags.writeable is False
    assert frame.landmark_points_reconstruction.flags.writeable is False
    assert frame.landmark_colors_rgb is not None
    assert frame.landmark_colors_rgb.flags.writeable is False
    with pytest.raises(ValueError):
        frame.landmark_points_reconstruction[0, 0] = 99.0


def test_i6_6d_omits_color_buffer_if_samples_are_not_uniformly_colored() -> None:
    snapshot = make_progressive_view_snapshot(include_colors=False)
    frame = build_progressive_sfm_viewer_frame(snapshot)
    assert frame.landmark_colors_rgb is None
    assert frame.landmark_points_reconstruction.shape[0] == 5


def test_i6_6d_geometry_absence_projects_empty_replace_buffers() -> None:
    snapshot = ProgressiveSfmSnapshot(
        session_id=SESSION_ID,
        revision=1,
        timestamp_utc="2026-09-17T12:00:00+00:00",
        elapsed_ns=1,
        stage=ProgressiveSfmStage.MAPPING,
        stage_detail="mapping_without_preview_geometry",
        backend_id=I66A_PYCOLMAP_BACKEND_ID,
        backend_version="1",
        request_fingerprint_sha256=REQUEST_SHA,
        statistics=ProgressiveSfmStatistics(
            input_views=48,
            registered_cameras=2,
            sparse_landmarks=0,
            observations=0,
        ),
        progress_fraction=0.2,
        geometry=None,
    )
    frame = build_progressive_sfm_viewer_frame(snapshot)
    assert frame.has_geometry is False
    assert frame.camera_matrices_reconstruction.shape == (0, 4, 4)
    assert frame.landmark_points_reconstruction.shape == (0, 3)
    assert frame.camera_ids == ()


def test_i6_6d_rejects_non_snapshot_input() -> None:
    with pytest.raises(TypeError):
        build_progressive_sfm_viewer_frame({})  # type: ignore[arg-type]
