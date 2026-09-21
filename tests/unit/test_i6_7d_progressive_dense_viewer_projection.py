from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_progressive_dense_viewer import (
    I67D_MAX_DISPLAY_POINTS,
    build_progressive_dense_viewer_frame,
)
from core.reconstruction.reconstruction_progressive_dense_transport import I67C_MAX_GEOMETRY_PREVIEW_POINTS
from tests.i6_7c_helpers import make_snapshot


def test_status_only_revisions_allocate_no_gpu_buffers():
    frame = build_progressive_dense_viewer_frame(make_snapshot(revision=1))
    assert frame.points_viewer is None
    assert frame.colors_rgba is None
    assert frame.has_geometry_update is False
    assert frame.display_point_count == 0
    assert frame.provisional and frame.viewer_profile.startswith("provisional_")


def test_preview_projection_preserves_identity_geometry_color_and_authority():
    snap = make_snapshot(revision=3, geometry_points=3)
    frame = build_progressive_dense_viewer_frame(snap)
    assert frame.dense_run_id == snap.dense_run_id
    assert frame.source_sparse_result_sha256 == snap.source_sparse_result_sha256
    assert frame.reconstruction_frame_id == snap.reconstruction_frame_id
    assert frame.current_dense_points == snap.statistics.current_dense_points
    assert frame.points_viewer.dtype == np.float32
    assert frame.colors_rgba.dtype == np.float32
    assert frame.points_viewer.flags.c_contiguous and not frame.points_viewer.flags.writeable
    assert frame.colors_rgba.flags.c_contiguous and not frame.colors_rgba.flags.writeable
    np.testing.assert_array_equal(frame.points_viewer[0], [0., -0.5, -0.25])
    np.testing.assert_array_equal(frame.points_viewer[1], [1., 0.5, -1.25])
    np.testing.assert_allclose(frame.colors_rgba[0, :3], [0, 0, 0])
    np.testing.assert_allclose(frame.colors_rgba[1, :3], [1/255, 3/255, 7/255])
    assert frame.colors_rgba[0, 3] == pytest.approx(0.88)


def test_viewer_budget_is_frozen_transport_budget():
    assert I67D_MAX_DISPLAY_POINTS == I67C_MAX_GEOMETRY_PREVIEW_POINTS == 4096
    frame = build_progressive_dense_viewer_frame(make_snapshot(geometry_points=4096, dense_points=4096))
    assert frame.points_viewer.shape == (4096, 3)
    assert frame.colors_rgba.shape == (4096, 4)
    assert frame.points_viewer.nbytes + frame.colors_rgba.nbytes == 4096 * 28


def test_invalid_source_type_rejected():
    with pytest.raises(TypeError):
        build_progressive_dense_viewer_frame(object())
