"""Dependency-light I6.7D bounded staging and repeated status-only frames."""
from __future__ import annotations

import numpy as np

from app.reconstruction_progressive_dense_viewer import build_progressive_dense_viewer_frame
from tests.i6_7c_helpers import make_snapshot


def test_status_only_updates_have_zero_staging_arrays() -> None:
    for revision in range(1, 501):
        frame = build_progressive_dense_viewer_frame(make_snapshot(revision=revision))
        assert frame.revision == revision
        assert frame.points_viewer is None
        assert frame.colors_rgba is None


def test_maximum_preview_staging_is_one_bounded_float32_frame() -> None:
    full = make_snapshot(revision=2, geometry_points=4096, dense_points=1_686_160)
    for _ in range(15):
        frame = build_progressive_dense_viewer_frame(full)
        assert frame.display_point_count == 4096
        assert frame.points_viewer.flags.c_contiguous
        assert frame.colors_rgba.flags.c_contiguous
        assert frame.points_viewer.nbytes + frame.colors_rgba.nbytes == 114_688
        assert np.isfinite(frame.points_viewer).all()
