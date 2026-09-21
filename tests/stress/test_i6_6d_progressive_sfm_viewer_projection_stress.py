from __future__ import annotations

from app.reconstruction_progressive_sfm_viewer import build_progressive_sfm_viewer_frame
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6d_helpers import make_progressive_view_snapshot


def test_i6_6d_repeated_projection_retains_only_current_frame_shape() -> None:
    latest = None
    for revision in range(1, 1001):
        latest = build_progressive_sfm_viewer_frame(
            make_progressive_view_snapshot(
                revision=revision,
                camera_sample_count=4,
                landmark_sample_count=7,
                registered_camera_total=12,
                landmark_total=250,
            )
        )
        assert latest.camera_matrices_reconstruction.shape == (4, 4, 4)
        assert latest.landmark_points_reconstruction.shape == (7, 3)
    assert latest is not None
    assert latest.revision == 1000
    assert latest.registered_camera_total == 12
    assert latest.landmark_total == 250


def test_i6_6d_finalizing_projection_does_not_gain_final_authority() -> None:
    frame = build_progressive_sfm_viewer_frame(
        make_progressive_view_snapshot(
            revision=2,
            stage=ProgressiveSfmStage.FINALIZING,
        )
    )
    assert frame.stage is ProgressiveSfmStage.FINALIZING
    assert frame.provisional is True
