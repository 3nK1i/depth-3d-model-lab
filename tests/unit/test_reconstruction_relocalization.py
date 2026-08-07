from __future__ import annotations

import numpy as np

from core.reconstruction import RelocalizationAcceptancePolicy, RelocalizationStatus


def test_relocalization_policy_accepts_strong_geometric_match() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 2.5
    result = RelocalizationAcceptancePolicy().evaluate(
        query_frame_id="frame-100",
        map_revision=7,
        matched_keyframe_id="kf-12",
        confidence=0.91,
        inlier_count=88,
        T_reconstruction_from_camera=pose,
    )
    assert result.status is RelocalizationStatus.SUCCESS
    assert result.matched_keyframe_id == "kf-12"
    assert result.T_reconstruction_from_camera is not None
    assert result.gauge.unit_to_metre is None


def test_relocalization_policy_rejects_low_confidence_without_pose_claim() -> None:
    result = RelocalizationAcceptancePolicy().evaluate(
        query_frame_id="frame-101",
        map_revision=7,
        matched_keyframe_id="kf-12",
        confidence=0.2,
        inlier_count=100,
        T_reconstruction_from_camera=np.eye(4),
    )
    assert result.status is RelocalizationStatus.REJECTED
    assert result.T_reconstruction_from_camera is None
    assert result.matched_keyframe_id == ""
