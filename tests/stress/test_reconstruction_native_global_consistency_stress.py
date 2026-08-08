from __future__ import annotations

import numpy as np

from core.reconstruction import native_global_consistency_status_from_payload
from tests.reconstruction_helpers import SESSION_ID


def test_native_global_consistency_status_parsing_is_bounded_and_deterministic() -> None:
    identity = np.eye(4, dtype=np.float64).tolist()
    for revision in range(1, 501):
        payload = {
            "loop_detector_enabled": True,
            "global_optimization_enabled": True,
            "loop_ba_running": revision % 7 == 0,
            "map_revision": revision,
            "optimized_pose_revision": revision,
            "keyframe_count": 2,
            "landmark_count": 200 + revision,
            "accepted_loop_closure_requests": revision // 100,
            "relocalization_count": revision // 125,
            "last_relocalization_frame_id": "" if revision < 125 else f"frame-{revision}",
            "optimized_keyframe_poses": [
                {
                    "keyframe_id": "stella-keyframe-0",
                    "frame_id": "stella-keyframe-0",
                    "timestamp_ns": 1,
                    "source_pose_id": "stella-native-keyframe-pose-0",
                    "pose_convention": "T_reconstruction_from_camera",
                    "T_reconstruction_from_camera": identity,
                },
                {
                    "keyframe_id": "stella-keyframe-1",
                    "frame_id": "stella-keyframe-1",
                    "timestamp_ns": 2,
                    "source_pose_id": "stella-native-keyframe-pose-1",
                    "pose_convention": "T_reconstruction_from_camera",
                    "T_reconstruction_from_camera": identity,
                },
            ],
        }
        status = native_global_consistency_status_from_payload(
            payload, expected_session_id=SESSION_ID
        )
        assert status.optimized_pose_revision == revision
        assert status.keyframe_count == 2
        assert status.gauge.unit_to_metre is None
