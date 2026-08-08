from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.reconstruction import (
    ExternalStellaLiveSlamBackend,
    LiveSharedMemoryFrameRing,
    LiveSlamMessageType,
    LiveSlamRequest,
    NativeGlobalConsistencyStatus,
    ReconstructionValidationError,
    STELLA_LIVE_BACKEND_ID,
    make_global_consistency_status_request_message,
    make_native_loop_closure_request_message,
    native_global_consistency_status_from_payload,
    stella_monocular_config_text,
)
from tests.reconstruction_helpers import SESSION_ID
from tests.reconstruction_live_helpers import LIVE_IMAGE_SIZE, make_live_request


def test_generic_live_request_allows_phase8_5_features_with_dependency() -> None:
    base = make_live_request()
    loop_only = replace(base, enable_loop_closure=True)
    assert loop_only.enable_loop_closure is True
    assert loop_only.enable_global_optimization is False

    with pytest.raises(ReconstructionValidationError, match="requires loop closure"):
        LiveSlamRequest(
            session_id=base.session_id,
            source=base.source,
            camera=base.camera,
            profile=base.profile,
            backend_id=base.backend_id,
            enable_global_optimization=True,
        )


def test_stella_phase8_5_config_tracks_explicit_loop_enablement() -> None:
    base = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    disabled = stella_monocular_config_text(base)
    enabled = stella_monocular_config_text(
        replace(base, enable_loop_closure=True, enable_global_optimization=True)
    )
    assert "LoopDetector:\n  enabled: false" in disabled
    assert "LoopDetector:\n  enabled: true" in enabled


def test_stella_backend_advertises_loop_closure_but_keeps_default_off(tmp_path) -> None:
    ring = LiveSharedMemoryFrameRing.create(
        slot_count=2,
        max_payload_bytes=LIVE_IMAGE_SIZE.width * LIVE_IMAGE_SIZE.height * 3,
    )
    try:
        backend = ExternalStellaLiveSlamBackend(
            install_prefix=tmp_path / "prefix",
            worker_executable=tmp_path / "worker",
            vocabulary_path=tmp_path / "vocab",
            frame_ring_spec=ring.spec,
            require_qualified_build=False,
        )
        assert backend.capabilities.supports_loop_closure is True
        request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
        assert request.enable_loop_closure is False
        assert request.enable_global_optimization is False
    finally:
        ring.shutdown()


def test_phase8_5_protocol_builds_native_control_messages() -> None:
    loop = make_native_loop_closure_request_message(
        SESSION_ID,
        from_keyframe_id="stella-keyframe-2",
        to_keyframe_id="stella-keyframe-19",
        sequence_number=7,
    )
    status = make_global_consistency_status_request_message(SESSION_ID, sequence_number=8)
    assert loop.message_type is LiveSlamMessageType.REQUEST_LOOP_CLOSURE
    assert status.message_type is LiveSlamMessageType.GLOBAL_CONSISTENCY_STATUS_REQUEST


def test_native_status_parser_preserves_optimized_layer_and_arbitrary_gauge() -> None:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = 0.25
    status = native_global_consistency_status_from_payload(
        {
            "loop_detector_enabled": True,
            "global_optimization_enabled": True,
            "loop_ba_running": False,
            "map_revision": 8,
            "optimized_pose_revision": 3,
            "keyframe_count": 1,
            "landmark_count": 123,
            "accepted_loop_closure_requests": 1,
            "relocalization_count": 2,
            "last_relocalization_frame_id": "live-frame-91",
            "optimized_keyframe_poses": [
                {
                    "keyframe_id": "stella-keyframe-7",
                    "frame_id": "stella-keyframe-7",
                    "timestamp_ns": 1_500_000_000,
                    "source_pose_id": "stella-native-keyframe-pose-7",
                    "pose_convention": "T_reconstruction_from_camera",
                    "T_reconstruction_from_camera": matrix.tolist(),
                }
            ],
        },
        expected_session_id=SESSION_ID,
    )
    assert isinstance(status, NativeGlobalConsistencyStatus)
    assert status.gauge.unit_to_metre is None
    assert status.optimized_pose_revision == 3
    assert status.optimized_keyframe_poses[0].as_dict()["pose_layer"] == "optimized"
    assert np.allclose(status.optimized_keyframe_poses[0].T_reconstruction_from_camera, matrix)
