"""Deterministic dependency-free live SLAM backend used before native integration."""

from __future__ import annotations

import numpy as np

from core.reconstruction.backends import BackendCapabilities, BackendOperationContext
from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.live_contracts import (
    LiveMapSnapshot,
    LiveSlamFrame,
    LiveSlamObservation,
    LiveSlamPose,
    LiveSlamRequest,
    LiveTrackingQuality,
    LiveTrackingState,
)
from core.reconstruction.profiles import ReconstructionQualityProfile

MOCK_LIVE_SLAM_BACKEND_ID = "development-live-slam-mock"
MOCK_LIVE_SLAM_BACKEND_VERSION = "1"


class DeterministicMockLiveSlamBackend:
    def __init__(
        self,
        *,
        initialization_frames: int = 2,
        keyframe_interval: int = 5,
        degraded_sequences: tuple[int, ...] = (),
        lost_sequences: tuple[int, ...] = (),
    ) -> None:
        if initialization_frames < 1 or keyframe_interval < 1:
            raise ReconstructionValidationError(
                "Mock live SLAM initialization_frames and keyframe_interval must be positive."
            )
        self._initialization_frames = int(initialization_frames)
        self._keyframe_interval = int(keyframe_interval)
        self._degraded = frozenset(int(value) for value in degraded_sequences)
        self._lost = frozenset(int(value) for value in lost_sequences)
        if self._degraded & self._lost:
            raise ReconstructionValidationError(
                "Mock degraded and lost frame sequences must not overlap."
            )
        self._request: LiveSlamRequest | None = None
        self._keyframes = 0
        self._landmarks = 0
        self._revision = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        from core.reconstruction.backends import ReconstructionBackendKind

        return BackendCapabilities(
            backend_id=MOCK_LIVE_SLAM_BACKEND_ID,
            display_name="Development Live SLAM Mock",
            backend_version=MOCK_LIVE_SLAM_BACKEND_VERSION,
            kinds=(ReconstructionBackendKind.LIVE_SLAM,),
            source_kinds=(
                InputSourceKind.USB_CAMERA,
                InputSourceKind.RTSP_STREAM,
                InputSourceKind.RECORDED_VIDEO,
            ),
            quality_profiles=(
                ReconstructionQualityProfile.EFFICIENT,
                ReconstructionQualityProfile.BALANCED,
                ReconstructionQualityProfile.HIGH_QUALITY,
            ),
            supports_cancellation=True,
            supports_pause=True,
            supports_checkpoint=False,
            supports_resume=False,
            supports_metric_scale=False,
            supports_loop_closure=False,
            supports_cpu=True,
            requires_cuda=False,
        )

    def prepare(self, context: BackendOperationContext) -> None:
        _require_live_context(context)

    def start_live_session(
        self,
        request: LiveSlamRequest,
        context: BackendOperationContext,
    ) -> None:
        _require_live_context(context)
        if not isinstance(request, LiveSlamRequest):
            raise TypeError("request must be LiveSlamRequest.")
        if request.backend_id != self.capabilities.backend_id:
            raise ReconstructionValidationError(
                "Live SLAM request backend_id does not match mock backend."
            )
        if request.session_id != context.session_id:
            raise ReconstructionValidationError(
                "Live SLAM request session_id does not match context."
            )
        self._request = request
        self._keyframes = 0
        self._landmarks = 0
        self._revision = 0

    def track_live_frame(
        self,
        frame: LiveSlamFrame,
        context: BackendOperationContext,
    ) -> LiveSlamObservation:
        _require_live_context(context)
        if self._request is None:
            raise ReconstructionValidationError("Mock live SLAM session has not been started.")
        if frame.session_id != self._request.session_id:
            raise ReconstructionValidationError(
                "Live frame session_id does not match active mock session."
            )
        sequence = frame.sequence_index
        if sequence < self._initialization_frames:
            state = LiveTrackingState.INITIALIZING
        elif sequence in self._lost:
            state = LiveTrackingState.LOST
        elif sequence in self._degraded:
            state = LiveTrackingState.DEGRADED
        else:
            state = LiveTrackingState.TRACKING

        if state is LiveTrackingState.INITIALIZING:
            tracked, inliers, confidence, reason = 70 + sequence * 5, 0, None, "initializing"
        elif state is LiveTrackingState.LOST:
            tracked, inliers, confidence, reason = 25, 4, 0.1, "mock_tracking_lost"
        elif state is LiveTrackingState.DEGRADED:
            tracked, inliers, confidence, reason = 90, 45, 0.5, "mock_low_inlier_support"
        else:
            tracked, inliers, confidence, reason = 180, 150, 0.93, ""

        pose = None
        if state in {LiveTrackingState.TRACKING, LiveTrackingState.DEGRADED}:
            matrix = np.eye(4, dtype=np.float64)
            matrix[0, 3] = sequence * 0.05
            matrix[1, 3] = sequence * 0.01
            pose = LiveSlamPose(
                pose_id=f"live-pose-{sequence}",
                camera_id=frame.camera_id,
                frame_id=frame.frame_id,
                timestamp_ns=frame.capture_timestamp_ns,
                camera_frame_id=self._request.camera.camera_frame.frame_id,
                T_reconstruction_from_camera=matrix,
            )

        is_keyframe = bool(
            pose is not None
            and self._request.enable_mapping
            and (sequence - self._initialization_frames) % self._keyframe_interval == 0
        )
        keyframe_id = ""
        if is_keyframe:
            keyframe_id = f"live-keyframe-{self._keyframes}"
            self._keyframes += 1
            self._landmarks += 120
            self._revision += 1

        return LiveSlamObservation(
            session_id=frame.session_id,
            frame_id=frame.frame_id,
            sequence_index=sequence,
            tracking_state=state,
            quality=LiveTrackingQuality(
                tracked_features=tracked,
                inlier_features=inliers,
                processing_latency_ns=3_000_000 + sequence * 10_000,
                tracking_confidence=confidence,
                reason=reason,
            ),
            pose=pose,
            is_keyframe=is_keyframe,
            keyframe_id=keyframe_id,
            map_snapshot=LiveMapSnapshot(
                revision=self._revision,
                keyframe_count=self._keyframes,
                landmark_count=self._landmarks,
            ),
        )

    def stop_live_session(self, context: BackendOperationContext) -> None:
        _require_live_context(context)
        self._request = None

    def shutdown(self) -> None:
        self._request = None


def _require_live_context(context: BackendOperationContext) -> None:
    from core.reconstruction.events import ReconstructionStage

    if not isinstance(context, BackendOperationContext):
        raise TypeError("context must be BackendOperationContext.")
    if context.stage is not ReconstructionStage.LIVE_TRACKING:
        raise ReconstructionValidationError("Live SLAM backend requires LIVE_TRACKING stage context.")
