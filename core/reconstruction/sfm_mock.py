"""Deterministic dependency-free offline SfM backend used before real PyCOLMAP execution."""

from __future__ import annotations

from time import monotonic_ns

import numpy as np

from core.reconstruction.backends import (
    BackendCapabilities,
    BackendOperationContext,
    ReconstructionBackendKind,
)
from core.reconstruction.contracts import InputSourceKind
from core.reconstruction.events import (
    ReconstructionEventRecorder,
    ReconstructionEventType,
    ReconstructionStage,
)
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.sfm_contracts import (
    OfflineSfmRequest,
    OfflineSfmResult,
    ReconstructionGauge,
    SfmCameraEstimate,
    SfmLandmarkEstimate,
    SfmObservation,
    SfmStageStatistics,
)


class DeterministicMockSfmBackend:
    """Create a repeatable arbitrary-gauge sparse model for contract validation."""

    def __init__(self) -> None:
        self._prepared = False
        self._events: ReconstructionEventRecorder | None = None
        self._started_ns = 0

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_id="development-mock-offline-sfm",
            display_name="Development Mock Offline SfM",
            backend_version="1",
            kinds=(ReconstructionBackendKind.OFFLINE_SFM,),
            source_kinds=(InputSourceKind.IMAGE_SEQUENCE, InputSourceKind.RECORDED_VIDEO),
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
            supports_cpu=True,
            requires_cuda=False,
        )

    def _emit(
        self,
        context: BackendOperationContext,
        *,
        event_type: ReconstructionEventType,
        message: str,
        progress: float | None = None,
    ) -> None:
        if self._events is None:
            self._events = ReconstructionEventRecorder(context.session_id)
        event = self._events.emit(
            event_type=event_type,
            stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
            message=message,
            elapsed_ns=max(0, monotonic_ns() - self._started_ns),
            progress_fraction=progress,
        )
        context.emit_event(event)

    def prepare(self, context: BackendOperationContext) -> None:
        context.control.checkpoint()
        self._started_ns = monotonic_ns()
        self._events = ReconstructionEventRecorder(context.session_id)
        self._prepared = True
        self._emit(
            context,
            event_type=ReconstructionEventType.STATUS,
            message="Mock offline SfM backend prepared.",
        )

    def run_sparse_reconstruction(
        self,
        request: OfflineSfmRequest,
        context: BackendOperationContext,
    ) -> OfflineSfmResult:
        if not self._prepared:
            self.prepare(context)
        if request.session_id != context.session_id:
            raise ValueError("SfM request and backend context session IDs must match.")

        images = request.workspace.images
        cameras: list[SfmCameraEstimate] = []
        self._emit(
            context,
            event_type=ReconstructionEventType.PROGRESS,
            message="Generating deterministic mock camera trajectory.",
            progress=0.1,
        )
        for index, image in enumerate(images):
            context.control.checkpoint()
            matrix = np.eye(4, dtype=np.float64)
            matrix[0, 3] = float(index) * 0.75
            matrix[1, 3] = float(index % 2) * 0.05
            cameras.append(
                SfmCameraEstimate(
                    estimate_id=f"mock-pose-{index:06d}",
                    camera_id="mock-camera-0",
                    keyframe_id=image.keyframe_id,
                    frame_id=image.frame_id,
                    backend_image_id=index + 1,
                    T_reconstruction_from_camera=matrix,
                )
            )

        self._emit(
            context,
            event_type=ReconstructionEventType.PROGRESS,
            message="Generating deterministic mock sparse landmarks.",
            progress=0.55,
        )
        landmarks: list[SfmLandmarkEstimate] = []
        landmark_count = max(1, len(images) - 1)
        for index in range(landmark_count):
            context.control.checkpoint()
            left = images[index % (len(images) - 1)]
            right = images[(index % (len(images) - 1)) + 1]
            landmark_id = f"mock-landmark-{index:06d}"
            observations = (
                SfmObservation(
                    observation_id=f"mock-observation-{index:06d}-a",
                    landmark_id=landmark_id,
                    keyframe_id=left.keyframe_id,
                    feature_xy=(left.image_size.width * 0.45, left.image_size.height * 0.5),
                    feature_index=index * 2,
                    backend_point2d_index=index * 2,
                ),
                SfmObservation(
                    observation_id=f"mock-observation-{index:06d}-b",
                    landmark_id=landmark_id,
                    keyframe_id=right.keyframe_id,
                    feature_xy=(right.image_size.width * 0.55, right.image_size.height * 0.5),
                    feature_index=index * 2 + 1,
                    backend_point2d_index=index * 2 + 1,
                ),
            )
            landmarks.append(
                SfmLandmarkEstimate(
                    landmark_id=landmark_id,
                    track_id=f"mock-track-{index:06d}",
                    position_reconstruction=np.array(
                        [index * 0.3 + 0.1, ((index % 3) - 1) * 0.12, 2.0 + index * 0.08],
                        dtype=np.float64,
                    ),
                    observations=observations,
                    color_rgb=(80 + index % 120, 140, 210),
                )
            )

        elapsed = max(0, monotonic_ns() - self._started_ns)
        statistics = SfmStageStatistics(
            input_keyframes=len(images),
            registered_keyframes=len(cameras),
            landmarks=len(landmarks),
            observations=sum(len(item.observations) for item in landmarks),
            elapsed_ns=elapsed,
            mean_reprojection_error_px=0.5,
        )
        self._emit(
            context,
            event_type=ReconstructionEventType.PROGRESS,
            message="Mock sparse reconstruction complete.",
            progress=1.0,
        )
        return OfflineSfmResult(
            session_id=request.session_id,
            backend_id=self.capabilities.backend_id,
            backend_version=self.capabilities.backend_version,
            gauge=ReconstructionGauge(),
            cameras=tuple(cameras),
            landmarks=tuple(landmarks),
            statistics=statistics,
            workspace=request.workspace,
            warnings=(
                "Development mock output only; coordinates have arbitrary scale and are not real SfM.",
            ),
            metadata={"deterministic": True, "real_sfm": False},
        )

    def shutdown(self) -> None:
        self._prepared = False
        self._events = None
        self._started_ns = 0
