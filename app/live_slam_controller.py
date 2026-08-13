"""Qt bridge from the existing capture signal to Phase 8 live SLAM.

The bridge is intentionally UI-agnostic. Phase 8.4C wired it into the
application while Reconstruction remained disabled; Phase 8.8A2 now drives
its public lifecycle through ``ReconstructionController`` without changing
capture ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from app.application_controller import ApplicationController
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.capture.rtsp_source import RtspSource
from core.capture.video_source import VideoSource
from core.reconstruction import (
    STELLA_LIVE_BACKEND_ID,
    CameraIntrinsicsPolicy,
    ExternalStellaLiveSlamBackend,
    LiveCaptureRuntimeState,
    LiveCaptureSlamRuntime,
    LiveSlamObservation,
    LiveSlamRecoveryPolicy,
    LiveSlamRequest,
    ReconstructionCamera,
    ReconstructionQualityProfile,
    make_recorded_video_live_metadata,
    make_rtsp_source_metadata_from_identity,
    make_usb_source_metadata,
    new_reconstruction_session_id,
)
from core.reconstruction.errors import ReconstructionValidationError


@dataclass(frozen=True, slots=True)
class LiveStellaLaunchSettings:
    install_prefix: Path
    worker_executable: Path
    vocabulary_path: Path
    startup_timeout_s: float = 15.0
    frame_timeout_s: float = 5.0
    shutdown_timeout_s: float = 5.0
    require_qualified_build: bool = True
    max_worker_restarts: int = 2
    restart_backoff_s: float = 0.10
    max_consecutive_stale_frames: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "install_prefix", Path(self.install_prefix).expanduser().resolve())
        object.__setattr__(
            self, "worker_executable", Path(self.worker_executable).expanduser().resolve()
        )
        object.__setattr__(
            self, "vocabulary_path", Path(self.vocabulary_path).expanduser().resolve()
        )
        for name in ("startup_timeout_s", "frame_timeout_s", "shutdown_timeout_s"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ReconstructionValidationError(f"{name} must be positive.")
            object.__setattr__(self, name, value)
        policy = LiveSlamRecoveryPolicy(
            max_worker_restarts=self.max_worker_restarts,
            restart_backoff_s=self.restart_backoff_s,
            max_consecutive_stale_frames=self.max_consecutive_stale_frames,
        )
        object.__setattr__(self, "max_worker_restarts", policy.max_worker_restarts)
        object.__setattr__(self, "restart_backoff_s", policy.restart_backoff_s)
        object.__setattr__(
            self,
            "max_consecutive_stale_frames",
            policy.max_consecutive_stale_frames,
        )


class LiveSlamCaptureController(QObject):
    """Dormant-until-started bridge that taps ApplicationController.frame_captured."""

    state_changed = pyqtSignal(object)  # LiveCaptureRuntimeState
    observation_ready = pyqtSignal(object)  # LiveSlamObservation
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        application_controller: ApplicationController,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(application_controller, ApplicationController):
            raise TypeError("application_controller must be ApplicationController.")
        self._application_controller = application_controller
        self._runtime: LiveCaptureSlamRuntime | None = None
        application_controller.frame_captured.connect(self.on_source_frame)
        application_controller.source_type_changed.connect(self._on_source_type_changed)
        application_controller.source_status_changed.connect(self._on_source_status_changed)

    @property
    def state(self) -> LiveCaptureRuntimeState:
        runtime = self._runtime
        return runtime.state if runtime is not None else LiveCaptureRuntimeState.IDLE

    @property
    def runtime(self) -> LiveCaptureSlamRuntime | None:
        return self._runtime

    @property
    def is_active(self) -> bool:
        runtime = self._runtime
        return runtime is not None and runtime.is_active

    def start_stella_for_active_source(
        self,
        *,
        camera: ReconstructionCamera,
        settings: LiveStellaLaunchSettings,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        session_id: str | None = None,
        camera_intrinsics_policy: CameraIntrinsicsPolicy = CameraIntrinsicsPolicy.CALIBRATED_ONLY,
    ) -> str:
        if self.is_active:
            raise ReconstructionValidationError("A live SLAM capture session is already active.")
        if not isinstance(camera, ReconstructionCamera):
            raise TypeError("camera must be ReconstructionCamera.")
        if not isinstance(settings, LiveStellaLaunchSettings):
            raise TypeError("settings must be LiveStellaLaunchSettings.")
        if not isinstance(profile, ReconstructionQualityProfile):
            raise TypeError("profile must be ReconstructionQualityProfile.")
        if not isinstance(camera_intrinsics_policy, CameraIntrinsicsPolicy):
            raise TypeError("camera_intrinsics_policy must be CameraIntrinsicsPolicy.")

        source, simulated_live = self._build_active_source_metadata(camera)
        sid = session_id or new_reconstruction_session_id()
        request = LiveSlamRequest(
            session_id=sid,
            source=source,
            camera=camera,
            profile=profile,
            backend_id=STELLA_LIVE_BACKEND_ID,
            simulated_live=simulated_live,
            enable_mapping=True,
            enable_loop_closure=False,
            enable_global_optimization=False,
            metadata={"capture_owner": "ApplicationController", "phase": "8.4D"},
            camera_intrinsics_policy=camera_intrinsics_policy,
        )

        def backend_factory(ring_spec):
            return ExternalStellaLiveSlamBackend(
                install_prefix=settings.install_prefix,
                worker_executable=settings.worker_executable,
                vocabulary_path=settings.vocabulary_path,
                frame_ring_spec=ring_spec,
                startup_timeout_s=settings.startup_timeout_s,
                frame_timeout_s=settings.frame_timeout_s,
                shutdown_timeout_s=settings.shutdown_timeout_s,
                require_qualified_build=settings.require_qualified_build,
            )

        runtime = LiveCaptureSlamRuntime(
            request=request,
            backend_factory=backend_factory,
            recovery_policy=LiveSlamRecoveryPolicy(
                max_worker_restarts=settings.max_worker_restarts,
                restart_backoff_s=settings.restart_backoff_s,
                max_consecutive_stale_frames=settings.max_consecutive_stale_frames,
            ),
            observation_callback=self._emit_observation,
            state_callback=self._emit_state,
            error_callback=self.error_occurred.emit,
        )
        self._runtime = runtime
        runtime.start()
        return sid

    def on_source_frame(self, frame: object) -> None:
        runtime = self._runtime
        if runtime is None or not runtime.is_active:
            return
        try:
            runtime.offer_source_frame(frame)  # type: ignore[arg-type]
        except (TypeError, ValueError, ReconstructionValidationError) as exc:
            self.error_occurred.emit(str(exc))
            runtime.request_stop()

    def pause(self) -> bool:
        runtime = self._runtime
        return bool(runtime is not None and runtime.pause())

    def request_stop(self) -> bool:
        """Request cooperative stop without blocking the caller/UI thread."""
        runtime = self._runtime
        return bool(runtime is not None and runtime.request_stop())

    def resume(self) -> bool:
        runtime = self._runtime
        return bool(runtime is not None and runtime.resume())

    def stop(self, *, timeout_s: float = 7.0) -> bool:
        runtime = self._runtime
        if runtime is None:
            return True
        stopped = runtime.stop(timeout_s=timeout_s)
        if stopped:
            self._runtime = None
        return stopped

    def shutdown(self) -> None:
        self.stop(timeout_s=7.0)

    def _build_active_source_metadata(self, camera: ReconstructionCamera):
        source = self._application_controller.active_source
        latest = self._application_controller.latest_source_frame
        if source is None:
            raise ReconstructionValidationError("Connect a camera, RTSP stream, or video first.")
        if latest is not None and (
            latest.width != camera.image_size.width or latest.height != camera.image_size.height
        ):
            raise ReconstructionValidationError(
                "Active capture resolution does not match the selected calibrated camera profile."
            )
        if isinstance(source, CameraSource):
            return (
                make_usb_source_metadata(
                    source.device_index,
                    width=camera.image_size.width,
                    height=camera.image_size.height,
                    frame_rate_hz=source.requested_fps,
                ),
                False,
            )
        if isinstance(source, RtspSource):
            return (
                make_rtsp_source_metadata_from_identity(
                    sanitized_url=source.sanitized_url,
                    source_fingerprint=source.source_fingerprint,
                    expected_image_size=camera.image_size,
                    frame_rate_hz=source.native_fps,
                ),
                False,
            )
        if isinstance(source, VideoSource):
            return (
                make_recorded_video_live_metadata(
                    source.path,
                    expected_image_size=camera.image_size,
                    frame_rate_hz=source.native_fps,
                ),
                True,
            )
        raise ReconstructionValidationError(
            "Live SLAM accepts only USB camera, RTSP, or recorded-video sources."
        )

    def _on_source_type_changed(self, _source_type: object) -> None:
        runtime = self._runtime
        if runtime is not None and runtime.is_active:
            runtime.request_stop()

    def _on_source_status_changed(self, status: object) -> None:
        runtime = self._runtime
        if runtime is None or not isinstance(status, SourceStatus):
            return
        if status is SourceStatus.RECONNECTING:
            runtime.pause()
        elif status is SourceStatus.CONNECTED:
            runtime.resume()
        elif status in {SourceStatus.ERROR, SourceStatus.STOPPED}:
            runtime.request_stop()

    def _emit_observation(self, observation: LiveSlamObservation) -> None:
        self.observation_ready.emit(observation)

    def _emit_state(self, state: LiveCaptureRuntimeState) -> None:
        self.state_changed.emit(state)
