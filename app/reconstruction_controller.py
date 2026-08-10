"""Qt application coordinator for Phase 8 reconstruction.

Phase 8.8A1 established one dormant application-facing ownership boundary.
Phase 8.8A2 exposed the existing live-SLAM lifecycle. Phase 8.8A3 now composes
an offline process bridge for real image-sequence/recorded-video ingestion and
isolated PyCOLMAP execution without enabling Reconstruction in the UI.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QObject, pyqtSignal

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveSlamCaptureController, LiveStellaLaunchSettings
from app.offline_reconstruction_controller import (
    OFFLINE_RECONSTRUCTION_WORKER,
    OfflineReconstructionConfiguration,
    OfflineReconstructionController,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    build_offline_reconstruction_command,
)
from core.reconstruction import (
    LiveCaptureRuntimeState,
    LiveCaptureRuntimeStats,
    ReconstructionCamera,
    ReconstructionQualityProfile,
)
from core.reconstruction.errors import ReconstructionValidationError


class ReconstructionController(QObject):
    """Application-level reconstruction coordinator."""

    live_state_changed = pyqtSignal(object)  # LiveCaptureRuntimeState
    live_observation_ready = pyqtSignal(object)  # LiveSlamObservation
    offline_state_changed = pyqtSignal(object)  # OfflineReconstructionRunState
    offline_event_received = pyqtSignal(object)
    offline_log_received = pyqtSignal(str)
    offline_result_ready = pyqtSignal(object, str)
    offline_run_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        application_controller: ApplicationController,
        parent: QObject | None = None,
        *,
        offline_worker_script: Path = OFFLINE_RECONSTRUCTION_WORKER,
    ) -> None:
        super().__init__(parent)
        if not isinstance(application_controller, ApplicationController):
            raise TypeError("application_controller must be ApplicationController.")

        self._application_controller = application_controller
        self._shutdown_started = False
        self._live_session_id: str | None = None
        self._live_slam_controller = LiveSlamCaptureController(
            application_controller,
            self,
        )
        self._live_slam_controller.state_changed.connect(self.live_state_changed.emit)
        self._live_slam_controller.observation_ready.connect(self.live_observation_ready.emit)
        self._live_slam_controller.error_occurred.connect(self.error_occurred.emit)

        self._offline_controller = OfflineReconstructionController(
            self,
            worker_script=offline_worker_script,
        )
        self._offline_controller.state_changed.connect(self.offline_state_changed.emit)
        self._offline_controller.event_received.connect(self.offline_event_received.emit)
        self._offline_controller.log_received.connect(self.offline_log_received.emit)
        self._offline_controller.result_ready.connect(self.offline_result_ready.emit)
        self._offline_controller.run_finished.connect(self.offline_run_finished.emit)
        self._offline_controller.error_occurred.connect(self.error_occurred.emit)

    @property
    def live_slam_controller(self) -> LiveSlamCaptureController:
        return self._live_slam_controller

    @property
    def offline_controller(self) -> OfflineReconstructionController:
        return self._offline_controller

    @property
    def live_state(self) -> LiveCaptureRuntimeState:
        return self._live_slam_controller.state

    @property
    def live_stats(self) -> LiveCaptureRuntimeStats | None:
        runtime = self._live_slam_controller.runtime
        return runtime.stats if runtime is not None else None

    @property
    def live_session_id(self) -> str | None:
        return self._live_session_id

    @property
    def offline_state(self) -> OfflineReconstructionRunState:
        return self._offline_controller.state

    @property
    def offline_session_id(self) -> str | None:
        return self._offline_controller.session_id

    @property
    def offline_output_dir(self) -> Path | None:
        return self._offline_controller.output_dir

    @property
    def offline_is_running(self) -> bool:
        return self._offline_controller.is_running

    @property
    def is_active(self) -> bool:
        return self._live_slam_controller.is_active or self._offline_controller.is_active

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    def start_live(
        self,
        *,
        camera: ReconstructionCamera,
        settings: LiveStellaLaunchSettings,
        profile: ReconstructionQualityProfile = ReconstructionQualityProfile.EFFICIENT,
        session_id: str | None = None,
    ) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start a live session."
            )
        if self._offline_controller.is_running:
            raise ReconstructionValidationError(
                "Offline reconstruction is active; stop it before starting live reconstruction."
            )
        sid = self._live_slam_controller.start_stella_for_active_source(
            camera=camera,
            settings=settings,
            profile=profile,
            session_id=session_id,
        )
        self._live_session_id = sid
        return sid

    def pause_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.pause()

    def resume_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.resume()

    def request_stop_live(self) -> bool:
        if self._shutdown_started:
            return False
        return self._live_slam_controller.request_stop()

    def stop_live(self, *, timeout_s: float = 7.0) -> bool:
        return self._live_slam_controller.stop(timeout_s=timeout_s)

    def start_offline(
        self,
        configuration: OfflineReconstructionConfiguration,
        *,
        session_id: str | None = None,
    ) -> str:
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start an offline session."
            )
        if self._live_slam_controller.is_active:
            raise ReconstructionValidationError(
                "Live reconstruction is active; stop it before starting offline reconstruction."
            )
        return self._offline_controller.start(configuration, session_id=session_id)

    def cancel_offline(self) -> bool:
        if self._shutdown_started:
            return False
        return self._offline_controller.cancel()

    def shutdown(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._offline_controller.shutdown()
        self._live_slam_controller.shutdown()


__all__ = [
    "ReconstructionController",
    "OfflineReconstructionConfiguration",
    "OfflineReconstructionRunState",
    "OfflineReconstructionSourceKind",
    "build_offline_reconstruction_command",
]
