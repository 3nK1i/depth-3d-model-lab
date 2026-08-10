"""Qt application coordinator for Phase 8 reconstruction.

Phase 8.8A1 established one dormant application-facing ownership boundary.
Phase 8.8A2 adds the public live-reconstruction lifecycle on top of the already
qualified :class:`LiveSlamCaptureController` without duplicating capture,
translating the authoritative live runtime state, or enabling Reconstruction in
the UI. Offline reconstruction remains deferred to Phase 8.8A3.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveSlamCaptureController, LiveStellaLaunchSettings
from core.reconstruction import (
    LiveCaptureRuntimeState,
    LiveCaptureRuntimeStats,
    ReconstructionCamera,
    ReconstructionQualityProfile,
)
from core.reconstruction.errors import ReconstructionValidationError


class ReconstructionController(QObject):
    """Application-level reconstruction coordinator.

    The controller owns the existing live-SLAM capture bridge. Public live
    lifecycle methods are intentionally thin delegates so the established
    runtime remains the only live state machine and ``ApplicationController``
    remains the only capture owner.
    """

    live_state_changed = pyqtSignal(object)  # LiveCaptureRuntimeState
    live_observation_ready = pyqtSignal(object)  # LiveSlamObservation
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
        self._shutdown_started = False
        self._live_session_id: str | None = None
        self._live_slam_controller = LiveSlamCaptureController(
            application_controller,
            self,
        )
        self._live_slam_controller.state_changed.connect(self.live_state_changed.emit)
        self._live_slam_controller.observation_ready.connect(
            self.live_observation_ready.emit
        )
        self._live_slam_controller.error_occurred.connect(self.error_occurred.emit)

    @property
    def live_slam_controller(self) -> LiveSlamCaptureController:
        """Return the existing Phase 8.4 live-SLAM bridge owned by this coordinator."""
        return self._live_slam_controller

    @property
    def live_state(self) -> LiveCaptureRuntimeState:
        """Expose the authoritative live runtime state without translating it."""
        return self._live_slam_controller.state

    @property
    def live_stats(self) -> LiveCaptureRuntimeStats | None:
        """Return current/last live runtime statistics while its runtime is retained."""
        runtime = self._live_slam_controller.runtime
        return runtime.stats if runtime is not None else None

    @property
    def live_session_id(self) -> str | None:
        """Identity of the most recently started live reconstruction session."""
        return self._live_session_id

    @property
    def is_active(self) -> bool:
        """Whether any currently integrated reconstruction runtime is active."""
        return self._live_slam_controller.is_active

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
        """Start live reconstruction for the source already owned by the application.

        Source validation, metadata construction, shared-memory publication and
        isolated stella worker launch remain owned by ``LiveSlamCaptureController``.
        This method intentionally adds no second capture or reconstruction state
        machine.
        """
        if self._shutdown_started:
            raise ReconstructionValidationError(
                "Reconstruction controller is shut down and cannot start a live session."
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
        """Pause live reconstruction without pausing or replacing capture ownership."""
        if self._shutdown_started:
            return False
        return self._live_slam_controller.pause()

    def resume_live(self) -> bool:
        """Resume a paused live reconstruction runtime."""
        if self._shutdown_started:
            return False
        return self._live_slam_controller.resume()

    def request_stop_live(self) -> bool:
        """Request non-blocking live stop for future UI actions."""
        if self._shutdown_started:
            return False
        return self._live_slam_controller.request_stop()

    def stop_live(self, *, timeout_s: float = 7.0) -> bool:
        """Synchronously finish live teardown for controlled non-UI boundaries/tests."""
        return self._live_slam_controller.stop(timeout_s=timeout_s)

    def shutdown(self) -> None:
        """Deterministically stop owned reconstruction resources once."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._live_slam_controller.shutdown()
