"""Dormant Qt application coordinator for Phase 8 reconstruction.

Phase 8.8A1 establishes one application-facing ownership boundary without
starting reconstruction work or enabling Reconstruction in the UI.  The
controller owns the already-qualified :class:`LiveSlamCaptureController` and
forwards its existing runtime contracts; later 8.8A substeps can add live and
offline lifecycle entry points here without duplicating capture ownership or
introducing a second reconstruction state machine.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveSlamCaptureController
from core.reconstruction import LiveCaptureRuntimeState


class ReconstructionController(QObject):
    """Application-level reconstruction coordinator, dormant until commanded."""

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
    def is_active(self) -> bool:
        """Whether any currently integrated reconstruction runtime is active."""
        return self._live_slam_controller.is_active

    @property
    def is_shutdown(self) -> bool:
        return self._shutdown_started

    def shutdown(self) -> None:
        """Deterministically stop owned reconstruction resources once."""
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._live_slam_controller.shutdown()
