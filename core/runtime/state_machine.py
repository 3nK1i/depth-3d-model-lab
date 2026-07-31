"""The application's explicit state machine (spec §14).

Every runtime button's enabled/disabled state derives from exactly one
place — this class — rather than being toggled ad hoc across widgets.
This is what the Phase 1 demo lifecycle in main_window.py was always
documented as a stand-in for.
"""

from __future__ import annotations

from enum import Enum

from PyQt6.QtCore import QObject, pyqtSignal


class ApplicationState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    SOURCE_READY = "source_ready"
    LOADING_MODEL = "loading_model"
    MODEL_READY = "model_ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    SWITCHING_MODEL = "switching_model"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


# Explicit transition table: {from_state: {allowed_to_states}}. A
# transition not listed here is rejected — invalid transitions are a
# programming error, not something to silently allow.
_ALLOWED_TRANSITIONS: dict[ApplicationState, frozenset[ApplicationState]] = {
    ApplicationState.DISCONNECTED: frozenset({ApplicationState.CONNECTING}),
    ApplicationState.CONNECTING: frozenset(
        {ApplicationState.SOURCE_READY, ApplicationState.ERROR, ApplicationState.DISCONNECTED}
    ),
    ApplicationState.SOURCE_READY: frozenset(
        {
            ApplicationState.LOADING_MODEL,
            ApplicationState.DISCONNECTED,
            ApplicationState.RECONNECTING,
            ApplicationState.ERROR,
            ApplicationState.SHUTTING_DOWN,
        }
    ),
    ApplicationState.LOADING_MODEL: frozenset(
        {ApplicationState.MODEL_READY, ApplicationState.ERROR, ApplicationState.SOURCE_READY}
    ),
    ApplicationState.MODEL_READY: frozenset(
        {
            ApplicationState.RUNNING,
            ApplicationState.SWITCHING_MODEL,
            ApplicationState.DISCONNECTED,
            ApplicationState.RECONNECTING,
            ApplicationState.ERROR,
            ApplicationState.SHUTTING_DOWN,
        }
    ),
    ApplicationState.RUNNING: frozenset(
        {
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.RECONNECTING,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.PAUSED: frozenset(
        {ApplicationState.RUNNING, ApplicationState.STOPPING, ApplicationState.ERROR}
    ),
    ApplicationState.STOPPING: frozenset(
        {ApplicationState.MODEL_READY, ApplicationState.ERROR}
    ),
    ApplicationState.SWITCHING_MODEL: frozenset(
        {ApplicationState.MODEL_READY, ApplicationState.ERROR, ApplicationState.SOURCE_READY}
    ),
    ApplicationState.RECONNECTING: frozenset(
        {
            ApplicationState.SOURCE_READY,
            ApplicationState.MODEL_READY,
            ApplicationState.DISCONNECTED,
            ApplicationState.ERROR,
        }
    ),
    ApplicationState.ERROR: frozenset(
        {ApplicationState.DISCONNECTED, ApplicationState.SOURCE_READY, ApplicationState.SHUTTING_DOWN}
    ),
    ApplicationState.SHUTTING_DOWN: frozenset(),
}


class InvalidTransitionError(Exception):
    """Raised when a transition isn't in the explicit allowed table."""


class ButtonStates:
    """Snapshot of which runtime actions are available in a given state.

    One dataclass-like bundle rather than scattering ``state ==
    ApplicationState.X`` checks across widget code (spec §14: "centralize
    state transitions").
    """

    __slots__ = (
        "connect_enabled",
        "disconnect_enabled",
        "start_enabled",
        "pause_resume_enabled",
        "stop_enabled",
        "is_paused",
    )

    def __init__(
        self,
        *,
        connect_enabled: bool,
        disconnect_enabled: bool,
        start_enabled: bool,
        pause_resume_enabled: bool,
        stop_enabled: bool,
        is_paused: bool = False,
    ) -> None:
        self.connect_enabled = connect_enabled
        self.disconnect_enabled = disconnect_enabled
        self.start_enabled = start_enabled
        self.pause_resume_enabled = pause_resume_enabled
        self.stop_enabled = stop_enabled
        self.is_paused = is_paused


def button_states_for(state: ApplicationState) -> ButtonStates:
    """The single source of truth for which runtime buttons are enabled."""
    if state == ApplicationState.DISCONNECTED:
        return ButtonStates(
            connect_enabled=True,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    if state in (ApplicationState.CONNECTING, ApplicationState.RECONNECTING):
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    if state in (
        ApplicationState.SOURCE_READY,
        ApplicationState.LOADING_MODEL,
        ApplicationState.MODEL_READY,
    ):
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=(state == ApplicationState.MODEL_READY or state == ApplicationState.SOURCE_READY),
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    if state == ApplicationState.RUNNING:
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=False,
            pause_resume_enabled=True,
            stop_enabled=True,
            is_paused=False,
        )
    if state == ApplicationState.PAUSED:
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=False,
            pause_resume_enabled=True,
            stop_enabled=True,
            is_paused=True,
        )
    if state == ApplicationState.STOPPING:
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    if state == ApplicationState.SWITCHING_MODEL:
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    if state == ApplicationState.ERROR:
        return ButtonStates(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )
    # SHUTTING_DOWN and any future state: everything disabled.
    return ButtonStates(
        connect_enabled=False,
        disconnect_enabled=False,
        start_enabled=False,
        pause_resume_enabled=False,
        stop_enabled=False,
    )


class StateMachine(QObject):
    """Owns the current :class:`ApplicationState` and validates transitions."""

    state_changed = pyqtSignal(object, object)  # (old_state, new_state)
    transition_rejected = pyqtSignal(object, object)  # (from_state, attempted_to)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._state = ApplicationState.DISCONNECTED

    @property
    def state(self) -> ApplicationState:
        return self._state

    @property
    def button_states(self) -> ButtonStates:
        return button_states_for(self._state)

    def can_transition_to(self, target: ApplicationState) -> bool:
        return target in _ALLOWED_TRANSITIONS.get(self._state, frozenset())

    def transition_to(self, target: ApplicationState, *, strict: bool = True) -> bool:
        """Move to ``target`` if allowed from the current state.

        Returns True on success. If the transition isn't allowed: raises
        :class:`InvalidTransitionError` when ``strict`` (the default —
        an invalid transition is a bug), or returns False and emits
        ``transition_rejected`` when ``strict=False`` (for call sites
        that want to handle it gracefully, e.g. a race between a user
        click and an async event).
        """
        if not self.can_transition_to(target):
            self.transition_rejected.emit(self._state, target)
            if strict:
                raise InvalidTransitionError(
                    f"Cannot transition from {self._state} to {target}"
                )
            return False

        old_state = self._state
        self._state = target
        self.state_changed.emit(old_state, target)
        return True