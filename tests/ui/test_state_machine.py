"""Explicit state-machine and runtime-button tests from Step 0 onward."""

from __future__ import annotations

import pytest

from core.runtime.state_machine import (
    ApplicationState,
    InvalidTransitionError,
    StateMachine,
    button_states_for,
)


def test_happy_path_connect_load_run_pause_stop() -> None:
    machine = StateMachine()

    for target in (
        ApplicationState.CONNECTING,
        ApplicationState.SOURCE_READY,
        ApplicationState.LOADING_MODEL,
        ApplicationState.MODEL_READY,
        ApplicationState.RUNNING,
        ApplicationState.PAUSED,
        ApplicationState.RUNNING,
        ApplicationState.STOPPING,
        ApplicationState.MODEL_READY,
    ):
        assert machine.transition_to(target)


def test_static_image_result_ready_can_run_again() -> None:
    machine = StateMachine()
    machine.transition_to(ApplicationState.CONNECTING)
    machine.transition_to(ApplicationState.SOURCE_READY)
    machine.transition_to(ApplicationState.LOADING_MODEL)
    machine.transition_to(ApplicationState.MODEL_READY)
    machine.transition_to(ApplicationState.RUNNING)
    machine.transition_to(ApplicationState.RESULT_READY)

    assert machine.button_states.start_enabled
    assert not machine.button_states.pause_resume_enabled
    assert machine.transition_to(ApplicationState.RUNNING)


def test_invalid_transition_raises_and_keeps_state(qtbot) -> None:
    machine = StateMachine()

    with qtbot.waitSignal(machine.transition_rejected, timeout=500) as rejected:
        with pytest.raises(InvalidTransitionError):
            machine.transition_to(ApplicationState.RUNNING)

    assert rejected.args == [ApplicationState.DISCONNECTED, ApplicationState.RUNNING]
    assert machine.state is ApplicationState.DISCONNECTED


def test_non_strict_invalid_transition_returns_false() -> None:
    machine = StateMachine()
    assert machine.transition_to(ApplicationState.PAUSED, strict=False) is False
    assert machine.state is ApplicationState.DISCONNECTED


@pytest.mark.parametrize(
    ("state", "connect", "disconnect", "start", "pause", "stop", "is_paused"),
    [
        (ApplicationState.DISCONNECTED, True, False, False, False, False, False),
        (ApplicationState.SOURCE_READY, False, True, True, False, False, False),
        (ApplicationState.MODEL_READY, False, True, True, False, False, False),
        (ApplicationState.RESULT_READY, False, True, True, False, False, False),
        (ApplicationState.RUNNING, False, True, False, True, True, False),
        (ApplicationState.PAUSED, False, True, False, True, True, True),
        (ApplicationState.STOPPING, False, False, False, False, False, False),
        (ApplicationState.ERROR, False, True, False, False, False, False),
        (ApplicationState.SHUTTING_DOWN, False, False, False, False, False, False),
    ],
)
def test_button_states_are_centralized(
    state, connect, disconnect, start, pause, stop, is_paused
) -> None:
    buttons = button_states_for(state)
    assert buttons.connect_enabled is connect
    assert buttons.disconnect_enabled is disconnect
    assert buttons.start_enabled is start
    assert buttons.pause_resume_enabled is pause
    assert buttons.stop_enabled is stop
    assert buttons.is_paused is is_paused
