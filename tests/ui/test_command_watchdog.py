"""Worker-command timeout tests without waiting several seconds."""

from __future__ import annotations

from core.runtime.messages import CommandType
from core.runtime.watchdog import CommandWatchdog


def test_watchdog_emits_pending_command_on_timeout(qtbot) -> None:
    watchdog = CommandWatchdog(timeout_ms=20)
    with qtbot.waitSignal(watchdog.timeout_exceeded, timeout=500) as signal:
        watchdog.watch(CommandType.STOP)
    assert signal.args == [CommandType.STOP]
    assert watchdog.is_watching


def test_matching_ack_cancels_timeout(qtbot) -> None:
    watchdog = CommandWatchdog(timeout_ms=30)
    watchdog.watch(CommandType.STOP)
    watchdog.acknowledge(CommandType.STOP)
    qtbot.wait(60)
    assert not watchdog.is_watching


def test_nonmatching_ack_does_not_cancel_current_watch() -> None:
    watchdog = CommandWatchdog(timeout_ms=1000)
    watchdog.watch(CommandType.STOP)
    watchdog.acknowledge(CommandType.PAUSE)
    assert watchdog.is_watching
    watchdog.cancel()
    assert not watchdog.is_watching
