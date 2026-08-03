"""Session and frame ID regression tests from the Phase 5 Step 0 lifecycle."""

from __future__ import annotations

from core.runtime.session import SessionTracker


def test_new_sessions_increment_and_reset_frame_ids() -> None:
    tracker = SessionTracker()

    first = tracker.start_new_session()
    assert first == 1
    assert tracker.next_frame_id() == 0
    assert tracker.next_frame_id() == 1

    second = tracker.start_new_session()
    assert second == 2
    assert tracker.next_frame_id() == 0


def test_invalidate_rejects_results_from_old_session() -> None:
    tracker = SessionTracker()
    session_id = tracker.start_new_session()
    assert tracker.is_current(session_id)

    tracker.invalidate()

    assert tracker.active_session_id == 0
    assert not tracker.is_current(session_id)


def test_model_id_is_checked_when_both_ids_are_supplied() -> None:
    tracker = SessionTracker()
    session_id = tracker.start_new_session()

    assert tracker.is_current(session_id, "moge-2", "moge-2")
    assert not tracker.is_current(session_id, "development-mock", "moge-2")
    assert tracker.is_current(session_id, model_id=None, active_model_id="moge-2")
