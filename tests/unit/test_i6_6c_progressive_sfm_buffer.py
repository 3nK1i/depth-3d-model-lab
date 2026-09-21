from __future__ import annotations

from app.reconstruction_progressive_sfm_state import (
    ProgressiveSfmOfferDisposition,
    ProgressiveSfmSnapshotBuffer,
)
from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmStage
from tests.i6_6c_helpers import SESSION_ID, make_snapshot


def test_buffer_accepts_and_delivers_one_snapshot() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    snapshot = make_snapshot(revision=1)
    assert buffer.offer(snapshot) is ProgressiveSfmOfferDisposition.ACCEPTED
    assert buffer.take_pending() is snapshot
    assert buffer.stats.accepted == 1
    assert buffer.stats.delivered == 1


def test_buffer_coalesces_pending_updates_to_latest_revision() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    first = make_snapshot(revision=1)
    latest = make_snapshot(revision=4)
    assert buffer.offer(first) is ProgressiveSfmOfferDisposition.ACCEPTED
    assert buffer.offer(latest) is ProgressiveSfmOfferDisposition.COALESCED
    assert buffer.take_pending() is latest
    assert buffer.stats.coalesced == 1
    assert buffer.stats.delivered == 1


def test_buffer_rejects_stale_duplicate_cross_session_and_request_change() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    assert buffer.offer(make_snapshot(revision=2)) is ProgressiveSfmOfferDisposition.ACCEPTED
    assert buffer.take_pending() is not None
    assert buffer.offer(make_snapshot(revision=2)) is ProgressiveSfmOfferDisposition.REJECTED
    assert buffer.offer(
        make_snapshot(
            revision=3,
            session_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        )
    ) is ProgressiveSfmOfferDisposition.REJECTED
    assert buffer.offer(
        make_snapshot(revision=3, request_sha="e" * 64)
    ) is ProgressiveSfmOfferDisposition.REJECTED
    assert buffer.stats.rejected == 3


def test_buffer_allows_revision_gaps_and_finalizing_transition() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    assert buffer.offer(make_snapshot(revision=1)) is ProgressiveSfmOfferDisposition.ACCEPTED
    assert buffer.take_pending() is not None
    finalizing = make_snapshot(revision=8, stage=ProgressiveSfmStage.FINALIZING)
    assert buffer.offer(finalizing) is ProgressiveSfmOfferDisposition.ACCEPTED
    assert buffer.take_pending() is finalizing


def test_reset_invalidates_previous_session_pending_state() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    assert buffer.offer(make_snapshot(revision=1)) is ProgressiveSfmOfferDisposition.ACCEPTED
    new_session = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
    buffer.reset(new_session)
    assert buffer.pending is None
    assert buffer.last_delivered is None
    assert buffer.stats.accepted == 0
    assert buffer.offer(make_snapshot(revision=2)) is ProgressiveSfmOfferDisposition.REJECTED
