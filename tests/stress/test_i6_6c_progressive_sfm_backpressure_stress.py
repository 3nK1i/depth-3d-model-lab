from __future__ import annotations

from app.reconstruction_progressive_sfm_state import (
    ProgressiveSfmOfferDisposition,
    ProgressiveSfmSnapshotBuffer,
)
from tests.i6_6c_helpers import SESSION_ID, make_snapshot


def test_latest_only_buffer_coalesces_large_revision_burst_without_growth() -> None:
    buffer = ProgressiveSfmSnapshotBuffer()
    buffer.reset(SESSION_ID)
    for revision in range(1, 2001):
        disposition = buffer.offer(make_snapshot(revision=revision))
        if revision == 1:
            assert disposition is ProgressiveSfmOfferDisposition.ACCEPTED
        else:
            assert disposition is ProgressiveSfmOfferDisposition.COALESCED
        assert buffer.pending is not None
        assert buffer.pending.revision == revision
    final = buffer.take_pending()
    assert final is not None and final.revision == 2000
    assert buffer.stats.accepted == 1
    assert buffer.stats.coalesced == 1999
    assert buffer.stats.delivered == 1
    assert buffer.stats.rejected == 0
