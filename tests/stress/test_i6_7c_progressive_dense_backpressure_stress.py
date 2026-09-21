from __future__ import annotations

from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_RETAINED_PAYLOAD_BYTES,
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES,
    ProgressiveDenseOfferDisposition,
    ProgressiveDenseSnapshotBuffer,
    ProgressiveDenseTransportBinding,
)
from tests.i6_7c_helpers import make_snapshot


def test_i67c_thousands_of_status_updates_remain_latest_only_and_memory_bounded() -> None:
    first = make_snapshot(revision=1)
    buffer = ProgressiveDenseSnapshotBuffer()
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(first))
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED

    for revision in range(2, 5001):
        assert buffer.offer(make_snapshot(revision=revision)) is ProgressiveDenseOfferDisposition.COALESCED
        assert buffer.pending is not None
        assert buffer.stats.current_pending_payload_bytes <= I67C_MAX_TRANSPORT_SNAPSHOT_BYTES
        assert buffer.stats.current_retained_payload_bytes <= I67C_MAX_RETAINED_PAYLOAD_BYTES

    latest = buffer.take_pending()
    assert latest is not None
    assert latest.revision == 5000
    assert buffer.stats.accepted == 1
    assert buffer.stats.coalesced == 4999
    assert buffer.stats.delivered == 1
    assert buffer.stats.peak_retained_payload_bytes <= I67C_MAX_RETAINED_PAYLOAD_BYTES
