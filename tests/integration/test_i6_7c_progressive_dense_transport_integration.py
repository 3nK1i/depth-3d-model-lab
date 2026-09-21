from __future__ import annotations

from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseStage
from core.reconstruction.reconstruction_progressive_dense_transport import (
    ProgressiveDenseOfferDisposition,
    ProgressiveDenseSnapshotBuffer,
    ProgressiveDenseTransportBinding,
    ProgressiveDenseTransportLineSink,
)
from tests.i6_7c_helpers import make_snapshot


def test_i67b_sink_to_i67c_line_to_latest_only_buffer_preserves_identity_and_geometry() -> None:
    first = make_snapshot(revision=1)
    buffer = ProgressiveDenseSnapshotBuffer()
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(first))

    dispositions = []
    sink = ProgressiveDenseTransportLineSink(
        lambda line: dispositions.append(buffer.offer_line(line))
    )
    sink.publish(first)
    sink.publish(make_snapshot(revision=2))
    sink.publish(make_snapshot(revision=3, stage=ProgressiveDenseStage.FUSION_FILTERING))
    final = make_snapshot(
        revision=4,
        stage=ProgressiveDenseStage.FINALIZATION,
        geometry_points=32,
        dense_points=10000,
    )
    sink.publish(final)

    assert dispositions == [
        ProgressiveDenseOfferDisposition.ACCEPTED,
        ProgressiveDenseOfferDisposition.COALESCED,
        ProgressiveDenseOfferDisposition.COALESCED,
        ProgressiveDenseOfferDisposition.COALESCED,
    ]
    assert buffer.pending is not None
    assert buffer.pending.revision == 4
    assert buffer.pending.geometry is not None
    delivered = buffer.take_pending()
    assert delivered is not None
    assert delivered.snapshot_sha256 == final.snapshot_sha256
    assert buffer.stats.coalesced == 3
    assert buffer.stats.delivered == 1
    assert buffer.stats.peak_pending_payload_bytes <= 4 * 1024 * 1024
    assert buffer.stats.peak_retained_payload_bytes <= 8 * 1024 * 1024


def test_i67c_malformed_transport_is_fail_soft_at_buffer_boundary() -> None:
    first = make_snapshot(revision=1)
    buffer = ProgressiveDenseSnapshotBuffer()
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(first))
    assert buffer.offer_line("EIA_DENSE_PROGRESSIVE {") is ProgressiveDenseOfferDisposition.REJECTED
    assert buffer.pending is None
    assert buffer.stats.rejected == 1
