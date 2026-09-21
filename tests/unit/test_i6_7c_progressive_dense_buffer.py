from __future__ import annotations

from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseStage
from core.reconstruction.reconstruction_progressive_dense_transport import (
    ProgressiveDenseOfferDisposition,
    ProgressiveDenseSnapshotBuffer,
    ProgressiveDenseTransportBinding,
    ProgressiveDenseTransportPolicy,
    progressive_dense_transport_payload_bytes,
)
from tests.i6_7c_helpers import DENSE_RUN_ID, make_snapshot


def _buffer(first=None, *, policy=ProgressiveDenseTransportPolicy()):
    first = first or make_snapshot(revision=1)
    buffer = ProgressiveDenseSnapshotBuffer(policy=policy)
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(first))
    return buffer, first


def test_i67c_latest_only_buffer_coalesces_pending_updates() -> None:
    buffer, first = _buffer()
    latest = make_snapshot(revision=4)
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert buffer.offer(latest) is ProgressiveDenseOfferDisposition.COALESCED
    assert buffer.pending is latest
    assert buffer.take_pending() is latest
    assert buffer.stats.accepted == 1
    assert buffer.stats.coalesced == 1
    assert buffer.stats.delivered == 1


def test_i67c_buffer_rejects_cross_run_and_stale_updates() -> None:
    buffer, first = _buffer()
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert buffer.take_pending() is first

    cross_run = make_snapshot(
        revision=2,
        dense_run_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
    )
    assert buffer.offer(cross_run) is ProgressiveDenseOfferDisposition.REJECTED
    assert "binding" in buffer.last_rejection_reason

    duplicate = make_snapshot(revision=1)
    assert buffer.offer(duplicate) is ProgressiveDenseOfferDisposition.REJECTED
    assert "strictly increase" in buffer.last_rejection_reason


def test_i67c_buffer_preserves_frozen_stage_regression_invariant() -> None:
    first = make_snapshot(revision=10, stage=ProgressiveDenseStage.FUSION_FILTERING)
    buffer, _ = _buffer(first)
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert buffer.take_pending() is first
    regressed = make_snapshot(revision=11, stage=ProgressiveDenseStage.PROVIDER_EXECUTION)
    assert buffer.offer(regressed) is ProgressiveDenseOfferDisposition.REJECTED
    assert "stage cannot regress" in buffer.last_rejection_reason


def test_i67c_final_geometry_survives_backpressure_coalescing() -> None:
    first = make_snapshot(revision=1)
    buffer, _ = _buffer(first)
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert buffer.take_pending() is first

    for revision in range(2, 23):
        assert buffer.offer(make_snapshot(revision=revision)) in {
            ProgressiveDenseOfferDisposition.ACCEPTED,
            ProgressiveDenseOfferDisposition.COALESCED,
        }
    final = make_snapshot(
        revision=23,
        stage=ProgressiveDenseStage.FINALIZATION,
        geometry_points=16,
        dense_points=1686160,
    )
    assert buffer.offer(final) is ProgressiveDenseOfferDisposition.COALESCED
    delivered = buffer.take_pending()
    assert delivered is final
    assert delivered.geometry is not None
    assert len(delivered.geometry.point_samples) == 16
    assert delivered.statistics.current_dense_points == 1686160
    assert buffer.stats.coalesced == 21
    assert buffer.stats.delivered == 2


def test_i67c_retained_transport_payload_is_hard_bounded_to_two_snapshots() -> None:
    first = make_snapshot(revision=1, geometry_points=64)
    buffer, _ = _buffer(first)
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    assert buffer.take_pending() is first
    second = make_snapshot(revision=2, geometry_points=64)
    assert buffer.offer(second) is ProgressiveDenseOfferDisposition.ACCEPTED
    stats = buffer.stats
    assert stats.current_pending_payload_bytes == progressive_dense_transport_payload_bytes(second)
    assert stats.current_retained_payload_bytes == (
        progressive_dense_transport_payload_bytes(first)
        + progressive_dense_transport_payload_bytes(second)
    )
    assert stats.current_retained_payload_bytes <= ProgressiveDenseTransportPolicy().max_retained_payload_bytes


def test_i67c_reset_invalidates_previous_dense_run_binding() -> None:
    first = make_snapshot(revision=1)
    buffer, _ = _buffer(first)
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.ACCEPTED
    new_first = make_snapshot(
        revision=1,
        dense_run_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
    )
    buffer.reset(ProgressiveDenseTransportBinding.from_snapshot(new_first))
    assert buffer.pending is None
    assert buffer.last_delivered is None
    assert buffer.offer(first) is ProgressiveDenseOfferDisposition.REJECTED
    assert buffer.offer(new_first) is ProgressiveDenseOfferDisposition.ACCEPTED
