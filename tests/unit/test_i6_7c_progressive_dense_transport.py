from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_dense_contract import ProgressiveDenseStage
from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_GEOMETRY_PREVIEW_POINTS,
    I67C_MAX_PENDING_SNAPSHOTS,
    I67C_MAX_RETAINED_SNAPSHOTS,
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES,
    PROGRESSIVE_DENSE_EVENT_PREFIX,
    ProgressiveDenseTransportLineSink,
    ProgressiveDenseTransportPolicy,
    parse_progressive_dense_transport_line,
    progressive_dense_transport_line,
    progressive_dense_transport_payload_bytes,
)
from tests.i6_7c_helpers import make_snapshot


def test_i67c_transport_policy_is_bounded_and_visualization_only() -> None:
    policy = ProgressiveDenseTransportPolicy()
    assert policy.max_snapshot_bytes == I67C_MAX_TRANSPORT_SNAPSHOT_BYTES == 4 * 1024 * 1024
    assert policy.max_geometry_preview_points == I67C_MAX_GEOMETRY_PREVIEW_POINTS == 4096
    assert I67C_MAX_PENDING_SNAPSHOTS == 1
    assert I67C_MAX_RETAINED_SNAPSHOTS == 2
    assert policy.max_retained_payload_bytes == 8 * 1024 * 1024


def test_i67c_dedicated_transport_line_round_trips_contract_snapshot() -> None:
    snapshot = make_snapshot(revision=7, stage=ProgressiveDenseStage.FUSION_FILTERING)
    line = progressive_dense_transport_line(snapshot)
    assert line.startswith(PROGRESSIVE_DENSE_EVENT_PREFIX)
    assert progressive_dense_transport_payload_bytes(snapshot) == len(line.encode("utf-8"))
    restored = parse_progressive_dense_transport_line(line)
    assert restored == snapshot
    assert restored.snapshot_sha256 == snapshot.snapshot_sha256


def test_i67c_unrelated_process_lines_are_ignored() -> None:
    assert parse_progressive_dense_transport_line("native COLMAP output") is None
    assert parse_progressive_dense_transport_line('{"protocol":"eia.reconstruction.offline.v1"}') is None


def test_i67c_malformed_or_authority_upgraded_payload_fails_closed() -> None:
    with pytest.raises(ReconstructionValidationError):
        parse_progressive_dense_transport_line(PROGRESSIVE_DENSE_EVENT_PREFIX + "{")

    payload = make_snapshot().as_dict()
    payload["authority"]["final_dense_result_admitted"] = True
    with pytest.raises(ReconstructionValidationError):
        parse_progressive_dense_transport_line(
            PROGRESSIVE_DENSE_EVENT_PREFIX + json.dumps(payload)
        )


def test_i67c_transport_rejects_snapshot_over_explicit_byte_budget() -> None:
    snapshot = make_snapshot(warnings=("x" * 2048,))
    policy = ProgressiveDenseTransportPolicy(max_snapshot_bytes=1024)
    with pytest.raises(ReconstructionValidationError, match="byte bound"):
        progressive_dense_transport_line(snapshot, policy=policy)


def test_i67c_transport_rejects_geometry_over_explicit_preview_bound() -> None:
    snapshot = make_snapshot(geometry_points=5)
    policy = ProgressiveDenseTransportPolicy(max_geometry_preview_points=4)
    with pytest.raises(ReconstructionValidationError, match="point bound"):
        progressive_dense_transport_line(snapshot, policy=policy)


def test_i67c_line_sink_is_directly_compatible_with_i67b_snapshot_sink() -> None:
    lines: list[str] = []
    sink = ProgressiveDenseTransportLineSink(lines.append)
    snapshot = make_snapshot(revision=3)
    sink.publish(snapshot)
    assert len(lines) == 1
    assert parse_progressive_dense_transport_line(lines[0]) == snapshot
