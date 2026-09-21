from __future__ import annotations

import json

import pytest

from core.reconstruction.errors import SfmWorkerProtocolError
from core.reconstruction.reconstruction_progressive_sfm_transport import (
    PROGRESSIVE_SFM_EVENT_PREFIX,
    parse_progressive_sfm_worker_line,
    progressive_sfm_transport_line,
)
from tests.i6_6c_helpers import make_snapshot


def test_transport_roundtrip_preserves_frozen_i66a_snapshot() -> None:
    snapshot = make_snapshot()
    line = progressive_sfm_transport_line(snapshot)
    assert line.startswith(PROGRESSIVE_SFM_EVENT_PREFIX)
    restored = parse_progressive_sfm_worker_line(line)
    assert restored is not None
    assert restored.as_dict() == snapshot.as_dict()


def test_transport_ignores_ordinary_worker_and_log_lines() -> None:
    assert parse_progressive_sfm_worker_line("EIA_SFM_EVENT {}") is None
    assert parse_progressive_sfm_worker_line('{"protocol":"eia.reconstruction.offline.v1"}') is None
    assert parse_progressive_sfm_worker_line("native colmap log") is None


def test_transport_rejects_malformed_json() -> None:
    with pytest.raises(SfmWorkerProtocolError):
        parse_progressive_sfm_worker_line(PROGRESSIVE_SFM_EVENT_PREFIX + "{")


def test_transport_rejects_authority_upgrade() -> None:
    payload = make_snapshot().as_dict()
    payload.pop("snapshot_sha256", None)
    payload["authority"]["final_result_admitted"] = True
    with pytest.raises(SfmWorkerProtocolError):
        parse_progressive_sfm_worker_line(
            PROGRESSIVE_SFM_EVENT_PREFIX + json.dumps(payload)
        )
