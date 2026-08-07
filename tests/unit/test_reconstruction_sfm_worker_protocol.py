from __future__ import annotations

import json

import pytest

from core.reconstruction import (
    SFM_WORKER_EVENT_PREFIX,
    SFM_WORKER_SCHEMA_VERSION,
    SfmWorkerEventType,
    SfmWorkerProtocolError,
    parse_worker_event_line,
)


def test_worker_protocol_parses_structured_event_and_ignores_logs() -> None:
    payload = {
        "schema_version": SFM_WORKER_SCHEMA_VERSION,
        "event_type": "progress",
        "stage": "matching",
        "message": "matching",
        "progress_fraction": 0.4,
        "metadata": {"pairs": 12},
    }
    event = parse_worker_event_line(SFM_WORKER_EVENT_PREFIX + json.dumps(payload))
    assert event is not None
    assert event.event_type is SfmWorkerEventType.PROGRESS
    assert event.stage == "matching"
    assert event.progress_fraction == pytest.approx(0.4)
    assert dict(event.metadata) == {"pairs": 12}
    assert parse_worker_event_line("COLMAP ordinary log output") is None


def test_worker_protocol_rejects_bad_schema_and_invalid_json() -> None:
    with pytest.raises(SfmWorkerProtocolError):
        parse_worker_event_line(SFM_WORKER_EVENT_PREFIX + "{")
    with pytest.raises(SfmWorkerProtocolError):
        parse_worker_event_line(
            SFM_WORKER_EVENT_PREFIX
            + json.dumps(
                {
                    "schema_version": 999,
                    "event_type": "status",
                    "stage": "startup",
                    "message": "bad schema",
                    "metadata": {},
                }
            )
        )
