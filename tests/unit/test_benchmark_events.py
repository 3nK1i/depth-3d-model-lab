"""Structured benchmark progress protocol tests."""

from __future__ import annotations

import json

import pytest

from core.benchmark.events import EVENT_PREFIX, encode_event, parse_event_line


def test_event_round_trip_is_line_oriented_and_deterministic() -> None:
    line = encode_event("model_started", model_id="development-mock", index=1)

    assert line.startswith(EVENT_PREFIX)
    assert parse_event_line(line) == {
        "event": "model_started",
        "index": 1,
        "model_id": "development-mock",
    }


def test_normal_console_output_is_not_an_event() -> None:
    assert parse_event_line("[MoGe-2] running…") is None


def test_invalid_event_payload_is_rejected() -> None:
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_event_line(EVENT_PREFIX + "[]")
