"""Structured progress events shared by the benchmark CLI and Qt UI."""

from __future__ import annotations

import json
from typing import Any

EVENT_PREFIX = "DEPTH3D_BENCHMARK_EVENT "


def encode_event(event: str, **payload: object) -> str:
    """Serialize one line-oriented benchmark event."""
    body: dict[str, object] = {"event": str(event)}
    body.update(payload)
    return EVENT_PREFIX + json.dumps(body, sort_keys=True, separators=(",", ":"))


def parse_event_line(line: str) -> dict[str, Any] | None:
    """Parse a structured event line, returning ``None`` for normal log text."""
    text = str(line).strip()
    if not text.startswith(EVENT_PREFIX):
        return None
    raw = text[len(EVENT_PREFIX) :]
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("event"), str):
        raise ValueError("Invalid benchmark event payload.")
    return payload
