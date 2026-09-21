"""I6.6C transport helpers for provisional progressive sparse-SfM snapshots.

This module is deliberately dependency-light and owns only the JSON-line transport
boundary between the isolated PyCOLMAP worker and the application process.  The
frozen I6.6A contract remains the semantic authority for every payload.
"""

from __future__ import annotations

import json
from typing import Mapping

from core.reconstruction.errors import SfmWorkerProtocolError
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
)

PROGRESSIVE_SFM_EVENT_PREFIX = "EIA_SFM_PROGRESSIVE "


def parse_progressive_sfm_worker_line(line: str) -> ProgressiveSfmSnapshot | None:
    """Parse one progressive worker line, ignoring all unrelated stdout."""

    text = str(line).rstrip("\r\n")
    if not text.startswith(PROGRESSIVE_SFM_EVENT_PREFIX):
        return None
    raw = text[len(PROGRESSIVE_SFM_EVENT_PREFIX) :]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SfmWorkerProtocolError(
            f"Invalid progressive SfM worker JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SfmWorkerProtocolError(
            "Progressive SfM worker JSON must contain an object."
        )
    try:
        return ProgressiveSfmSnapshot.from_payload(payload)
    except (TypeError, ValueError) as exc:
        raise SfmWorkerProtocolError(
            f"Invalid progressive SfM worker payload: {exc}"
        ) from exc


def progressive_sfm_transport_line(snapshot: ProgressiveSfmSnapshot) -> str:
    """Serialize one already-validated snapshot onto the dedicated transport line."""

    if not isinstance(snapshot, ProgressiveSfmSnapshot):
        raise TypeError("snapshot must be ProgressiveSfmSnapshot.")
    return PROGRESSIVE_SFM_EVENT_PREFIX + json.dumps(
        snapshot.as_dict(),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


__all__ = [
    "PROGRESSIVE_SFM_EVENT_PREFIX",
    "parse_progressive_sfm_worker_line",
    "progressive_sfm_transport_line",
]
