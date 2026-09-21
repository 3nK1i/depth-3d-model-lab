"""I6.6C controller-side buffering for provisional progressive sparse SfM.

The buffer is intentionally Qt-free.  It validates session/request/revision ordering
against the frozen I6.6A contract and keeps at most one pending snapshot.  Replacing
that pending snapshot is the explicit backpressure/coalescing mechanism used by the
Qt process bridge; revision gaps remain legal by contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_progressive_sfm_contract import (
    ProgressiveSfmSnapshot,
    validate_progressive_sfm_snapshot_transition,
)


class ProgressiveSfmOfferDisposition(str, Enum):
    ACCEPTED = "accepted"
    COALESCED = "coalesced"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ProgressiveSfmBufferStats:
    accepted: int
    coalesced: int
    rejected: int
    delivered: int


class ProgressiveSfmSnapshotBuffer:
    """One-session, latest-only provisional snapshot buffer."""

    def __init__(self) -> None:
        self._active_session_id = ""
        self._pending: ProgressiveSfmSnapshot | None = None
        self._last_delivered: ProgressiveSfmSnapshot | None = None
        self._accepted = 0
        self._coalesced = 0
        self._rejected = 0
        self._delivered = 0
        self._last_rejection_reason = ""

    @property
    def active_session_id(self) -> str:
        return self._active_session_id

    @property
    def pending(self) -> ProgressiveSfmSnapshot | None:
        return self._pending

    @property
    def last_delivered(self) -> ProgressiveSfmSnapshot | None:
        return self._last_delivered

    @property
    def last_rejection_reason(self) -> str:
        return self._last_rejection_reason

    @property
    def stats(self) -> ProgressiveSfmBufferStats:
        return ProgressiveSfmBufferStats(
            accepted=self._accepted,
            coalesced=self._coalesced,
            rejected=self._rejected,
            delivered=self._delivered,
        )

    def reset(self, session_id: str) -> None:
        value = str(session_id or "").strip()
        if not value:
            raise ValueError("session_id must not be empty.")
        self._active_session_id = value
        self._pending = None
        self._last_delivered = None
        self._accepted = 0
        self._coalesced = 0
        self._rejected = 0
        self._delivered = 0
        self._last_rejection_reason = ""

    def clear(self) -> None:
        self._active_session_id = ""
        self._pending = None
        self._last_delivered = None
        self._last_rejection_reason = ""

    def reject(self, reason: object) -> ProgressiveSfmOfferDisposition:
        self._rejected += 1
        self._last_rejection_reason = str(reason or "progressive_sfm_snapshot_rejected").strip()
        return ProgressiveSfmOfferDisposition.REJECTED

    def offer(self, snapshot: ProgressiveSfmSnapshot) -> ProgressiveSfmOfferDisposition:
        if not isinstance(snapshot, ProgressiveSfmSnapshot):
            raise TypeError("snapshot must be ProgressiveSfmSnapshot.")
        if not self._active_session_id:
            return self.reject("no_active_offline_session")
        if snapshot.session_id != self._active_session_id:
            return self.reject("snapshot_session_mismatch")

        anchor = self._pending or self._last_delivered
        if anchor is not None:
            try:
                validate_progressive_sfm_snapshot_transition(anchor, snapshot)
            except ReconstructionValidationError as exc:
                return self.reject(str(exc))

        disposition = (
            ProgressiveSfmOfferDisposition.COALESCED
            if self._pending is not None
            else ProgressiveSfmOfferDisposition.ACCEPTED
        )
        if disposition is ProgressiveSfmOfferDisposition.COALESCED:
            self._coalesced += 1
        else:
            self._accepted += 1
        self._pending = snapshot
        self._last_rejection_reason = ""
        return disposition

    def take_pending(self) -> ProgressiveSfmSnapshot | None:
        snapshot = self._pending
        if snapshot is None:
            return None
        self._pending = None
        self._last_delivered = snapshot
        self._delivered += 1
        return snapshot


__all__ = [
    "ProgressiveSfmBufferStats",
    "ProgressiveSfmOfferDisposition",
    "ProgressiveSfmSnapshotBuffer",
]
