"""I6.7C transport, backpressure, and memory bounds for progressive dense snapshots.

The frozen I6.7A contract remains the semantic authority and the frozen I6.7B
emitter remains the producer.  This module adds only a dedicated JSON-line
transport boundary plus a dependency-light latest-only buffer.  It does not
render, admit, persist, or promote provisional dense geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Callable, Mapping

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.validation import require_sha256, require_uuid
from core.reconstruction.reconstruction_progressive_dense_contract import (
    ProgressiveDenseSnapshot,
    validate_progressive_dense_snapshot_transition,
)
from core.reconstruction.reconstruction_progressive_dense_emission import (
    I67B_MAX_PREVIEW_POINTS,
)

I67C_PHASE = "I6.7C_DENSE_TRANSPORT_BACKPRESSURE_AND_MEMORY_BOUNDS"
PROGRESSIVE_DENSE_EVENT_PREFIX = "EIA_DENSE_PROGRESSIVE "

# Transport-only limits.  These are not reconstruction thresholds and do not
# alter provider execution, scientific geometry, or the I6.7B source artifact.
I67C_MAX_TRANSPORT_SNAPSHOT_BYTES = 4 * 1024 * 1024
I67C_MAX_PENDING_SNAPSHOTS = 1
I67C_MAX_RETAINED_SNAPSHOTS = 2  # last delivered + latest pending
I67C_MAX_RETAINED_PAYLOAD_BYTES = (
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES * I67C_MAX_RETAINED_SNAPSHOTS
)
I67C_MAX_GEOMETRY_PREVIEW_POINTS = I67B_MAX_PREVIEW_POINTS


class ProgressiveDenseOfferDisposition(str, Enum):
    ACCEPTED = "accepted"
    COALESCED = "coalesced"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ProgressiveDenseTransportBinding:
    """Expected identity for one provisional dense transport run."""

    source_sfm_session_id: str
    dense_run_id: str
    provider_id: str
    implementation: DenseProviderImplementation
    request_fingerprint_sha256: str
    source_sparse_result_sha256: str
    reconstruction_frame_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_sfm_session_id",
            require_uuid(self.source_sfm_session_id, field_name="source_sfm_session_id"),
        )
        object.__setattr__(
            self,
            "dense_run_id",
            require_uuid(self.dense_run_id, field_name="dense_run_id"),
        )
        object.__setattr__(self, "provider_id", require_nonempty(self.provider_id, field_name="provider_id"))
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        object.__setattr__(
            self,
            "request_fingerprint_sha256",
            require_sha256(
                self.request_fingerprint_sha256,
                field_name="request_fingerprint_sha256",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "source_sparse_result_sha256",
            require_sha256(
                self.source_sparse_result_sha256,
                field_name="source_sparse_result_sha256",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )

    @classmethod
    def from_snapshot(cls, snapshot: ProgressiveDenseSnapshot) -> "ProgressiveDenseTransportBinding":
        if not isinstance(snapshot, ProgressiveDenseSnapshot):
            raise TypeError("snapshot must be ProgressiveDenseSnapshot.")
        return cls(
            source_sfm_session_id=snapshot.source_sfm_session_id,
            dense_run_id=snapshot.dense_run_id,
            provider_id=snapshot.provider_id,
            implementation=snapshot.implementation,
            request_fingerprint_sha256=snapshot.request_fingerprint_sha256,
            source_sparse_result_sha256=snapshot.source_sparse_result_sha256,
            reconstruction_frame_id=snapshot.reconstruction_frame_id,
        )

    def matches(self, snapshot: ProgressiveDenseSnapshot) -> bool:
        if not isinstance(snapshot, ProgressiveDenseSnapshot):
            return False
        return (
            snapshot.source_sfm_session_id == self.source_sfm_session_id
            and snapshot.dense_run_id == self.dense_run_id
            and snapshot.provider_id == self.provider_id
            and snapshot.implementation is self.implementation
            and snapshot.request_fingerprint_sha256 == self.request_fingerprint_sha256
            and snapshot.source_sparse_result_sha256 == self.source_sparse_result_sha256
            and snapshot.reconstruction_frame_id == self.reconstruction_frame_id
        )


@dataclass(frozen=True, slots=True)
class ProgressiveDenseTransportPolicy:
    """Operational transport/memory policy; never a scientific quality policy."""

    max_snapshot_bytes: int = I67C_MAX_TRANSPORT_SNAPSHOT_BYTES
    max_geometry_preview_points: int = I67C_MAX_GEOMETRY_PREVIEW_POINTS

    def __post_init__(self) -> None:
        for name, value in (
            ("max_snapshot_bytes", self.max_snapshot_bytes),
            ("max_geometry_preview_points", self.max_geometry_preview_points),
        ):
            if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                raise ReconstructionValidationError(f"{name} must be a positive integer.")
        object.__setattr__(self, "max_snapshot_bytes", int(self.max_snapshot_bytes))
        object.__setattr__(
            self, "max_geometry_preview_points", int(self.max_geometry_preview_points)
        )

    @property
    def max_retained_payload_bytes(self) -> int:
        return self.max_snapshot_bytes * I67C_MAX_RETAINED_SNAPSHOTS


@dataclass(frozen=True, slots=True)
class ProgressiveDenseTransportStats:
    accepted: int
    coalesced: int
    rejected: int
    delivered: int
    current_pending_payload_bytes: int
    peak_pending_payload_bytes: int
    current_retained_payload_bytes: int
    peak_retained_payload_bytes: int


def _snapshot_payload_json(snapshot: ProgressiveDenseSnapshot) -> str:
    if not isinstance(snapshot, ProgressiveDenseSnapshot):
        raise TypeError("snapshot must be ProgressiveDenseSnapshot.")
    return json.dumps(
        snapshot.as_dict(),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def progressive_dense_transport_payload_bytes(snapshot: ProgressiveDenseSnapshot) -> int:
    """Return UTF-8 bytes for the dedicated transport line, excluding newline."""

    text = PROGRESSIVE_DENSE_EVENT_PREFIX + _snapshot_payload_json(snapshot)
    return len(text.encode("utf-8"))


def _validate_transport_policy(
    snapshot: ProgressiveDenseSnapshot,
    policy: ProgressiveDenseTransportPolicy,
) -> int:
    if snapshot.geometry is not None:
        point_count = len(snapshot.geometry.point_samples)
        if point_count > policy.max_geometry_preview_points:
            raise ReconstructionValidationError(
                "Progressive dense geometry preview exceeds the I6.7C transport point bound."
            )
    payload_bytes = progressive_dense_transport_payload_bytes(snapshot)
    if payload_bytes > policy.max_snapshot_bytes:
        raise ReconstructionValidationError(
            "Progressive dense transport payload exceeds the I6.7C per-snapshot byte bound."
        )
    return payload_bytes


def progressive_dense_transport_line(
    snapshot: ProgressiveDenseSnapshot,
    *,
    policy: ProgressiveDenseTransportPolicy = ProgressiveDenseTransportPolicy(),
) -> str:
    """Serialize one validated provisional dense snapshot onto its dedicated channel."""

    _validate_transport_policy(snapshot, policy)
    return PROGRESSIVE_DENSE_EVENT_PREFIX + _snapshot_payload_json(snapshot)


def parse_progressive_dense_transport_line(
    line: str,
    *,
    policy: ProgressiveDenseTransportPolicy = ProgressiveDenseTransportPolicy(),
) -> ProgressiveDenseSnapshot | None:
    """Parse one dense transport line, ignoring unrelated process output."""

    text = str(line).rstrip("\r\n")
    if not text.startswith(PROGRESSIVE_DENSE_EVENT_PREFIX):
        return None
    if len(text.encode("utf-8")) > policy.max_snapshot_bytes:
        raise ReconstructionValidationError(
            "Progressive dense transport line exceeds the I6.7C per-snapshot byte bound."
        )
    raw = text[len(PROGRESSIVE_DENSE_EVENT_PREFIX) :]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReconstructionValidationError(
            f"Invalid progressive dense transport JSON: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ReconstructionValidationError(
            "Progressive dense transport JSON must contain an object."
        )
    snapshot = ProgressiveDenseSnapshot.from_payload(payload)
    _validate_transport_policy(snapshot, policy)
    return snapshot


class ProgressiveDenseTransportLineSink:
    """I6.7B-compatible snapshot sink writing the dedicated I6.7C line protocol."""

    def __init__(
        self,
        writer: Callable[[str], None],
        *,
        policy: ProgressiveDenseTransportPolicy = ProgressiveDenseTransportPolicy(),
    ) -> None:
        if not callable(writer):
            raise TypeError("writer must be callable.")
        self._writer = writer
        self._policy = policy

    def publish(self, snapshot: ProgressiveDenseSnapshot) -> None:
        self._writer(progressive_dense_transport_line(snapshot, policy=self._policy))


class ProgressiveDenseSnapshotBuffer:
    """One-run latest-only buffer with explicit transport-memory accounting."""

    def __init__(
        self,
        *,
        policy: ProgressiveDenseTransportPolicy = ProgressiveDenseTransportPolicy(),
    ) -> None:
        self._policy = policy
        self._binding: ProgressiveDenseTransportBinding | None = None
        self._pending: ProgressiveDenseSnapshot | None = None
        self._pending_payload_bytes = 0
        self._last_delivered: ProgressiveDenseSnapshot | None = None
        self._last_delivered_payload_bytes = 0
        self._accepted = 0
        self._coalesced = 0
        self._rejected = 0
        self._delivered = 0
        self._peak_pending_payload_bytes = 0
        self._peak_retained_payload_bytes = 0
        self._last_rejection_reason = ""

    @property
    def binding(self) -> ProgressiveDenseTransportBinding | None:
        return self._binding

    @property
    def pending(self) -> ProgressiveDenseSnapshot | None:
        return self._pending

    @property
    def last_delivered(self) -> ProgressiveDenseSnapshot | None:
        return self._last_delivered

    @property
    def last_rejection_reason(self) -> str:
        return self._last_rejection_reason

    @property
    def stats(self) -> ProgressiveDenseTransportStats:
        current_retained = self._pending_payload_bytes + self._last_delivered_payload_bytes
        return ProgressiveDenseTransportStats(
            accepted=self._accepted,
            coalesced=self._coalesced,
            rejected=self._rejected,
            delivered=self._delivered,
            current_pending_payload_bytes=self._pending_payload_bytes,
            peak_pending_payload_bytes=self._peak_pending_payload_bytes,
            current_retained_payload_bytes=current_retained,
            peak_retained_payload_bytes=self._peak_retained_payload_bytes,
        )

    def reset(self, binding: ProgressiveDenseTransportBinding) -> None:
        if not isinstance(binding, ProgressiveDenseTransportBinding):
            raise TypeError("binding must be ProgressiveDenseTransportBinding.")
        self._binding = binding
        self._pending = None
        self._pending_payload_bytes = 0
        self._last_delivered = None
        self._last_delivered_payload_bytes = 0
        self._accepted = 0
        self._coalesced = 0
        self._rejected = 0
        self._delivered = 0
        self._peak_pending_payload_bytes = 0
        self._peak_retained_payload_bytes = 0
        self._last_rejection_reason = ""

    def clear(self) -> None:
        self._binding = None
        self._pending = None
        self._pending_payload_bytes = 0
        self._last_delivered = None
        self._last_delivered_payload_bytes = 0
        self._last_rejection_reason = ""

    def reject(self, reason: object) -> ProgressiveDenseOfferDisposition:
        self._rejected += 1
        self._last_rejection_reason = str(
            reason or "progressive_dense_snapshot_rejected"
        ).strip()
        return ProgressiveDenseOfferDisposition.REJECTED

    def _update_peaks(self) -> None:
        self._peak_pending_payload_bytes = max(
            self._peak_pending_payload_bytes, self._pending_payload_bytes
        )
        retained = self._pending_payload_bytes + self._last_delivered_payload_bytes
        self._peak_retained_payload_bytes = max(
            self._peak_retained_payload_bytes, retained
        )
        if retained > self._policy.max_retained_payload_bytes:
            raise RuntimeError("I6.7C retained transport payload exceeded its hard memory bound.")

    def offer(self, snapshot: ProgressiveDenseSnapshot) -> ProgressiveDenseOfferDisposition:
        if not isinstance(snapshot, ProgressiveDenseSnapshot):
            raise TypeError("snapshot must be ProgressiveDenseSnapshot.")
        if self._binding is None:
            return self.reject("no_active_dense_transport_binding")
        if not self._binding.matches(snapshot):
            return self.reject("snapshot_dense_run_binding_mismatch")

        try:
            payload_bytes = _validate_transport_policy(snapshot, self._policy)
        except ReconstructionValidationError as exc:
            return self.reject(str(exc))

        anchor = self._pending or self._last_delivered
        if anchor is not None:
            try:
                validate_progressive_dense_snapshot_transition(anchor, snapshot)
            except ReconstructionValidationError as exc:
                return self.reject(str(exc))

        disposition = (
            ProgressiveDenseOfferDisposition.COALESCED
            if self._pending is not None
            else ProgressiveDenseOfferDisposition.ACCEPTED
        )
        if disposition is ProgressiveDenseOfferDisposition.COALESCED:
            self._coalesced += 1
        else:
            self._accepted += 1
        self._pending = snapshot
        self._pending_payload_bytes = payload_bytes
        self._last_rejection_reason = ""
        self._update_peaks()
        return disposition

    def offer_line(self, line: str) -> ProgressiveDenseOfferDisposition | None:
        try:
            snapshot = parse_progressive_dense_transport_line(line, policy=self._policy)
        except ReconstructionValidationError as exc:
            return self.reject(str(exc))
        if snapshot is None:
            return None
        return self.offer(snapshot)

    def take_pending(self) -> ProgressiveDenseSnapshot | None:
        snapshot = self._pending
        if snapshot is None:
            return None
        self._last_delivered = snapshot
        self._last_delivered_payload_bytes = self._pending_payload_bytes
        self._pending = None
        self._pending_payload_bytes = 0
        self._delivered += 1
        self._update_peaks()
        return snapshot


__all__ = [
    "I67C_PHASE",
    "PROGRESSIVE_DENSE_EVENT_PREFIX",
    "I67C_MAX_TRANSPORT_SNAPSHOT_BYTES",
    "I67C_MAX_PENDING_SNAPSHOTS",
    "I67C_MAX_RETAINED_SNAPSHOTS",
    "I67C_MAX_RETAINED_PAYLOAD_BYTES",
    "I67C_MAX_GEOMETRY_PREVIEW_POINTS",
    "ProgressiveDenseOfferDisposition",
    "ProgressiveDenseTransportBinding",
    "ProgressiveDenseTransportPolicy",
    "ProgressiveDenseTransportStats",
    "ProgressiveDenseTransportLineSink",
    "ProgressiveDenseSnapshotBuffer",
    "progressive_dense_transport_payload_bytes",
    "progressive_dense_transport_line",
    "parse_progressive_dense_transport_line",
]
