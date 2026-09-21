"""I6.5A adaptive frame/view planning contract and provenance model.

This module is deliberately planning-contract only.  It freezes the product
semantics needed to make the existing adaptive pre-SfM behavior inspectable and
later controllable without changing the reconstruction algorithm in I6.5A.

The contract keeps four facts separate:

* source provenance -- which immutable source the request belongs to;
* requested plan -- the user's/product's requested planning mode;
* effective plan -- the bounded target actually resolved for pre-SfM planning;
* realization -- which observed source views were eligible and which were
  actually planned for submission to SfM.

A planned source view is *not* a registered camera.  PyCOLMAP SfM remains the
sole canonical camera/global-frame authority.  This module does not execute
sampling, change ingestion behavior, select a dense provider, activate Auto,
start reconstruction, alter scientific thresholds, or establish metric scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import isfinite
import re
from typing import Any

from core.reconstruction.dense_provider_architecture import I43A_AUTHORITATIVE_FRAME_OWNER
from core.reconstruction.errors import ReconstructionValidationError

I65A_SCHEMA_VERSION = 1
I65A_PHASE = "I6.5A"
I65A_CAMERA_AUTHORITY_OWNER = I43A_AUTHORITATIVE_FRAME_OWNER
I65A_PERMANENT_GLOBAL_VIEW_CAP = None
I65A_MINIMUM_RECONSTRUCTION_VIEWS = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FrameViewPlanMode(str, Enum):
    """Requested planning mode; none of these values executes reconstruction."""

    ADAPTIVE = "adaptive"
    ALL_ELIGIBLE = "all_eligible"
    MANUAL_LIMIT = "manual_limit"


class FrameViewRecordDisposition(str, Enum):
    """Observed per-view outcome before SfM registration."""

    PLANNED = "planned"
    ELIGIBLE_NOT_PLANNED = "eligible_not_planned"
    INELIGIBLE = "ineligible"


def _nonempty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ReconstructionValidationError(f"{field_name} must be non-empty.")
    return text


def _reason_codes(values: tuple[str, ...] | list[str], *, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value or "").strip() for value in values)
    if not result or any(not value for value in result):
        raise ReconstructionValidationError(f"{field_name} requires non-empty reason codes.")
    if len(result) != len(set(result)):
        raise ReconstructionValidationError(f"{field_name} reason codes must be unique.")
    return result


def _sha256(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ReconstructionValidationError(f"{field_name} must be a lowercase SHA-256 digest.")
    return text


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class FrameViewSourceProvenance:
    """Immutable source identity used to bind one frame/view plan."""

    source_id: str
    source_kind: str
    source_uri: str
    source_fingerprint_sha256: str
    estimated_total_views: int | None
    expected_frame_rate_hz: float | None
    schema_version: int = I65A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_id = _nonempty(self.source_id, field_name="source_id")
        source_kind = _nonempty(self.source_kind, field_name="source_kind")
        source_uri = _nonempty(self.source_uri, field_name="source_uri")
        fingerprint = _sha256(
            self.source_fingerprint_sha256,
            field_name="source_fingerprint_sha256",
        )
        total = self.estimated_total_views
        if total is not None:
            if isinstance(total, bool) or int(total) <= 0:
                raise ReconstructionValidationError(
                    "estimated_total_views must be positive when known."
                )
            total = int(total)
        fps = self.expected_frame_rate_hz
        if fps is not None:
            fps = float(fps)
            if not isfinite(fps) or fps <= 0.0:
                raise ReconstructionValidationError(
                    "expected_frame_rate_hz must be finite and positive when known."
                )
        if int(self.schema_version) != I65A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5A source-provenance schema version.")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "source_uri", source_uri)
        object.__setattr__(self, "source_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "estimated_total_views", total)
        object.__setattr__(self, "expected_frame_rate_hz", fps)
        object.__setattr__(self, "schema_version", I65A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_uri": self.source_uri,
            "source_fingerprint_sha256": self.source_fingerprint_sha256,
            "estimated_total_views": self.estimated_total_views,
            "expected_frame_rate_hz": self.expected_frame_rate_hz,
        }


@dataclass(frozen=True, slots=True)
class FrameViewPlanRequest:
    """Explicit product/user request before resource-aware resolution."""

    mode: FrameViewPlanMode
    requested_max_views: int | None = None
    schema_version: int = I65A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.mode, FrameViewPlanMode):
            raise TypeError("mode must be FrameViewPlanMode.")
        requested = self.requested_max_views
        if self.mode is FrameViewPlanMode.MANUAL_LIMIT:
            if isinstance(requested, bool) or requested is None or int(requested) < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
                raise ReconstructionValidationError(
                    f"MANUAL_LIMIT requires requested_max_views >= {I65A_MINIMUM_RECONSTRUCTION_VIEWS}."
                )
            requested = int(requested)
        elif requested is not None:
            raise ReconstructionValidationError(
                "Only MANUAL_LIMIT requests may carry requested_max_views."
            )
        if int(self.schema_version) != I65A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5A request schema version.")
        object.__setattr__(self, "requested_max_views", requested)
        object.__setattr__(self, "schema_version", I65A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode.value,
            "requested_max_views": self.requested_max_views,
        }


@dataclass(frozen=True, slots=True)
class FrameViewPlanResolution:
    """Effective pre-SfM plan with explicit request and resource provenance."""

    source: FrameViewSourceProvenance
    request: FrameViewPlanRequest
    effective_target_views: int | None
    resource_target_views: int | None
    reason_codes: tuple[str, ...]
    permanent_global_view_cap: int | None = I65A_PERMANENT_GLOBAL_VIEW_CAP
    schema_version: int = I65A_SCHEMA_VERSION
    plan_input_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, FrameViewSourceProvenance):
            raise TypeError("source must be FrameViewSourceProvenance.")
        if not isinstance(self.request, FrameViewPlanRequest):
            raise TypeError("request must be FrameViewPlanRequest.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I6.5 forbids a permanent global frame/view cap.")

        target = self.effective_target_views
        if target is not None:
            if isinstance(target, bool) or int(target) < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
                raise ReconstructionValidationError(
                    f"effective_target_views must be >= {I65A_MINIMUM_RECONSTRUCTION_VIEWS} when present."
                )
            target = int(target)
        resource_target = self.resource_target_views
        if resource_target is not None:
            if isinstance(resource_target, bool) or int(resource_target) < I65A_MINIMUM_RECONSTRUCTION_VIEWS:
                raise ReconstructionValidationError(
                    f"resource_target_views must be >= {I65A_MINIMUM_RECONSTRUCTION_VIEWS} when present."
                )
            resource_target = int(resource_target)

        known_total = self.source.estimated_total_views
        if known_total is not None:
            if target is not None and target > known_total:
                raise ReconstructionValidationError(
                    "effective_target_views cannot exceed a known source-view count."
                )
            if resource_target is not None and resource_target > known_total:
                # Resource capacity may naturally exceed the source.  Clamp-like
                # semantics belong to the resolver; the product contract records
                # an effective resource target no larger than a known source.
                raise ReconstructionValidationError(
                    "resource_target_views cannot exceed a known source-view count."
                )

        if self.request.mode is FrameViewPlanMode.ALL_ELIGIBLE and target is not None:
            raise ReconstructionValidationError(
                "ALL_ELIGIBLE must not carry an effective view cap."
            )
        if self.request.mode is FrameViewPlanMode.MANUAL_LIMIT:
            requested = self.request.requested_max_views
            if target is None:
                raise ReconstructionValidationError(
                    "MANUAL_LIMIT requires an explicit effective_target_views value."
                )
            if requested is None or target > requested:
                raise ReconstructionValidationError(
                    "A manual effective target cannot exceed requested_max_views."
                )

        reasons = _reason_codes(self.reason_codes, field_name="reason_codes")
        if int(self.schema_version) != I65A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5A plan-resolution schema version.")

        object.__setattr__(self, "effective_target_views", target)
        object.__setattr__(self, "resource_target_views", resource_target)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I65A_SCHEMA_VERSION)
        object.__setattr__(
            self,
            "plan_input_sha256",
            _canonical_sha256(self._digest_payload()),
        )

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "phase": I65A_PHASE,
            "schema_version": I65A_SCHEMA_VERSION,
            "source": self.source.as_dict(),
            "request": self.request.as_dict(),
            "effective_target_views": self.effective_target_views,
            "resource_target_views": self.resource_target_views,
            "reason_codes": list(self.reason_codes),
            "permanent_global_view_cap": None,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "plan_input_sha256": self.plan_input_sha256,
            "effects": {
                "sampling_executed": False,
                "ingestion_modified": False,
                "provider_selected": False,
                "provider_activated": False,
                "production_auto_activated": False,
                "execution_started": False,
                "scientific_threshold_modified": False,
                "production_default_modified": False,
            },
            "authority": {
                "camera_pose_authority_owner": I65A_CAMERA_AUTHORITY_OWNER,
                "planned_views_are_registered_cameras": False,
                "pycolmap_registration_outcome_claimed": False,
                "metric_scale_claimed": False,
                "certified_metrology": False,
            },
        }


@dataclass(frozen=True, slots=True)
class FrameViewRecord:
    """Per-source-view provenance after eligibility/planning is realized."""

    source_view_id: str
    sequence_index: int
    source_locator: str
    source_timestamp_ns: int | None
    disposition: FrameViewRecordDisposition
    reason_codes: tuple[str, ...]
    schema_version: int = I65A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_view_id = _nonempty(self.source_view_id, field_name="source_view_id")
        source_locator = _nonempty(self.source_locator, field_name="source_locator")
        if isinstance(self.sequence_index, bool) or int(self.sequence_index) < 0:
            raise ReconstructionValidationError("sequence_index must be a non-negative integer.")
        sequence_index = int(self.sequence_index)
        timestamp = self.source_timestamp_ns
        if timestamp is not None:
            if isinstance(timestamp, bool) or int(timestamp) != timestamp or int(timestamp) < 0:
                raise ReconstructionValidationError(
                    "source_timestamp_ns must be a non-negative integer when present."
                )
            timestamp = int(timestamp)
        if not isinstance(self.disposition, FrameViewRecordDisposition):
            raise TypeError("disposition must be FrameViewRecordDisposition.")
        reasons = _reason_codes(self.reason_codes, field_name="reason_codes")
        if int(self.schema_version) != I65A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5A frame/view-record schema version.")
        object.__setattr__(self, "source_view_id", source_view_id)
        object.__setattr__(self, "sequence_index", sequence_index)
        object.__setattr__(self, "source_locator", source_locator)
        object.__setattr__(self, "source_timestamp_ns", timestamp)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I65A_SCHEMA_VERSION)

    @property
    def eligible(self) -> bool:
        return self.disposition is not FrameViewRecordDisposition.INELIGIBLE

    @property
    def planned(self) -> bool:
        return self.disposition is FrameViewRecordDisposition.PLANNED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_view_id": self.source_view_id,
            "sequence_index": self.sequence_index,
            "source_locator": self.source_locator,
            "source_timestamp_ns": self.source_timestamp_ns,
            "disposition": self.disposition.value,
            "eligible": self.eligible,
            "planned": self.planned,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class FrameViewPlanRealization:
    """Observed source-view realization before canonical SfM registration."""

    resolution: FrameViewPlanResolution
    records: tuple[FrameViewRecord, ...]
    schema_version: int = I65A_SCHEMA_VERSION
    plan_provenance_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.resolution, FrameViewPlanResolution):
            raise TypeError("resolution must be FrameViewPlanResolution.")
        records = tuple(self.records)
        if not records:
            raise ReconstructionValidationError("A realized frame/view plan requires at least one source record.")
        if any(not isinstance(item, FrameViewRecord) for item in records):
            raise TypeError("records must contain FrameViewRecord values.")

        indices = tuple(item.sequence_index for item in records)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ReconstructionValidationError(
                "Frame/view records must preserve strictly unique source sequence order."
            )
        ids = tuple(item.source_view_id for item in records)
        if len(ids) != len(set(ids)):
            raise ReconstructionValidationError("Frame/view source_view_id values must be unique.")

        known_total = self.resolution.source.estimated_total_views
        if known_total is not None and len(records) > known_total:
            raise ReconstructionValidationError(
                "Observed frame/view records cannot exceed the known source-view count."
            )
        target = self.resolution.effective_target_views
        planned_count = sum(item.planned for item in records)
        if target is not None and planned_count > target:
            raise ReconstructionValidationError(
                "Realized planned-view count cannot exceed effective_target_views."
            )
        if self.resolution.request.mode is FrameViewPlanMode.ALL_ELIGIBLE:
            eligible_count = sum(item.eligible for item in records)
            if planned_count != eligible_count:
                raise ReconstructionValidationError(
                    "ALL_ELIGIBLE realization must plan every eligible observed view."
                )

        if int(self.schema_version) != I65A_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I6.5A plan-realization schema version.")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "schema_version", I65A_SCHEMA_VERSION)
        object.__setattr__(
            self,
            "plan_provenance_sha256",
            _canonical_sha256(self._digest_payload()),
        )

    @property
    def observed_source_view_count(self) -> int:
        return len(self.records)

    @property
    def eligible_view_count(self) -> int:
        return sum(item.eligible for item in self.records)

    @property
    def planned_view_count(self) -> int:
        return sum(item.planned for item in self.records)

    @property
    def planned_source_view_ids(self) -> tuple[str, ...]:
        return tuple(item.source_view_id for item in self.records if item.planned)

    def _digest_payload(self) -> dict[str, Any]:
        return {
            "phase": I65A_PHASE,
            "schema_version": I65A_SCHEMA_VERSION,
            "plan_input_sha256": self.resolution.plan_input_sha256,
            "records": [item.as_dict() for item in self.records],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._digest_payload(),
            "plan_provenance_sha256": self.plan_provenance_sha256,
            "resolution": self.resolution.as_dict(),
            "counts": {
                "observed_source_views": self.observed_source_view_count,
                "eligible_views": self.eligible_view_count,
                "planned_views": self.planned_view_count,
            },
            "planned_source_view_ids": list(self.planned_source_view_ids),
            "authority": {
                "camera_pose_authority_owner": I65A_CAMERA_AUTHORITY_OWNER,
                "planned_views_are_registered_cameras": False,
                "pycolmap_registration_outcome_claimed": False,
                "metric_scale_claimed": False,
                "certified_metrology": False,
            },
            "effects": {
                "provider_selected": False,
                "provider_activated": False,
                "production_auto_activated": False,
                "execution_started": False,
                "scientific_threshold_modified": False,
                "production_default_modified": False,
            },
        }


__all__ = [
    "I65A_SCHEMA_VERSION",
    "I65A_PHASE",
    "I65A_CAMERA_AUTHORITY_OWNER",
    "I65A_PERMANENT_GLOBAL_VIEW_CAP",
    "I65A_MINIMUM_RECONSTRUCTION_VIEWS",
    "FrameViewPlanMode",
    "FrameViewRecordDisposition",
    "FrameViewSourceProvenance",
    "FrameViewPlanRequest",
    "FrameViewPlanResolution",
    "FrameViewRecord",
    "FrameViewPlanRealization",
]
