"""I6.5C controller-facing frame/view planning state.

This module composes the frozen I6.5A provenance contract with the frozen
I6.5B deterministic resource-aware resolver.  It represents *planning intent*
only: no source sampling, worker mutation, SfM execution, provider selection,
Auto activation, or camera-registration claim occurs here.

The state deliberately keeps user intent separate from the effective plan.  A
manual limit is a request, not an exact execution count; ``ALL_ELIGIBLE`` stays
uncapped even when resource pressure is reported; and a source/resource change
rebuilds a new deterministic plan rather than retaining stale effective state.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_frame_view_plan_contract import (
    FrameViewPlanMode,
    FrameViewPlanRequest,
    FrameViewSourceProvenance,
)
from core.reconstruction.reconstruction_frame_view_planning import (
    DeterministicFrameViewPlan,
    resolve_deterministic_frame_view_plan,
)
from core.reconstruction.sources import SUPPORTED_IMAGE_EXTENSIONS

I65C_SCHEMA_VERSION = 1
I65C_PHASE = "I6.5C"
I65C_COMPOSITION_PROFILE = "controller_frame_view_intent_only"
I65C_SOURCE_FINGERPRINT_PROFILE = "selection_metadata_manifest_v1"


class FrameViewPlanStateStatus(str, Enum):
    """Passive controller planning state; never an execution state."""

    WAITING_SOURCE = "waiting_source"
    UNSUPPORTED_WORKFLOW = "unsupported_workflow"
    WAITING_RESOURCES = "waiting_resources"
    BLOCKED = "blocked"
    RESOLVED = "resolved"


class FrameViewPlanIntentOrigin(str, Enum):
    """Whether the request is the product default or explicit user intent."""

    DEFAULT_ADAPTIVE = "default_adaptive"
    EXPLICIT_USER_OVERRIDE = "explicit_user_override"


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _workflow_value(source_selection: Any) -> str:
    raw = getattr(source_selection, "workflow", "none")
    value = getattr(raw, "value", raw)
    return str(value or "none").strip().lower()


def _selection_path(source_selection: Any) -> Path | None:
    raw = getattr(source_selection, "path", None)
    if raw is None:
        return None
    try:
        return Path(raw).expanduser().resolve()
    except (TypeError, ValueError, OSError):
        return None


def _image_sequence_provenance(path: Path) -> FrameViewSourceProvenance:
    if not path.is_dir():
        raise ReconstructionValidationError(f"Image-sequence directory does not exist: {path}")
    records: list[dict[str, Any]] = []
    for item in sorted(path.iterdir(), key=lambda value: value.name.casefold()):
        if not item.is_file() or item.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        stat = item.stat()
        records.append(
            {
                "relative_name": item.name,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    if len(records) < 3:
        raise ReconstructionValidationError(
            "Image-sequence frame/view planning requires at least 3 supported images."
        )
    fingerprint = _canonical_sha256(
        {
            "profile": I65C_SOURCE_FINGERPRINT_PROFILE,
            "source_kind": "image_sequence",
            "source_uri": str(path),
            "records": records,
        }
    )
    return FrameViewSourceProvenance(
        source_id=f"image-sequence:{fingerprint[:16]}",
        source_kind="image_sequence",
        source_uri=str(path),
        source_fingerprint_sha256=fingerprint,
        estimated_total_views=len(records),
        expected_frame_rate_hz=None,
    )


def _recorded_video_provenance(path: Path) -> FrameViewSourceProvenance:
    if not path.is_file():
        raise ReconstructionValidationError(f"Recorded video does not exist: {path}")
    stat = path.stat()
    fingerprint = _canonical_sha256(
        {
            "profile": I65C_SOURCE_FINGERPRINT_PROFILE,
            "source_kind": "recorded_video",
            "source_uri": str(path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    )
    return FrameViewSourceProvenance(
        source_id=f"recorded-video:{fingerprint[:16]}",
        source_kind="recorded_video",
        source_uri=str(path),
        source_fingerprint_sha256=fingerprint,
        # File-count=1 is not a video frame count.  I6.5C refuses to invent one.
        estimated_total_views=None,
        expected_frame_rate_hz=None,
    )


def source_provenance_from_selection(source_selection: Any) -> FrameViewSourceProvenance | None:
    """Build deterministic planning provenance from one validated offline selection.

    This is a metadata-manifest identity for planning invalidation, not a content
    evidence hash and not a PyCOLMAP camera-registration fingerprint.
    """

    if not bool(getattr(source_selection, "ready", False)):
        return None
    workflow = _workflow_value(source_selection)
    path = _selection_path(source_selection)
    if workflow == "image_sequence" and path is not None:
        return _image_sequence_provenance(path)
    if workflow == "recorded_video" and path is not None:
        return _recorded_video_provenance(path)
    return None


@dataclass(frozen=True, slots=True)
class ReconstructionFrameViewPlanState:
    """Controller-facing I6.5 planning snapshot with zero runtime effect."""

    request: FrameViewPlanRequest
    intent_origin: FrameViewPlanIntentOrigin
    status: FrameViewPlanStateStatus
    reason_codes: tuple[str, ...]
    source: FrameViewSourceProvenance | None = None
    plan: DeterministicFrameViewPlan | None = None
    schema_version: int = I65C_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.request, FrameViewPlanRequest):
            raise TypeError("request must be FrameViewPlanRequest.")
        if not isinstance(self.intent_origin, FrameViewPlanIntentOrigin):
            raise TypeError("intent_origin must be FrameViewPlanIntentOrigin.")
        if not isinstance(self.status, FrameViewPlanStateStatus):
            raise TypeError("status must be FrameViewPlanStateStatus.")
        reasons = tuple(str(value or "").strip() for value in self.reason_codes)
        if any(not value for value in reasons) or len(reasons) != len(set(reasons)):
            raise ValueError("reason_codes must be unique non-empty values.")
        if self.status is FrameViewPlanStateStatus.RESOLVED:
            if self.source is None or self.plan is None:
                raise ValueError("RESOLVED state requires source and plan.")
        elif self.plan is not None:
            raise ValueError("Only RESOLVED state may carry a deterministic plan.")
        if self.source is not None and not isinstance(self.source, FrameViewSourceProvenance):
            raise TypeError("source must be FrameViewSourceProvenance or None.")
        if self.plan is not None and not isinstance(self.plan, DeterministicFrameViewPlan):
            raise TypeError("plan must be DeterministicFrameViewPlan or None.")
        if int(self.schema_version) != I65C_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.5C state schema version.")
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I65C_SCHEMA_VERSION)

    @property
    def resolved(self) -> bool:
        return self.status is FrameViewPlanStateStatus.RESOLVED

    @property
    def effective_target_views(self) -> int | None:
        if self.plan is None:
            return None
        return self.plan.resolution.effective_target_views

    @property
    def resource_target_views(self) -> int | None:
        if self.plan is None:
            return None
        return self.plan.resolution.resource_target_views

    @property
    def planning_provenance_sha256(self) -> str:
        return "" if self.plan is None else self.plan.planning_provenance_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I65C_PHASE,
            "composition_profile": I65C_COMPOSITION_PROFILE,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "intent_origin": self.intent_origin.value,
            "request": self.request.as_dict(),
            "source": None if self.source is None else self.source.as_dict(),
            "plan": None if self.plan is None else self.plan.as_dict(),
            "effects": {
                "planning_intent_only": True,
                "source_sampling_executed": False,
                "worker_modified": False,
                "sfm_execution_started": False,
                "provider_selection_effect": False,
                "production_auto_activated": False,
                "settings_mutation": False,
                "progressive_sfm_visualization": False,
            },
            "authority": {
                "planned_views_are_registered_cameras": False,
                "pycolmap_registration_outcome_claimed": False,
                "metric_scale_claimed": False,
                "certified_metrology": False,
            },
        }


def build_reconstruction_frame_view_plan_state(
    *,
    source_selection: Any,
    request: FrameViewPlanRequest,
    intent_origin: FrameViewPlanIntentOrigin,
    resources: AdaptiveSelectionResourceSnapshot | None,
) -> ReconstructionFrameViewPlanState:
    """Compose one passive I6.5C state from explicit source/request/resources."""

    if not isinstance(request, FrameViewPlanRequest):
        raise TypeError("request must be FrameViewPlanRequest.")
    if not isinstance(intent_origin, FrameViewPlanIntentOrigin):
        raise TypeError("intent_origin must be FrameViewPlanIntentOrigin.")
    if resources is not None and not isinstance(resources, AdaptiveSelectionResourceSnapshot):
        raise TypeError("resources must be AdaptiveSelectionResourceSnapshot or None.")

    if not bool(getattr(source_selection, "ready", False)):
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.WAITING_SOURCE,
            reason_codes=("source_not_ready",),
        )

    workflow = _workflow_value(source_selection)
    if workflow not in {"image_sequence", "recorded_video"}:
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.UNSUPPORTED_WORKFLOW,
            reason_codes=("offline_pre_sfm_planning_not_supported_for_workflow",),
        )

    try:
        source = source_provenance_from_selection(source_selection)
    except ReconstructionValidationError as exc:
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.BLOCKED,
            reason_codes=("source_provenance_invalid",),
        )
    if source is None:  # fail closed for malformed future adapters
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.WAITING_SOURCE,
            reason_codes=("source_provenance_unresolved",),
        )

    if resources is None:
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.WAITING_RESOURCES,
            reason_codes=("resource_snapshot_unresolved",),
            source=source,
        )

    try:
        plan = resolve_deterministic_frame_view_plan(
            source=source,
            request=request,
            resources=resources,
        )
    except ReconstructionValidationError as exc:
        return ReconstructionFrameViewPlanState(
            request=request,
            intent_origin=intent_origin,
            status=FrameViewPlanStateStatus.BLOCKED,
            reason_codes=("plan_resolution_blocked",),
            source=source,
        )
    reasons = tuple(plan.resolution.reason_codes)
    return ReconstructionFrameViewPlanState(
        request=request,
        intent_origin=intent_origin,
        status=FrameViewPlanStateStatus.RESOLVED,
        reason_codes=reasons,
        source=source,
        plan=plan,
    )


__all__ = [
    "I65C_SCHEMA_VERSION",
    "I65C_PHASE",
    "I65C_COMPOSITION_PROFILE",
    "I65C_SOURCE_FINGERPRINT_PROFILE",
    "FrameViewPlanStateStatus",
    "FrameViewPlanIntentOrigin",
    "ReconstructionFrameViewPlanState",
    "source_provenance_from_selection",
    "build_reconstruction_frame_view_plan_state",
]
