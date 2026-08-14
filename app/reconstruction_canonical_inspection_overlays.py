"""Canonical object inspection/measurement overlays for Phase 9.3E.

This module is presentation-only.  It binds already-authored Phase 8.8 E2/E3/E4
measurement/validity/QC evidence to the exact Phase 9.3D persisted semantic
object selected in the authoritative canonical scene.

Numeric measurement values are allowed only from E3 entries whose source status
is ``available`` and which carry source-authored uncertainty.  Rejected or
unavailable entries are rendered as status/reason text only.  Semantic support
bounds are never reinterpreted as measurement boxes, and Phase 7.5 camera-space
box geometry is not drawn because no authoritative camera-to-canonical transform
is part of the E2/E3 presentation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from app.reconstruction_canonical_semantics import CanonicalSemanticScenePresentation
from app.reconstruction_inspection_qc_presentation import ReconstructionInspectionQcPresentation
from app.reconstruction_inspection_selection import ReconstructionInspectionSelection
from app.reconstruction_measurement_presentation import ReconstructionMeasurementPresentation
from app.reconstruction_measurement_validity_presentation import (
    ReconstructionMeasurementValidityEntry,
    ReconstructionMeasurementValidityPresentation,
)

MAX_CANONICAL_MEASUREMENT_OVERLAY_ENTRIES = 3
MAX_CANONICAL_QC_FINDINGS = 2


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalObjectInspectionOverlayPresentation:
    """One bounded, source-authored overlay for the selected canonical object."""

    scene_id: str
    session_id: str
    object_id: str
    target_label: str
    anchor_viewer: np.ndarray
    lines: tuple[str, ...]
    numeric_measurement_count: int
    rejected_measurement_count: int
    unavailable_measurement_count: int
    inspection_result_attached: bool
    inspection_result_available: bool
    scene_metric_geometry_status: str
    measurement_source: str = ""
    measurement_observation_id: str = ""
    inspection_source: str = ""
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for name in ("scene_id", "session_id", "object_id", "target_label"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        anchor = np.asarray(self.anchor_viewer, dtype=np.float32)
        if anchor.shape != (3,) or not np.isfinite(anchor).all():
            raise ValueError("anchor_viewer must be one finite 3D point.")
        anchor = np.ascontiguousarray(anchor)
        anchor.setflags(write=False)
        object.__setattr__(self, "anchor_viewer", anchor)

        lines = tuple(str(value).strip() for value in self.lines if str(value).strip())
        if not lines:
            raise ValueError("Canonical inspection overlay requires at least one source-evidence line.")
        object.__setattr__(self, "lines", lines)
        for name in (
            "numeric_measurement_count",
            "rejected_measurement_count",
            "unavailable_measurement_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
            object.__setattr__(self, name, int(value))
        status = str(self.scene_metric_geometry_status).strip().lower() or "unknown"
        if status not in {"arbitrary", "provisional_metric", "validated_metric", "unknown"}:
            raise ValueError(f"Unsupported scene metric geometry status: {status!r}")
        object.__setattr__(self, "scene_metric_geometry_status", status)
        object.__setattr__(self, "measurement_source", str(self.measurement_source).strip())
        object.__setattr__(
            self, "measurement_observation_id", str(self.measurement_observation_id).strip()
        )
        object.__setattr__(self, "inspection_source", str(self.inspection_source).strip())
        object.__setattr__(self, "inspection_result_attached", bool(self.inspection_result_attached))
        object.__setattr__(self, "inspection_result_available", bool(self.inspection_result_available))
        if self.inspection_result_available and not self.inspection_result_attached:
            raise ValueError("Available inspection result requires attached inspection evidence.")
        if bool(self.certified_metrology):
            raise ValueError("Phase 9.3E overlays cannot claim certified metrology.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def overlay_text(self) -> str:
        return "\n".join(self.lines)

    @property
    def measurement_status_text(self) -> str:
        parts = []
        if self.numeric_measurement_count:
            parts.append(f"{self.numeric_measurement_count} available")
        if self.rejected_measurement_count:
            parts.append(f"{self.rejected_measurement_count} rejected")
        if self.unavailable_measurement_count:
            parts.append(f"{self.unavailable_measurement_count} unavailable")
        return " • ".join(parts) if parts else "No qualified measurement overlay"

    @property
    def authority_text(self) -> str:
        return (
            "Source-authored measurement/QC evidence • canonical geometry unchanged • "
            "not certified metrology"
        )


def build_canonical_object_inspection_overlay(
    *,
    selection: ReconstructionInspectionSelection,
    canonical_semantics: CanonicalSemanticScenePresentation | None,
    measurement: ReconstructionMeasurementPresentation | None = None,
    validity: ReconstructionMeasurementValidityPresentation | None = None,
    inspection_qc: ReconstructionInspectionQcPresentation | None = None,
) -> CanonicalObjectInspectionOverlayPresentation | None:
    """Build the 9.3E overlay for the exact selected persisted semantic object.

    E2 alone is insufficient for numeric overlay output.  E3 validity must be
    present and must match the exact E2 target/source/observation.  E4 QC remains
    independent source authority and is copied only for the exact selected
    object; scene decisions are never reused as object decisions.
    """

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_object or canonical_semantics is None:
        return None
    if not isinstance(canonical_semantics, CanonicalSemanticScenePresentation):
        raise TypeError("canonical_semantics must be CanonicalSemanticScenePresentation or None.")
    if canonical_semantics.certified_metrology:
        raise ValueError("Canonical semantic presentation unexpectedly claims certified metrology.")

    session_id = selection.session_id or selection.scene_id
    if canonical_semantics.session_id != session_id:
        raise ValueError("Canonical semantic session does not match the active inspection target.")
    if selection.scene_id and canonical_semantics.scene_id != selection.scene_id:
        raise ValueError("Canonical semantic scene does not match the active inspection target.")

    matches = [item for item in canonical_semantics.objects if item.object_id == selection.object_id]
    if len(matches) != 1:
        if not matches:
            return None
        raise ValueError("Canonical semantic object IDs are unexpectedly duplicated.")
    semantic = matches[0]
    if semantic.certified_metrology:
        raise ValueError("Canonical semantic object unexpectedly claims certified metrology.")

    lines: list[str] = []
    available_count = 0
    rejected_count = 0
    unavailable_count = 0
    measurement_source = ""
    measurement_observation_id = ""

    if validity is not None:
        if measurement is None:
            raise ValueError("E3 validity overlay requires the exact E2 measurement presentation.")
        _validate_measurement_binding(selection, measurement, validity)
        if validity.certified_metrology or measurement.certified_metrology:
            raise ValueError("Measurement presentation unexpectedly claims certified metrology.")
        if validity.scene_metric_geometry_status != canonical_semantics.metric_geometry_status:
            raise ValueError(
                "Measurement validity scene-scale status does not match canonical semantic scene scale."
            )
        measurement_source = validity.source
        measurement_observation_id = validity.source_observation_id
        measurement_lines, available_count, rejected_count, unavailable_count = _measurement_lines(
            validity.entries
        )
        lines.extend(measurement_lines)
        if validity.entries:
            lines.append(_scale_qualifier(validity))
    elif measurement is not None:
        # E2 values are intentionally omitted from the viewport until E3 supplies
        # source-authored validity/uncertainty.  This is not an error: the sidebar
        # keeps the legacy E2 presentation while 9.3E stays fail-closed visually.
        _validate_e2_target(selection, measurement)

    inspection_attached = inspection_qc is not None
    inspection_available = False
    inspection_source = ""
    if inspection_qc is not None:
        _validate_qc_binding(selection, inspection_qc)
        if inspection_qc.certified_metrology:
            raise ValueError("Inspection/QC presentation unexpectedly claims certified metrology.")
        inspection_source = inspection_qc.source
        inspection_available = bool(inspection_qc.target_result_available)
        lines.append(_inspection_line(inspection_qc))

    if not lines:
        return None

    # Always keep the viewport authority qualifier explicit.  The measurement
    # values may be metric source evidence even when the canonical scene remains
    # arbitrary gauge, and neither E3 nor E4 certifies metrology.
    lines.append("Evidence overlay • canonical geometry unchanged • metrology not certified")

    return CanonicalObjectInspectionOverlayPresentation(
        scene_id=canonical_semantics.scene_id,
        session_id=canonical_semantics.session_id,
        object_id=selection.object_id,
        target_label=selection.object_text,
        anchor_viewer=_overlay_anchor(semantic.centroid_viewer, semantic.surface_vertices_viewer),
        lines=tuple(lines),
        numeric_measurement_count=available_count,
        rejected_measurement_count=rejected_count,
        unavailable_measurement_count=unavailable_count,
        inspection_result_attached=inspection_attached,
        inspection_result_available=inspection_available,
        scene_metric_geometry_status=canonical_semantics.metric_geometry_status,
        measurement_source=measurement_source,
        measurement_observation_id=measurement_observation_id,
        inspection_source=inspection_source,
        certified_metrology=False,
    )


def _validate_e2_target(
    selection: ReconstructionInspectionSelection,
    measurement: ReconstructionMeasurementPresentation,
) -> None:
    if not isinstance(measurement, ReconstructionMeasurementPresentation):
        raise TypeError("measurement must be ReconstructionMeasurementPresentation or None.")
    session_id = selection.session_id or selection.scene_id
    if measurement.session_id != session_id or measurement.target_object_id != selection.object_id:
        raise ValueError("E2 measurement presentation is not bound to the active canonical object.")


def _validate_measurement_binding(
    selection: ReconstructionInspectionSelection,
    measurement: ReconstructionMeasurementPresentation,
    validity: ReconstructionMeasurementValidityPresentation,
) -> None:
    _validate_e2_target(selection, measurement)
    if not isinstance(validity, ReconstructionMeasurementValidityPresentation):
        raise TypeError("validity must be ReconstructionMeasurementValidityPresentation or None.")
    session_id = selection.session_id or selection.scene_id
    if validity.session_id != session_id or validity.target_object_id != selection.object_id:
        raise ValueError("E3 validity presentation is not bound to the active canonical object.")
    if validity.source != measurement.source:
        raise ValueError("E3 validity source does not match the exact E2 measurement source.")
    if validity.source_observation_id != measurement.source_observation_id:
        raise ValueError("E3 validity observation does not match the exact E2 measurement observation.")


def _validate_qc_binding(
    selection: ReconstructionInspectionSelection,
    qc: ReconstructionInspectionQcPresentation,
) -> None:
    if not isinstance(qc, ReconstructionInspectionQcPresentation):
        raise TypeError("inspection_qc must be ReconstructionInspectionQcPresentation or None.")
    session_id = selection.session_id or selection.scene_id
    if qc.session_id != session_id:
        raise ValueError("Inspection/QC session does not match the active canonical object.")
    if qc.target_scope != "object" or qc.target_object_id != selection.object_id:
        raise ValueError("Inspection/QC presentation is not bound to the active canonical object.")


def _measurement_lines(
    entries: tuple[ReconstructionMeasurementValidityEntry, ...],
) -> tuple[tuple[str, ...], int, int, int]:
    lines: list[str] = []
    available = rejected = unavailable = 0
    bounded = tuple(entries[:MAX_CANONICAL_MEASUREMENT_OVERLAY_ENTRIES])
    for entry in bounded:
        if entry.status == "available":
            available += 1
            reliability = entry.reliability.title()
            if entry.reliability_score is not None:
                reliability += f" {entry.reliability_score * 100.0:.0f}%"
            lines.append(f"{entry.qualified_value_text} • {reliability}")
            continue
        if entry.status == "rejected":
            rejected += 1
        else:
            unavailable += 1
        reason = _entry_reason(entry)
        suffix = f" • {reason}" if reason else ""
        lines.append(f"{entry.label}: {entry.status.upper()}{suffix}")
    omitted = len(entries) - len(bounded)
    if omitted > 0:
        lines.append(f"+{omitted} additional measurement status(es) in sidebar")
    # Counts must describe the full source evidence, not only the bounded text.
    if omitted > 0:
        available = sum(entry.status == "available" for entry in entries)
        rejected = sum(entry.status == "rejected" for entry in entries)
        unavailable = sum(entry.status == "unavailable" for entry in entries)
    return tuple(lines), available, rejected, unavailable


def _entry_reason(entry: ReconstructionMeasurementValidityEntry) -> str:
    values = [
        *(str(value).strip() for value in entry.rejection_reasons if str(value).strip()),
        *(str(value).strip() for value in entry.warnings if str(value).strip()),
    ]
    if not values:
        return ""
    return "; ".join(dict.fromkeys(_humanize(value) for value in values))


def _scale_qualifier(validity: ReconstructionMeasurementValidityPresentation) -> str:
    source = {
        "metric": "metric depth",
        "relative": "relative depth",
        "synthetic": "synthetic depth",
        "unknown": "unknown depth scale",
    }[validity.context.depth_scale]
    scene = {
        "arbitrary": "arbitrary scene gauge",
        "provisional_metric": "provisional metric scene",
        "validated_metric": "validated metric scene",
        "unknown": "unknown scene scale",
    }[validity.scene_metric_geometry_status]
    return f"Measurement source: {source} • canonical scene: {scene}"


def _inspection_line(qc: ReconstructionInspectionQcPresentation) -> str:
    if not qc.target_result_available:
        return "QC: NOT EVALUATED • no exact selected-object result in source report"
    decision = str(qc.target_decision).strip().upper().replace("_", " ")
    findings = tuple(_humanize(value) for value in qc.findings[:MAX_CANONICAL_QC_FINDINGS])
    suffix = f" • {'; '.join(findings)}" if findings else ""
    return f"QC: {decision}{suffix}"


def _overlay_anchor(centroid_viewer: np.ndarray, surface_vertices_viewer: np.ndarray) -> np.ndarray:
    centroid = np.asarray(centroid_viewer, dtype=np.float64)
    if centroid.shape != (3,) or not np.isfinite(centroid).all():
        raise ValueError("Canonical semantic centroid is invalid for overlay anchoring.")
    anchor = np.array(centroid, copy=True)
    vertices = np.asarray(surface_vertices_viewer, dtype=np.float64)
    if vertices.ndim == 2 and vertices.shape[1:] == (3,) and vertices.shape[0] and np.isfinite(vertices).all():
        span = np.ptp(vertices, axis=0)
        local_scale = float(np.max(span))
        top_z = float(np.max(vertices[:, 2]))
        # Keep the larger 9.3E evidence callout visually separated from the
        # selected surface while remaining tightly anchored to that object.
        offset = max(local_scale * 0.20, np.finfo(np.float64).eps * 1024.0)
        if isfinite(top_z) and isfinite(offset):
            anchor[2] = top_z + offset
    return anchor.astype(np.float32)


def _humanize(value: str) -> str:
    return str(value).strip().replace("_", " ").replace("-", " ").capitalize()


__all__ = [
    "CanonicalObjectInspectionOverlayPresentation",
    "MAX_CANONICAL_MEASUREMENT_OVERLAY_ENTRIES",
    "build_canonical_object_inspection_overlay",
]
