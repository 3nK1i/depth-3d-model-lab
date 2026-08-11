"""Presentation-only reconstruction measurement bridge for Phase 8.8E2.

E2 reuses already-computed Phase 7.5 measurement evidence for the active E1
reconstruction object.  It does not run perception, depth, geometry fitting,
measurement computation, uncertainty evaluation, QC, registration, or reference
comparison.  E3 owns validity/confidence/uncertainty presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, isfinite
from typing import Any, Mapping, Sequence

from app.reconstruction_inspection_selection import ReconstructionInspectionSelection
from core.inspection.contracts import InspectionObjectSummary


_ALLOWED_SOURCE_STATUSES = {"available", "unavailable", "rejected"}


@dataclass(frozen=True, slots=True)
class ReconstructionMeasurementValue:
    """One source-authored measurement value shown by E2 without E3 confidence."""

    kind: str
    value_si: float
    unit_si: str
    method: str

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        unit = str(self.unit_si).strip()
        method = str(self.method).strip()
        value = float(self.value_si)
        if not kind:
            raise ValueError("Measurement kind must be non-empty.")
        if not unit:
            raise ValueError("Measurement unit must be non-empty.")
        if not method:
            raise ValueError("Measurement method must be non-empty.")
        if not isfinite(value):
            raise ValueError("Measurement value must be finite.")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit_si", unit)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "value_si", value)

    @property
    def label(self) -> str:
        return self.kind.replace("_", " ").title()

    @property
    def value_text(self) -> str:
        """Human-friendly value only; E3 adds uncertainty/validity context."""

        if self.kind in {"width", "height", "depth", "camera_distance", "centre_x", "centre_y", "centre_z"} and self.unit_si == "m":
            return f"{self.value_si * 1000.0:.1f} mm"
        if self.kind.startswith("orientation_") and self.unit_si == "rad":
            return f"{degrees(self.value_si):.2f}°"
        if self.kind == "volume" and self.unit_si == "m^3":
            return f"{self.value_si * 1000.0:.2f} L"
        return f"{self.value_si:.4g} {self.unit_si}"


@dataclass(frozen=True, slots=True)
class ReconstructionMeasurementPresentation:
    """Bounded E2 UI state for the current selected reconstruction object."""

    session_id: str
    target_object_id: str
    target_label: str
    source: str
    source_observation_id: str
    values: tuple[ReconstructionMeasurementValue, ...]
    source_measurement_count: int
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        session_id = str(self.session_id).strip()
        target_object_id = str(self.target_object_id).strip()
        target_label = str(self.target_label).strip()
        source = str(self.source).strip()
        source_observation_id = str(self.source_observation_id).strip()
        if not session_id or not target_object_id or not target_label:
            raise ValueError("E2 measurement presentation requires a bound E1 object target.")
        if not source or not source_observation_id:
            raise ValueError("E2 measurement presentation requires measurement provenance.")
        values = tuple(self.values)
        if any(not isinstance(value, ReconstructionMeasurementValue) for value in values):
            raise TypeError("values must contain ReconstructionMeasurementValue entries.")
        count = int(self.source_measurement_count)
        if count < len(values):
            raise ValueError("source_measurement_count cannot be smaller than presented values.")
        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E2 cannot claim certified metrology.")
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "target_object_id", target_object_id)
        object.__setattr__(self, "target_label", target_label)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_observation_id", source_observation_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "source_measurement_count", count)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def values_text(self) -> str:
        if not self.values:
            return "No displayable values"
        return "\n".join(f"{value.label}: {value.value_text}" for value in self.values)

    @property
    def state_text(self) -> str:
        shown = len(self.values)
        if shown == 0:
            return f"0/{self.source_measurement_count} values presented"
        return f"{shown}/{self.source_measurement_count} values presented"

    @property
    def authority_text(self) -> str:
        return "Existing measurement evidence • validity/confidence deferred to E3"

    @property
    def provenance_tooltip(self) -> str:
        return (
            f"source={self.source} • observation_id={self.source_observation_id} • "
            f"target_object_id={self.target_object_id}"
        )


def build_reconstruction_measurement_presentation(
    selection: ReconstructionInspectionSelection,
    measurement_object: InspectionObjectSummary | Mapping[str, Any],
    *,
    source: str = "phase7_5_measurement",
) -> ReconstructionMeasurementPresentation:
    """Bind existing measurement evidence to the current E1 object target.

    Only source measurements already marked ``available`` and carrying a finite
    value are shown.  E2 intentionally does not expose reliability, uncertainty,
    rejection reasons, calibration qualification, or QC decisions; those are E3/E4.
    """

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_object:
        raise ValueError("E2 measurement evidence requires an active E1 object target.")

    source = str(source).strip()
    if not source:
        raise ValueError("Measurement source must be non-empty.")

    observation_id, measurements = _measurement_payload(measurement_object)
    values: list[ReconstructionMeasurementValue] = []
    for raw in measurements:
        status = str(raw.get("status", "unavailable")).strip().lower()
        if status not in _ALLOWED_SOURCE_STATUSES:
            raise ValueError(f"Unsupported source measurement status: {status!r}")
        if status != "available" or raw.get("value_si") is None:
            continue
        values.append(
            ReconstructionMeasurementValue(
                kind=str(raw.get("kind", "")),
                value_si=float(raw["value_si"]),
                unit_si=str(raw.get("unit_si", "")),
                method=str(raw.get("method", "existing_measurement_evidence")),
            )
        )

    return ReconstructionMeasurementPresentation(
        session_id=selection.session_id or selection.scene_id,
        target_object_id=selection.object_id,
        target_label=selection.object_text,
        source=source,
        source_observation_id=observation_id,
        values=tuple(values),
        source_measurement_count=len(measurements),
        certified_metrology=False,
    )


def empty_reconstruction_measurement_text(selection: ReconstructionInspectionSelection) -> tuple[str, str, str, str, str]:
    """Return stable E2 empty-state rows for the current E1 target."""

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_scene:
        return (
            "No target",
            "No measurement evidence",
            "—",
            "Waiting for reconstruction target",
            "Existing measurements only • E2 does not compute measurements",
        )
    if not selection.has_object:
        return (
            "Scene",
            "No object measurement selected",
            "—",
            "Select a semantic object",
            "Existing measurements only • E2 does not compute measurements",
        )
    return (
        selection.object_text,
        "No measurement evidence attached",
        "—",
        "No values for selected object",
        "Existing measurements only • validity/confidence deferred to E3",
    )


def _measurement_payload(
    value: InspectionObjectSummary | Mapping[str, Any],
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    if isinstance(value, InspectionObjectSummary):
        return value.observation_id, tuple(dict(item) for item in value.measurements)
    if not isinstance(value, Mapping):
        raise TypeError("measurement_object must be InspectionObjectSummary or mapping.")
    observation_id = str(value.get("observation_id", "")).strip()
    if not observation_id:
        raise ValueError("Measurement evidence mapping requires observation_id.")
    raw_measurements = value.get("measurements", ())
    if not isinstance(raw_measurements, Sequence) or isinstance(
        raw_measurements, (str, bytes, bytearray)
    ):
        raise ValueError("Measurement evidence requires a sequence of measurements.")
    measurements: list[Mapping[str, Any]] = []
    for item in raw_measurements:
        if not isinstance(item, Mapping):
            raise ValueError("Each measurement evidence entry must be a mapping.")
        measurements.append(dict(item))
    return observation_id, tuple(measurements)


__all__ = [
    "ReconstructionMeasurementPresentation",
    "ReconstructionMeasurementValue",
    "build_reconstruction_measurement_presentation",
    "empty_reconstruction_measurement_text",
]
