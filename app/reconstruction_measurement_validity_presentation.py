"""Presentation-only measurement validity bridge for Phase 8.8E3.

E3 qualifies the exact Phase 7.5 measurement evidence already bound by E2.
It reads source-authored status, expanded uncertainty, bounds, reliability,
support, calibration/scale context, warnings, and rejection reasons.  It does
not recompute measurements or uncertainty, score reliability, run inspection
QC, perform reference comparison/alignment, or promote reconstruction geometry
to metric authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, isfinite
from typing import Any, Mapping, Sequence

from app.reconstruction_inspection_selection import ReconstructionInspectionSelection
from core.inspection.contracts import InspectionBundleSummary, InspectionObjectSummary


_ALLOWED_STATUSES = {"available", "unavailable", "rejected"}
_ALLOWED_RELIABILITY = {"high", "medium", "low", "rejected", "unavailable"}
_ALLOWED_DEPTH_SCALES = {"metric", "relative", "synthetic", "unknown"}
_ALLOWED_INTRINSICS = {"model_predicted", "estimated", "calibrated", "none", "unknown"}
_ALLOWED_CALIBRATION = {"calibrated", "not_calibrated", "unknown"}
_ALLOWED_CONFIDENCE_KINDS = {"none", "confidence", "uncertainty"}


@dataclass(frozen=True, slots=True)
class ReconstructionMeasurementValidityContext:
    """Source-authored Phase 7.5 measurement context; never inferred by E3."""

    depth_scale: str = "unknown"
    intrinsics_source: str = "unknown"
    calibration_status: str = "unknown"
    confidence_kind: str = "none"

    def __post_init__(self) -> None:
        depth_scale = str(self.depth_scale).strip().lower() or "unknown"
        intrinsics = str(self.intrinsics_source).strip().lower() or "unknown"
        calibration = str(self.calibration_status).strip().lower() or "unknown"
        confidence = str(self.confidence_kind).strip().lower() or "none"
        if depth_scale not in _ALLOWED_DEPTH_SCALES:
            raise ValueError(f"Unsupported measurement depth scale: {depth_scale!r}")
        if intrinsics not in _ALLOWED_INTRINSICS:
            raise ValueError(f"Unsupported intrinsics source: {intrinsics!r}")
        if calibration not in _ALLOWED_CALIBRATION:
            raise ValueError(f"Unsupported calibration status: {calibration!r}")
        if confidence not in _ALLOWED_CONFIDENCE_KINDS:
            raise ValueError(f"Unsupported confidence kind: {confidence!r}")
        object.__setattr__(self, "depth_scale", depth_scale)
        object.__setattr__(self, "intrinsics_source", intrinsics)
        object.__setattr__(self, "calibration_status", calibration)
        object.__setattr__(self, "confidence_kind", confidence)


@dataclass(frozen=True, slots=True)
class ReconstructionMeasurementValidityEntry:
    """One source-authored per-kind validity/uncertainty record."""

    kind: str
    status: str
    unit_si: str
    method: str
    reliability: str
    value_si: float | None = None
    expanded_uncertainty_si: float | None = None
    lower_bound_si: float | None = None
    upper_bound_si: float | None = None
    reliability_score: float | None = None
    valid_observation_count: int = 0
    rejection_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    uncertainty_basis: str = "unknown"
    uncertainty_qualification: str = ""

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        status = str(self.status).strip().lower()
        unit = str(self.unit_si).strip()
        method = str(self.method).strip()
        reliability = str(self.reliability).strip().lower() or "unavailable"
        if not kind or not unit or not method:
            raise ValueError("E3 validity entries require kind, unit, and method.")
        if status not in _ALLOWED_STATUSES:
            raise ValueError(f"Unsupported measurement status: {status!r}")
        if reliability not in _ALLOWED_RELIABILITY:
            raise ValueError(f"Unsupported measurement reliability: {reliability!r}")
        score = self.reliability_score
        if score is not None:
            score = float(score)
            if not isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("reliability_score must remain within [0, 1].")
        count = int(self.valid_observation_count)
        if isinstance(self.valid_observation_count, bool) or count < 0:
            raise ValueError("valid_observation_count must be non-negative.")
        reasons = tuple(str(value).strip() for value in self.rejection_reasons if str(value).strip())
        warnings = tuple(str(value).strip() for value in self.warnings if str(value).strip())

        value = _optional_finite(self.value_si, "value_si")
        uncertainty = _optional_finite(self.expanded_uncertainty_si, "expanded_uncertainty_si")
        lower = _optional_finite(self.lower_bound_si, "lower_bound_si")
        upper = _optional_finite(self.upper_bound_si, "upper_bound_si")
        if uncertainty is not None and uncertainty < 0.0:
            raise ValueError("expanded_uncertainty_si must be non-negative.")
        if lower is not None and upper is not None and upper < lower:
            raise ValueError("upper_bound_si must not be smaller than lower_bound_si.")

        if status == "available":
            if value is None or uncertainty is None:
                raise ValueError("Available E3 measurements require source value and uncertainty.")
            if reliability not in {"high", "medium", "low"}:
                raise ValueError("Available E3 measurements require available source reliability.")
            if reasons:
                raise ValueError("Available E3 measurements cannot carry rejection reasons.")
            if (lower is None) != (upper is None):
                raise ValueError("Source measurement bounds must be provided as a pair.")
            if lower is not None and upper is not None and not (lower <= value <= upper):
                raise ValueError("Source measurement bounds must contain value_si.")
        else:
            if value is not None or uncertainty is not None or lower is not None or upper is not None:
                raise ValueError("Unavailable/rejected E3 measurements must not expose numeric values.")
            if not reasons:
                raise ValueError("Unavailable/rejected E3 measurements require source rejection reasons.")
            expected_reliability = "rejected" if status == "rejected" else "unavailable"
            if reliability != expected_reliability:
                raise ValueError(
                    f"{status.title()} E3 measurements require {expected_reliability!r} source reliability."
                )

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unit_si", unit)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "reliability", reliability)
        object.__setattr__(self, "value_si", value)
        object.__setattr__(self, "expanded_uncertainty_si", uncertainty)
        object.__setattr__(self, "lower_bound_si", lower)
        object.__setattr__(self, "upper_bound_si", upper)
        object.__setattr__(self, "reliability_score", score)
        object.__setattr__(self, "valid_observation_count", count)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "uncertainty_basis", str(self.uncertainty_basis).strip().lower() or "unknown")
        object.__setattr__(self, "uncertainty_qualification", str(self.uncertainty_qualification).strip())

    @property
    def label(self) -> str:
        return self.kind.replace("_", " ").title()

    @property
    def qualified_value_text(self) -> str:
        if self.status != "available" or self.value_si is None or self.expanded_uncertainty_si is None:
            return f"{self.label}: —"
        value = _format_quantity(self.kind, self.value_si, self.unit_si)
        uncertainty = _format_quantity(self.kind, self.expanded_uncertainty_si, self.unit_si, uncertainty=True)
        text = f"{self.label}: {value} ± {uncertainty}"
        if self.lower_bound_si is not None and self.upper_bound_si is not None:
            lower = _format_quantity(self.kind, self.lower_bound_si, self.unit_si)
            upper = _format_quantity(self.kind, self.upper_bound_si, self.unit_si)
            text += f" [{lower}–{upper}]"
        return text

    @property
    def reliability_text(self) -> str:
        label = self.reliability.title()
        if self.reliability_score is not None:
            label += f" {self.reliability_score * 100.0:.0f}%"
        return f"{self.label}: {label}"

    @property
    def status_text(self) -> str:
        return f"{self.label}: {self.status.title()}"

    @property
    def support_text(self) -> str:
        return f"{self.label}: {self.valid_observation_count} obs"

    @property
    def reason_text(self) -> str:
        details = [*_humanize_many(self.rejection_reasons), *_humanize_many(self.warnings)]
        if not details:
            return ""
        return f"{self.label}: " + "; ".join(dict.fromkeys(details))


@dataclass(frozen=True, slots=True)
class ReconstructionMeasurementValidityPresentation:
    """Bounded E3 presentation for the exact E2-bound measurement observation."""

    session_id: str
    target_object_id: str
    target_label: str
    source: str
    source_observation_id: str
    entries: tuple[ReconstructionMeasurementValidityEntry, ...]
    context: ReconstructionMeasurementValidityContext
    scene_metric_geometry_status: str
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for name in ("session_id", "target_object_id", "target_label", "source", "source_observation_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"E3 validity presentation requires {name}.")
            object.__setattr__(self, name, value)
        entries = tuple(self.entries)
        if not entries or any(not isinstance(value, ReconstructionMeasurementValidityEntry) for value in entries):
            raise ValueError("E3 validity presentation requires source measurement entries.")
        if not isinstance(self.context, ReconstructionMeasurementValidityContext):
            raise TypeError("context must be ReconstructionMeasurementValidityContext.")
        scene_status = str(self.scene_metric_geometry_status).strip().lower() or "unknown"
        if scene_status not in {"arbitrary", "provisional_metric", "validated_metric", "unknown"}:
            raise ValueError(f"Unsupported reconstruction metric geometry status: {scene_status!r}")
        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E3 cannot claim certified metrology.")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "scene_metric_geometry_status", scene_status)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def qualified_values_text(self) -> str:
        available = [entry.qualified_value_text for entry in self.entries if entry.status == "available"]
        return "\n".join(available) if available else "No available values"

    @property
    def validity_text(self) -> str:
        counts = {name: sum(entry.status == name for entry in self.entries) for name in _ALLOWED_STATUSES}
        parts = []
        for name in ("available", "unavailable", "rejected"):
            if counts[name]:
                parts.append(f"{counts[name]} {name}")
        return " • ".join(parts) if parts else "No measurement status"

    @property
    def reliability_text(self) -> str:
        return "\n".join(entry.reliability_text for entry in self.entries)

    @property
    def uncertainty_text(self) -> str:
        qualifications = tuple(
            dict.fromkeys(
                entry.uncertainty_qualification
                for entry in self.entries
                if entry.status == "available" and entry.uncertainty_qualification
            )
        )
        if len(qualifications) == 1:
            qualification = qualifications[0]
        elif len(qualifications) > 1:
            qualification = "Mixed source uncertainty qualifications"
        else:
            qualification = "Source uncertainty qualification unavailable"
        confidence = {
            "confidence": "depth confidence channel",
            "uncertainty": "depth uncertainty channel",
            "none": "no depth confidence channel",
        }[self.context.confidence_kind]
        return f"{qualification} • {confidence}"

    @property
    def scale_text(self) -> str:
        source_scale = {
            "metric": "Measurement: Metric depth",
            "relative": "Measurement: Relative depth",
            "synthetic": "Measurement: Synthetic depth",
            "unknown": "Measurement: Unknown depth scale",
        }[self.context.depth_scale]
        scene = {
            "arbitrary": "scene: Arbitrary gauge",
            "provisional_metric": "scene: Provisional metric",
            "validated_metric": "scene: Validated metric",
            "unknown": "scene: Unknown scale",
        }[self.scene_metric_geometry_status]
        return f"{source_scale} • {scene}"

    @property
    def calibration_text(self) -> str:
        intrinsics = {
            "model_predicted": "Model-predicted intrinsics",
            "estimated": "Estimated intrinsics",
            "calibrated": "Calibrated intrinsics",
            "none": "No intrinsics source",
            "unknown": "Unknown intrinsics source",
        }[self.context.intrinsics_source]
        calibration = {
            "calibrated": "Calibrated",
            "not_calibrated": "Not Calibrated",
            "unknown": "Calibration unknown",
        }[self.context.calibration_status]
        return f"{intrinsics} • {calibration}"

    @property
    def support_text(self) -> str:
        return "\n".join(entry.support_text for entry in self.entries)

    @property
    def reasons_text(self) -> str:
        reasons = [entry.reason_text for entry in self.entries if entry.reason_text]
        return "\n".join(reasons) if reasons else "No source warnings/rejections"

    @property
    def authority_text(self) -> str:
        return "Source validity/uncertainty • scene scale unchanged • not QC decision"

    @property
    def provenance_tooltip(self) -> str:
        return (
            f"source={self.source} • observation_id={self.source_observation_id} • "
            f"target_object_id={self.target_object_id} • depth_scale={self.context.depth_scale} • "
            f"intrinsics={self.context.intrinsics_source} • calibration={self.context.calibration_status}"
        )


def build_reconstruction_measurement_validity_presentation(
    selection: ReconstructionInspectionSelection,
    measurement_object: InspectionObjectSummary | Mapping[str, Any],
    *,
    source: str = "phase7_5_measurement",
    context: ReconstructionMeasurementValidityContext | InspectionBundleSummary | Mapping[str, Any] | None = None,
) -> ReconstructionMeasurementValidityPresentation:
    """Present existing E3 qualification fields without deriving new evidence."""

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_object:
        raise ValueError("E3 validity evidence requires an active E1 object target.")
    source = str(source).strip()
    if not source:
        raise ValueError("Measurement validity source must be non-empty.")

    observation_id, measurements = _measurement_payload(measurement_object)
    entries = tuple(_entry_from_payload(raw) for raw in measurements)
    if not entries:
        raise ValueError("E3 validity evidence requires at least one source measurement.")
    validity_context = _validity_context(context)

    return ReconstructionMeasurementValidityPresentation(
        session_id=selection.session_id or selection.scene_id,
        target_object_id=selection.object_id,
        target_label=selection.object_text,
        source=source,
        source_observation_id=observation_id,
        entries=entries,
        context=validity_context,
        scene_metric_geometry_status=selection.metric_geometry_status,
        certified_metrology=False,
    )


def empty_reconstruction_measurement_validity_text(
    selection: ReconstructionInspectionSelection,
    *,
    has_e2_evidence: bool = False,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Stable E3 empty rows following E1/E2 target state."""

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    authority = "Existing validity evidence only • E3 does not compute confidence/uncertainty"
    if not selection.has_scene:
        return ("—", "Waiting for reconstruction target", "—", "—", "—", "—", "—", authority)
    if not selection.has_object:
        return ("—", "Select a semantic object", "—", "—", "—", "—", "—", authority)
    if not has_e2_evidence:
        return ("—", "Measurement evidence required", "—", "—", "—", "—", "—", authority)
    return (
        "—",
        "Validity evidence not attached",
        "—",
        "—",
        "—",
        "—",
        "—",
        "Source-authored validity only • exact E2 provenance required",
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
        raise ValueError("E3 measurement evidence mapping requires observation_id.")
    raw_measurements = value.get("measurements", ())
    if not isinstance(raw_measurements, Sequence) or isinstance(raw_measurements, (str, bytes, bytearray)):
        raise ValueError("E3 measurement evidence requires a sequence of measurements.")
    values = []
    for item in raw_measurements:
        if not isinstance(item, Mapping):
            raise ValueError("Each E3 measurement entry must be a mapping.")
        values.append(dict(item))
    return observation_id, tuple(values)


def _entry_from_payload(raw: Mapping[str, Any]) -> ReconstructionMeasurementValidityEntry:
    return ReconstructionMeasurementValidityEntry(
        kind=str(raw.get("kind", "")),
        status=str(raw.get("status", "unavailable")),
        unit_si=str(raw.get("unit_si", "")),
        method=str(raw.get("method", "existing_measurement_evidence")),
        reliability=str(raw.get("reliability", "unavailable")),
        value_si=raw.get("value_si"),
        expanded_uncertainty_si=raw.get("expanded_uncertainty_si"),
        lower_bound_si=raw.get("lower_bound_si"),
        upper_bound_si=raw.get("upper_bound_si"),
        reliability_score=raw.get("reliability_score"),
        valid_observation_count=raw.get("valid_observation_count", 0),
        rejection_reasons=tuple(raw.get("rejection_reasons", ())),
        warnings=tuple(raw.get("warnings", ())),
        uncertainty_basis=str(raw.get("uncertainty_basis", "unknown")),
        uncertainty_qualification=str(
            raw.get("uncertainty_qualification")
            or raw.get("uncertainty_display_qualification")
            or ""
        ),
    )


def _validity_context(
    value: ReconstructionMeasurementValidityContext | InspectionBundleSummary | Mapping[str, Any] | None,
) -> ReconstructionMeasurementValidityContext:
    if value is None:
        return ReconstructionMeasurementValidityContext()
    if isinstance(value, ReconstructionMeasurementValidityContext):
        return value
    if isinstance(value, InspectionBundleSummary):
        return ReconstructionMeasurementValidityContext(
            depth_scale=value.depth_scale,
            intrinsics_source=value.intrinsics_source,
            calibration_status=value.calibration_status,
            confidence_kind=value.confidence_kind,
        )
    if not isinstance(value, Mapping):
        raise TypeError("context must be E3 validity context, InspectionBundleSummary, mapping, or None.")
    return ReconstructionMeasurementValidityContext(
        depth_scale=str(value.get("depth_scale", "unknown")),
        intrinsics_source=str(value.get("intrinsics_source", "unknown")),
        calibration_status=str(value.get("calibration_status", "unknown")),
        confidence_kind=str(value.get("confidence_kind", "none")),
    )


def _format_quantity(kind: str, value: float, unit_si: str, *, uncertainty: bool = False) -> str:
    number = float(value)
    if kind in {"width", "height", "depth", "camera_distance", "centre_x", "centre_y", "centre_z"} and unit_si == "m":
        return f"{number * 1000.0:.1f} mm"
    if kind.startswith("orientation_") and unit_si == "rad":
        return f"{degrees(number):.2f}°"
    if kind == "volume" and unit_si in {"m3", "m^3"}:
        return f"{number * 1000.0:.2f} L"
    precision = ".3g" if uncertainty else ".4g"
    return f"{format(number, precision)} {unit_si}"


def _optional_finite(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    return number


def _humanize_many(values: Sequence[str]) -> list[str]:
    return [str(value).replace("_", " ").strip().capitalize() for value in values if str(value).strip()]


__all__ = [
    "ReconstructionMeasurementValidityContext",
    "ReconstructionMeasurementValidityEntry",
    "ReconstructionMeasurementValidityPresentation",
    "build_reconstruction_measurement_validity_presentation",
    "empty_reconstruction_measurement_validity_text",
]
