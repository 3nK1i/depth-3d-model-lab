"""Reference-comparison workflow presentation for Phase 8.8E5.

E5 is an application/UI integration layer over already-qualified Phase 8.7C
reference and inspection contracts.  It can load the JSON emitted by
``write_reference_assembly_json`` and pair that exact reference identity with
an already-produced Phase 8.7C inspection report for the active E1 scene or
object target.

This module deliberately does *not* parse CAD meshes, build reference
assemblies, register geometry, apply tolerances, compute deviations, infer QC
from semantic/measurement confidence, or certify metrology.  Those operations
remain owned by the existing reconstruction/reference/QC layers.  E5 only
validates provenance/identity and presents source-authored comparison evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from app.reconstruction_inspection_selection import ReconstructionInspectionSelection

_ALLOWED_REFERENCE_SOURCES = {
    "golden_reconstruction",
    "cad_feature_model",
    "manual_reference",
}
_ALLOWED_METRIC_STATUS = {"arbitrary", "provisional_metric", "validated_metric"}
_ALLOWED_REGISTRATION_STATUS = {
    "success",
    "insufficient_evidence",
    "degenerate",
    "residual_too_high",
}


@dataclass(frozen=True, slots=True)
class ReconstructionReferenceDescriptor:
    """Bounded metadata view of an existing Phase 8.7C ReferenceAssembly."""

    reference_id: str
    source: str
    scene_profile: str
    reference_frame_id: str
    metric_geometry_status: str
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    feature_count: int
    registration_anchor_count: int
    provenance: str
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "reference_id",
            "source",
            "scene_profile",
            "reference_frame_id",
            "metric_geometry_status",
            "provenance",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"E5 reference descriptor requires {field_name}.")
            object.__setattr__(self, field_name, value)

        source = self.source.lower()
        if source not in _ALLOWED_REFERENCE_SOURCES:
            raise ValueError(f"Unsupported reference source: {source!r}")
        object.__setattr__(self, "source", source)

        metric = _metric_status(self.metric_geometry_status, "metric_geometry_status")
        object.__setattr__(self, "metric_geometry_status", metric)

        unit = self.unit_to_metre
        uncertainty = self.relative_scale_uncertainty
        if metric == "arbitrary":
            if unit is not None or uncertainty is not None:
                raise ValueError("Arbitrary E5 reference cannot expose metric scale values.")
        else:
            if unit is None:
                raise ValueError("Metric E5 reference requires unit_to_metre.")
            unit = _positive_float(unit, "unit_to_metre")
            if uncertainty is not None:
                uncertainty = _nonnegative_float(
                    uncertainty, "relative_scale_uncertainty"
                )
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)

        feature_count = _positive_int(self.feature_count, "feature_count")
        anchor_count = _positive_int(
            self.registration_anchor_count, "registration_anchor_count"
        )
        if anchor_count > feature_count:
            raise ValueError("registration_anchor_count cannot exceed feature_count.")
        object.__setattr__(self, "feature_count", feature_count)
        object.__setattr__(self, "registration_anchor_count", anchor_count)

        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E5 cannot claim certified metrology.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def reference_text(self) -> str:
        return f"{self.reference_id} • {_humanize(self.source)}"

    @property
    def structure_text(self) -> str:
        return (
            f"{self.feature_count} features • {self.registration_anchor_count} anchors • "
            f"{_humanize(self.scene_profile)}"
        )

    @property
    def scale_text(self) -> str:
        if self.metric_geometry_status == "arbitrary":
            return "Arbitrary reference gauge • no metric unit claim"
        unit = float(self.unit_to_metre or 0.0)
        label = _metric_status_label(self.metric_geometry_status)
        uncertainty = ""
        if self.relative_scale_uncertainty is not None:
            uncertainty = f" • scale ±{self.relative_scale_uncertainty * 100.0:.1f}%"
        return f"{label} • 1 ref unit = {unit:.9g} m{uncertainty}"


@dataclass(frozen=True, slots=True)
class ReconstructionReferenceComparisonPresentation:
    """Read-only E5 pairing of one reference descriptor and one 8.7C report."""

    session_id: str
    target_scope: str
    target_object_id: str
    target_label: str
    reference: ReconstructionReferenceDescriptor
    report_source: str
    report_attached: bool
    report_reference_match: bool
    comparison_available: bool
    comparison_metric_available: bool
    candidate_metric_geometry_status: str
    registration_status: str
    registration_method: str
    registration_confidence: float
    registration_reason: str
    registration_rms_reference_units: float | None
    registration_rms_m: float | None
    registration_normalized_rms_fraction: float | None
    matched_reference_feature_id: str
    target_result_available: bool
    position_error_reference_units: float | None
    position_error_m: float | None
    orientation_error_degrees: float | None
    extent_relative_error: float | None
    geometry_quality: float | None
    uncertainty_reference_units: float | None
    uncertainty_m: float | None
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for field_name in ("session_id", "target_scope", "target_label"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"E5 comparison presentation requires {field_name}.")
            object.__setattr__(self, field_name, value)
        scope = self.target_scope.lower()
        if scope not in {"scene", "object"}:
            raise ValueError("E5 target_scope must be scene or object.")
        object.__setattr__(self, "target_scope", scope)
        object.__setattr__(self, "target_object_id", str(self.target_object_id).strip())
        if scope == "object" and not self.target_object_id:
            raise ValueError("Object-scope E5 presentation requires target_object_id.")

        if not isinstance(self.reference, ReconstructionReferenceDescriptor):
            raise TypeError("reference must be ReconstructionReferenceDescriptor.")
        object.__setattr__(self, "report_source", str(self.report_source).strip())
        object.__setattr__(self, "report_attached", bool(self.report_attached))
        object.__setattr__(self, "report_reference_match", bool(self.report_reference_match))
        object.__setattr__(self, "comparison_available", bool(self.comparison_available))
        object.__setattr__(self, "comparison_metric_available", bool(self.comparison_metric_available))

        if self.comparison_available and not (
            self.report_attached and self.report_reference_match
        ):
            raise ValueError("Available E5 comparison requires an attached matching report.")

        candidate_metric = str(self.candidate_metric_geometry_status).strip().lower()
        if candidate_metric:
            candidate_metric = _metric_status(
                candidate_metric, "candidate_metric_geometry_status"
            )
        object.__setattr__(self, "candidate_metric_geometry_status", candidate_metric)

        registration_status = str(self.registration_status).strip().lower()
        if registration_status and registration_status not in _ALLOWED_REGISTRATION_STATUS:
            raise ValueError(f"Unsupported E5 registration status: {registration_status!r}")
        object.__setattr__(self, "registration_status", registration_status)
        object.__setattr__(self, "registration_method", str(self.registration_method).strip().lower())
        confidence = 0.0
        if registration_status:
            confidence = _probability(self.registration_confidence, "registration_confidence")
        object.__setattr__(self, "registration_confidence", confidence)
        object.__setattr__(self, "registration_reason", str(self.registration_reason).strip())

        for field_name in (
            "registration_rms_reference_units",
            "registration_rms_m",
            "registration_normalized_rms_fraction",
            "position_error_reference_units",
            "position_error_m",
            "orientation_error_degrees",
            "extent_relative_error",
            "uncertainty_reference_units",
            "uncertainty_m",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _nonnegative_float(value, field_name))
        quality = self.geometry_quality
        if quality is not None:
            object.__setattr__(self, "geometry_quality", _probability(quality, "geometry_quality"))

        object.__setattr__(
            self, "matched_reference_feature_id", str(self.matched_reference_feature_id).strip()
        )
        object.__setattr__(self, "target_result_available", bool(self.target_result_available))
        if self.target_result_available and scope != "object":
            raise ValueError("Target feature result is only valid for object-scope E5 presentation.")

        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E5 cannot claim certified metrology.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def target_text(self) -> str:
        return self.target_label

    @property
    def state_text(self) -> str:
        if not self.report_attached:
            return "Reference attached • comparison report required"
        if not self.report_reference_match:
            return "Reference/report mismatch • comparison unavailable"
        if not self.comparison_available:
            return "Matching report attached • comparison unavailable"
        if self.target_scope == "object" and not self.target_result_available:
            return "Matching scene comparison • selected object not evaluated"
        return "Comparison evidence available"

    @property
    def registration_text(self) -> str:
        if not self.comparison_available:
            return "—"
        status = _humanize(self.registration_status)
        method = _humanize(self.registration_method) if self.registration_method else "Method unavailable"
        text = f"{status} • {self.registration_confidence * 100.0:.0f}% • {method}"
        residual = self._registration_residual_text()
        if residual:
            text += f" • RMS {residual}"
        if self.registration_reason:
            text += f" • {_humanize(self.registration_reason)}"
        return text

    @property
    def deviations_text(self) -> str:
        if not self.comparison_available:
            return "—"
        if self.target_scope == "scene":
            residual = self._registration_residual_text()
            normalized = ""
            if self.registration_normalized_rms_fraction is not None:
                normalized = f" • normalized {self.registration_normalized_rms_fraction * 100.0:.2f}%"
            return f"Registration RMS {residual or 'unavailable'}{normalized}"
        if not self.target_result_available:
            return "No exact selected-object deviation result"
        values: list[str] = []
        position = self._position_error_text()
        if position:
            values.append(f"Position {position}")
        if self.orientation_error_degrees is not None:
            values.append(f"Orientation {self.orientation_error_degrees:.2f}°")
        if self.extent_relative_error is not None:
            values.append(f"Extent {self.extent_relative_error * 100.0:.2f}%")
        return " • ".join(values) if values else "No quantitative deviation in source report"

    @property
    def evidence_text(self) -> str:
        if not self.comparison_available:
            return "—"
        values: list[str] = []
        if self.target_scope == "object" and self.matched_reference_feature_id:
            values.append(f"Reference feature {self.matched_reference_feature_id}")
        if self.geometry_quality is not None:
            values.append(f"geometry quality {self.geometry_quality * 100.0:.1f}%")
        uncertainty = self._uncertainty_text()
        if uncertainty:
            values.append(f"uncertainty {uncertainty}")
        return " • ".join(values) if values else "Source report context only"

    @property
    def scale_text(self) -> str:
        if not self.report_attached or not self.report_reference_match:
            return self.reference.scale_text
        if self.comparison_metric_available:
            return (
                f"Metric-compatible comparison • candidate {_metric_status_label(self.candidate_metric_geometry_status)} • "
                f"reference {_metric_status_label(self.reference.metric_geometry_status)}"
            )
        candidate = (
            _metric_status_label(self.candidate_metric_geometry_status)
            if self.candidate_metric_geometry_status
            else "Unknown"
        )
        return (
            f"Relative-only comparison • candidate {candidate} • "
            f"reference {_metric_status_label(self.reference.metric_geometry_status)}"
        )

    @property
    def authority_text(self) -> str:
        return (
            "Reference identity + source 8.7C comparison evidence • E5 does not register/apply tolerances • "
            "not certified metrology"
        )

    @property
    def provenance_tooltip(self) -> str:
        target = f" • object_id={self.target_object_id}" if self.target_object_id else ""
        report = f" • report_source={self.report_source}" if self.report_source else ""
        return (
            f"session_id={self.session_id}{target} • reference_id={self.reference.reference_id} • "
            f"reference_source={self.reference.provenance}{report}"
        )

    def _registration_residual_text(self) -> str:
        if self.comparison_metric_available and self.registration_rms_m is not None:
            return _metres_as_engineering_length(self.registration_rms_m)
        if self.registration_rms_reference_units is not None:
            return f"{self.registration_rms_reference_units:.5g} reference units"
        return ""

    def _position_error_text(self) -> str:
        if self.comparison_metric_available and self.position_error_m is not None:
            return _metres_as_engineering_length(self.position_error_m)
        if self.position_error_reference_units is not None:
            return f"{self.position_error_reference_units:.5g} reference units"
        return ""

    def _uncertainty_text(self) -> str:
        if self.comparison_metric_available and self.uncertainty_m is not None:
            return _metres_as_engineering_length(self.uncertainty_m)
        if self.uncertainty_reference_units is not None:
            return f"{self.uncertainty_reference_units:.5g} reference units"
        return ""


def reference_descriptor_from_payload(
    reference: object,
    *,
    provenance: str = "phase8_7c.reference_assembly",
) -> ReconstructionReferenceDescriptor:
    """Validate/reference metadata from a ReferenceAssembly or serialized mapping."""

    payload = _mapping_payload(reference, "reference")
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("E5 requires ReferenceAssembly schema_version=1.")
    if bool(payload.get("certified_metrology", False)):
        raise ValueError("Phase 8.8E5 cannot attach a certified metrology reference claim.")

    reference_id = _nonempty(payload.get("reference_id"), "reference_id")
    source = _nonempty(payload.get("source"), "source").lower()
    scene_profile = _nonempty(payload.get("scene_profile"), "scene_profile")
    frame_id = _nonempty(payload.get("reference_frame_id"), "reference_frame_id")
    metric = _metric_status(payload.get("metric_geometry_status"), "metric_geometry_status")
    features = payload.get("features")
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("E5 reference requires a non-empty features list.")
    anchors = 0
    feature_ids: set[str] = set()
    for index, item in enumerate(features):
        feature = _mapping(item, f"features[{index}]")
        feature_id = _nonempty(feature.get("feature_id"), f"features[{index}].feature_id")
        if feature_id in feature_ids:
            raise ValueError("E5 reference feature IDs must be unique.")
        feature_ids.add(feature_id)
        if bool(feature.get("registration_anchor", False)):
            anchors += 1
    if anchors < 1:
        raise ValueError("E5 reference requires at least one registration anchor.")

    return ReconstructionReferenceDescriptor(
        reference_id=reference_id,
        source=source,
        scene_profile=scene_profile,
        reference_frame_id=frame_id,
        metric_geometry_status=metric,
        unit_to_metre=payload.get("unit_to_metre"),
        relative_scale_uncertainty=payload.get("relative_scale_uncertainty"),
        feature_count=len(features),
        registration_anchor_count=anchors,
        provenance=_nonempty(provenance, "provenance"),
        certified_metrology=False,
    )


def load_reference_descriptor_json(path: str | Path) -> ReconstructionReferenceDescriptor:
    """Load only ReferenceAssembly metadata needed by E5 from its JSON artifact."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Reference JSON does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reference JSON: {exc}") from exc
    return reference_descriptor_from_payload(payload, provenance=str(resolved))


def build_reconstruction_reference_comparison_presentation(
    selection: ReconstructionInspectionSelection,
    reference: ReconstructionReferenceDescriptor | object,
    report: object | None = None,
    *,
    report_source: str = "phase8_7c.inspection_report",
) -> ReconstructionReferenceComparisonPresentation:
    """Pair the active E1 target with an exact reference and optional 8.7C report."""

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_scene:
        raise ValueError("E5 reference comparison requires an active E1 scene/object target.")
    descriptor = (
        reference
        if isinstance(reference, ReconstructionReferenceDescriptor)
        else reference_descriptor_from_payload(reference)
    )
    session_id = str(selection.session_id or selection.scene_id).strip()
    target_scope = "object" if selection.has_object else "scene"
    target_label = selection.object_text if selection.has_object else selection.scene_text

    if report is None:
        return ReconstructionReferenceComparisonPresentation(
            session_id=session_id,
            target_scope=target_scope,
            target_object_id=selection.object_id,
            target_label=target_label,
            reference=descriptor,
            report_source="",
            report_attached=False,
            report_reference_match=False,
            comparison_available=False,
            comparison_metric_available=False,
            candidate_metric_geometry_status=(
                "" if selection.metric_geometry_status == "unknown" else selection.metric_geometry_status
            ),
            registration_status="",
            registration_method="",
            registration_confidence=0.0,
            registration_reason="",
            registration_rms_reference_units=None,
            registration_rms_m=None,
            registration_normalized_rms_fraction=None,
            matched_reference_feature_id="",
            target_result_available=False,
            position_error_reference_units=None,
            position_error_m=None,
            orientation_error_degrees=None,
            extent_relative_error=None,
            geometry_quality=None,
            uncertainty_reference_units=None,
            uncertainty_m=None,
            certified_metrology=False,
        )

    payload = _mapping_payload(report, "inspection_report")
    if int(payload.get("schema_version", 0)) != 1 or payload.get("inspection_qc_enabled") is not True:
        raise ValueError("E5 requires an enabled Phase 8.7C inspection report schema_version=1.")
    if bool(payload.get("certified_metrology", False)):
        raise ValueError("Phase 8.8E5 cannot present a certified metrology report claim.")
    report_session = _nonempty(payload.get("session_id"), "inspection_report.session_id")
    if report_session != session_id:
        raise ValueError("E5 inspection report session does not match the active reconstruction target.")
    source = _nonempty(report_source, "report_source")
    report_reference_id = _nonempty(payload.get("reference_id"), "inspection_report.reference_id")
    reference_match = report_reference_id == descriptor.reference_id

    candidate_metric = _metric_status(
        payload.get("candidate_metric_geometry_status"),
        "candidate_metric_geometry_status",
    )
    report_reference_metric = _metric_status(
        payload.get("reference_metric_geometry_status"),
        "reference_metric_geometry_status",
    )
    if reference_match and report_reference_metric != descriptor.metric_geometry_status:
        raise ValueError("E5 matching report/reference disagree on reference metric status.")

    if not reference_match:
        return ReconstructionReferenceComparisonPresentation(
            session_id=session_id,
            target_scope=target_scope,
            target_object_id=selection.object_id,
            target_label=target_label,
            reference=descriptor,
            report_source=source,
            report_attached=True,
            report_reference_match=False,
            comparison_available=False,
            comparison_metric_available=False,
            candidate_metric_geometry_status=candidate_metric,
            registration_status="",
            registration_method="",
            registration_confidence=0.0,
            registration_reason="",
            registration_rms_reference_units=None,
            registration_rms_m=None,
            registration_normalized_rms_fraction=None,
            matched_reference_feature_id="",
            target_result_available=False,
            position_error_reference_units=None,
            position_error_m=None,
            orientation_error_degrees=None,
            extent_relative_error=None,
            geometry_quality=None,
            uncertainty_reference_units=None,
            uncertainty_m=None,
            certified_metrology=False,
        )

    registration = _mapping(payload.get("registration"), "inspection_report.registration")
    registration_status = _nonempty(registration.get("status"), "registration.status").lower()
    if registration_status not in _ALLOWED_REGISTRATION_STATUS:
        raise ValueError("E5 report has unsupported registration status.")
    registration_method_raw = registration.get("method")
    registration_method = "" if registration_method_raw is None else str(registration_method_raw).strip().lower()
    comparison_available = registration_status == "success"
    comparison_metric = bool(payload.get("comparison_metric_available", False)) if comparison_available else False

    result: Mapping[str, Any] | None = None
    if selection.has_object:
        seen_ids: set[str] = set()
        for index, item in enumerate(payload.get("feature_results", ())):
            feature_result = _mapping(item, f"feature_results[{index}]")
            candidate_id = str(feature_result.get("candidate_object_id", "")).strip()
            if not candidate_id:
                continue
            if candidate_id in seen_ids:
                raise ValueError("E5 report contains duplicate candidate object results.")
            seen_ids.add(candidate_id)
            if candidate_id == selection.object_id:
                result = feature_result

    return ReconstructionReferenceComparisonPresentation(
        session_id=session_id,
        target_scope=target_scope,
        target_object_id=selection.object_id,
        target_label=target_label,
        reference=descriptor,
        report_source=source,
        report_attached=True,
        report_reference_match=True,
        comparison_available=comparison_available,
        comparison_metric_available=comparison_metric,
        candidate_metric_geometry_status=candidate_metric,
        registration_status=registration_status,
        registration_method=registration_method,
        registration_confidence=registration.get("confidence", 0.0),
        registration_reason=str(registration.get("reason", "")).strip(),
        registration_rms_reference_units=registration.get("rms_residual_reference_units"),
        registration_rms_m=registration.get("rms_residual_m"),
        registration_normalized_rms_fraction=registration.get("normalized_rms_fraction"),
        matched_reference_feature_id=(
            str(result.get("reference_feature_id", "")).strip() if result is not None else ""
        ),
        target_result_available=result is not None,
        position_error_reference_units=(
            result.get("position_error_reference_units") if result is not None else None
        ),
        position_error_m=result.get("position_error_m") if result is not None else None,
        orientation_error_degrees=(
            result.get("orientation_error_degrees") if result is not None else None
        ),
        extent_relative_error=result.get("extent_relative_error") if result is not None else None,
        geometry_quality=result.get("geometry_quality") if result is not None else None,
        uncertainty_reference_units=(
            result.get("uncertainty_reference_units") if result is not None else None
        ),
        uncertainty_m=result.get("uncertainty_m") if result is not None else None,
        certified_metrology=False,
    )


def empty_reference_comparison_text(
    selection: ReconstructionInspectionSelection,
) -> tuple[str, str, str, str, str, str, str, str]:
    """Return explicit E5 empty-state rows without manufacturing comparison evidence."""

    target = selection.object_text if selection.has_object else selection.scene_text
    if not selection.has_scene:
        target = "No reconstruction target"
        state = "Waiting for reconstruction target"
    else:
        state = "No reference attached"
    return (
        target,
        "No reference attached",
        "—",
        "—",
        "—",
        "—",
        state,
        "Reference workflow only • E5 does not register/apply tolerances",
    )


def _mapping_payload(value: object, field_name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError(f"{field_name} must be a mapping or expose as_dict().")


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _nonempty(value: object, field_name: str) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _metric_status(value: object, field_name: str) -> str:
    text = str(getattr(value, "value", value)).strip().lower()
    # The 8.7C contract calls the validated status calibrated_metric while the
    # D5/E1 presentation contract historically used validated_metric.  E5
    # accepts both display vocabularies without changing source authority.
    if text == "calibrated_metric":
        text = "validated_metric"
    if text == "estimated_metric":
        text = "provisional_metric"
    if text not in _ALLOWED_METRIC_STATUS:
        raise ValueError(f"Unsupported {field_name}: {text!r}")
    return text


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if result < 1:
        raise ValueError(f"{field_name} must be a positive integer.")
    return result


def _positive_float(value: object, field_name: str) -> float:
    result = _nonnegative_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return result


def _nonnegative_float(value: object, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite and non-negative.") from exc
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return result


def _probability(value: object, field_name: str) -> float:
    result = _nonnegative_float(value, field_name)
    if result > 1.0:
        raise ValueError(f"{field_name} must be in [0,1].")
    return result


def _humanize(value: str) -> str:
    return str(value).replace("_", " ").strip().capitalize()


def _metric_status_label(value: str) -> str:
    labels = {
        "arbitrary": "Arbitrary gauge",
        "provisional_metric": "Provisional metric",
        "validated_metric": "Validated metric",
    }
    return labels.get(str(value).strip().lower(), "Unknown")


def _metres_as_engineering_length(value_m: float) -> str:
    value = float(value_m)
    if value < 0.01:
        return f"{value * 1000.0:.3f} mm"
    if value < 1.0:
        return f"{value * 100.0:.3f} cm"
    return f"{value:.5g} m"


__all__ = [
    "ReconstructionReferenceDescriptor",
    "ReconstructionReferenceComparisonPresentation",
    "reference_descriptor_from_payload",
    "load_reference_descriptor_json",
    "build_reconstruction_reference_comparison_presentation",
    "empty_reference_comparison_text",
]
