"""Presentation-only inspection/QC bridge for Phase 8.8E4.

E4 exposes authoritative decisions that already exist in a Phase 8.7C
inspection report. It does not run registration, apply tolerances, compare
reference geometry, recompute deviations, infer PASS/FAIL from measurements or
semantic score, or certify metrology.

The active E1 scene/object selection is used only to bind an exact
source-authored result by session and candidate object identity. Reference
loading, selection, quantitative comparison/deviation, and comparison
visualization remain the Phase 8.8E5 reference workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    ReconstructionInspectionSelectionScope,
)

_ALLOWED_DECISIONS = {"pass", "warn", "reject", "insufficient_evidence"}
_ALLOWED_REGISTRATION_STATUS = {
    "success",
    "insufficient_evidence",
    "degenerate",
    "residual_too_high",
}
_ALLOWED_METRIC_STATUS = {"arbitrary", "provisional_metric", "validated_metric"}


@dataclass(frozen=True, slots=True)
class ReconstructionInspectionQcPresentation:
    """Bounded read-only view of an existing Phase 8.7C inspection report."""

    session_id: str
    reference_id: str
    source: str
    target_scope: str
    target_object_id: str
    target_label: str
    overall_decision: str
    target_result_available: bool
    target_decision: str
    findings: tuple[str, ...]
    registration_status: str
    registration_confidence: float
    registration_reason: str
    comparison_metric_available: bool
    candidate_metric_geometry_status: str
    reference_metric_geometry_status: str
    reference_feature_count: int
    matched_feature_count: int
    missing_required_count: int
    extra_candidate_count: int
    pass_count: int
    warn_count: int
    reject_count: int
    insufficient_evidence_count: int
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        for field_name in ("session_id", "reference_id", "source", "target_scope", "target_label"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"E4 QC presentation requires {field_name}.")
            object.__setattr__(self, field_name, value)

        scope = self.target_scope.lower()
        if scope not in {"scene", "object"}:
            raise ValueError("E4 target_scope must be scene or object.")
        object.__setattr__(self, "target_scope", scope)
        object.__setattr__(self, "target_object_id", str(self.target_object_id).strip())

        overall = _decision(self.overall_decision, "overall_decision")
        object.__setattr__(self, "overall_decision", overall)
        target_decision = str(self.target_decision).strip().lower()
        if target_decision:
            target_decision = _decision(target_decision, "target_decision")
        object.__setattr__(self, "target_decision", target_decision)
        object.__setattr__(self, "target_result_available", bool(self.target_result_available))
        if self.target_result_available and not target_decision:
            raise ValueError("Available E4 target result requires a source-authored decision.")
        if not self.target_result_available and target_decision:
            raise ValueError("Unavailable E4 target result cannot carry a target decision.")
        object.__setattr__(self, "findings", tuple(str(v).strip().lower() for v in self.findings if str(v).strip()))

        registration = str(self.registration_status).strip().lower()
        if registration not in _ALLOWED_REGISTRATION_STATUS:
            raise ValueError(f"Unsupported registration status: {registration!r}")
        object.__setattr__(self, "registration_status", registration)
        object.__setattr__(
            self,
            "registration_confidence",
            _probability(self.registration_confidence, "registration_confidence"),
        )
        object.__setattr__(self, "registration_reason", str(self.registration_reason).strip())

        for field_name in ("candidate_metric_geometry_status", "reference_metric_geometry_status"):
            value = _metric_status(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "comparison_metric_available", bool(self.comparison_metric_available))

        for field_name in (
            "reference_feature_count",
            "matched_feature_count",
            "missing_required_count",
            "extra_candidate_count",
            "pass_count",
            "warn_count",
            "reject_count",
            "insufficient_evidence_count",
        ):
            object.__setattr__(self, field_name, _nonnegative_int(getattr(self, field_name), field_name))

        if bool(self.certified_metrology):
            raise ValueError("Phase 8.8E4 cannot claim certified metrology.")
        object.__setattr__(self, "certified_metrology", False)

    @property
    def decision_text(self) -> str:
        if self.target_scope == "scene":
            return f"{_decision_label(self.overall_decision)} • Scene"
        if self.target_result_available:
            return f"{_decision_label(self.target_decision)} • Object"
        return "NOT EVALUATED • Object"

    @property
    def findings_text(self) -> str:
        if self.target_scope == "scene":
            return "See source report statistics"
        if not self.target_result_available:
            return "No selected-object result in attached report"
        if not self.findings:
            return "No source-authored findings"
        return " • ".join(_humanize(value) for value in self.findings)

    @property
    def coverage_text(self) -> str:
        return (
            f"{self.matched_feature_count}/{self.reference_feature_count} matched • "
            f"{self.missing_required_count} missing • {self.extra_candidate_count} extra"
        )

    @property
    def decisions_text(self) -> str:
        return (
            f"{self.pass_count} pass • {self.warn_count} warn • "
            f"{self.reject_count} reject • {self.insufficient_evidence_count} insufficient"
        )

    @property
    def registration_text(self) -> str:
        text = f"{_humanize(self.registration_status)} • {self.registration_confidence * 100.0:.0f}%"
        if self.registration_reason:
            text += f" • {_humanize(self.registration_reason)}"
        return text

    @property
    def comparison_text(self) -> str:
        kind = "Metric comparison" if self.comparison_metric_available else "Relative-only comparison"
        return (
            f"{kind} • candidate {_metric_status_label(self.candidate_metric_geometry_status)} • "
            f"reference {_metric_status_label(self.reference_metric_geometry_status)}"
        )

    @property
    def authority_text(self) -> str:
        return (
            "Source inspection/QC decision • existing 8.7C report • not certified metrology • "
            "reference workflow deferred to E5"
        )

    @property
    def provenance_tooltip(self) -> str:
        target = f" • object_id={self.target_object_id}" if self.target_object_id else ""
        return (
            f"session_id={self.session_id}{target} • reference_id={self.reference_id} • "
            f"source={self.source}"
        )


def build_reconstruction_inspection_qc_presentation(
    selection: ReconstructionInspectionSelection,
    report: object,
    *,
    source: str = "phase8_7c.inspection_report",
) -> ReconstructionInspectionQcPresentation:
    """Present source inspection/QC decisions without rerunning the QC engine."""

    if not isinstance(selection, ReconstructionInspectionSelection):
        raise TypeError("selection must be ReconstructionInspectionSelection.")
    if not selection.has_scene:
        raise ValueError("E4 inspection/QC evidence requires an active E1 scene or object target.")
    source = str(source).strip()
    if not source:
        raise ValueError("Inspection/QC source must be non-empty.")

    payload = _report_mapping(report)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("E4 requires Phase 8.7C inspection report schema_version=1.")
    if payload.get("inspection_qc_enabled") is not True:
        raise ValueError("E4 requires an inspection_qc_enabled Phase 8.7C report.")
    if bool(payload.get("certified_metrology", False)):
        raise ValueError("Phase 8.8E4 cannot present a certified metrology claim.")

    session_id = str(payload.get("session_id", "")).strip()
    active_session = str(selection.session_id or selection.scene_id).strip()
    if not session_id or not active_session or session_id != active_session:
        raise ValueError("E4 inspection report session does not match the active E1 reconstruction target.")
    reference_id = str(payload.get("reference_id", "")).strip()
    if not reference_id:
        raise ValueError("E4 inspection report requires reference_id provenance.")

    overall_decision = _decision(payload.get("decision"), "decision")
    registration = _mapping(payload.get("registration"), "registration")
    registration_status = str(registration.get("status", "")).strip().lower()
    if registration_status not in _ALLOWED_REGISTRATION_STATUS:
        raise ValueError("E4 inspection report has unsupported registration status.")
    registration_confidence = _probability(registration.get("confidence", 0.0), "registration.confidence")
    registration_reason = str(registration.get("reason", "")).strip()

    statistics = _mapping(payload.get("statistics"), "statistics")
    counts = {
        name: _nonnegative_int(statistics.get(name), f"statistics.{name}")
        for name in (
            "reference_feature_count",
            "matched_feature_count",
            "missing_required_count",
            "extra_candidate_count",
            "pass_count",
            "warn_count",
            "reject_count",
            "insufficient_evidence_count",
        )
    }

    feature_results = payload.get("feature_results", ())
    if not isinstance(feature_results, (list, tuple)):
        raise ValueError("E4 inspection report feature_results must be a sequence.")

    target_scope = selection.scope.value
    target_object_id = selection.object_id if selection.has_object else ""
    target_label = selection.object_text if selection.has_object else selection.scene_text
    target_result_available = False
    target_decision = ""
    findings: tuple[str, ...] = ()

    if selection.scope is ReconstructionInspectionSelectionScope.OBJECT:
        matches: list[Mapping[str, Any]] = []
        for value in feature_results:
            item = _mapping(value, "feature_result")
            if str(item.get("candidate_object_id", "")).strip() == selection.object_id:
                matches.append(item)
        if len(matches) > 1:
            raise ValueError("E4 inspection report contains duplicate results for the active candidate object ID.")
        if matches:
            item = matches[0]
            target_result_available = True
            target_decision = _decision(item.get("decision"), "feature_result.decision")
            raw_findings = item.get("findings", ())
            if not isinstance(raw_findings, (list, tuple)):
                raise ValueError("E4 feature findings must be a sequence.")
            findings = tuple(str(value).strip().lower() for value in raw_findings if str(value).strip())
            if not findings:
                raise ValueError("E4 matched object result requires source-authored findings.")

    return ReconstructionInspectionQcPresentation(
        session_id=session_id,
        reference_id=reference_id,
        source=source,
        target_scope=target_scope,
        target_object_id=target_object_id,
        target_label=target_label,
        overall_decision=overall_decision,
        target_result_available=target_result_available,
        target_decision=target_decision,
        findings=findings,
        registration_status=registration_status,
        registration_confidence=registration_confidence,
        registration_reason=registration_reason,
        comparison_metric_available=bool(payload.get("comparison_metric_available", False)),
        candidate_metric_geometry_status=_metric_status(
            payload.get("candidate_metric_geometry_status"), "candidate_metric_geometry_status"
        ),
        reference_metric_geometry_status=_metric_status(
            payload.get("reference_metric_geometry_status"), "reference_metric_geometry_status"
        ),
        reference_feature_count=counts["reference_feature_count"],
        matched_feature_count=counts["matched_feature_count"],
        missing_required_count=counts["missing_required_count"],
        extra_candidate_count=counts["extra_candidate_count"],
        pass_count=counts["pass_count"],
        warn_count=counts["warn_count"],
        reject_count=counts["reject_count"],
        insufficient_evidence_count=counts["insufficient_evidence_count"],
        certified_metrology=False,
    )


def empty_reconstruction_inspection_qc_text(
    selection: ReconstructionInspectionSelection,
) -> tuple[str, str, str, str, str, str, str]:
    """Stable E4 empty state; never manufactures a QC decision."""

    if not selection.has_scene:
        return (
            "No target",
            "Not evaluated",
            "Waiting for reconstruction target",
            "—",
            "No inspection/QC report attached",
            "—",
            "Existing QC only • E4 does not run inspection • reference workflow deferred to E5",
        )
    target = selection.object_text if selection.has_object else selection.scene_text
    return (
        target,
        "Not evaluated",
        "No attached inspection/QC result",
        "—",
        "No inspection/QC report attached",
        "—",
        "Existing QC only • E4 does not run inspection • reference workflow deferred to E5",
    )


def _report_mapping(report: object) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    as_dict = getattr(report, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("E4 inspection report must be a mapping or expose as_dict().")


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"E4 inspection report {field_name} must be a mapping.")
    return value


def _decision(value: object, field_name: str) -> str:
    text = str(value).strip().lower()
    if text not in _ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported {field_name}: {text!r}")
    return text


def _metric_status(value: object, field_name: str) -> str:
    text = str(value).strip().lower()
    if text not in _ALLOWED_METRIC_STATUS:
        raise ValueError(f"Unsupported {field_name}: {text!r}")
    return text


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return parsed


def _probability(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite in [0, 1].") from exc
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be finite in [0, 1].")
    return parsed


def _decision_label(value: str) -> str:
    return {
        "pass": "PASS",
        "warn": "WARN",
        "reject": "REJECT",
        "insufficient_evidence": "INSUFFICIENT EVIDENCE",
    }[value]


def _metric_status_label(value: str) -> str:
    return {
        "arbitrary": "Arbitrary gauge",
        "provisional_metric": "Provisional metric",
        "validated_metric": "Validated metric",
    }[value]


def _humanize(value: str) -> str:
    return str(value).strip().replace("_", " ").replace("-", " ").capitalize()
