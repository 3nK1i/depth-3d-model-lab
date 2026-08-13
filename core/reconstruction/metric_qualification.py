"""Phase 9.1 metric-scale evidence qualification built on Phase 8 scale contracts.

This module is intentionally additive.  ``ScaleReferenceProfile`` remains the
low-level scale authority used by reconstruction geometry.  Phase 9.1E adds a
fail-closed evidence layer that decides whether a candidate source may produce
that authority, preserves provenance/uncertainty/rejection state, and keeps
measurement-system benchmark evidence supporting-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from math import isfinite
from typing import Any, Mapping

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.scale_reference import (
    ScaleReferenceProfile,
    external_metric_prior_scale_reference,
    known_dimension_scale_reference,
    reference_target_scale_reference,
)
from core.reconstruction.validation import require_iso8601_utc


METRIC_QUALIFICATION_SCHEMA_VERSION = 1


class MetricEvidenceKind(str, Enum):
    """Supported and explicitly reserved metric-evidence source classes."""

    KNOWN_SIZE_REFERENCE = "known_size_reference"
    KNOWN_DISTANCE_REFERENCE = "known_distance_reference"
    REFERENCE_TARGET = "reference_target"
    QUALIFIED_METRIC_DEPTH = "qualified_metric_depth"
    EXTERNAL_METRIC_PRIOR = "external_metric_prior"
    KNOWN_SIZE_BENCHMARK = "known_size_benchmark"
    FUTURE_STEREO = "future_stereo"
    FUTURE_RGBD = "future_rgbd"


class MetricEvidenceValidationState(str, Enum):
    """Validation state of the evidence source itself."""

    UNVALIDATED = "unvalidated"
    VALIDATED = "validated"
    REJECTED = "rejected"


class MetricQualificationState(str, Enum):
    """Result of applying one evidence record to metric-scale authority."""

    PROVISIONAL = "provisional"
    QUALIFIED = "qualified"
    SUPPORTING_ONLY = "supporting_only"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"


_SCALE_DIMENSION_KINDS = {
    MetricEvidenceKind.KNOWN_SIZE_REFERENCE,
    MetricEvidenceKind.KNOWN_DISTANCE_REFERENCE,
    MetricEvidenceKind.REFERENCE_TARGET,
}
_DIRECT_SCALE_KINDS = {
    MetricEvidenceKind.QUALIFIED_METRIC_DEPTH,
    MetricEvidenceKind.EXTERNAL_METRIC_PRIOR,
}
_FUTURE_KINDS = {
    MetricEvidenceKind.FUTURE_STEREO,
    MetricEvidenceKind.FUTURE_RGBD,
}


@dataclass(frozen=True, slots=True)
class MetricQualificationEvidence:
    """Traceable candidate evidence that may establish reconstruction scale.

    Structural validity is different from qualification.  Invalid numbers or
    contradictory fields are rejected immediately; a structurally valid record
    may still be unvalidated, rejected, supporting-only, or unsupported.
    """

    evidence_id: str
    kind: MetricEvidenceKind
    validation_state: MetricEvidenceValidationState
    provenance: str
    reconstruction_frame_id: str = ""
    reference_id: str = ""
    validation_record_id: str = ""
    reference_length_m: float | None = None
    measured_length_reconstruction_units: float | None = None
    reference_relative_uncertainty: float | None = None
    measurement_relative_uncertainty: float | None = None
    unit_to_metre: float | None = None
    relative_scale_uncertainty: float | None = None
    rejection_reasons: tuple[str, ...] = ()
    created_at_utc: str = ""
    notes: str = ""
    source_report_sha256: str = ""
    schema_version: int = METRIC_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            require_nonempty(self.evidence_id, field_name="evidence_id"),
        )
        if not isinstance(self.kind, MetricEvidenceKind):
            raise TypeError("kind must be MetricEvidenceKind.")
        if not isinstance(self.validation_state, MetricEvidenceValidationState):
            raise TypeError("validation_state must be MetricEvidenceValidationState.")
        if int(self.schema_version) != METRIC_QUALIFICATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported metric-qualification schema version.")

        provenance = require_nonempty(self.provenance, field_name="provenance")
        frame_id = str(self.reconstruction_frame_id).strip()
        reference_id = str(self.reference_id).strip()
        validation_record_id = str(self.validation_record_id).strip()
        notes = str(self.notes).strip()
        created = self.created_at_utc or datetime.now(timezone.utc).isoformat()
        created = require_iso8601_utc(created, field_name="created_at_utc")
        reasons = tuple(
            require_nonempty(value, field_name="rejection_reason")
            for value in self.rejection_reasons
        )
        digest = str(self.source_report_sha256).strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ReconstructionValidationError(
                "source_report_sha256 must be a 64-character hexadecimal digest."
            )

        ref_length = _optional_positive(self.reference_length_m, "reference_length_m")
        measured = _optional_positive(
            self.measured_length_reconstruction_units,
            "measured_length_reconstruction_units",
        )
        ref_uncertainty = _optional_relative_uncertainty(
            self.reference_relative_uncertainty,
            "reference_relative_uncertainty",
        )
        measurement_uncertainty = _optional_relative_uncertainty(
            self.measurement_relative_uncertainty,
            "measurement_relative_uncertainty",
        )
        unit_to_metre = _optional_positive(self.unit_to_metre, "unit_to_metre")
        scale_uncertainty = _optional_relative_uncertainty(
            self.relative_scale_uncertainty,
            "relative_scale_uncertainty",
        )

        if self.validation_state is MetricEvidenceValidationState.REJECTED:
            if not reasons:
                raise ReconstructionValidationError(
                    "Rejected metric evidence requires at least one rejection reason."
                )
        elif reasons:
            raise ReconstructionValidationError(
                "rejection_reasons are valid only when validation_state=REJECTED."
            )

        if self.kind in _SCALE_DIMENSION_KINDS:
            if unit_to_metre is not None or scale_uncertainty is not None:
                raise ReconstructionValidationError(
                    "Dimension-based metric evidence derives scale; do not also declare unit_to_metre."
                )
            if self.validation_state is not MetricEvidenceValidationState.REJECTED:
                if not frame_id:
                    raise ReconstructionValidationError(
                        "Dimension-based metric evidence requires reconstruction_frame_id."
                    )
                if not reference_id:
                    raise ReconstructionValidationError(
                        "Dimension-based metric evidence requires reference_id."
                    )
                if None in (ref_length, measured, ref_uncertainty, measurement_uncertainty):
                    raise ReconstructionValidationError(
                        "Dimension-based metric evidence requires explicit physical length, "
                        "reconstruction length, reference uncertainty, and measurement uncertainty."
                    )
        elif self.kind in _DIRECT_SCALE_KINDS:
            if any(value is not None for value in (ref_length, measured, ref_uncertainty, measurement_uncertainty)):
                raise ReconstructionValidationError(
                    "Direct metric-scale evidence must not also declare a dimension pair."
                )
            if self.validation_state is not MetricEvidenceValidationState.REJECTED:
                if not frame_id:
                    raise ReconstructionValidationError(
                        "Direct metric-scale evidence requires reconstruction_frame_id."
                    )
                if not reference_id:
                    raise ReconstructionValidationError(
                        "Direct metric-scale evidence requires reference_id."
                    )
                if unit_to_metre is None or scale_uncertainty is None:
                    raise ReconstructionValidationError(
                        "Direct metric-scale evidence requires explicit unit_to_metre and relative_scale_uncertainty."
                    )
        elif self.kind is MetricEvidenceKind.KNOWN_SIZE_BENCHMARK:
            if not digest:
                raise ReconstructionValidationError(
                    "Known-size benchmark evidence requires source_report_sha256."
                )
            if any(
                value is not None
                for value in (
                    ref_length,
                    measured,
                    ref_uncertainty,
                    measurement_uncertainty,
                    unit_to_metre,
                    scale_uncertainty,
                )
            ):
                raise ReconstructionValidationError(
                    "Known-size benchmark evidence is supporting-only and must not declare scene scale."
                )
        elif self.kind in _FUTURE_KINDS:
            if any(
                value is not None
                for value in (
                    ref_length,
                    measured,
                    ref_uncertainty,
                    measurement_uncertainty,
                    unit_to_metre,
                    scale_uncertainty,
                )
            ):
                raise ReconstructionValidationError(
                    "Future stereo/RGB-D evidence is reserved architecture and cannot declare scale in Phase 9.1E."
                )

        if (
            self.validation_state is MetricEvidenceValidationState.VALIDATED
            and self.kind in ({MetricEvidenceKind.REFERENCE_TARGET} | _DIRECT_SCALE_KINDS)
            and not validation_record_id
        ):
            raise ReconstructionValidationError(
                "Validated reference-target/direct metric evidence requires validation_record_id."
            )

        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "validation_record_id", validation_record_id)
        object.__setattr__(self, "reference_length_m", ref_length)
        object.__setattr__(self, "measured_length_reconstruction_units", measured)
        object.__setattr__(self, "reference_relative_uncertainty", ref_uncertainty)
        object.__setattr__(self, "measurement_relative_uncertainty", measurement_uncertainty)
        object.__setattr__(self, "unit_to_metre", unit_to_metre)
        object.__setattr__(self, "relative_scale_uncertainty", scale_uncertainty)
        object.__setattr__(self, "rejection_reasons", reasons)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "notes", notes)
        object.__setattr__(self, "source_report_sha256", digest)
        object.__setattr__(self, "schema_version", METRIC_QUALIFICATION_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "validation_state": self.validation_state.value,
            "provenance": self.provenance,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "reference_id": self.reference_id,
            "validation_record_id": self.validation_record_id,
            "reference_length_m": self.reference_length_m,
            "measured_length_reconstruction_units": self.measured_length_reconstruction_units,
            "reference_relative_uncertainty": self.reference_relative_uncertainty,
            "measurement_relative_uncertainty": self.measurement_relative_uncertainty,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "rejection_reasons": list(self.rejection_reasons),
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
            "source_report_sha256": self.source_report_sha256,
        }


@dataclass(frozen=True, slots=True)
class MetricQualificationAssessment:
    """Authority decision produced from one evidence record."""

    evidence_id: str
    state: MetricQualificationState
    reconstruction_frame_id: str
    scale_profile: ScaleReferenceProfile | None
    authorizes_metric_coordinates: bool
    authorizes_calibrated_metric_scale: bool
    certified_metrology: bool
    message: str
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_id",
            require_nonempty(self.evidence_id, field_name="evidence_id"),
        )
        if not isinstance(self.state, MetricQualificationState):
            raise TypeError("state must be MetricQualificationState.")
        frame_id = str(self.reconstruction_frame_id).strip()
        if self.scale_profile is not None and not isinstance(self.scale_profile, ScaleReferenceProfile):
            raise TypeError("scale_profile must be ScaleReferenceProfile or None.")
        metric = bool(self.authorizes_metric_coordinates)
        calibrated = bool(self.authorizes_calibrated_metric_scale)
        if calibrated and not metric:
            raise ReconstructionValidationError(
                "Calibrated metric-scale authority implies metric-coordinate authority."
            )
        if self.scale_profile is None and (metric or calibrated):
            raise ReconstructionValidationError(
                "Metric authority requires an emitted ScaleReferenceProfile."
            )
        if self.scale_profile is not None:
            if not self.scale_profile.is_metric or not metric:
                raise ReconstructionValidationError(
                    "An emitted metric ScaleReferenceProfile must authorize metric coordinates."
                )
            if calibrated != self.scale_profile.is_calibrated_metric:
                raise ReconstructionValidationError(
                    "Assessment calibrated-scale authority must match ScaleReferenceProfile."
                )
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Phase 9.1 metric qualification cannot claim certified metrology."
            )
        object.__setattr__(self, "reconstruction_frame_id", frame_id)
        object.__setattr__(self, "authorizes_metric_coordinates", metric)
        object.__setattr__(self, "authorizes_calibrated_metric_scale", calibrated)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "message", require_nonempty(self.message, field_name="message"))
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(value).strip() for value in self.rejection_reasons if str(value).strip()),
        )

    @property
    def relative_scale_uncertainty(self) -> float | None:
        if self.scale_profile is None:
            return None
        return self.scale_profile.gauge.relative_scale_uncertainty

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "state": self.state.value,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "scale_profile": self.scale_profile.as_dict() if self.scale_profile else None,
            "authorizes_metric_coordinates": self.authorizes_metric_coordinates,
            "authorizes_calibrated_metric_scale": self.authorizes_calibrated_metric_scale,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "certified_metrology": self.certified_metrology,
            "message": self.message,
            "rejection_reasons": list(self.rejection_reasons),
        }


def known_size_reference_evidence(
    *,
    evidence_id: str,
    reconstruction_frame_id: str,
    reference_id: str,
    known_length_m: float,
    measured_length_reconstruction_units: float,
    known_length_relative_uncertainty: float,
    measurement_relative_uncertainty: float,
    provenance: str,
    validation_state: MetricEvidenceValidationState = MetricEvidenceValidationState.UNVALIDATED,
    validation_record_id: str = "",
    rejection_reasons: tuple[str, ...] = (),
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.KNOWN_SIZE_REFERENCE,
        validation_state=validation_state,
        provenance=provenance,
        reconstruction_frame_id=reconstruction_frame_id,
        reference_id=reference_id,
        validation_record_id=validation_record_id,
        rejection_reasons=rejection_reasons,
        reference_length_m=known_length_m,
        measured_length_reconstruction_units=measured_length_reconstruction_units,
        reference_relative_uncertainty=known_length_relative_uncertainty,
        measurement_relative_uncertainty=measurement_relative_uncertainty,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def known_distance_reference_evidence(
    *,
    evidence_id: str,
    reconstruction_frame_id: str,
    reference_id: str,
    known_distance_m: float,
    measured_distance_reconstruction_units: float,
    reference_relative_uncertainty: float,
    measurement_relative_uncertainty: float,
    provenance: str,
    validation_state: MetricEvidenceValidationState = MetricEvidenceValidationState.UNVALIDATED,
    validation_record_id: str = "",
    rejection_reasons: tuple[str, ...] = (),
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.KNOWN_DISTANCE_REFERENCE,
        validation_state=validation_state,
        provenance=provenance,
        reconstruction_frame_id=reconstruction_frame_id,
        reference_id=reference_id,
        validation_record_id=validation_record_id,
        rejection_reasons=rejection_reasons,
        reference_length_m=known_distance_m,
        measured_length_reconstruction_units=measured_distance_reconstruction_units,
        reference_relative_uncertainty=reference_relative_uncertainty,
        measurement_relative_uncertainty=measurement_relative_uncertainty,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def reference_target_metric_evidence(
    *,
    evidence_id: str,
    reconstruction_frame_id: str,
    reference_id: str,
    target_length_m: float,
    measured_length_reconstruction_units: float,
    target_relative_uncertainty: float,
    measurement_relative_uncertainty: float,
    provenance: str,
    validation_state: MetricEvidenceValidationState,
    validation_record_id: str = "",
    rejection_reasons: tuple[str, ...] = (),
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.REFERENCE_TARGET,
        validation_state=validation_state,
        provenance=provenance,
        reconstruction_frame_id=reconstruction_frame_id,
        reference_id=reference_id,
        validation_record_id=validation_record_id,
        rejection_reasons=rejection_reasons,
        reference_length_m=target_length_m,
        measured_length_reconstruction_units=measured_length_reconstruction_units,
        reference_relative_uncertainty=target_relative_uncertainty,
        measurement_relative_uncertainty=measurement_relative_uncertainty,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def metric_depth_scale_evidence(
    *,
    evidence_id: str,
    reconstruction_frame_id: str,
    reference_id: str,
    unit_to_metre: float,
    relative_scale_uncertainty: float,
    provenance: str,
    validation_state: MetricEvidenceValidationState,
    validation_record_id: str = "",
    rejection_reasons: tuple[str, ...] = (),
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.QUALIFIED_METRIC_DEPTH,
        validation_state=validation_state,
        provenance=provenance,
        reconstruction_frame_id=reconstruction_frame_id,
        reference_id=reference_id,
        validation_record_id=validation_record_id,
        rejection_reasons=rejection_reasons,
        unit_to_metre=unit_to_metre,
        relative_scale_uncertainty=relative_scale_uncertainty,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def external_metric_prior_evidence(
    *,
    evidence_id: str,
    reconstruction_frame_id: str,
    reference_id: str,
    unit_to_metre: float,
    relative_scale_uncertainty: float,
    provenance: str,
    validation_state: MetricEvidenceValidationState,
    validation_record_id: str = "",
    rejection_reasons: tuple[str, ...] = (),
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.EXTERNAL_METRIC_PRIOR,
        validation_state=validation_state,
        provenance=provenance,
        reconstruction_frame_id=reconstruction_frame_id,
        reference_id=reference_id,
        validation_record_id=validation_record_id,
        rejection_reasons=rejection_reasons,
        unit_to_metre=unit_to_metre,
        relative_scale_uncertainty=relative_scale_uncertainty,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def future_metric_source_evidence(
    *,
    evidence_id: str,
    kind: MetricEvidenceKind,
    provenance: str,
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    if kind not in _FUTURE_KINDS:
        raise ReconstructionValidationError(
            "future_metric_source_evidence supports FUTURE_STEREO or FUTURE_RGBD only."
        )
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=kind,
        validation_state=MetricEvidenceValidationState.UNVALIDATED,
        provenance=provenance,
        notes=notes,
        created_at_utc=created_at_utc,
    )


def known_size_benchmark_supporting_evidence(
    report_payload: Mapping[str, Any],
    *,
    evidence_id: str,
    provenance: str,
    validation_record_id: str = "",
    notes: str = "",
    created_at_utc: str = "",
) -> MetricQualificationEvidence:
    """Import the existing known-size benchmark report as supporting evidence only.

    A passing benchmark validates measurement-system behavior against known
    dimensions.  It does not identify a scene-specific reconstruction scale and
    therefore can never emit ``ScaleReferenceProfile`` authority by itself.
    """

    if not isinstance(report_payload, Mapping):
        raise TypeError("report_payload must be a mapping.")
    if bool(report_payload.get("certified_metrology", False)):
        raise ReconstructionValidationError(
            "Known-size benchmark reports must not claim certified metrology."
        )
    if "passed" not in report_payload or not isinstance(report_payload.get("summary"), Mapping):
        raise ReconstructionValidationError(
            "Known-size benchmark report is missing passed/summary fields."
        )
    canonical = json.dumps(
        dict(report_payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    passed = bool(report_payload["passed"])
    state = (
        MetricEvidenceValidationState.VALIDATED
        if passed
        else MetricEvidenceValidationState.REJECTED
    )
    reasons = () if passed else ("known_size_benchmark_failed",)
    return MetricQualificationEvidence(
        evidence_id=evidence_id,
        kind=MetricEvidenceKind.KNOWN_SIZE_BENCHMARK,
        validation_state=state,
        provenance=provenance,
        validation_record_id=validation_record_id,
        rejection_reasons=reasons,
        created_at_utc=created_at_utc,
        notes=notes,
        source_report_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def assess_metric_qualification(
    evidence: MetricQualificationEvidence,
    *,
    scale_profile_id: str,
) -> MetricQualificationAssessment:
    if not isinstance(evidence, MetricQualificationEvidence):
        raise TypeError("evidence must be MetricQualificationEvidence.")
    profile_id = require_nonempty(scale_profile_id, field_name="scale_profile_id")

    if evidence.validation_state is MetricEvidenceValidationState.REJECTED:
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=MetricQualificationState.REJECTED,
            reconstruction_frame_id=evidence.reconstruction_frame_id,
            scale_profile=None,
            authorizes_metric_coordinates=False,
            authorizes_calibrated_metric_scale=False,
            certified_metrology=False,
            message="Metric evidence was explicitly rejected and cannot establish scene scale.",
            rejection_reasons=evidence.rejection_reasons,
        )

    if evidence.kind is MetricEvidenceKind.KNOWN_SIZE_BENCHMARK:
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=MetricQualificationState.SUPPORTING_ONLY,
            reconstruction_frame_id="",
            scale_profile=None,
            authorizes_metric_coordinates=False,
            authorizes_calibrated_metric_scale=False,
            certified_metrology=False,
            message=(
                "The known-size benchmark supports measurement-system validation only; "
                "it does not establish scene-specific reconstruction scale."
            ),
        )

    if evidence.kind in _FUTURE_KINDS:
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=MetricQualificationState.UNSUPPORTED,
            reconstruction_frame_id=evidence.reconstruction_frame_id,
            scale_profile=None,
            authorizes_metric_coordinates=False,
            authorizes_calibrated_metric_scale=False,
            certified_metrology=False,
            message=(
                "Stereo/RGB-D metric evidence is reserved for a future qualified provider; "
                "Phase 9.1E does not grant scale authority from it."
            ),
        )

    if evidence.kind in {
        MetricEvidenceKind.KNOWN_SIZE_REFERENCE,
        MetricEvidenceKind.KNOWN_DISTANCE_REFERENCE,
    }:
        profile = known_dimension_scale_reference(
            profile_id=profile_id,
            reference_id=evidence.reference_id,
            known_length_m=_required(evidence.reference_length_m, "reference_length_m"),
            measured_length_reconstruction_units=_required(
                evidence.measured_length_reconstruction_units,
                "measured_length_reconstruction_units",
            ),
            known_length_relative_uncertainty=_required(
                evidence.reference_relative_uncertainty,
                "reference_relative_uncertainty",
            ),
            measurement_relative_uncertainty=_required(
                evidence.measurement_relative_uncertainty,
                "measurement_relative_uncertainty",
            ),
            provenance=evidence.provenance,
            notes=_profile_notes(evidence),
        )
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=MetricQualificationState.PROVISIONAL,
            reconstruction_frame_id=evidence.reconstruction_frame_id,
            scale_profile=profile,
            authorizes_metric_coordinates=True,
            authorizes_calibrated_metric_scale=False,
            certified_metrology=False,
            message=(
                "Known-size/distance evidence establishes provisional metric scale with explicit "
                "uncertainty; it is not a calibrated reference-target qualification."
            ),
        )

    if evidence.kind is MetricEvidenceKind.REFERENCE_TARGET:
        validated = evidence.validation_state is MetricEvidenceValidationState.VALIDATED
        profile = reference_target_scale_reference(
            profile_id=profile_id,
            reference_id=evidence.reference_id,
            target_length_m=_required(evidence.reference_length_m, "reference_length_m"),
            measured_length_reconstruction_units=_required(
                evidence.measured_length_reconstruction_units,
                "measured_length_reconstruction_units",
            ),
            target_relative_uncertainty=_required(
                evidence.reference_relative_uncertainty,
                "reference_relative_uncertainty",
            ),
            measurement_relative_uncertainty=_required(
                evidence.measurement_relative_uncertainty,
                "measurement_relative_uncertainty",
            ),
            validated_target=validated,
            provenance=evidence.provenance,
            notes=_profile_notes(evidence),
        )
        state = MetricQualificationState.QUALIFIED if validated else MetricQualificationState.PROVISIONAL
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=state,
            reconstruction_frame_id=evidence.reconstruction_frame_id,
            scale_profile=profile,
            authorizes_metric_coordinates=True,
            authorizes_calibrated_metric_scale=validated,
            certified_metrology=False,
            message=(
                "Validated reference-target evidence establishes calibrated metric scale."
                if validated
                else "Reference-target evidence is present but not validated; metric scale remains provisional."
            ),
        )

    if evidence.kind in _DIRECT_SCALE_KINDS:
        validated = evidence.validation_state is MetricEvidenceValidationState.VALIDATED
        profile = external_metric_prior_scale_reference(
            profile_id=profile_id,
            reference_id=evidence.reference_id,
            unit_to_metre=_required(evidence.unit_to_metre, "unit_to_metre"),
            relative_scale_uncertainty=_required(
                evidence.relative_scale_uncertainty,
                "relative_scale_uncertainty",
            ),
            validated_prior=validated,
            provenance=evidence.provenance,
            notes=_profile_notes(evidence),
        )
        state = MetricQualificationState.QUALIFIED if validated else MetricQualificationState.PROVISIONAL
        source_name = (
            "metric-depth source"
            if evidence.kind is MetricEvidenceKind.QUALIFIED_METRIC_DEPTH
            else "external metric prior"
        )
        return MetricQualificationAssessment(
            evidence_id=evidence.evidence_id,
            state=state,
            reconstruction_frame_id=evidence.reconstruction_frame_id,
            scale_profile=profile,
            authorizes_metric_coordinates=True,
            authorizes_calibrated_metric_scale=validated,
            certified_metrology=False,
            message=(
                f"Validated {source_name} establishes calibrated metric scale."
                if validated
                else f"Unvalidated {source_name} provides provisional metric scale only."
            ),
        )

    raise ReconstructionValidationError(f"Unhandled metric evidence kind: {evidence.kind.value}")


def metric_qualification_evidence_to_json(
    evidence: MetricQualificationEvidence,
    *,
    indent: int | None = 2,
) -> str:
    if not isinstance(evidence, MetricQualificationEvidence):
        raise TypeError("evidence must be MetricQualificationEvidence.")
    return json.dumps(
        evidence.as_dict(),
        indent=indent,
        sort_keys=True,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    )


def metric_qualification_evidence_from_json(
    payload: str | bytes | bytearray,
) -> MetricQualificationEvidence:
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ReconstructionValidationError("Metric-qualification evidence must be valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Metric-qualification evidence root must be an object.")
    return metric_qualification_evidence_from_dict(raw)


def metric_qualification_evidence_from_dict(
    raw: Mapping[str, Any],
) -> MetricQualificationEvidence:
    try:
        if int(raw["schema_version"]) != METRIC_QUALIFICATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported metric-qualification schema version.")
        return MetricQualificationEvidence(
            evidence_id=str(raw["evidence_id"]),
            kind=MetricEvidenceKind(str(raw["kind"])),
            validation_state=MetricEvidenceValidationState(str(raw["validation_state"])),
            provenance=str(raw["provenance"]),
            reconstruction_frame_id=str(raw.get("reconstruction_frame_id", "")),
            reference_id=str(raw.get("reference_id", "")),
            validation_record_id=str(raw.get("validation_record_id", "")),
            reference_length_m=_optional_float(raw.get("reference_length_m")),
            measured_length_reconstruction_units=_optional_float(
                raw.get("measured_length_reconstruction_units")
            ),
            reference_relative_uncertainty=_optional_float(
                raw.get("reference_relative_uncertainty")
            ),
            measurement_relative_uncertainty=_optional_float(
                raw.get("measurement_relative_uncertainty")
            ),
            unit_to_metre=_optional_float(raw.get("unit_to_metre")),
            relative_scale_uncertainty=_optional_float(raw.get("relative_scale_uncertainty")),
            rejection_reasons=tuple(str(v) for v in raw.get("rejection_reasons", ())),
            created_at_utc=str(raw.get("created_at_utc", "")),
            notes=str(raw.get("notes", "")),
            source_report_sha256=str(raw.get("source_report_sha256", "")),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Metric-qualification evidence is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Metric-qualification evidence contains an invalid value: {exc}."
        ) from exc


def _positive(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return result


def _optional_positive(value: float | None, field_name: str) -> float | None:
    return None if value is None else _positive(value, field_name)


def _relative_uncertainty(value: float, field_name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0 or result >= 1.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and in [0, 1).")
    return result


def _optional_relative_uncertainty(value: float | None, field_name: str) -> float | None:
    return None if value is None else _relative_uncertainty(value, field_name)


def _required(value: float | None, field_name: str) -> float:
    if value is None:
        raise ReconstructionValidationError(f"{field_name} is required.")
    return float(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _profile_notes(evidence: MetricQualificationEvidence) -> str:
    fields = [f"phase9_metric_evidence_id={evidence.evidence_id}"]
    if evidence.validation_record_id:
        fields.append(f"validation_record_id={evidence.validation_record_id}")
    if evidence.notes:
        fields.append(evidence.notes)
    return "; ".join(fields)
