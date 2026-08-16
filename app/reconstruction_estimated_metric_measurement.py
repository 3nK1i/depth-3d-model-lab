"""Estimated metric ruler presentation for Phase 9.6E0.

The canonical surface remains the only picking/measurement geometry authority.
This module adds a deliberately non-authoritative *estimated scale hint* that may
convert canonical reconstruction-unit distances into a useful metre estimate
before scene scale is formally qualified.  The estimate is explicitly
uncalibrated, carries uncertainty/provenance, and never promotes the canonical
scene to metric authority.

A later provisional/qualified scene scale may refine the current readout while
this module retains the original estimate for provenance.  Persistence of this
session-side estimate is intentionally deferred to Phase 9.6F.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

import numpy as np

from app.reconstruction_canonical_scene_presentation import CanonicalScenePresentation
from core.measurement.interactive_distance import (
    InteractiveDistanceEstimate,
    InteractiveDistanceState,
    estimate_reconstruction_distance,
    format_interactive_distance,
)


ESTIMATED_SCALE_SOURCE_MANUAL_RULER = "manual_ruler_reference"
_SURFACE_REFERENCE_RELATIVE_UNCERTAINTY = 0.01


def _positive(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _relative_uncertainty(value: float, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0 or number >= 1.0:
        raise ValueError(f"{name} must be finite and in (0, 1).")
    return number


def _point(value: Sequence[float], name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector.")
    return tuple(float(v) for v in array)


@dataclass(frozen=True, slots=True)
class EstimatedMetricScaleHint:
    """One scene-bound non-authoritative estimate of metres per reconstruction unit."""

    scene_id: str
    source_kind: str
    unit_to_metre_estimate: float
    relative_scale_uncertainty: float
    provenance: str
    reference_distance_reconstruction_units: float | None = None
    reference_distance_m: float | None = None
    reference_relative_uncertainty: float | None = None
    calibrated: bool = False
    metric_scale_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        source_kind = str(self.source_kind).strip()
        provenance = str(self.provenance).strip()
        if not scene_id:
            raise ValueError("scene_id is required for an estimated scale hint.")
        if not source_kind:
            raise ValueError("source_kind is required for an estimated scale hint.")
        if not provenance:
            raise ValueError("provenance is required for an estimated scale hint.")
        unit = _positive(self.unit_to_metre_estimate, "unit_to_metre_estimate")
        uncertainty = _relative_uncertainty(
            self.relative_scale_uncertainty,
            "relative_scale_uncertainty",
        )
        reference_units = self.reference_distance_reconstruction_units
        reference_m = self.reference_distance_m
        reference_uncertainty = self.reference_relative_uncertainty
        if (reference_units is None) != (reference_m is None):
            raise ValueError("Estimated ruler references require both reconstruction and metre distances.")
        if reference_units is not None:
            reference_units = _positive(reference_units, "reference_distance_reconstruction_units")
            reference_m = _positive(reference_m, "reference_distance_m")  # type: ignore[arg-type]
            if reference_uncertainty is None:
                raise ValueError("Estimated ruler references require reference_relative_uncertainty.")
            reference_uncertainty = _relative_uncertainty(
                reference_uncertainty,
                "reference_relative_uncertainty",
            )
        elif reference_uncertainty is not None:
            raise ValueError("reference_relative_uncertainty requires a ruler reference pair.")
        if self.calibrated or self.metric_scale_authoritative or self.certified_metrology:
            raise ValueError(
                "Estimated scale hints are uncalibrated, non-authoritative, and cannot certify metrology."
            )
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "unit_to_metre_estimate", unit)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "reference_distance_reconstruction_units", reference_units)
        object.__setattr__(self, "reference_distance_m", reference_m)
        object.__setattr__(self, "reference_relative_uncertainty", reference_uncertainty)
        object.__setattr__(self, "calibrated", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def scale_text(self) -> str:
        return (
            f"{self.unit_to_metre_estimate:.6g} m / reconstruction unit "
            f"• ±{self.relative_scale_uncertainty * 100.0:.1f}%"
        )

    @property
    def reference_text(self) -> str:
        if self.reference_distance_reconstruction_units is None or self.reference_distance_m is None:
            return self.source_kind.replace("_", " ").title()
        return (
            f"{self.reference_distance_m:.3f} m ↔ "
            f"{self.reference_distance_reconstruction_units:.4f} reconstruction units"
        )


def estimated_scale_from_ruler_reference(
    *,
    scene_id: str,
    distance_reconstruction_units: float,
    approximate_distance_m: float,
    reference_relative_uncertainty: float,
    provenance: str = "User-supplied approximate physical distance bound to the last canonical ruler segment",
) -> EstimatedMetricScaleHint:
    """Build a non-authoritative scene scale estimate from one approximate ruler reference."""

    native = _positive(distance_reconstruction_units, "distance_reconstruction_units")
    distance_m = _positive(approximate_distance_m, "approximate_distance_m")
    reference_rel = _relative_uncertainty(
        reference_relative_uncertainty,
        "reference_relative_uncertainty",
    )
    unit = distance_m / native
    # The user/reference estimate and canonical surface endpoint localization are
    # independent heuristic components.  Keep the surface term explicit instead
    # of silently treating the picked reconstruction segment as exact.
    scale_rel = hypot(reference_rel, _SURFACE_REFERENCE_RELATIVE_UNCERTAINTY)
    return EstimatedMetricScaleHint(
        scene_id=scene_id,
        source_kind=ESTIMATED_SCALE_SOURCE_MANUAL_RULER,
        unit_to_metre_estimate=unit,
        relative_scale_uncertainty=scale_rel,
        provenance=provenance,
        reference_distance_reconstruction_units=native,
        reference_distance_m=distance_m,
        reference_relative_uncertainty=reference_rel,
    )


@dataclass(frozen=True, slots=True)
class ReconstructionRulerMetricRecord:
    """Current ruler result plus immutable provenance of its first metric estimate."""

    scene_id: str
    point_a_reconstruction: tuple[float, float, float]
    point_b_reconstruction: tuple[float, float, float]
    current: InteractiveDistanceEstimate
    current_scale_source: str
    current_scale_provenance: str
    original_estimate: InteractiveDistanceEstimate | None = None
    original_scale_source: str = ""
    original_scale_provenance: str = ""
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        scene_id = str(self.scene_id).strip()
        if not scene_id:
            raise ValueError("scene_id is required for ruler metric provenance.")
        if not isinstance(self.current, InteractiveDistanceEstimate):
            raise TypeError("current must be InteractiveDistanceEstimate.")
        if self.original_estimate is not None and not isinstance(
            self.original_estimate,
            InteractiveDistanceEstimate,
        ):
            raise TypeError("original_estimate must be InteractiveDistanceEstimate or None.")
        if self.original_estimate is not None and not self.original_estimate.has_metric_estimate:
            raise ValueError("original_estimate must contain a metric estimate when present.")
        if self.certified_metrology:
            raise ValueError("Interactive ruler provenance cannot certify metrology.")
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(
            self,
            "point_a_reconstruction",
            _point(self.point_a_reconstruction, "point_a_reconstruction"),
        )
        object.__setattr__(
            self,
            "point_b_reconstruction",
            _point(self.point_b_reconstruction, "point_b_reconstruction"),
        )
        object.__setattr__(self, "current_scale_source", str(self.current_scale_source).strip())
        object.__setattr__(
            self,
            "current_scale_provenance",
            str(self.current_scale_provenance).strip(),
        )
        object.__setattr__(self, "original_scale_source", str(self.original_scale_source).strip())
        object.__setattr__(
            self,
            "original_scale_provenance",
            str(self.original_scale_provenance).strip(),
        )
        object.__setattr__(self, "certified_metrology", False)

    @property
    def original_preserved(self) -> bool:
        return self.original_estimate is not None and self.current is not self.original_estimate

    @property
    def primary_text(self) -> str:
        primary, _ = format_interactive_distance(self.current)
        if self.current.has_metric_estimate and self.current.state is InteractiveDistanceState.ESTIMATED:
            return f"Estimated {primary}"
        if self.current.has_metric_estimate and self.current.state is InteractiveDistanceState.QUALIFIED:
            return f"Qualified {primary}"
        return primary

    @property
    def secondary_text(self) -> str:
        raw = self.current.distance_source_units
        if self.current.has_metric_estimate:
            status = "QUALIFIED" if self.current.state is InteractiveDistanceState.QUALIFIED else "ESTIMATED"
            source = self.current_scale_source.replace("_", " ") if self.current_scale_source else "scale evidence"
            suffix = " • original estimate preserved" if self.original_preserved else ""
            raw_text = f" • raw {raw:.4f} reconstruction units" if raw is not None else ""
            return (
                f"{status} • {source}{raw_text}{suffix} • certified metrology: no"
            )
        _, secondary = format_interactive_distance(self.current)
        return secondary

    @property
    def provenance_tooltip(self) -> str:
        lines = [
            f"Current scale source: {self.current_scale_source or 'none'}",
            f"Current provenance: {self.current_scale_provenance or 'no metric scale evidence'}",
        ]
        if self.current.distance_source_units is not None:
            lines.append(
                f"Canonical distance: {self.current.distance_source_units:.9g} reconstruction units"
            )
        if self.original_estimate is not None:
            assert self.original_estimate.distance_m is not None
            assert self.original_estimate.expanded_uncertainty_m is not None
            lines.extend(
                [
                    (
                        "Original estimate: "
                        f"{self.original_estimate.distance_m:.6g} m ± "
                        f"{self.original_estimate.expanded_uncertainty_m:.6g} m"
                    ),
                    f"Original scale source: {self.original_scale_source}",
                    f"Original provenance: {self.original_scale_provenance}",
                ]
            )
        lines.append("Canonical geometry remains picking/measurement authority; metrology is not certified.")
        return "\n".join(lines)


def _same_endpoints(
    previous: ReconstructionRulerMetricRecord | None,
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
) -> bool:
    if previous is None:
        return False
    return bool(
        np.allclose(previous.point_a_reconstruction, point_a, rtol=0.0, atol=1e-12)
        and np.allclose(previous.point_b_reconstruction, point_b, rtol=0.0, atol=1e-12)
    )


def build_reconstruction_ruler_metric_record(
    *,
    scene_id: str,
    point_a_reconstruction: Sequence[float],
    point_b_reconstruction: Sequence[float],
    canonical_scene: CanonicalScenePresentation | None,
    estimated_scale_hint: EstimatedMetricScaleHint | None = None,
    previous: ReconstructionRulerMetricRecord | None = None,
) -> ReconstructionRulerMetricRecord:
    """Build the best truthful current ruler readout without changing scale authority."""

    scene = str(scene_id).strip()
    if not scene:
        raise ValueError("scene_id is required for an interactive reconstruction ruler result.")
    point_a = _point(point_a_reconstruction, "point_a_reconstruction")
    point_b = _point(point_b_reconstruction, "point_b_reconstruction")
    if canonical_scene is not None and canonical_scene.available and canonical_scene.scene_id != scene:
        raise ValueError("Canonical scene presentation does not match ruler scene_id.")
    if estimated_scale_hint is not None and estimated_scale_hint.scene_id != scene:
        raise ValueError("Estimated scale hint does not match ruler scene_id.")

    if canonical_scene is not None and canonical_scene.scale_unit_to_metre is not None:
        current = estimate_reconstruction_distance(
            point_a_reconstruction=point_a,
            point_b_reconstruction=point_b,
            unit_to_metre=canonical_scene.scale_unit_to_metre,
            relative_scale_uncertainty=canonical_scene.scale_relative_uncertainty,
            scale_state=canonical_scene.scale_state.value,
            calibration_available=canonical_scene.calibration_available,
        )
        current_source = (
            "qualified_scene_scale"
            if current.state is InteractiveDistanceState.QUALIFIED
            else "provisional_scene_scale"
        )
        current_provenance = (
            f"scene-bound metric evidence {canonical_scene.metric_evidence_id or '(unspecified)'}"
        )
    elif estimated_scale_hint is not None:
        current = estimate_reconstruction_distance(
            point_a_reconstruction=point_a,
            point_b_reconstruction=point_b,
            unit_to_metre=estimated_scale_hint.unit_to_metre_estimate,
            relative_scale_uncertainty=estimated_scale_hint.relative_scale_uncertainty,
            scale_state="estimated",
            calibration_available=False,
        )
        current_source = estimated_scale_hint.source_kind
        current_provenance = estimated_scale_hint.provenance
    else:
        current = estimate_reconstruction_distance(
            point_a_reconstruction=point_a,
            point_b_reconstruction=point_b,
            unit_to_metre=None,
            relative_scale_uncertainty=None,
            scale_state=(
                canonical_scene.scale_state.value
                if canonical_scene is not None
                else "not_qualified"
            ),
            calibration_available=bool(
                canonical_scene is not None and canonical_scene.calibration_available
            ),
        )
        current_source = ""
        current_provenance = "No scene-bound metric or estimated scale evidence is available"

    original = None
    original_source = ""
    original_provenance = ""
    if _same_endpoints(previous, point_a, point_b) and previous is not None:
        original = previous.original_estimate
        original_source = previous.original_scale_source
        original_provenance = previous.original_scale_provenance
    if original is None and current.has_metric_estimate and current.state is InteractiveDistanceState.ESTIMATED:
        original = current
        original_source = current_source
        original_provenance = current_provenance

    return ReconstructionRulerMetricRecord(
        scene_id=scene,
        point_a_reconstruction=point_a,
        point_b_reconstruction=point_b,
        current=current,
        current_scale_source=current_source,
        current_scale_provenance=current_provenance,
        original_estimate=original,
        original_scale_source=original_source,
        original_scale_provenance=original_provenance,
        certified_metrology=False,
    )


__all__ = [
    "ESTIMATED_SCALE_SOURCE_MANUAL_RULER",
    "EstimatedMetricScaleHint",
    "ReconstructionRulerMetricRecord",
    "build_reconstruction_ruler_metric_record",
    "estimated_scale_from_ruler_reference",
]
