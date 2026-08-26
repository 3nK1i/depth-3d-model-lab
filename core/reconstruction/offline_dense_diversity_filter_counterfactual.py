"""Phase I2.5 conservative diversity-aware filtering counterfactuals.

I2.5 is deliberately non-production.  It consumes the already-retained F
Extended geometry together with I2.4 support-diversity evidence and evaluates a
small ladder of conservative *post-F* rejection rules.  The rules never use the
I2 peripheral/far labels as filtering inputs; those labels are evaluation-only.

No source geometry, F policy, G persistence, SfM, DA3, metric authority,
measurement authority, or mesh authority is changed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError


I25_DIVERSITY_FILTER_COUNTERFACTUAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DiversityFilterScenario:
    """One conservative diagnostic rejection rule.

    ``diversity_scale`` multiplies the I2.4 central-Core reference thresholds.
    Values below one therefore require *more extremely* narrow support geometry.
    Optional topology/confidence constraints are conjunctive and are evaluated
    using the already-authored F evidence for the retained Extended cloud.
    """

    name: str
    maximum_support: int
    diversity_scale: float
    maximum_high_support_neighbors: int | None = None
    maximum_radius2_neighbors: int | None = None
    maximum_confidence_reference_quantile: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ReconstructionValidationError("I2.5 scenario name cannot be empty.")
        maximum_support = int(self.maximum_support)
        if maximum_support < 2 or maximum_support > 3:
            raise ReconstructionValidationError("I2.5 maximum_support must be 2 or 3.")
        scale = float(self.diversity_scale)
        if not isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ReconstructionValidationError("I2.5 diversity_scale must be finite in (0, 1].")
        if self.maximum_high_support_neighbors is not None and int(self.maximum_high_support_neighbors) < 0:
            raise ReconstructionValidationError("I2.5 maximum_high_support_neighbors must be non-negative.")
        if self.maximum_radius2_neighbors is not None and int(self.maximum_radius2_neighbors) < 0:
            raise ReconstructionValidationError("I2.5 maximum_radius2_neighbors must be non-negative.")
        q = self.maximum_confidence_reference_quantile
        if q is not None:
            q = float(q)
            if not isfinite(q) or not 0.0 < q < 1.0:
                raise ReconstructionValidationError(
                    "I2.5 maximum_confidence_reference_quantile must be finite in (0, 1)."
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "maximum_support", maximum_support)
        object.__setattr__(self, "diversity_scale", scale)
        if self.maximum_high_support_neighbors is not None:
            object.__setattr__(self, "maximum_high_support_neighbors", int(self.maximum_high_support_neighbors))
        if self.maximum_radius2_neighbors is not None:
            object.__setattr__(self, "maximum_radius2_neighbors", int(self.maximum_radius2_neighbors))
        object.__setattr__(self, "maximum_confidence_reference_quantile", q)
        object.__setattr__(self, "note", str(self.note).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "maximum_support": self.maximum_support,
            "diversity_scale": self.diversity_scale,
            "maximum_high_support_neighbors": self.maximum_high_support_neighbors,
            "maximum_radius2_neighbors": self.maximum_radius2_neighbors,
            "maximum_confidence_reference_quantile": self.maximum_confidence_reference_quantile,
            "note": self.note,
        }


def default_diversity_filter_scenarios() -> tuple[DiversityFilterScenario, ...]:
    """Return the fixed conservative I2.5 policy ladder.

    The ladder intentionally spans different safeguards instead of sweeping a
    fine numeric grid, which would overfit the current scene.
    """

    return (
        DiversityFilterScenario(
            "s2_lowdiv_anchor1",
            maximum_support=2,
            diversity_scale=1.0,
            maximum_high_support_neighbors=1,
            note="support-2 + I2.4 low diversity + weakest retained >=3-view anchoring",
        ),
        DiversityFilterScenario(
            "s2_lowdiv_anchor2",
            maximum_support=2,
            diversity_scale=1.0,
            maximum_high_support_neighbors=2,
            note="support-2 + I2.4 low diversity + weak/moderate >=3-view anchoring",
        ),
        DiversityFilterScenario(
            "s2_lowdiv_anchor2_radius12",
            maximum_support=2,
            diversity_scale=1.0,
            maximum_high_support_neighbors=2,
            maximum_radius2_neighbors=12,
            note="support-2 low diversity constrained to locally sparse/weakly anchored geometry",
        ),
        DiversityFilterScenario(
            "s2_075_anchor2",
            maximum_support=2,
            diversity_scale=0.75,
            maximum_high_support_neighbors=2,
            note="stricter-than-reference diversity with weak/moderate anchoring",
        ),
        DiversityFilterScenario(
            "s2_verylow_anyanchor",
            maximum_support=2,
            diversity_scale=0.50,
            note="only very-low-diversity support-2 geometry, independent of anchor count",
        ),
        DiversityFilterScenario(
            "s23_075_anchor1_lowconf25",
            maximum_support=3,
            diversity_scale=0.75,
            maximum_high_support_neighbors=1,
            maximum_confidence_reference_quantile=0.25,
            note="support<=3, stricter diversity, weakest anchor and low central-Core-relative confidence",
        ),
    )


def _aligned_1d(name: str, value: object, length: int, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (int(length),):
        raise ReconstructionValidationError(f"I2.5 {name} must be an aligned 1D vector.")
    return array


def confidence_reference_threshold(
    confidence: object,
    *,
    reference_mask: object,
    quantile: float,
) -> float:
    values = np.asarray(confidence, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_mask, dtype=bool).reshape(-1)
    if values.shape != reference.shape or len(values) == 0:
        raise ReconstructionValidationError("I2.5 confidence/reference arrays must align and be non-empty.")
    selected = values[reference & np.isfinite(values)]
    if len(selected) < 2:
        raise ReconstructionValidationError("I2.5 has insufficient central-Core confidence reference values.")
    q = float(quantile)
    if not isfinite(q) or not 0.0 < q < 1.0:
        raise ReconstructionValidationError("I2.5 confidence quantile must be finite in (0, 1).")
    return float(np.quantile(selected, q))


def diversity_filter_rejection_mask(
    *,
    scenario: DiversityFilterScenario,
    support: object,
    maximum_pair_parallax_degrees: object,
    maximum_baseline_to_range_ratio: object,
    high_support_neighbors: object,
    radius2_neighbors: object,
    confidence: object,
    reference_mask: object,
    parallax_reference_threshold_degrees: float,
    baseline_to_range_reference_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a candidate post-F rejection mask and resolved thresholds."""

    if not isinstance(scenario, DiversityFilterScenario):
        raise TypeError("scenario must be DiversityFilterScenario.")
    support_arr = np.asarray(support, dtype=np.int64).reshape(-1)
    n = len(support_arr)
    if n == 0:
        raise ReconstructionValidationError("I2.5 requires retained Extended geometry.")
    angle = _aligned_1d("parallax", maximum_pair_parallax_degrees, n, np.float64)
    ratio = _aligned_1d("baseline/range", maximum_baseline_to_range_ratio, n, np.float64)
    high = _aligned_1d("high-support neighbors", high_support_neighbors, n, np.int64)
    radius2 = _aligned_1d("radius2 neighbors", radius2_neighbors, n, np.int64)
    conf = _aligned_1d("confidence", confidence, n, np.float64)
    reference = _aligned_1d("reference mask", reference_mask, n, bool)
    if np.any(support_arr < 2):
        raise ReconstructionValidationError("I2.5 input must already be retained >=2-view F geometry.")
    if np.any(~np.isfinite(angle)) or np.any(angle < 0.0):
        raise ReconstructionValidationError("I2.5 parallax evidence must be finite and non-negative.")
    if np.any(~np.isfinite(ratio)) or np.any(ratio < 0.0):
        raise ReconstructionValidationError("I2.5 baseline/range evidence must be finite and non-negative.")
    if np.any(high < 0) or np.any(radius2 < 0):
        raise ReconstructionValidationError("I2.5 topology evidence must be non-negative.")
    if np.any(~np.isfinite(conf)) or np.any((conf < 0.0) | (conf > 1.0)):
        raise ReconstructionValidationError("I2.5 confidence evidence must be finite in [0,1].")

    parallax_ref = float(parallax_reference_threshold_degrees)
    ratio_ref = float(baseline_to_range_reference_threshold)
    if not isfinite(parallax_ref) or parallax_ref < 0.0 or not isfinite(ratio_ref) or ratio_ref < 0.0:
        raise ReconstructionValidationError("I2.5 reference diversity thresholds must be finite and non-negative.")
    parallax_limit = parallax_ref * float(scenario.diversity_scale)
    ratio_limit = ratio_ref * float(scenario.diversity_scale)

    reject = (
        (support_arr <= int(scenario.maximum_support))
        & (angle <= parallax_limit)
        & (ratio <= ratio_limit)
    )
    if scenario.maximum_high_support_neighbors is not None:
        reject &= high <= int(scenario.maximum_high_support_neighbors)
    if scenario.maximum_radius2_neighbors is not None:
        reject &= radius2 <= int(scenario.maximum_radius2_neighbors)

    confidence_limit: float | None = None
    if scenario.maximum_confidence_reference_quantile is not None:
        confidence_limit = confidence_reference_threshold(
            conf,
            reference_mask=reference,
            quantile=float(scenario.maximum_confidence_reference_quantile),
        )
        reject &= conf <= confidence_limit

    resolved = {
        **scenario.as_dict(),
        "resolved_parallax_limit_degrees": float(parallax_limit),
        "resolved_baseline_to_range_limit": float(ratio_limit),
        "resolved_confidence_limit": confidence_limit,
        "rejected_points": int(np.count_nonzero(reject)),
        "rejected_fraction_of_extended": float(np.mean(reject)),
    }
    return reject.astype(bool), resolved


def policy_effect_counts(
    reject_mask: object,
    *,
    is_core: object,
    is_peripheral: object,
    is_far_peripheral: object,
    is_weak_peripheral: object,
) -> dict[str, Any]:
    reject = np.asarray(reject_mask, dtype=bool).reshape(-1)
    masks = [np.asarray(v, dtype=bool).reshape(-1) for v in (is_core, is_peripheral, is_far_peripheral, is_weak_peripheral)]
    if any(mask.shape != reject.shape for mask in masks):
        raise ReconstructionValidationError("I2.5 evaluation masks must align.")
    core, peripheral, far, weak = masks
    central_core = core & ~peripheral

    def _fraction_removed(region: np.ndarray) -> float:
        count = int(np.count_nonzero(region))
        return int(np.count_nonzero(reject & region)) / float(count) if count else 0.0

    return {
        "rejected_points": int(np.count_nonzero(reject)),
        "core_rejected_points": int(np.count_nonzero(reject & core)),
        "central_core_rejected_points": int(np.count_nonzero(reject & central_core)),
        "peripheral_rejected_points": int(np.count_nonzero(reject & peripheral)),
        "far_peripheral_rejected_points": int(np.count_nonzero(reject & far)),
        "weak_peripheral_rejected_points": int(np.count_nonzero(reject & weak)),
        "core_loss_fraction": _fraction_removed(core),
        "central_core_loss_fraction": _fraction_removed(central_core),
        "peripheral_reduction_fraction": _fraction_removed(peripheral),
        "far_peripheral_reduction_fraction": _fraction_removed(far),
        "weak_peripheral_reduction_fraction": _fraction_removed(weak),
    }


def counterfactual_quality_comparison(
    *,
    baseline_extended_points: int,
    baseline_core_points: int,
    baseline_full_to_robust_ratio: float,
    baseline_robust_diagonal: float,
    candidate_extended_points: int,
    candidate_core_points: int,
    candidate_full_to_robust_ratio: float,
    candidate_robust_diagonal: float,
    candidate_dominant_component_fraction: float,
    effect_counts: Mapping[str, Any],
) -> dict[str, Any]:
    values = (
        baseline_extended_points,
        baseline_core_points,
        candidate_extended_points,
        candidate_core_points,
    )
    if any(int(v) <= 0 for v in values):
        raise ReconstructionValidationError("I2.5 comparison point counts must be positive.")
    base_ratio = float(baseline_full_to_robust_ratio)
    cand_ratio = float(candidate_full_to_robust_ratio)
    base_robust = float(baseline_robust_diagonal)
    cand_robust = float(candidate_robust_diagonal)
    dominant = float(candidate_dominant_component_fraction)
    if any(not isfinite(v) or v <= 0.0 for v in (base_ratio, cand_ratio, base_robust, cand_robust)):
        raise ReconstructionValidationError("I2.5 envelope comparison values must be finite and positive.")
    if not isfinite(dominant) or not 0.0 <= dominant <= 1.0:
        raise ReconstructionValidationError("I2.5 dominant-component fraction must be finite in [0,1].")

    core_loss = 1.0 - int(candidate_core_points) / float(int(baseline_core_points))
    extended_loss = 1.0 - int(candidate_extended_points) / float(int(baseline_extended_points))
    ratio_improvement = 1.0 - cand_ratio / base_ratio
    robust_change = cand_robust / base_robust - 1.0
    central_core_loss = float(effect_counts["central_core_loss_fraction"])
    peripheral_reduction = float(effect_counts["peripheral_reduction_fraction"])
    far_reduction = float(effect_counts["far_peripheral_reduction_fraction"])
    weak_reduction = float(effect_counts["weak_peripheral_reduction_fraction"])

    score = (
        2.25 * far_reduction
        + 1.25 * peripheral_reduction
        + 0.50 * weak_reduction
        + 1.25 * ratio_improvement
        - 3.50 * max(core_loss, 0.0)
        - 4.00 * max(central_core_loss, 0.0)
        - 1.25 * max(extended_loss, 0.0)
        - 1.00 * max(-robust_change - 0.05, 0.0)
    )
    balanced = bool(
        core_loss <= 0.05
        and central_core_loss <= 0.03
        and extended_loss <= 0.08
        and peripheral_reduction >= 0.10
        and far_reduction >= 0.20
        and ratio_improvement > 0.0
        and robust_change >= -0.10
        and dominant >= 0.90
    )
    preferred = bool(
        core_loss <= 0.03
        and central_core_loss <= 0.02
        and extended_loss <= 0.06
        and peripheral_reduction >= 0.15
        and far_reduction >= 0.30
        and ratio_improvement >= 0.03
        and robust_change >= -0.08
        and dominant >= 0.90
    )
    return {
        "core_loss_fraction": float(core_loss),
        "central_core_loss_fraction": float(central_core_loss),
        "extended_loss_fraction": float(extended_loss),
        "peripheral_reduction_fraction": float(peripheral_reduction),
        "far_peripheral_reduction_fraction": float(far_reduction),
        "weak_peripheral_reduction_fraction": float(weak_reduction),
        "full_to_robust_ratio_improvement_fraction": float(ratio_improvement),
        "extended_robust_diagonal_change_fraction": float(robust_change),
        "candidate_dominant_component_fraction": float(dominant),
        "diagnostic_score": float(score),
        "balanced_for_manual_review": balanced,
        "preferred_for_manual_review": preferred,
    }


def pareto_front(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Non-dominated policies for far/peripheral reduction vs central/Core loss."""

    result: list[str] = []
    for candidate in records:
        c = candidate["comparison"]
        dominated = False
        for other in records:
            if other is candidate:
                continue
            o = other["comparison"]
            no_worse = (
                float(o["core_loss_fraction"]) <= float(c["core_loss_fraction"])
                and float(o["central_core_loss_fraction"]) <= float(c["central_core_loss_fraction"])
                and float(o["far_peripheral_reduction_fraction"]) >= float(c["far_peripheral_reduction_fraction"])
                and float(o["peripheral_reduction_fraction"]) >= float(c["peripheral_reduction_fraction"])
                and float(o["full_to_robust_ratio_improvement_fraction"])
                >= float(c["full_to_robust_ratio_improvement_fraction"])
            )
            strictly_better = (
                float(o["core_loss_fraction"]) < float(c["core_loss_fraction"])
                or float(o["central_core_loss_fraction"]) < float(c["central_core_loss_fraction"])
                or float(o["far_peripheral_reduction_fraction"]) > float(c["far_peripheral_reduction_fraction"])
                or float(o["peripheral_reduction_fraction"]) > float(c["peripheral_reduction_fraction"])
                or float(o["full_to_robust_ratio_improvement_fraction"])
                > float(c["full_to_robust_ratio_improvement_fraction"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["name"]))
    return tuple(result)
