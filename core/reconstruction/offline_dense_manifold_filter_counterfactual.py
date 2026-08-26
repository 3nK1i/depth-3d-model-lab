"""I2.5.3 conservative manifold-attachment filtering counterfactuals.

The helpers in this module evaluate a narrow family of post-F candidate
rejection masks derived only from I2.5.2 local-manifold evidence.  Scene
position labels (peripheral/far), I2.5.1 extremum-driver labels, source-view
blacklists, and metric-distance thresholds are deliberately excluded from the
filtering rule and are evaluation-only.

This module is counterfactual only: it does not mutate F, G persistence,
metric/measurement authority, or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError


I253_MANIFOLD_FILTER_COUNTERFACTUAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ManifoldFilterScenario:
    """Scene-relative local-manifold rejection rule.

    ``anchor_distance_scale`` and ``knn_spacing_scale`` multiply the central
    Core non-driver reference quantiles learned by I2.5.2.  Optional linearity
    protection further narrows the rule.  No spatial/extremum label is accepted
    by the rule itself.
    """

    name: str
    anchor_distance_scale: float
    knn_spacing_scale: float = 1.0
    require_linearity: bool = False
    linearity_scale: float = 1.0
    note: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ReconstructionValidationError("I2.5.3 scenario name must be non-empty.")
        for name in ("anchor_distance_scale", "knn_spacing_scale", "linearity_scale"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ReconstructionValidationError(f"I2.5.3 {name} must be finite and positive.")
        if float(self.anchor_distance_scale) < 1.0:
            raise ReconstructionValidationError("I2.5.3 anchor-distance scale must be >= 1.")

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def default_manifold_filter_scenarios() -> tuple[ManifoldFilterScenario, ...]:
    """Small fixed ladder; intentionally not a dense threshold sweep."""

    return (
        ManifoldFilterScenario(
            "d4_sparse",
            anchor_distance_scale=4.0,
            knn_spacing_scale=1.0,
            note="distance > 4x central-Core q90 AND locally sparse",
        ),
        ManifoldFilterScenario(
            "d6_sparse",
            anchor_distance_scale=6.0,
            knn_spacing_scale=1.0,
            note="distance > 6x central-Core q90 AND locally sparse",
        ),
        ManifoldFilterScenario(
            "d8_sparse",
            anchor_distance_scale=8.0,
            knn_spacing_scale=1.0,
            note="distance > 8x central-Core q90 AND locally sparse",
        ),
        ManifoldFilterScenario(
            "d12_sparse",
            anchor_distance_scale=12.0,
            knn_spacing_scale=1.0,
            note="distance > 12x central-Core q90 AND locally sparse",
        ),
        ManifoldFilterScenario(
            "d4_sparse_linear",
            anchor_distance_scale=4.0,
            knn_spacing_scale=1.0,
            require_linearity=True,
            linearity_scale=1.0,
            note="4x manifold detachment + sparse + central-Core-q90 linearity guard",
        ),
    )


def _aligned(name: str, value: object, length: int, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (int(length),):
        raise ReconstructionValidationError(f"I2.5.3 {name} must align with Extended geometry.")
    return array


def manifold_rejection_mask(
    *,
    scenario: ManifoldFilterScenario,
    nearest_strong_anchor_distance_voxels: object,
    knn_median_distance_voxels: object,
    local_linearity: object,
    anchor_distance_reference_q90_voxels: float,
    knn_spacing_reference_q90_voxels: float,
    linearity_reference_q90: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve and evaluate one I2.5.3 candidate rule.

    Only local-manifold evidence is accepted.  In particular, no peripheral,
    far, extremum-driver, source-view, or camera-order input exists here.
    """

    if not isinstance(scenario, ManifoldFilterScenario):
        raise TypeError("scenario must be ManifoldFilterScenario.")
    distance = np.asarray(nearest_strong_anchor_distance_voxels, dtype=np.float64).reshape(-1)
    n = len(distance)
    if n == 0:
        raise ReconstructionValidationError("I2.5.3 requires retained Extended geometry.")
    knn = _aligned("KNN spacing", knn_median_distance_voxels, n, np.float64)
    linearity = _aligned("local linearity", local_linearity, n, np.float64)
    for name, values in (("anchor distance", distance), ("KNN spacing", knn), ("linearity", linearity)):
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ReconstructionValidationError(f"I2.5.3 {name} evidence must be finite and non-negative.")

    refs = {
        "anchor_distance": float(anchor_distance_reference_q90_voxels),
        "knn_spacing": float(knn_spacing_reference_q90_voxels),
        "linearity": float(linearity_reference_q90),
    }
    if any(not isfinite(v) or v <= 0.0 for v in refs.values()):
        raise ReconstructionValidationError("I2.5.3 reference thresholds must be finite and positive.")

    distance_limit = refs["anchor_distance"] * float(scenario.anchor_distance_scale)
    knn_limit = refs["knn_spacing"] * float(scenario.knn_spacing_scale)
    linearity_limit = refs["linearity"] * float(scenario.linearity_scale)

    reject = (distance > distance_limit) & (knn > knn_limit)
    if bool(scenario.require_linearity):
        reject &= linearity > linearity_limit

    resolved = {
        **scenario.as_dict(),
        "resolved_anchor_distance_limit_voxels": float(distance_limit),
        "resolved_knn_spacing_limit_voxels": float(knn_limit),
        "resolved_linearity_limit": float(linearity_limit) if scenario.require_linearity else None,
        "uses_peripheral_or_far_labels": False,
        "uses_extremum_driver_labels": False,
        "uses_source_view_blacklist": False,
        "uses_metric_distance": False,
        "rejected_points": int(np.count_nonzero(reject)),
        "rejected_fraction_of_extended": float(np.mean(reject)),
    }
    return reject.astype(bool), resolved


def manifold_effect_counts(
    reject_mask: object,
    *,
    is_core: object,
    is_peripheral: object,
    is_far_peripheral: object,
    is_weak_peripheral: object,
    is_extremum_driver: object,
    is_stubborn_extremum_survivor: object,
) -> dict[str, Any]:
    reject = np.asarray(reject_mask, dtype=bool).reshape(-1)
    masks = {
        "core": _aligned("Core", is_core, len(reject), bool),
        "peripheral": _aligned("peripheral", is_peripheral, len(reject), bool),
        "far": _aligned("far-peripheral", is_far_peripheral, len(reject), bool),
        "weak": _aligned("weak-peripheral", is_weak_peripheral, len(reject), bool),
        "drivers": _aligned("extremum-driver", is_extremum_driver, len(reject), bool),
        "stubborn": _aligned("stubborn-extremum", is_stubborn_extremum_survivor, len(reject), bool),
    }
    central_core = masks["core"] & ~masks["peripheral"]

    def removed(region: np.ndarray) -> tuple[int, float]:
        count = int(np.count_nonzero(region))
        hits = int(np.count_nonzero(reject & region))
        return hits, (hits / float(count) if count else 0.0)

    values: dict[str, Any] = {"rejected_points": int(np.count_nonzero(reject))}
    for key, region in (
        ("core", masks["core"]),
        ("central_core", central_core),
        ("peripheral", masks["peripheral"]),
        ("far_peripheral", masks["far"]),
        ("weak_peripheral", masks["weak"]),
        ("extremum_driver", masks["drivers"]),
        ("stubborn_extremum", masks["stubborn"]),
    ):
        hits, fraction = removed(region)
        values[key + "_rejected_points"] = hits
        values[key + "_rejection_fraction"] = fraction
    return values


def manifold_counterfactual_comparison(
    *,
    baseline_extended_points: int,
    baseline_core_points: int,
    baseline_full_diagonal: float,
    baseline_robust_diagonal: float,
    candidate_extended_points: int,
    candidate_core_points: int,
    candidate_full_diagonal: float,
    candidate_robust_diagonal: float,
    candidate_dominant_component_fraction: float,
    effects: Mapping[str, Any],
) -> dict[str, Any]:
    counts = (baseline_extended_points, baseline_core_points, candidate_extended_points, candidate_core_points)
    if any(int(v) <= 0 for v in counts):
        raise ReconstructionValidationError("I2.5.3 comparison point counts must be positive.")
    values = tuple(float(v) for v in (baseline_full_diagonal, baseline_robust_diagonal, candidate_full_diagonal, candidate_robust_diagonal))
    if any(not isfinite(v) or v <= 0.0 for v in values):
        raise ReconstructionValidationError("I2.5.3 envelope diagonals must be finite and positive.")
    dominant = float(candidate_dominant_component_fraction)
    if not isfinite(dominant) or not 0.0 <= dominant <= 1.0:
        raise ReconstructionValidationError("I2.5.3 dominant-component fraction must be in [0,1].")

    base_full, base_robust, cand_full, cand_robust = values
    base_ratio = base_full / base_robust
    cand_ratio = cand_full / cand_robust
    core_loss = 1.0 - int(candidate_core_points) / float(int(baseline_core_points))
    extended_loss = 1.0 - int(candidate_extended_points) / float(int(baseline_extended_points))
    full_shrink = 1.0 - cand_full / base_full
    robust_shrink = 1.0 - cand_robust / base_robust
    ratio_improvement = 1.0 - cand_ratio / base_ratio

    central_core_loss = float(effects["central_core_rejection_fraction"])
    peripheral = float(effects["peripheral_rejection_fraction"])
    far = float(effects["far_peripheral_rejection_fraction"])
    driver = float(effects["extremum_driver_rejection_fraction"])
    stubborn = float(effects["stubborn_extremum_rejection_fraction"])

    score = (
        2.50 * stubborn
        + 1.75 * driver
        + 1.50 * far
        + 1.00 * peripheral
        + 2.00 * ratio_improvement
        + 0.75 * full_shrink
        - 5.00 * max(core_loss, 0.0)
        - 6.00 * max(central_core_loss, 0.0)
        - 1.50 * max(extended_loss, 0.0)
        - 2.50 * max(robust_shrink - 0.03, 0.0)
    )
    balanced = bool(
        core_loss <= 0.02
        and central_core_loss <= 0.01
        and extended_loss <= 0.05
        and driver >= 0.50
        and stubborn >= 0.50
        and far >= 0.20
        and ratio_improvement > 0.0
        and robust_shrink <= 0.03
        and dominant >= 0.90
    )
    preferred = bool(
        core_loss <= 0.01
        and central_core_loss <= 0.005
        and extended_loss <= 0.03
        and driver >= 0.70
        and stubborn >= 0.65
        and far >= 0.30
        and ratio_improvement >= 0.05
        and robust_shrink <= 0.02
        and dominant >= 0.90
    )
    return {
        "core_loss_fraction": float(core_loss),
        "central_core_loss_fraction": float(central_core_loss),
        "extended_loss_fraction": float(extended_loss),
        "peripheral_reduction_fraction": peripheral,
        "far_peripheral_reduction_fraction": far,
        "extremum_driver_rejection_fraction": driver,
        "stubborn_extremum_rejection_fraction": stubborn,
        "baseline_full_diagonal": base_full,
        "candidate_full_diagonal": cand_full,
        "full_diagonal_shrink_fraction": float(full_shrink),
        "baseline_robust_diagonal": base_robust,
        "candidate_robust_diagonal": cand_robust,
        "robust_diagonal_shrink_fraction": float(robust_shrink),
        "baseline_full_to_robust_ratio": float(base_ratio),
        "candidate_full_to_robust_ratio": float(cand_ratio),
        "full_to_robust_ratio_improvement_fraction": float(ratio_improvement),
        "candidate_dominant_component_fraction": dominant,
        "diagnostic_score": float(score),
        "balanced_for_manual_review": balanced,
        "preferred_for_manual_review": preferred,
    }


def manifold_pareto_front(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Non-dominated candidate set for preservation + extremum removal + ratio."""
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
                and float(o["stubborn_extremum_rejection_fraction"]) >= float(c["stubborn_extremum_rejection_fraction"])
                and float(o["extremum_driver_rejection_fraction"]) >= float(c["extremum_driver_rejection_fraction"])
                and float(o["full_to_robust_ratio_improvement_fraction"]) >= float(c["full_to_robust_ratio_improvement_fraction"])
            )
            strictly = (
                float(o["core_loss_fraction"]) < float(c["core_loss_fraction"])
                or float(o["central_core_loss_fraction"]) < float(c["central_core_loss_fraction"])
                or float(o["stubborn_extremum_rejection_fraction"]) > float(c["stubborn_extremum_rejection_fraction"])
                or float(o["extremum_driver_rejection_fraction"]) > float(c["extremum_driver_rejection_fraction"])
                or float(o["full_to_robust_ratio_improvement_fraction"]) > float(c["full_to_robust_ratio_improvement_fraction"])
            )
            if no_worse and strictly:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["name"]))
    return tuple(result)
