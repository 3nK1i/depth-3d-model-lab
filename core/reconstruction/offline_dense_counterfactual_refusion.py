"""Phase I2.2 narrow exact counterfactual-refusion contracts.

I2.2 is a diagnostic experiment stage. It generates isolated exact refusions
from the already-aligned I1 source views while excluding only narrowly selected
suspect views/pairs identified by I2.1. Each scenario reruns voxel fusion and
formal F filtering so support, centroids, dispersion, topology, Core/Extended
classification, and spatial diagnostics are recomputed from scratch.

The source I1/G scene is never modified and no metric/measurement/mesh
authority is promoted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_quality import I2QualityPolicy, classify_peripheral_geometry, spatial_envelope


I22_COUNTERFACTUAL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CounterfactualScenario:
    name: str
    excluded_orders: tuple[int, ...]
    source: str

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        excluded = tuple(sorted({int(v) for v in self.excluded_orders}))
        source = str(self.source).strip()
        if not name or not source or not excluded:
            raise ReconstructionValidationError("I2.2 scenario requires name, source, and excluded orders.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "excluded_orders", excluded)
        object.__setattr__(self, "source", source)

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "excluded_orders": list(self.excluded_orders), "source": self.source}


def scenarios_from_i21(
    i21_summary: Mapping[str, Any],
    selected_orders: Sequence[int],
    *,
    top_single_views: int = 5,
    top_pairs: int = 5,
) -> tuple[CounterfactualScenario, ...]:
    selected = {int(v) for v in selected_orders}
    if len(selected) < 10:
        raise ReconstructionValidationError("I2.2 requires at least 10 selected baseline views.")
    findings = i21_summary.get("diagnostic_findings")
    pairs = i21_summary.get("top_support2_cosupport_pairs")
    if not isinstance(findings, Mapping) or not isinstance(pairs, list):
        raise ReconstructionValidationError("I2.2 requires I2.1 single-view and support-2 pair diagnostics.")
    singles = findings.get("top_single_view_dependencies")
    if not isinstance(singles, list):
        raise ReconstructionValidationError("I2.2 requires I2.1 ranked single-view dependencies.")

    scenarios: list[CounterfactualScenario] = []
    seen: set[tuple[int, ...]] = set()
    for item in singles[: max(0, int(top_single_views))]:
        if not isinstance(item, Mapping):
            continue
        order = int(item.get("workspace_order", -1))
        excluded = (order,)
        if order not in selected or excluded in seen:
            continue
        seen.add(excluded)
        scenarios.append(CounterfactualScenario(f"minus_{order}", excluded, "i2.1_top_single_view_dependency"))

    for item in pairs[: max(0, int(top_pairs))]:
        if not isinstance(item, Mapping):
            continue
        a = int(item.get("order_a", -1))
        b = int(item.get("order_b", -1))
        excluded = tuple(sorted((a, b)))
        if a == b or a not in selected or b not in selected or excluded in seen:
            continue
        seen.add(excluded)
        scenarios.append(CounterfactualScenario(f"minus_{a}_{b}", excluded, "i2.1_top_support2_cosupport_pair"))

    if not scenarios:
        raise ReconstructionValidationError("I2.2 could not derive any narrow counterfactual scenarios.")
    return tuple(scenarios)


def validate_scenario_against_baseline(scenario: CounterfactualScenario, selected_orders: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(int(v) for v in selected_orders)
    selected_set = set(selected)
    missing = [v for v in scenario.excluded_orders if v not in selected_set]
    if missing:
        raise ReconstructionValidationError(f"I2.2 scenario excludes non-selected orders: {missing}")
    remaining = tuple(v for v in selected if v not in set(scenario.excluded_orders))
    if len(remaining) < 10:
        raise ReconstructionValidationError("I2.2 exact F counterfactual must retain at least 10 views.")
    return remaining


def scenario_quality_profile(
    *,
    extended_points: object,
    core_points: object,
    support: object,
    confidence: object,
    dispersion: object,
    component_size: object,
    voxel_size: float,
    baseline_core_points: object,
    baseline_core_confidence: object,
    baseline_core_dispersion: object,
    policy: I2QualityPolicy,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    extended = np.asarray(extended_points, dtype=np.float64)
    core = np.asarray(core_points, dtype=np.float64)
    support_arr = np.asarray(support, dtype=np.uint32).reshape(-1)
    confidence_arr = np.asarray(confidence, dtype=np.float32).reshape(-1)
    dispersion_arr = np.asarray(dispersion, dtype=np.float32).reshape(-1)
    component_arr = np.asarray(component_size, dtype=np.int64).reshape(-1)
    if extended.ndim != 2 or extended.shape[1] != 3 or core.ndim != 2 or core.shape[1] != 3:
        raise ReconstructionValidationError("I2.2 scenario geometry must be Nx3.")
    if len(extended) == 0 or len(core) == 0 or not np.isfinite(extended).all() or not np.isfinite(core).all():
        raise ReconstructionValidationError("I2.2 scenario geometry must be finite and non-empty.")
    if not all(len(a) == len(extended) for a in (support_arr, confidence_arr, dispersion_arr, component_arr)):
        raise ReconstructionValidationError("I2.2 scenario Extended evidence arrays are not aligned.")
    voxel = float(voxel_size)
    if not isfinite(voxel) or voxel <= 0.0:
        raise ReconstructionValidationError("I2.2 voxel size must be finite and positive.")

    baseline_core = np.asarray(baseline_core_points, dtype=np.float64)
    base_conf = np.asarray(baseline_core_confidence, dtype=np.float64).reshape(-1)
    base_disp = np.asarray(baseline_core_dispersion, dtype=np.float64).reshape(-1)
    if baseline_core.ndim != 2 or baseline_core.shape[1] != 3 or len(baseline_core) == 0:
        raise ReconstructionValidationError("I2.2 requires a valid baseline Core anchor.")
    if len(base_conf) == 0 or len(base_disp) == 0:
        raise ReconstructionValidationError("I2.2 requires baseline Core confidence/dispersion evidence.")

    # Native profile shows what I2 would see if this scenario became the scene.
    native_peripheral, native_far, _, native_core_env = classify_peripheral_geometry(
        extended, core, voxel_size=voxel, policy=policy
    )
    # Baseline-anchor profile keeps the spatial yardstick fixed across scenarios.
    baseline_peripheral, baseline_far, sector, baseline_env = classify_peripheral_geometry(
        extended, baseline_core, voxel_size=voxel, policy=policy
    )
    core_conf_p25 = float(np.quantile(base_conf[np.isfinite(base_conf)], 0.25))
    core_disp_p90 = float(np.quantile(base_disp[np.isfinite(base_disp)], 0.90))
    baseline_weak = baseline_peripheral & (
        (support_arr <= 2) | (confidence_arr < core_conf_p25) | (dispersion_arr > core_disp_p90)
    )
    ext_env = spatial_envelope(
        extended,
        lower_quantile=policy.robust_lower_quantile,
        upper_quantile=policy.robust_upper_quantile,
    )
    largest_component = int(component_arr.max(initial=0))
    dominant_fraction = float(np.mean(component_arr == largest_component)) if largest_component > 0 else 0.0

    profile = {
        "extended_points": int(len(extended)),
        "core_points": int(len(core)),
        "baseline_anchor": {
            "peripheral_points": int(np.count_nonzero(baseline_peripheral)),
            "far_peripheral_points": int(np.count_nonzero(baseline_far)),
            "weak_peripheral_points": int(np.count_nonzero(baseline_weak)),
            "peripheral_fraction": float(np.mean(baseline_peripheral)),
            "far_fraction": float(np.mean(baseline_far)),
            "weak_fraction": float(np.mean(baseline_weak)),
            "anchor_robust_diagonal": float(baseline_env.robust_diagonal),
        },
        "native": {
            "peripheral_points": int(np.count_nonzero(native_peripheral)),
            "far_peripheral_points": int(np.count_nonzero(native_far)),
            "core_robust_diagonal": float(native_core_env.robust_diagonal),
        },
        "extended_envelope": ext_env.as_dict(),
        "dominant_component_fraction_of_extended": dominant_fraction,
        "baseline_core_confidence_p25": core_conf_p25,
        "baseline_core_dispersion_p90": core_disp_p90,
    }
    masks = {
        "peripheral": baseline_peripheral,
        "far": baseline_far,
        "weak": baseline_weak,
        "sector": sector,
    }
    return profile, masks


def comparison_against_baseline(baseline: Mapping[str, Any], scenario: Mapping[str, Any]) -> dict[str, Any]:
    def _ratio_loss(name: str) -> float:
        base = float(baseline[name])
        current = float(scenario[name])
        return 1.0 - current / base if base > 0.0 else 0.0

    def _reduction(name: str) -> float:
        base = float(baseline["baseline_anchor"][name])
        current = float(scenario["baseline_anchor"][name])
        return 1.0 - current / base if base > 0.0 else 0.0

    baseline_ratio = float(baseline["extended_envelope"]["full_to_robust_diagonal_ratio"] or 0.0)
    scenario_ratio = float(scenario["extended_envelope"]["full_to_robust_diagonal_ratio"] or 0.0)
    ratio_improvement = 1.0 - scenario_ratio / baseline_ratio if baseline_ratio > 0.0 else 0.0
    base_robust = float(baseline["extended_envelope"]["robust_diagonal"])
    scenario_robust = float(scenario["extended_envelope"]["robust_diagonal"])
    robust_change = scenario_robust / base_robust - 1.0 if base_robust > 0.0 else 0.0

    core_loss = _ratio_loss("core_points")
    extended_loss = _ratio_loss("extended_points")
    peripheral_reduction = _reduction("peripheral_points")
    far_reduction = _reduction("far_peripheral_points")
    weak_reduction = _reduction("weak_peripheral_points")
    diagnostic_score = (
        2.0 * far_reduction
        + 1.0 * peripheral_reduction
        + 0.5 * weak_reduction
        + 0.5 * ratio_improvement
        - 3.0 * max(core_loss, 0.0)
        - 1.0 * max(extended_loss, 0.0)
        - 0.75 * max(-robust_change - 0.05, 0.0)
    )
    promising = bool(
        core_loss <= 0.05
        and extended_loss <= 0.10
        and robust_change >= -0.15
        and (far_reduction >= 0.15 or peripheral_reduction >= 0.10)
    )
    return {
        "core_loss_fraction": float(core_loss),
        "extended_loss_fraction": float(extended_loss),
        "peripheral_reduction_fraction": float(peripheral_reduction),
        "far_peripheral_reduction_fraction": float(far_reduction),
        "weak_peripheral_reduction_fraction": float(weak_reduction),
        "full_to_robust_ratio_improvement_fraction": float(ratio_improvement),
        "extended_robust_diagonal_change_fraction": float(robust_change),
        "diagnostic_score": float(diagnostic_score),
        "promising_for_manual_exact_review": promising,
    }


def pareto_front(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Return non-dominated scenario names for artifact reduction vs Core loss.

    We minimize Core loss and maximize far/peripheral reduction. This is a
    diagnostic Pareto front only; it does not authorize removing any views.
    """
    result: list[str] = []
    for candidate in records:
        name = str(candidate["name"])
        c = candidate["comparison"]
        dominated = False
        for other in records:
            if other is candidate:
                continue
            o = other["comparison"]
            no_worse = (
                float(o["core_loss_fraction"]) <= float(c["core_loss_fraction"])
                and float(o["far_peripheral_reduction_fraction"]) >= float(c["far_peripheral_reduction_fraction"])
                and float(o["peripheral_reduction_fraction"]) >= float(c["peripheral_reduction_fraction"])
            )
            strictly_better = (
                float(o["core_loss_fraction"]) < float(c["core_loss_fraction"])
                or float(o["far_peripheral_reduction_fraction"]) > float(c["far_peripheral_reduction_fraction"])
                or float(o["peripheral_reduction_fraction"]) > float(c["peripheral_reduction_fraction"])
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(name)
    return tuple(result)
