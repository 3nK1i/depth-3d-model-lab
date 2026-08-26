"""Phase I2.3 balanced cross-segment counterfactual contracts.

I2.3 consumes the exact I2.2 results and forms only cross-segment two-view
counterfactuals: one early single-view suspect that strongly reduces far-tail
artifacts plus one late single-view suspect that improves the global spatial
envelope. Exact refusion remains delegated to the proven I2.2 engine.

The stage is diagnostic only. It never mutates the source persistent scene and
never promotes metric, measurement, or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError


I23_BALANCED_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BalancedCrossSegmentScenario:
    name: str
    early_order: int
    late_order: int

    def __post_init__(self) -> None:
        early = int(self.early_order)
        late = int(self.late_order)
        if early == late:
            raise ReconstructionValidationError("I2.3 cross-segment scenario requires two distinct views.")
        if early > late:
            early, late = late, early
        name = str(self.name).strip() or f"minus_{early}_{late}"
        object.__setattr__(self, "early_order", early)
        object.__setattr__(self, "late_order", late)
        object.__setattr__(self, "name", name)

    @property
    def excluded_orders(self) -> tuple[int, int]:
        return (self.early_order, self.late_order)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "early_order": self.early_order,
            "late_order": self.late_order,
            "excluded_orders": [self.early_order, self.late_order],
        }


def _single_records(i22_summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if str(i22_summary.get("status")) != "exact_counterfactuals_ready":
        raise ReconstructionValidationError("I2.3 requires an exact-counterfactuals-ready I2.2 summary.")
    scenarios = i22_summary.get("scenarios")
    if not isinstance(scenarios, list):
        raise ReconstructionValidationError("I2.3 requires I2.2 scenario records.")
    singles: list[Mapping[str, Any]] = []
    for item in scenarios:
        if not isinstance(item, Mapping):
            continue
        excluded = item.get("excluded_orders")
        comparison = item.get("comparison")
        if isinstance(excluded, list) and len(excluded) == 1 and isinstance(comparison, Mapping):
            singles.append(item)
    if len(singles) < 2:
        raise ReconstructionValidationError("I2.3 requires at least two exact single-view I2.2 counterfactuals.")
    return singles


def _temporal_split(selected_orders: Sequence[int]) -> float:
    orders = tuple(sorted({int(v) for v in selected_orders}))
    if len(orders) < 10:
        raise ReconstructionValidationError("I2.3 requires at least 10 selected baseline views.")
    return 0.5 * (float(orders[0]) + float(orders[-1]))


def derive_balanced_cross_segment_scenarios(
    i22_summary: Mapping[str, Any],
    selected_orders: Sequence[int],
    *,
    max_early: int = 3,
    max_late: int = 2,
    max_scenarios: int = 6,
) -> tuple[BalancedCrossSegmentScenario, ...]:
    """Select a tiny cross-segment grid from exact promising single-view results.

    Early candidates are ranked primarily by far-peripheral reduction. Late
    candidates are ranked primarily by improvement to the full/robust spatial
    envelope. Only I2.2 singles already marked promising are eligible.
    """
    if max_early <= 0 or max_late <= 0 or max_scenarios <= 0:
        raise ReconstructionValidationError("I2.3 candidate limits must be positive.")
    selected = {int(v) for v in selected_orders}
    split = _temporal_split(tuple(selected))
    singles = _single_records(i22_summary)
    eligible: list[Mapping[str, Any]] = []
    for item in singles:
        order = int(item["excluded_orders"][0])
        comp = item["comparison"]
        if order not in selected or not bool(comp.get("promising_for_manual_exact_review")):
            continue
        eligible.append(item)
    if len(eligible) < 2:
        raise ReconstructionValidationError("I2.3 needs at least two promising I2.2 single-view candidates.")

    early = [item for item in eligible if int(item["excluded_orders"][0]) <= split]
    late = [item for item in eligible if int(item["excluded_orders"][0]) > split]
    if not early or not late:
        raise ReconstructionValidationError("I2.3 requires promising single-view candidates on both temporal sides.")

    early.sort(
        key=lambda item: (
            float(item["comparison"].get("far_peripheral_reduction_fraction", 0.0)),
            float(item["comparison"].get("peripheral_reduction_fraction", 0.0)),
            -float(item["comparison"].get("core_loss_fraction", 1.0)),
        ),
        reverse=True,
    )
    late.sort(
        key=lambda item: (
            float(item["comparison"].get("full_to_robust_ratio_improvement_fraction", 0.0)),
            float(item["comparison"].get("peripheral_reduction_fraction", 0.0)),
            -float(item["comparison"].get("core_loss_fraction", 1.0)),
        ),
        reverse=True,
    )
    early = early[: int(max_early)]
    late = late[: int(max_late)]

    result: list[BalancedCrossSegmentScenario] = []
    for e in early:
        e_order = int(e["excluded_orders"][0])
        for l in late:
            l_order = int(l["excluded_orders"][0])
            a, b = sorted((e_order, l_order))
            result.append(BalancedCrossSegmentScenario(f"minus_{a}_{b}", a, b))
    result = result[: int(max_scenarios)]
    if not result:
        raise ReconstructionValidationError("I2.3 could not derive any cross-segment exact scenarios.")
    return tuple(result)


def single_by_order(i22_summary: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for item in _single_records(i22_summary):
        result[int(item["excluded_orders"][0])] = item
    return result


def evaluate_balanced_result(
    result: Mapping[str, Any],
    early_single: Mapping[str, Any],
    late_single: Mapping[str, Any],
    baseline_profile: Mapping[str, Any],
) -> dict[str, Any]:
    comp = result.get("comparison")
    profile = result.get("quality_profile")
    if not isinstance(comp, Mapping) or not isinstance(profile, Mapping):
        raise ReconstructionValidationError("I2.3 exact scenario is missing comparison/quality profile evidence.")
    ec = early_single.get("comparison")
    lc = late_single.get("comparison")
    if not isinstance(ec, Mapping) or not isinstance(lc, Mapping):
        raise ReconstructionValidationError("I2.3 requires exact I2.2 single-view comparison evidence.")

    core_loss = float(comp["core_loss_fraction"])
    extended_loss = float(comp["extended_loss_fraction"])
    peripheral = float(comp["peripheral_reduction_fraction"])
    far = float(comp["far_peripheral_reduction_fraction"])
    weak = float(comp["weak_peripheral_reduction_fraction"])
    ratio_improvement = float(comp["full_to_robust_ratio_improvement_fraction"])
    robust_change = float(comp["extended_robust_diagonal_change_fraction"])
    ratio = float(profile["extended_envelope"]["full_to_robust_diagonal_ratio"])
    dominant = float(profile["dominant_component_fraction_of_extended"])
    baseline_dominant = float(baseline_profile.get("dominant_component_fraction_of_extended", dominant))

    values = [core_loss, extended_loss, peripheral, far, weak, ratio_improvement, robust_change, ratio, dominant]
    if not all(isfinite(v) for v in values):
        raise ReconstructionValidationError("I2.3 comparison contains non-finite evidence.")

    early_core = float(ec["core_loss_fraction"])
    late_core = float(lc["core_loss_fraction"])
    best_single_far = max(float(ec["far_peripheral_reduction_fraction"]), float(lc["far_peripheral_reduction_fraction"]))
    best_single_ratio = max(
        float(ec["full_to_robust_ratio_improvement_fraction"]),
        float(lc["full_to_robust_ratio_improvement_fraction"]),
    )
    core_overlap_savings = early_core + late_core - core_loss
    far_gain_over_best_single = far - best_single_far
    ratio_gain_over_best_single = ratio_improvement - best_single_ratio

    balanced_score = (
        2.25 * far
        + 1.25 * peripheral
        + 0.50 * weak
        + 1.75 * ratio_improvement
        - 4.0 * max(core_loss, 0.0)
        - 1.25 * max(extended_loss, 0.0)
        - 0.75 * max(-robust_change - 0.05, 0.0)
    )
    balanced_candidate = bool(
        core_loss <= 0.075
        and extended_loss <= 0.10
        and peripheral >= 0.15
        and far >= 0.25
        and ratio_improvement > 0.0
        and robust_change >= -0.15
        and dominant >= max(0.90, baseline_dominant - 0.05)
    )
    return {
        "balanced_score": float(balanced_score),
        "balanced_candidate_for_manual_review": balanced_candidate,
        "core_overlap_savings_fraction": float(core_overlap_savings),
        "far_gain_over_best_single_fraction": float(far_gain_over_best_single),
        "ratio_gain_over_best_single_fraction": float(ratio_gain_over_best_single),
        "combined_core_loss_fraction": core_loss,
        "combined_extended_loss_fraction": extended_loss,
        "combined_peripheral_reduction_fraction": peripheral,
        "combined_far_peripheral_reduction_fraction": far,
        "combined_weak_peripheral_reduction_fraction": weak,
        "combined_full_to_robust_ratio": ratio,
        "combined_full_to_robust_ratio_improvement_fraction": ratio_improvement,
        "combined_extended_robust_diagonal_change_fraction": robust_change,
        "combined_dominant_component_fraction": dominant,
        "early_single": {
            "name": str(early_single["name"]),
            "core_loss_fraction": early_core,
            "far_peripheral_reduction_fraction": float(ec["far_peripheral_reduction_fraction"]),
            "full_to_robust_ratio_improvement_fraction": float(ec["full_to_robust_ratio_improvement_fraction"]),
        },
        "late_single": {
            "name": str(late_single["name"]),
            "core_loss_fraction": late_core,
            "far_peripheral_reduction_fraction": float(lc["far_peripheral_reduction_fraction"]),
            "full_to_robust_ratio_improvement_fraction": float(lc["full_to_robust_ratio_improvement_fraction"]),
        },
    }


def balanced_pareto_front(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Pareto front: minimize Core loss, maximize far/peripheral/envelope gains."""
    result: list[str] = []
    for candidate in records:
        c = candidate["balanced_evaluation"]
        dominated = False
        for other in records:
            if other is candidate:
                continue
            o = other["balanced_evaluation"]
            no_worse = (
                float(o["combined_core_loss_fraction"]) <= float(c["combined_core_loss_fraction"])
                and float(o["combined_far_peripheral_reduction_fraction"]) >= float(c["combined_far_peripheral_reduction_fraction"])
                and float(o["combined_peripheral_reduction_fraction"]) >= float(c["combined_peripheral_reduction_fraction"])
                and float(o["combined_full_to_robust_ratio_improvement_fraction"]) >= float(c["combined_full_to_robust_ratio_improvement_fraction"])
            )
            strictly_better = any((
                float(o["combined_core_loss_fraction"]) < float(c["combined_core_loss_fraction"]),
                float(o["combined_far_peripheral_reduction_fraction"]) > float(c["combined_far_peripheral_reduction_fraction"]),
                float(o["combined_peripheral_reduction_fraction"]) > float(c["combined_peripheral_reduction_fraction"]),
                float(o["combined_full_to_robust_ratio_improvement_fraction"]) > float(c["combined_full_to_robust_ratio_improvement_fraction"]),
            ))
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(str(candidate["name"]))
    return tuple(result)
