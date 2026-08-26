"""I4.1.5 decoupled global/local sampling counterfactual.

Read-only planning and evidence scoring after I4.1.4 established that both
capture regime and local temporal density affect dense consistency.  This
module does not change production sampling.  It compares bounded candidate
plans while preserving a single authoritative global SfM frame.

The counterfactual explicitly distinguishes:
  * global SfM anchors: sparse/global coverage authority,
  * localized dense candidates: extra source observations that would need to
    be registered/localized into the existing SfM model,
  * final fusion views: still bounded by the existing production cap.

No threshold, SfM, DA3, fusion, filtering, persistence, metric, measurement, or
mesh policy is modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I4_1_5_SCHEMA_VERSION = 1
_EPS = 1e-9


@dataclass(frozen=True)
class CounterfactualPlan:
    name: str
    source_duration_seconds: float
    global_anchor_times_seconds: tuple[float, ...]
    dense_candidate_times_seconds: tuple[float, ...]
    final_fusion_cap: int
    dense_candidates_require_localization: bool
    note: str

    @property
    def global_anchor_count(self) -> int:
        return len(self.global_anchor_times_seconds)

    @property
    def dense_candidate_count(self) -> int:
        return len(self.dense_candidate_times_seconds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_duration_seconds": float(self.source_duration_seconds),
            "global_anchor_count": self.global_anchor_count,
            "dense_candidate_count": self.dense_candidate_count,
            "final_fusion_cap": int(self.final_fusion_cap),
            "dense_candidates_require_localization": bool(self.dense_candidates_require_localization),
            "global_anchor_times_seconds": [float(v) for v in self.global_anchor_times_seconds],
            "dense_candidate_times_seconds": [float(v) for v in self.dense_candidate_times_seconds],
            "note": self.note,
        }


def _validate_inputs(
    *,
    duration_seconds: float,
    global_anchor_cap: int,
    dense_candidate_cap: int,
    local_target_fps: float,
    final_fusion_cap: int,
) -> None:
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ReconstructionValidationError("I4.1.5 duration must be finite and positive.")
    if global_anchor_cap < 2:
        raise ReconstructionValidationError("I4.1.5 requires at least two global anchors.")
    if dense_candidate_cap < 2:
        raise ReconstructionValidationError("I4.1.5 requires at least two dense candidates.")
    if not isfinite(local_target_fps) or local_target_fps <= 0:
        raise ReconstructionValidationError("I4.1.5 local target FPS must be finite and positive.")
    if final_fusion_cap < 1 or final_fusion_cap > dense_candidate_cap:
        raise ReconstructionValidationError("I4.1.5 final fusion cap must be within the dense candidate budget.")


def uniform_times(duration_seconds: float, count: int) -> tuple[float, ...]:
    if not isfinite(duration_seconds) or duration_seconds <= 0 or count < 2:
        raise ReconstructionValidationError("I4.1.5 uniform schedule requires positive duration and count >= 2.")
    return tuple(float(v) for v in np.linspace(0.0, float(duration_seconds), int(count), dtype=np.float64))


def nominal_local_count(duration_seconds: float, target_fps: float) -> int:
    if not isfinite(duration_seconds) or duration_seconds <= 0 or not isfinite(target_fps) or target_fps <= 0:
        raise ReconstructionValidationError("I4.1.5 nominal local count inputs are invalid.")
    # Mirrors the intent of source-inclusive temporal sampling: first and last
    # source regions remain represented.
    return max(2, int(ceil(float(duration_seconds) * float(target_fps))))


def _dedupe_sorted(values: Sequence[float], *, tolerance_seconds: float = 1e-6) -> tuple[float, ...]:
    result: list[float] = []
    for value in sorted(float(v) for v in values):
        if not isfinite(value):
            raise ReconstructionValidationError("I4.1.5 candidate schedule contains a non-finite time.")
        if not result or abs(value - result[-1]) > tolerance_seconds:
            result.append(value)
    return tuple(result)


def _bounded_even_subset(values: Sequence[float], cap: int) -> tuple[float, ...]:
    data = _dedupe_sorted(values)
    if len(data) <= cap:
        return data
    indices = np.linspace(0, len(data) - 1, cap, dtype=np.float64)
    chosen = sorted({int(round(v)) for v in indices})
    # Rounding may collapse indices; fill deterministically if needed.
    if len(chosen) < cap:
        used = set(chosen)
        for idx in range(len(data)):
            if idx not in used:
                chosen.append(idx); used.add(idx)
                if len(chosen) == cap:
                    break
        chosen.sort()
    return tuple(data[idx] for idx in chosen[:cap])


def build_counterfactual_plans(
    *,
    duration_seconds: float,
    global_anchor_cap: int = 40,
    dense_candidate_cap: int = 120,
    local_target_fps: float = 2.0,
    final_fusion_cap: int = 40,
    local_neighbor_step_seconds: float | None = None,
) -> tuple[CounterfactualPlan, ...]:
    """Build bounded, non-authoritative sampling plans for comparison."""

    _validate_inputs(
        duration_seconds=duration_seconds,
        global_anchor_cap=global_anchor_cap,
        dense_candidate_cap=dense_candidate_cap,
        local_target_fps=local_target_fps,
        final_fusion_cap=final_fusion_cap,
    )
    duration = float(duration_seconds)
    anchors = uniform_times(duration, int(global_anchor_cap))
    step = 1.0 / float(local_target_fps) if local_neighbor_step_seconds is None else float(local_neighbor_step_seconds)
    if not isfinite(step) or step <= 0:
        raise ReconstructionValidationError("I4.1.5 local neighbor step must be finite and positive.")

    baseline = CounterfactualPlan(
        name="global_anchors_only",
        source_duration_seconds=duration,
        global_anchor_times_seconds=anchors,
        dense_candidate_times_seconds=anchors,
        final_fusion_cap=int(final_fusion_cap),
        dense_candidates_require_localization=False,
        note="Current one-set behavior: global anchors also serve as dense candidates.",
    )

    around: list[float] = []
    for anchor in anchors:
        around.extend((max(0.0, anchor - step), anchor, min(duration, anchor + step)))
    anchor_neighbors = CounterfactualPlan(
        name="global_anchors_plus_local_neighbors",
        source_duration_seconds=duration,
        global_anchor_times_seconds=anchors,
        dense_candidate_times_seconds=_bounded_even_subset(around, int(dense_candidate_cap)),
        final_fusion_cap=int(final_fusion_cap),
        dense_candidates_require_localization=True,
        note="Preserve global anchors; add bounded local temporal neighbors that would be localized into the same SfM frame.",
    )

    local_count = min(int(dense_candidate_cap), nominal_local_count(duration, float(local_target_fps)))
    uniform_local = CounterfactualPlan(
        name="uniform_localized_dense_pool",
        source_duration_seconds=duration,
        global_anchor_times_seconds=anchors,
        dense_candidate_times_seconds=uniform_times(duration, local_count),
        final_fusion_cap=int(final_fusion_cap),
        dense_candidates_require_localization=True,
        note="Use a separate full-span local-density pool; candidates would be localized into the authoritative global SfM model.",
    )
    return baseline, anchor_neighbors, uniform_local


def _gaps(values: Sequence[float]) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) < 2 or not np.isfinite(data).all():
        raise ReconstructionValidationError("I4.1.5 schedule must contain at least two finite times.")
    data = np.sort(data)
    gaps = np.diff(data)
    if np.any(gaps <= 0):
        raise ReconstructionValidationError("I4.1.5 schedule must be strictly increasing after deduplication.")
    return gaps


def plan_geometry_summary(plan: CounterfactualPlan, *, local_threshold_seconds: float = 0.75) -> dict[str, Any]:
    if local_threshold_seconds <= 0 or not isfinite(local_threshold_seconds):
        raise ReconstructionValidationError("I4.1.5 local threshold must be finite and positive.")
    dense = tuple(plan.dense_candidate_times_seconds)
    gaps = _gaps(dense)
    duration = float(plan.source_duration_seconds)
    source_span = float((max(dense) - min(dense)) / max(duration, _EPS))
    local_fraction = float(np.mean(gaps <= float(local_threshold_seconds) + 1e-12))
    # Mimic the existing local graph topology: up to three forward neighbours.
    pair_dts: list[float] = []
    for i in range(len(dense)):
        for gap in (1, 2, 3):
            j = i + gap
            if j < len(dense):
                pair_dts.append(float(dense[j] - dense[i]))
    pair_arr = np.asarray(pair_dts, dtype=np.float64)
    return {
        "name": plan.name,
        "global_anchor_count": plan.global_anchor_count,
        "dense_candidate_count": plan.dense_candidate_count,
        "final_fusion_cap": int(plan.final_fusion_cap),
        "source_span_fraction": source_span,
        "dense_adjacent_gap_seconds": {
            "p50": float(np.quantile(gaps, 0.50)),
            "p90": float(np.quantile(gaps, 0.90)),
            "max": float(np.max(gaps)),
        },
        "adjacent_local_fraction_le_threshold": local_fraction,
        "local_threshold_seconds": float(local_threshold_seconds),
        "candidate_graph_pair_count": len(pair_dts),
        "candidate_graph_delta_t_seconds": {
            "p50": float(np.quantile(pair_arr, 0.50)),
            "p90": float(np.quantile(pair_arr, 0.90)),
        },
        "dense_candidates_require_localization": bool(plan.dense_candidates_require_localization),
    }


def _bucket_probability(time_buckets: Mapping[str, Any], delta_t: float) -> float | None:
    if delta_t <= 0.75 + 1e-9:
        key = "le_0_75s"
    elif delta_t <= 1.50 + 1e-9:
        key = "gt_0_75_le_1_5s"
    elif delta_t <= 3.0 + 1e-9:
        key = "gt_1_5_le_3_0s"
    else:
        key = "gt_3_0s"
    record = time_buckets.get(key)
    if not isinstance(record, Mapping):
        return None
    value = record.get("pass_fraction")
    if value is None:
        return None
    p = float(value)
    if not isfinite(p) or p < 0 or p > 1:
        raise ReconstructionValidationError("I4.1.5 empirical pass fraction is invalid.")
    return p


def estimate_plan_pass_fraction(
    plan: CounterfactualPlan,
    *,
    evidence_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate local-graph pass rate from I4.1.4 temporal buckets.

    This is explicitly a counterfactual estimate, not a substitute for DA3 or
    dense-pair execution. Pairs whose temporal bucket has no empirical support
    remain unscored rather than being invented.
    """

    buckets = evidence_summary.get("time_buckets")
    if not isinstance(buckets, Mapping):
        raise ReconstructionValidationError("I4.1.5 evidence summary lacks I4.1.4 time buckets.")
    dense = tuple(plan.dense_candidate_times_seconds)
    probabilities: list[float] = []
    unscored = 0
    for i in range(len(dense)):
        for gap in (1, 2, 3):
            j = i + gap
            if j >= len(dense):
                continue
            p = _bucket_probability(buckets, float(dense[j] - dense[i]))
            if p is None:
                unscored += 1
            else:
                probabilities.append(p)
    total = len(probabilities) + unscored
    return {
        "estimated_pass_fraction": None if not probabilities else float(np.mean(probabilities)),
        "scored_pair_count": len(probabilities),
        "unscored_pair_count": int(unscored),
        "scored_fraction": 0.0 if total == 0 else float(len(probabilities) / total),
        "model": "I4.1.4 empirical temporal-bucket counterfactual",
        "actual_dense_execution": False,
    }


def _safe_pf(record: Mapping[str, Any]) -> float | None:
    value = record.get("estimated_pass_fraction")
    return None if value is None else float(value)


def build_i4_1_5_counterfactual(
    *,
    duration_seconds: float,
    i4_1_4_summary: Mapping[str, Any],
    global_anchor_cap: int = 40,
    dense_candidate_cap: int = 120,
    local_target_fps: float = 2.0,
    final_fusion_cap: int = 40,
) -> dict[str, Any]:
    diagnosis = i4_1_4_summary.get("diagnosis")
    if not isinstance(diagnosis, Mapping):
        raise ReconstructionValidationError("I4.1.5 requires an I4.1.4 diagnosis summary.")
    classification = str(diagnosis.get("classification", ""))
    if classification not in {
        "local_density_dominant",
        "mixed_capture_regime_and_local_density",
        "capture_regime_dominant_local_density_secondary",
        "capture_regime_dominant",
        "same_source_dense_consistency_poor_even_at_higher_local_density",
        "insufficient_or_mixed_evidence",
    }:
        raise ReconstructionValidationError("I4.1.5 received an unknown I4.1.4 classification.")

    current = i4_1_4_summary.get("current_full_span")
    prefix = i4_1_4_summary.get("same_source_dense_prefix")
    control = i4_1_4_summary.get("short_control")
    if not isinstance(current, Mapping) or not isinstance(prefix, Mapping):
        raise ReconstructionValidationError("I4.1.5 requires current and same-source-prefix I4.1.4 evidence.")

    plans = build_counterfactual_plans(
        duration_seconds=float(duration_seconds),
        global_anchor_cap=int(global_anchor_cap),
        dense_candidate_cap=int(dense_candidate_cap),
        local_target_fps=float(local_target_fps),
        final_fusion_cap=int(final_fusion_cap),
    )
    rows: list[dict[str, Any]] = []
    for plan in plans:
        geometry = plan_geometry_summary(plan)
        same_source_estimate = estimate_plan_pass_fraction(plan, evidence_summary=prefix)
        clean_estimate = None if not isinstance(control, Mapping) else estimate_plan_pass_fraction(plan, evidence_summary=control)
        rows.append({
            "plan": plan.as_dict(),
            "geometry": geometry,
            "same_source_horse_estimate": same_source_estimate,
            "clean_capture_estimate": clean_estimate,
        })

    baseline = rows[0]
    alternatives = rows[1:]
    baseline_local = float(baseline["geometry"]["adjacent_local_fraction_le_threshold"])
    viable: list[dict[str, Any]] = []
    for row in alternatives:
        local = float(row["geometry"]["adjacent_local_fraction_le_threshold"])
        same_pf = _safe_pf(row["same_source_horse_estimate"])
        base_pf = _safe_pf(baseline["same_source_horse_estimate"])
        bounded = int(row["plan"]["dense_candidate_count"]) <= int(dense_candidate_cap)
        full_span = float(row["geometry"]["source_span_fraction"]) >= 0.999
        locality_gain = local - baseline_local
        estimated_gain = None if same_pf is None or base_pf is None else same_pf - base_pf
        row["assessment"] = {
            "bounded": bounded,
            "full_span": full_span,
            "locality_gain": float(locality_gain),
            "same_source_estimated_pass_gain": estimated_gain,
            "counterfactual_candidate": bool(bounded and full_span and locality_gain >= 0.25 and (estimated_gain is None or estimated_gain >= 0.05)),
        }
        if row["assessment"]["counterfactual_candidate"]:
            viable.append(row)

    def rank_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
        est = row["assessment"].get("same_source_estimated_pass_gain")
        return (
            float(row["geometry"]["adjacent_local_fraction_le_threshold"]),
            -float(row["plan"]["dense_candidate_count"]),
            -1e9 if est is None else float(est),
        )

    recommended = max(viable, key=rank_key) if viable else None
    capture_signal = bool(diagnosis.get("capture_regime_signal", False))
    local_signal = bool(diagnosis.get("local_density_signal", False))
    architecture_candidate = recommended is not None and local_signal

    return {
        "phase": "I4.1.5",
        "schema_version": I4_1_5_SCHEMA_VERSION,
        "status": "counterfactual_ready",
        "inputs": {
            "source_duration_seconds": float(duration_seconds),
            "i4_1_4_classification": classification,
            "global_anchor_cap": int(global_anchor_cap),
            "dense_candidate_cap": int(dense_candidate_cap),
            "local_target_fps": float(local_target_fps),
            "final_fusion_cap": int(final_fusion_cap),
        },
        "plans": rows,
        "decision": {
            "decoupled_sampling_architecture_candidate": bool(architecture_candidate),
            "recommended_counterfactual_plan": None if recommended is None else str(recommended["plan"]["name"]),
            "local_density_signal": local_signal,
            "capture_regime_signal": capture_signal,
            "cleaner_static_capture_still_recommended": capture_signal,
            "production_sampling_integration_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "i4_1_6_authorized": bool(architecture_candidate),
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
            "next_action": (
                "If the bounded decoupled plan is a counterfactual candidate, implement I4.1.6 as an additive production integration using one authoritative SfM frame and explicit localization of extra dense candidates; otherwise keep I4.1.3 and move to a cleaner static orbit capture. Do not relax dense-consistency thresholds."
            ),
        },
        "architecture_guardrails": {
            "single_authoritative_sfm_frame": True,
            "global_full_span_sampling_preserved": True,
            "extra_dense_candidates_are_observations_not_second_map": True,
            "extra_dense_candidates_require_pose_localization": True,
            "final_fusion_cap_preserved": True,
            "short_video_production_behavior_changed": False,
        },
        "authority": {
            "production_code_changed": False,
            "sampling_policy_changed": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
    }
