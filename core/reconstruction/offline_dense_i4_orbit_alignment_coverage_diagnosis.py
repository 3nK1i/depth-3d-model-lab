"""I4.1.1 object-orbit alignment and coverage diagnosis.

This milestone is diagnostic only.  It explains why an I4.1 object-orbit
baseline was rejected by the ordinary I2 policy, while still evaluating the
object-specific camera-orbit evidence that I4.1 intentionally could not reach
once that prerequisite failed.

No reconstruction geometry, SfM, DA3, dense qualification, filtering,
persistence, metric/measurement authority, or mesh state is changed here.
"""
from __future__ import annotations

from collections import Counter
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
    robust_object_target,
    select_camera_poses,
)

I4_1_1_SCHEMA_VERSION = 1

_I2_COVERAGE_GATE_NAMES = (
    "minimum_registered_fraction",
    "minimum_dense_alignment_fraction",
    "minimum_selected_fraction_of_registered",
)
_I2_GEOMETRY_CONTRACT_GATE_NAMES = (
    "minimum_largest_component_fraction",
    "minimum_core_fraction_of_supported",
    "minimum_extended_fraction_of_supported",
    "maximum_rejected_supported_fraction",
    "exact_observation_accounting",
    "persistence_self_contained",
    "authority_boundaries_preserved",
)
_I4_SELECTED_ORBIT_GATE_NAMES = (
    "selected_view_count",
    "selected_azimuth_coverage",
    "selected_azimuth_bins",
    "selected_opposed_view_diversity",
    "selected_target_facing",
    "selected_standoff_consistency",
)


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.1 {name} must be numeric.") from exc
    if not isfinite(number) or number <= 0.0:
        raise ReconstructionValidationError(f"I4.1.1 {name} must be finite and positive.")
    return number


def _contiguous_clusters(values: Sequence[int]) -> list[list[int]]:
    orders = sorted({int(v) for v in values})
    if not orders:
        return []
    clusters: list[list[int]] = [[orders[0]]]
    for value in orders[1:]:
        if value == clusters[-1][-1] + 1:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {
        "p10": float(np.quantile(data, 0.10)),
        "p25": float(np.quantile(data, 0.25)),
        "p50": float(np.quantile(data, 0.50)),
        "p75": float(np.quantile(data, 0.75)),
        "p90": float(np.quantile(data, 0.90)),
    }


def _candidate_alignment_diagnosis(candidate_views: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not candidate_views:
        raise ReconstructionValidationError("I4.1.1 requires candidate alignment evidence.")
    records: list[dict[str, Any]] = []
    seen_orders: set[int] = set()
    accepted_orders: list[int] = []
    rejected_orders: list[int] = []
    accepted_keyframes: list[str] = []
    rejected_keyframes: list[str] = []
    scales: list[tuple[int, float]] = []
    rejection_reasons: Counter[str] = Counter()

    for raw in candidate_views:
        if not isinstance(raw, Mapping):
            raise ReconstructionValidationError("I4.1.1 candidate view record must be an object.")
        order = int(raw.get("workspace_order", -1))
        if order < 0 or order in seen_orders:
            raise ReconstructionValidationError("I4.1.1 candidate workspace orders must be unique and non-negative.")
        seen_orders.add(order)
        keyframe_id = str(raw.get("keyframe_id", "")).strip()
        if not keyframe_id:
            raise ReconstructionValidationError(f"I4.1.1 candidate order {order} has no keyframe ID.")
        accepted = bool(raw.get("alignment_accepted"))
        reason = None if accepted else str(raw.get("rejection_code") or "unknown")
        scale = raw.get("scale_to_reconstruction_units") if accepted else None
        if accepted:
            accepted_orders.append(order)
            accepted_keyframes.append(keyframe_id)
            scales.append((order, _finite_positive(scale, name=f"accepted scale order {order}")))
        else:
            rejected_orders.append(order)
            rejected_keyframes.append(keyframe_id)
            rejection_reasons[str(reason)] += 1
        records.append({
            "workspace_order": order,
            "keyframe_id": keyframe_id,
            "accepted": accepted,
            "rejection_code": reason,
            "scale_to_reconstruction_units": None if scale is None else float(scale),
            "valid_correspondences": int(raw.get("valid_correspondences", 0)),
            "inlier_correspondences": int(raw.get("inlier_correspondences", 0)),
            "median_relative_residual": raw.get("median_relative_residual"),
            "p90_relative_residual": raw.get("p90_relative_residual"),
        })

    scales.sort()
    scale_values = [v for _, v in scales]
    adjacent: list[dict[str, Any]] = []
    for (order_a, scale_a), (order_b, scale_b) in zip(scales, scales[1:]):
        ratio = max(scale_a / scale_b, scale_b / scale_a)
        adjacent.append({
            "order_a": int(order_a), "order_b": int(order_b),
            "scale_a": float(scale_a), "scale_b": float(scale_b),
            "symmetric_ratio": float(ratio),
        })
    transitions = [item for item in adjacent if float(item["symmetric_ratio"]) >= 1.50]

    sorted_records = sorted(records, key=lambda item: int(item["workspace_order"]))
    chunks = np.array_split(np.asarray(sorted_records, dtype=object), min(4, len(sorted_records)))
    quartiles: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        values = list(chunk.tolist())
        count = len(values)
        accepted_count = sum(1 for item in values if bool(item["accepted"]))
        quartiles.append({
            "quartile": index,
            "orders": [int(item["workspace_order"]) for item in values],
            "candidate_count": count,
            "accepted_count": accepted_count,
            "accepted_fraction": accepted_count / float(count) if count else 0.0,
        })

    return {
        "candidate_count": len(records),
        "accepted_count": len(accepted_orders),
        "rejected_count": len(rejected_orders),
        "accepted_fraction": len(accepted_orders) / float(len(records)),
        "accepted_orders": sorted(accepted_orders),
        "rejected_orders": sorted(rejected_orders),
        "accepted_keyframe_ids": accepted_keyframes,
        "rejected_keyframe_ids": rejected_keyframes,
        "accepted_clusters": _contiguous_clusters(accepted_orders),
        "rejected_clusters": _contiguous_clusters(rejected_orders),
        "rejection_reason_histogram": dict(sorted(rejection_reasons.items())),
        "accepted_scale_quantiles": _quantiles(scale_values),
        "accepted_scale_min": None if not scale_values else float(min(scale_values)),
        "accepted_scale_max": None if not scale_values else float(max(scale_values)),
        "accepted_scale_max_to_min_ratio": None if not scale_values else float(max(scale_values) / min(scale_values)),
        "largest_adjacent_accepted_scale_ratio": None if not adjacent else float(max(item["symmetric_ratio"] for item in adjacent)),
        "large_adjacent_scale_transitions_ge_1_5x": transitions,
        "candidate_quartile_acceptance": quartiles,
        "records": sorted_records,
    }


def _gate_subset(gates: Mapping[str, Any], names: Sequence[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name in names:
        if name not in gates:
            raise ReconstructionValidationError(f"I4.1.1 missing required gate {name!r}.")
        result[name] = bool(gates[name])
    return result


def _classification(
    *,
    i2_geometry_contract_pass: bool,
    i4_registration_pass: bool,
    i4_alignment_pass: bool,
    selected_orbit_pass: bool,
    i2_selected_fraction_pass: bool,
) -> str:
    if not i2_geometry_contract_pass:
        return "ordinary_geometry_or_contract_quality_failure"
    if not i4_registration_pass:
        return "registration_coverage_insufficient"
    if not i4_alignment_pass and not selected_orbit_pass:
        return "mixed_alignment_and_selected_orbit_coverage_failure"
    if not i4_alignment_pass:
        return "alignment_coverage_insufficient"
    if not selected_orbit_pass:
        return "selected_orbit_coverage_insufficient"
    if not i2_selected_fraction_pass:
        return "room_scale_selected_fraction_policy_mismatch_candidate"
    return "no_unexplained_failure"


def build_i4_1_1_diagnosis(
    *,
    source_i4_summary: Mapping[str, Any],
    sfm_summary: Mapping[str, Any],
    i1_summary: Mapping[str, Any],
    i2_summary: Mapping[str, Any],
    e3_summary: Mapping[str, Any],
    registered_camera_ids: Sequence[str],
    registered_camera_matrices: object,
    core_points: object,
    policy: I4OrbitQualificationPolicy | None = None,
) -> dict[str, Any]:
    """Build an exact, non-mutating I4.1 failure diagnosis."""

    if str(source_i4_summary.get("phase")) != "I4.1" or str(source_i4_summary.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.1 requires a completed DEGRADED I4.1 campaign.")
    if str(source_i4_summary.get("failed_stage")) != "object_orbit_baseline_quality":
        raise ReconstructionValidationError("I4.1.1 is only valid for the I4.1 ordinary-baseline-quality stop.")
    if str(i1_summary.get("phase")) != "I1" or str(i1_summary.get("status")) != "pass" or not bool(i1_summary.get("pipeline_execution_complete")):
        raise ReconstructionValidationError("I4.1.1 requires a complete passing I1 baseline.")
    if str(i2_summary.get("phase")) != "I2" or bool(i2_summary.get("automated_pass")):
        raise ReconstructionValidationError("I4.1.1 requires the actual degraded I2 baseline-quality summary.")
    if str(e3_summary.get("phase")) != "E3" or not bool(e3_summary.get("pass")):
        raise ReconstructionValidationError("I4.1.1 requires the passing I1 qualified-fusion E3 adapter summary.")

    p = policy or I4OrbitQualificationPolicy()
    candidate_views = e3_summary.get("candidate_views")
    if not isinstance(candidate_views, list):
        raise ReconstructionValidationError("I4.1.1 E3 summary is missing candidate_views.")
    alignment = _candidate_alignment_diagnosis(candidate_views)

    selected_ids = tuple(str(v) for v in e3_summary.get("selected_keyframe_ids", []))
    if len(selected_ids) < 3:
        raise ReconstructionValidationError("I4.1.1 requires at least three selected qualified cameras.")
    camera_ids = tuple(str(v) for v in registered_camera_ids)
    poses = np.asarray(registered_camera_matrices, dtype=np.float64)
    if poses.shape != (len(camera_ids), 4, 4):
        raise ReconstructionValidationError("I4.1.1 registered camera IDs/poses are misaligned.")

    aligned_ids = tuple(str(v) for v in alignment["accepted_keyframe_ids"])
    _, aligned_poses = select_camera_poses(camera_ids, poses, aligned_ids)
    _, selected_poses = select_camera_poses(camera_ids, poses, selected_ids)
    target = robust_object_target(core_points)
    registered_metrics = orbit_coverage_metrics(
        poses, target,
        azimuth_bin_count=p.azimuth_bin_count,
        facing_angle_degrees=p.maximum_facing_angle_degrees,
    )
    aligned_metrics = orbit_coverage_metrics(
        aligned_poses, target,
        azimuth_bin_count=p.azimuth_bin_count,
        facing_angle_degrees=p.maximum_facing_angle_degrees,
    )
    selected_metrics = orbit_coverage_metrics(
        selected_poses, target,
        azimuth_bin_count=p.azimuth_bin_count,
        facing_angle_degrees=p.maximum_facing_angle_degrees,
    )

    input_views = int(sfm_summary.get("input_keyframes", 0))
    registered_views = int(i1_summary.get("sfm", {}).get("registered_view_count", 0))
    candidate_views_count = len(tuple(i1_summary.get("sfm", {}).get("candidate_dense_orders", [])))
    aligned_views = int(i1_summary.get("dense_qualification", {}).get("alignment_accepted_views", 0))
    selected_views = len(selected_ids)
    if min(input_views, registered_views, candidate_views_count, aligned_views, selected_views) <= 0:
        raise ReconstructionValidationError("I4.1.1 source view counts must be positive.")
    if aligned_views != int(alignment["accepted_count"]):
        raise ReconstructionValidationError("I4.1.1 accepted alignment count disagrees with candidate evidence.")

    registered_fraction = registered_views / float(input_views)
    alignment_fraction_of_registered = aligned_views / float(registered_views)
    alignment_fraction_of_candidates = aligned_views / float(candidate_views_count)
    selected_fraction_of_registered = selected_views / float(registered_views)

    i2_gates_raw = i2_summary.get("automated_gates")
    if not isinstance(i2_gates_raw, Mapping):
        raise ReconstructionValidationError("I4.1.1 I2 summary is missing automated_gates.")
    i2_coverage_gates = _gate_subset(i2_gates_raw, _I2_COVERAGE_GATE_NAMES)
    i2_geometry_contract_gates = _gate_subset(i2_gates_raw, _I2_GEOMETRY_CONTRACT_GATE_NAMES)
    i2_failed = sorted(name for name, passed in i2_gates_raw.items() if not bool(passed))

    i4_gates = {
        "registered_fraction": registered_fraction >= p.minimum_registered_fraction,
        "dense_alignment_fraction": alignment_fraction_of_registered >= p.minimum_dense_alignment_fraction,
        "selected_view_count": selected_views >= p.minimum_selected_views,
        "selected_azimuth_coverage": float(selected_metrics["azimuth_coverage_degrees"]) >= p.minimum_selected_azimuth_coverage_degrees,
        "selected_azimuth_bins": int(selected_metrics["occupied_azimuth_bins"]) >= p.minimum_selected_occupied_azimuth_bins,
        "selected_opposed_view_diversity": float(selected_metrics["maximum_opposed_view_angle_degrees"]) >= p.minimum_opposed_view_angle_degrees,
        "selected_target_facing": float(selected_metrics["target_facing_fraction"]) >= p.minimum_target_facing_fraction,
        "selected_standoff_consistency": float(selected_metrics["standoff_p90_to_p10_ratio"]) <= p.maximum_standoff_p90_to_p10_ratio,
    }
    selected_orbit_gates = _gate_subset(i4_gates, _I4_SELECTED_ORBIT_GATE_NAMES)

    i2_geometry_contract_pass = all(i2_geometry_contract_gates.values())
    selected_orbit_pass = all(selected_orbit_gates.values())
    i4_capture_without_i2_prereq_pass = all(i4_gates.values())
    classification = _classification(
        i2_geometry_contract_pass=i2_geometry_contract_pass,
        i4_registration_pass=bool(i4_gates["registered_fraction"]),
        i4_alignment_pass=bool(i4_gates["dense_alignment_fraction"]),
        selected_orbit_pass=selected_orbit_pass,
        i2_selected_fraction_pass=bool(i2_coverage_gates["minimum_selected_fraction_of_registered"]),
    )

    return {
        "phase": "I4.1.1",
        "schema_version": I4_1_1_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "diagnosis_complete": True,
        "source_i4_1": {
            "status": str(source_i4_summary.get("status")),
            "failed_stage": str(source_i4_summary.get("failed_stage")),
            "reason": str(source_i4_summary.get("reason", "")),
        },
        "counts": {
            "input_views": input_views,
            "registered_views": registered_views,
            "candidate_dense_views": candidate_views_count,
            "alignment_accepted_views": aligned_views,
            "selected_qualified_views": selected_views,
            "registered_fraction": registered_fraction,
            "alignment_fraction_of_registered_i4_semantics": alignment_fraction_of_registered,
            "alignment_fraction_of_candidates_i2_semantics": alignment_fraction_of_candidates,
            "selected_fraction_of_registered_i2_semantics": selected_fraction_of_registered,
            "alignment_denominator_semantics_match_for_this_scene": registered_views == candidate_views_count,
        },
        "i2_policy_diagnosis": {
            "automated_pass": False,
            "failed_gate_names": i2_failed,
            "coverage_gates": i2_coverage_gates,
            "geometry_and_contract_gates": i2_geometry_contract_gates,
            "geometry_and_contract_gates_pass": i2_geometry_contract_pass,
            "policy": dict(i2_summary.get("policy", {})),
        },
        "alignment_diagnosis": alignment,
        "orbit_diagnosis": {
            "target_reconstruction": [float(v) for v in target],
            "registered": registered_metrics,
            "aligned": aligned_metrics,
            "selected": selected_metrics,
            "current_i4_policy": p.as_dict(),
            "current_i4_gates_without_ordinary_i2_prerequisite": i4_gates,
            "selected_orbit_gate_subset": selected_orbit_gates,
            "selected_orbit_coverage_pass": selected_orbit_pass,
            "i4_capture_gates_without_i2_prerequisite_pass": i4_capture_without_i2_prereq_pass,
        },
        "policy_compatibility": {
            "ordinary_i2_full_policy_pass": False,
            "ordinary_i2_geometry_and_contract_only_pass": i2_geometry_contract_pass,
            "room_scale_selected_fraction_gate_pass": bool(i2_coverage_gates["minimum_selected_fraction_of_registered"]),
            "current_i4_registration_gate_pass": bool(i4_gates["registered_fraction"]),
            "current_i4_alignment_gate_pass": bool(i4_gates["dense_alignment_fraction"]),
            "selected_orbit_coverage_pass": selected_orbit_pass,
            "current_i4_capture_gates_without_i2_prerequisite_pass": i4_capture_without_i2_prereq_pass,
            "failure_classification": classification,
            "policy_reassessment_candidate": classification == "room_scale_selected_fraction_policy_mismatch_candidate",
            "policy_reassessment_authorized": False,
            "threshold_relaxation_authorized": False,
        },
        "authority": {
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i4_1_policy_change": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Inspect selected/aligned/registered orbit fixed views and alignment rejection clusters; do not relax thresholds."
            ),
        },
    }


__all__ = [
    "I4_1_1_SCHEMA_VERSION",
    "build_i4_1_1_diagnosis",
]
