"""I4.1.7A lightweight diagnosis for final clean-static orbit evidence.

This module reuses already-produced I4.1.7 camera poses and retained Core
geometry.  It deliberately performs no SfM, DA3, fusion, filtering,
persistence, production binding, metric-scale promotion, measurement
authorization, or mesh work.

The goal is to distinguish a genuinely narrow reconstructed camera arc from a
provisional-target/plane-estimation collapse before any threshold or capture
change is considered.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import I417QualificationPolicy
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (
    coverage_by_plane_mode,
    optical_axis_consensus_target,
    trajectory_centered_coverage,
    trajectory_circle_target,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics, robust_object_target

I4_1_7A_SCHEMA_VERSION = 1
_EPS = 1e-9


def _matrices(value: object, *, name: str) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) < 3:
        raise ReconstructionValidationError(f"I4.1.7A {name} requires at least three 4x4 camera poses.")
    if not np.isfinite(matrices).all():
        raise ReconstructionValidationError(f"I4.1.7A {name} camera poses must be finite.")
    return matrices


def _target(value: object, *, name: str) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64).reshape(-1)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ReconstructionValidationError(f"I4.1.7A {name} target must be a finite 3-vector.")
    return target


def _core_target(core_points: object | None) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    if core_points is None:
        return None, None
    points = np.asarray(core_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 64 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.1.7A retained Core points must be finite Nx3 with at least 64 points.")
    target = robust_object_target(points)
    return target, {
        "source": "I1 retained Core geometry robust_object_target",
        "core_point_count": int(len(points)),
        "production_target_authority": False,
        "i4_1_8_recompute_from_retained_core_required": True,
    }


def _target_evaluation(
    matrices: np.ndarray,
    target: np.ndarray,
    *,
    policy: I417QualificationPolicy,
) -> dict[str, Any]:
    current = orbit_coverage_metrics(
        matrices,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    modes = coverage_by_plane_mode(
        matrices,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    azimuths = [float(item["azimuth_coverage_degrees"]) for item in modes.values()]
    bins = [int(item["occupied_azimuth_bins"]) for item in modes.values()]
    opposed = [float(item["maximum_opposed_view_angle_degrees"]) for item in modes.values()]
    robust_geometry_pass = bool(
        min(azimuths) >= float(policy.minimum_selected_azimuth_coverage_degrees)
        and min(bins) >= int(policy.minimum_selected_occupied_azimuth_bins)
        and min(opposed) >= float(policy.minimum_opposed_view_angle_degrees)
    )
    return {
        "current_metric": current,
        "plane_modes": modes,
        "azimuth_coverage_min_degrees": float(min(azimuths)),
        "azimuth_coverage_max_degrees": float(max(azimuths)),
        "occupied_azimuth_bins_min": int(min(bins)),
        "occupied_azimuth_bins_max": int(max(bins)),
        "opposed_view_angle_min_degrees": float(min(opposed)),
        "opposed_view_angle_max_degrees": float(max(opposed)),
        "robust_locked_orbit_geometry_pass": robust_geometry_pass,
    }


def _reproduction_exact(source_metrics: Mapping[str, Any], recomputed: Mapping[str, Any]) -> bool:
    try:
        return bool(
            np.isclose(
                float(source_metrics["azimuth_coverage_degrees"]),
                float(recomputed["azimuth_coverage_degrees"]),
                atol=_EPS,
                rtol=0.0,
            )
            and int(source_metrics["occupied_azimuth_bins"]) == int(recomputed["occupied_azimuth_bins"])
            and np.isclose(
                float(source_metrics["maximum_opposed_view_angle_degrees"]),
                float(recomputed["maximum_opposed_view_angle_degrees"]),
                atol=_EPS,
                rtol=0.0,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def build_i4_1_7a_orbit_geometry_diagnosis(
    *,
    source_i4_1_7_summary: Mapping[str, Any],
    registered_camera_matrices: object,
    selected_camera_matrices: object,
    provisional_target_reconstruction: object,
    retained_core_points: object | None = None,
    policy: I417QualificationPolicy | None = None,
) -> dict[str, Any]:
    """Diagnose final I4.1.7 orbit evidence without changing policy."""

    if not isinstance(source_i4_1_7_summary, Mapping):
        raise ReconstructionValidationError("I4.1.7A requires an I4.1.7 summary mapping.")
    if str(source_i4_1_7_summary.get("phase")) != "I4.1.7":
        raise ReconstructionValidationError("I4.1.7A requires I4.1.7 evidence.")
    if str(source_i4_1_7_summary.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.7A is only for a completed DEGRADED I4.1.7 campaign.")
    assessment = source_i4_1_7_summary.get("assessment")
    if not isinstance(assessment, Mapping):
        raise ReconstructionValidationError("I4.1.7A requires completed I4.1.7 assessment evidence.")
    source_selected = assessment.get("selected_orbit")
    if not isinstance(source_selected, Mapping):
        raise ReconstructionValidationError("I4.1.7A source selected-orbit evidence is missing.")

    p = policy or I417QualificationPolicy()
    registered = _matrices(registered_camera_matrices, name="registered")
    selected = _matrices(selected_camera_matrices, name="selected")
    provisional = _target(provisional_target_reconstruction, name="provisional")

    recomputed_optical, optical_diag = optical_axis_consensus_target(registered)
    circle_target, circle_diag = trajectory_circle_target(registered)
    core_target, core_diag = _core_target(retained_core_points)

    targets: dict[str, dict[str, Any]] = {
        "provisional_i4_1_7": {
            "target_reconstruction": [float(v) for v in provisional],
            "diagnostics": {"source": "saved I4.1.7 provisional target"},
        },
        "recomputed_optical_axis_consensus": {
            "target_reconstruction": [float(v) for v in recomputed_optical],
            "diagnostics": dict(optical_diag),
        },
        "registered_trajectory_circle_center": {
            "target_reconstruction": [float(v) for v in circle_target],
            "diagnostics": dict(circle_diag),
        },
    }
    if core_target is not None:
        targets["retained_core_target"] = {
            "target_reconstruction": [float(v) for v in core_target],
            "diagnostics": dict(core_diag or {}),
        }

    evaluations: dict[str, dict[str, Any]] = {
        "registered": {
            "camera_count": int(len(registered)),
            "trajectory_centered": trajectory_centered_coverage(
                registered, azimuth_bin_count=int(p.azimuth_bin_count)
            ),
            "targets": {},
        },
        "selected": {
            "camera_count": int(len(selected)),
            "trajectory_centered": trajectory_centered_coverage(
                selected, azimuth_bin_count=int(p.azimuth_bin_count)
            ),
            "targets": {},
        },
    }
    for set_name, matrices in (("registered", registered), ("selected", selected)):
        for target_name, record in targets.items():
            target = np.asarray(record["target_reconstruction"], dtype=np.float64)
            evaluations[set_name]["targets"][target_name] = _target_evaluation(matrices, target, policy=p)

    current_selected = evaluations["selected"]["targets"]["provisional_i4_1_7"]["current_metric"]
    reproduction_exact = _reproduction_exact(source_selected, current_selected)
    current_geometry_failures: list[str] = []
    if float(current_selected["azimuth_coverage_degrees"]) < float(p.minimum_selected_azimuth_coverage_degrees):
        current_geometry_failures.append("selected_azimuth_coverage")
    if int(current_selected["occupied_azimuth_bins"]) < int(p.minimum_selected_occupied_azimuth_bins):
        current_geometry_failures.append("selected_azimuth_bins")
    if float(current_selected["maximum_opposed_view_angle_degrees"]) < float(p.minimum_opposed_view_angle_degrees):
        current_geometry_failures.append("selected_opposed_view_diversity")

    alternative_targets = [name for name in targets if name != "provisional_i4_1_7"]
    robust_alternative_targets = [
        name
        for name in alternative_targets
        if bool(evaluations["selected"]["targets"][name]["robust_locked_orbit_geometry_pass"])
    ]
    trajectory = evaluations["selected"]["trajectory_centered"]
    trajectory_broad = bool(
        float(trajectory["azimuth_coverage_degrees"]) >= float(p.minimum_selected_azimuth_coverage_degrees)
        and int(trajectory["occupied_azimuth_bins"]) >= int(p.minimum_selected_occupied_azimuth_bins)
    )

    if not reproduction_exact:
        classification = "saved_evidence_reproduction_mismatch"
    elif current_geometry_failures and robust_alternative_targets:
        classification = "provisional_target_collapse_supported"
    elif current_geometry_failures:
        classification = "no_alternative_target_recovers_locked_orbit_geometry"
    else:
        classification = "current_provisional_orbit_geometry_already_satisfies_locked_gates"

    correction_candidate = bool(classification == "provisional_target_collapse_supported")
    if classification == "provisional_target_collapse_supported":
        next_action = (
            "Review a bounded I4.1.7 orbit-target correction using the invariant alternative-target evidence; "
            "do not relax azimuth/bin/opposed thresholds and do not authorize I4.1.8 yet."
        )
    elif classification == "no_alternative_target_recovers_locked_orbit_geometry":
        next_action = (
            "The tested saved-pose targets do not recover the locked orbit geometry; prefer one wider capture unless new target evidence is introduced. "
            "No orbit-threshold relaxation is authorized."
        )
    else:
        next_action = (
            "Inspect the diagnostic evidence before any code or capture change; no threshold or production-policy change is authorized."
        )

    return {
        "phase": "I4.1.7A",
        "schema_version": I4_1_7A_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "policy_reference": p.as_dict(),
        "source_i4_1_7": {
            "status": str(source_i4_1_7_summary.get("status")),
            "qualification_plan": str(source_i4_1_7_summary.get("qualification_plan", "")),
        },
        "counts": {
            "registered_cameras": int(len(registered)),
            "selected_cameras": int(len(selected)),
            "retained_core_points": None if retained_core_points is None else int(len(np.asarray(retained_core_points))),
        },
        "targets": targets,
        "camera_set_evaluations": evaluations,
        "current_i4_1_7_metric_reproduction_exact": reproduction_exact,
        "diagnosis": {
            "classification": classification,
            "current_locked_geometry_gate_failures": current_geometry_failures,
            "selected_trajectory_centered_broad": trajectory_broad,
            "robust_alternative_targets": robust_alternative_targets,
            "orbit_target_correction_candidate": correction_candidate,
        },
        "authority": {
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "production_binding_executed": False,
            "orbit_metric_policy_changed": False,
            "threshold_relaxation": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_orbit_metric_change": False,
            "authorizes_orbit_threshold_relaxation": False,
            "authorizes_i4_1_8_production_binding": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": next_action,
        },
    }


__all__ = [
    "I4_1_7A_SCHEMA_VERSION",
    "build_i4_1_7a_orbit_geometry_diagnosis",
]
