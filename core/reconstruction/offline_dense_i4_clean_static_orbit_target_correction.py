"""I4.1.7B bounded provisional orbit-target correction.

This sub-step consumes completed I4.1.7 + I4.1.7A evidence only.  It may
replace the *provisional qualification target* with a registered-camera
trajectory-circle center only when I4.1.7A reproduced the original metric,
diagnosed optical-axis target collapse, and the trajectory-circle target
robustly satisfies the unchanged locked orbit geometry across invariant plane
constructions.

The correction is deliberately narrow: no SfM/DA3/dense/fusion/filtering/
persistence rerun, no production target authority, no metric-scale promotion,
no threshold relaxation, and no I4.2/mesh authorization.  I4.1.8 must still
recompute the final target from retained Core geometry before final I4.1
qualification.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (
    I417QualificationPolicy,
    build_clean_static_orbit_assessment,
)
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import (
    trajectory_circle_target,
)

I4_1_7B_SCHEMA_VERSION = 1
_TARGET_MODE = "provisional_registered_trajectory_circle_center"
_ALLOWED_SOURCE_FALSE_GATES = frozenset(
    {
        "real_dense_plan_improves_over_global_only",
        "selected_azimuth_coverage",
        "selected_azimuth_bins",
        "selected_opposed_view_diversity",
        "selected_target_facing",
        "selected_standoff_consistency",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.1.7B {name} must be a mapping.")
    return value


def _matrices(value: object, *, name: str) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or len(matrices) < 3:
        raise ReconstructionValidationError(f"I4.1.7B {name} requires at least three 4x4 camera poses.")
    if not np.isfinite(matrices).all():
        raise ReconstructionValidationError(f"I4.1.7B {name} camera poses must be finite.")
    return matrices


def _source_failed_gates(source_assessment: Mapping[str, Any]) -> tuple[str, ...]:
    gates = _mapping(source_assessment.get("gates"), name="source assessment gates")
    return tuple(sorted(str(name) for name, value in gates.items() if not bool(value)))


def build_i4_1_7b_bounded_target_correction(
    *,
    source_i4_1_7_summary: Mapping[str, Any],
    source_i4_1_7a_summary: Mapping[str, Any],
    registered_camera_matrices: object,
    selected_camera_matrices: object,
    plan_payload: Mapping[str, Any],
    baseline_payload: Mapping[str, Any],
    selected_source_indices: Sequence[int],
    selected_pose_provenance: Sequence[str],
    policy: I417QualificationPolicy | None = None,
) -> dict[str, Any]:
    """Requalify saved evidence with a bounded provisional trajectory target.

    The target correction is eligible only for the single diagnosed collapse
    case.  Unrelated source failures remain blocking and the locked I4.1.7
    policy is reused unchanged.
    """

    source = _mapping(source_i4_1_7_summary, name="source I4.1.7 summary")
    diag_source = _mapping(source_i4_1_7a_summary, name="source I4.1.7A summary")
    if str(source.get("phase")) != "I4.1.7" or str(source.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.7B requires a completed DEGRADED I4.1.7 summary.")
    if str(diag_source.get("phase")) != "I4.1.7A" or str(diag_source.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("I4.1.7B requires completed I4.1.7A diagnosis evidence.")

    source_assessment = _mapping(source.get("assessment"), name="source I4.1.7 assessment")
    source_target = _mapping(source_assessment.get("target"), name="source provisional target")
    source_gates = _mapping(source_assessment.get("gates"), name="source gates")
    source_counts = _mapping(source_assessment.get("counts"), name="source counts")

    registered = _matrices(registered_camera_matrices, name="registered")
    selected = _matrices(selected_camera_matrices, name="selected")
    provisional = np.asarray(source_target.get("target_reconstruction"), dtype=np.float64).reshape(-1)
    if provisional.shape != (3,) or not np.isfinite(provisional).all():
        raise ReconstructionValidationError("I4.1.7B source provisional target must be a finite 3-vector.")

    p = policy or I417QualificationPolicy()
    recomputed_diag = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=registered,
        selected_camera_matrices=selected,
        provisional_target_reconstruction=provisional,
        retained_core_points=None,
        policy=p,
    )
    source_diagnosis = _mapping(diag_source.get("diagnosis"), name="I4.1.7A diagnosis")
    recomputed_diagnosis = _mapping(recomputed_diag.get("diagnosis"), name="recomputed I4.1.7A diagnosis")

    circle_target, circle_diag = trajectory_circle_target(registered)
    registered_eval = _mapping(
        _mapping(recomputed_diag["camera_set_evaluations"], name="recomputed camera evaluations")["registered"],
        name="registered evaluation",
    )
    selected_eval = _mapping(
        _mapping(recomputed_diag["camera_set_evaluations"], name="recomputed camera evaluations")["selected"],
        name="selected evaluation",
    )
    registered_circle = _mapping(
        _mapping(registered_eval.get("targets"), name="registered target evaluations").get(
            "registered_trajectory_circle_center"
        ),
        name="registered trajectory-circle evaluation",
    )
    selected_circle = _mapping(
        _mapping(selected_eval.get("targets"), name="selected target evaluations").get(
            "registered_trajectory_circle_center"
        ),
        name="selected trajectory-circle evaluation",
    )

    source_failed = _source_failed_gates(source_assessment)
    unrelated_source_failures = tuple(name for name in source_failed if name not in _ALLOWED_SOURCE_FALSE_GATES)
    source_diag_classification = str(source_diagnosis.get("classification", ""))
    recomputed_classification = str(recomputed_diagnosis.get("classification", ""))
    source_robust_alternatives = tuple(str(v) for v in source_diagnosis.get("robust_alternative_targets", []))

    preconditions = {
        "source_metric_reproduction_exact": bool(recomputed_diag.get("current_i4_1_7_metric_reproduction_exact")),
        "source_a_metric_reproduction_exact": bool(diag_source.get("current_i4_1_7_metric_reproduction_exact")),
        "source_a_classifies_provisional_target_collapse": source_diag_classification
        == "provisional_target_collapse_supported",
        "recomputed_classifies_provisional_target_collapse": recomputed_classification
        == "provisional_target_collapse_supported",
        "trajectory_circle_listed_as_source_robust_alternative": "registered_trajectory_circle_center"
        in source_robust_alternatives,
        "registered_trajectory_circle_geometry_robust": bool(
            registered_circle.get("robust_locked_orbit_geometry_pass")
        ),
        "selected_trajectory_circle_geometry_robust": bool(selected_circle.get("robust_locked_orbit_geometry_pass")),
        "selected_trajectory_centered_broad": bool(recomputed_diagnosis.get("selected_trajectory_centered_broad")),
        "no_unrelated_source_gate_failures": len(unrelated_source_failures) == 0,
        "source_static_capture_gate_passed": bool(source_gates.get("capture_confirmed_static")),
        "source_prior_guardrails_passed": bool(source_gates.get("prior_decoupled_guardrails")),
    }
    correction_preconditions_pass = all(bool(value) for value in preconditions.values())

    assessment = build_clean_static_orbit_assessment(
        input_global_views=int(source_counts.get("input_global_views", 0)),
        registered_global_views=int(source_counts.get("registered_global_views", 0)),
        registered_camera_matrices=registered,
        selected_camera_matrices=selected,
        selected_source_indices=selected_source_indices,
        selected_pose_provenance=selected_pose_provenance,
        provisional_target_reconstruction=circle_target,
        target_diagnostics={
            **dict(circle_diag),
            "source": "registered authoritative SfM camera-center trajectory circle fit",
            "bounded_fallback_reason": "I4.1.7A provisional optical-axis target collapse",
        },
        provisional_target_mode=_TARGET_MODE,
        target_correction_evidence={
            "source_i4_1_7a_classification": source_diag_classification,
            "recomputed_i4_1_7a_classification": recomputed_classification,
            "source_failed_gates": list(source_failed),
            "allowed_source_false_gates": sorted(_ALLOWED_SOURCE_FALSE_GATES),
            "unrelated_source_gate_failures": list(unrelated_source_failures),
            "registered_circle_robust_locked_geometry_pass": bool(
                registered_circle.get("robust_locked_orbit_geometry_pass")
            ),
            "selected_circle_robust_locked_geometry_pass": bool(
                selected_circle.get("robust_locked_orbit_geometry_pass")
            ),
            "production_target_authority": False,
        },
        plan_payload=plan_payload,
        baseline_payload=baseline_payload,
        prior_guardrails={"guardrails_pass": bool(source_gates.get("prior_decoupled_guardrails"))},
        capture_confirmed_static=bool(source_gates.get("capture_confirmed_static")),
        policy=p,
    )

    qualified = bool(correction_preconditions_pass and assessment.get("pass"))
    return {
        "phase": "I4.1.7B",
        "schema_version": I4_1_7B_SCHEMA_VERSION,
        "status": "qualified_for_i4_1_8" if qualified else "degraded",
        "policy_reference": p.as_dict(),
        "source": {
            "i4_1_7_status": str(source.get("status")),
            "i4_1_7a_status": str(diag_source.get("status")),
            "qualification_plan": str(source.get("qualification_plan", "")),
        },
        "source_failed_gates": list(source_failed),
        "unrelated_source_gate_failures": list(unrelated_source_failures),
        "preconditions": preconditions,
        "correction_preconditions_pass": correction_preconditions_pass,
        "target_correction": {
            "from_mode": str(source_target.get("mode", "")),
            "to_mode": _TARGET_MODE,
            "target_reconstruction": [float(v) for v in circle_target],
            "trajectory_circle_diagnostics": dict(circle_diag),
            "registered_invariant_evaluation": dict(registered_circle),
            "selected_invariant_evaluation": dict(selected_circle),
            "bounded_to_i4_1_7_provisional_qualification": True,
            "production_target_authority": False,
            "i4_1_8_recompute_from_retained_core_required": True,
        },
        "assessment": assessment,
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "production_binding_executed": False,
            "production_sampling_integration_authorized": False,
            "production_default_changed": False,
            "orbit_thresholds_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "clean_static_orbit_requalified": qualified,
            "i4_1_8_production_binding_authorized": qualified,
            "production_sampling_integration_authorized": False,
            "i4_1_8_core_target_requalification_required": True,
            "i4_2_detail_preservation_authorized": False,
            "mesh_authorization": False,
        },
    }


__all__ = [
    "I4_1_7B_SCHEMA_VERSION",
    "build_i4_1_7b_bounded_target_correction",
]
