"""I4.1.8C final retained-Core target binding and orbit-causality policy.

This step consumes the independently isolated retained-Core support from
I4.1.8B and compares the same object target against two already-existing pose
sets: the complete qualified decoupled path and the bounded production path.
It never changes orbit thresholds and never searches for a target or a camera
subset that happens to pass.

The result is a final closure *decision*:

* if the bound production path passes the locked object-orbit gates, the
  retained-Core target may become final I4.1 target authority and manual orbit
  review may close I4.1;
* if the complete qualified path passes but the bound path fails, production
  binding lost object-centric angular coverage and must be corrected before
  closure;
* if the complete qualified path also fails, the reconstructed capture itself
  does not provide the required object-centric orbit and no code/threshold
  workaround is authorized.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_geometry_sampling_diagnosis import coverage_by_plane_mode
from core.reconstruction.offline_dense_i4_orbit_qualification import (
    I4OrbitQualificationPolicy,
    orbit_coverage_metrics,
)
from core.reconstruction.offline_dense_i4_retained_core_object_support import _geometric_median

I4_1_8C_SCHEMA_VERSION = 1
_TARGET_DEPENDENT_FINAL_GATES = frozenset(
    {
        "selected_azimuth_coverage",
        "selected_azimuth_bins",
        "selected_opposed_view_diversity",
        "selected_target_facing",
        "selected_standoff_consistency",
    }
)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.1.8C {name} must be a mapping.")
    return value


def _poses(value: object, *, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 3 or not np.isfinite(poses).all():
        raise ReconstructionValidationError(f"I4.1.8C {name} poses must be finite Nx4x4 with N>=3.")
    if not np.allclose(poses[:, 3, :], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError(f"I4.1.8C {name} poses have invalid homogeneous rows.")
    rotations = poses[:, :3, :3]
    gram = np.einsum("nij,nkj->nik", rotations, rotations)
    if not np.allclose(gram, np.eye(3)[None, :, :], atol=1e-5):
        raise ReconstructionValidationError(f"I4.1.8C {name} camera rotations are not rigid.")
    return poses


def _sources(value: object, *, name: str, count: int) -> np.ndarray:
    sources = np.asarray(value, dtype=np.int64).reshape(-1)
    if len(sources) != int(count) or len(np.unique(sources)) != len(sources):
        raise ReconstructionValidationError(f"I4.1.8C {name} source indices must be unique and pose-aligned.")
    if np.any(np.diff(sources) <= 0):
        raise ReconstructionValidationError(f"I4.1.8C {name} source indices must be strictly increasing.")
    return sources


def _orbit_evaluation(
    poses: np.ndarray,
    target: np.ndarray,
    *,
    policy: I4OrbitQualificationPolicy,
) -> dict[str, Any]:
    metrics = orbit_coverage_metrics(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    gates = {
        "selected_view_count": int(len(poses)) >= int(policy.minimum_selected_views),
        "selected_azimuth_coverage": float(metrics["azimuth_coverage_degrees"])
        >= float(policy.minimum_selected_azimuth_coverage_degrees),
        "selected_azimuth_bins": int(metrics["occupied_azimuth_bins"])
        >= int(policy.minimum_selected_occupied_azimuth_bins),
        "selected_opposed_view_diversity": float(metrics["maximum_opposed_view_angle_degrees"])
        >= float(policy.minimum_opposed_view_angle_degrees),
        "selected_target_facing": float(metrics["target_facing_fraction"])
        >= float(policy.minimum_target_facing_fraction),
        "selected_standoff_consistency": float(metrics["standoff_p90_to_p10_ratio"])
        <= float(policy.maximum_standoff_p90_to_p10_ratio),
    }
    modes = coverage_by_plane_mode(
        poses,
        target,
        azimuth_bin_count=int(policy.azimuth_bin_count),
        facing_angle_degrees=float(policy.maximum_facing_angle_degrees),
    )
    # Coordinate-invariant plane variants are diagnostic only: the locked I4.1
    # gate semantics remain exactly orbit_coverage_metrics + policy above.
    invariant_ranges = {
        "azimuth_min_degrees": float(min(float(v["azimuth_coverage_degrees"]) for v in modes.values())),
        "azimuth_max_degrees": float(max(float(v["azimuth_coverage_degrees"]) for v in modes.values())),
        "bins_min": int(min(int(v["occupied_azimuth_bins"]) for v in modes.values())),
        "bins_max": int(max(int(v["occupied_azimuth_bins"]) for v in modes.values())),
        "opposed_min_degrees": float(min(float(v["maximum_opposed_view_angle_degrees"]) for v in modes.values())),
        "opposed_max_degrees": float(max(float(v["maximum_opposed_view_angle_degrees"]) for v in modes.values())),
    }
    return {
        "pass": bool(all(gates.values())),
        "metrics": metrics,
        "gates": gates,
        "failed_gates": sorted(name for name, value in gates.items() if not bool(value)),
        "invariant_plane_modes_diagnostic_only": modes,
        "invariant_ranges_diagnostic_only": invariant_ranges,
    }


def build_i4_1_8c_final_object_orbit_causality(
    *,
    source_i4_1_7b_summary: Mapping[str, Any],
    source_i4_1_8_summary: Mapping[str, Any],
    source_i4_1_8b_summary: Mapping[str, Any],
    qualified_camera_matrices: object,
    qualified_source_sequence_indices: object,
    production_bound_camera_matrices: object,
    production_bound_source_sequence_indices: object,
    isolated_core_points: object,
    retained_core_voxel_size: float,
    policy: I4OrbitQualificationPolicy | None = None,
) -> dict[str, Any]:
    """Classify whether object-orbit loss is upstream or caused by binding.

    The isolated support target is independently reproduced from saved retained
    Core points.  The same target and unchanged gate policy are then applied to
    both pose sets.  No pose subset is optimized here.
    """

    i417b = _mapping(source_i4_1_7b_summary, name="I4.1.7B summary")
    i418 = _mapping(source_i4_1_8_summary, name="I4.1.8 summary")
    i418b = _mapping(source_i4_1_8b_summary, name="I4.1.8B summary")
    if str(i417b.get("phase")) != "I4.1.7B" or str(i417b.get("status")) != "qualified_for_i4_1_8":
        raise ReconstructionValidationError("I4.1.8C requires qualified I4.1.7B evidence.")
    if str(i418.get("phase")) != "I4.1.8" or str(i418.get("status")) != "degraded":
        raise ReconstructionValidationError("I4.1.8C requires the completed DEGRADED I4.1.8 production run.")
    if str(i418b.get("phase")) != "I4.1.8B" or str(i418b.get("status")) != "object_support_isolated":
        raise ReconstructionValidationError("I4.1.8C requires I4.1.8B object_support_isolated evidence.")

    stage_gates = _mapping(i418.get("stage_gates"), name="I4.1.8 stage gates")
    if not all(bool(stage_gates.get(name)) for name in (
        "production_fusion", "formal_filtering", "canonical_persistence", "ordinary_i2_bound_scene_quality"
    )):
        raise ReconstructionValidationError("I4.1.8C requires the complete bound production F/G/I2 chain to have passed.")
    final_assessment = _mapping(i418.get("final_assessment"), name="I4.1.8 final assessment")
    failed = tuple(sorted(str(v) for v in final_assessment.get("failed_gates", [])))
    unrelated = tuple(name for name in failed if name not in _TARGET_DEPENDENT_FINAL_GATES)
    if unrelated:
        raise ReconstructionValidationError("I4.1.8C refuses unrelated I4.1.8 final failures.")

    b_diag = _mapping(i418b.get("diagnosis"), name="I4.1.8B diagnosis")
    b_selected = _mapping(i418b.get("selected_support"), name="I4.1.8B selected support")
    if str(b_diag.get("classification")) != "retained_core_object_support_isolated":
        raise ReconstructionValidationError("I4.1.8C requires independently isolated retained-Core object support.")
    if not bool(b_diag.get("object_support_isolation_ready")) or not bool(
        b_diag.get("recommends_i4_1_8c_final_target_binding_design")
    ):
        raise ReconstructionValidationError("I4.1.8C object-support evidence is not ready for final binding.")
    if bool(b_selected.get("posthoc_orbit_used_for_selection_or_readiness")):
        raise ReconstructionValidationError("I4.1.8C refuses object support selected using orbit-gate feedback.")
    readiness = _mapping(b_selected.get("readiness_gates"), name="I4.1.8B readiness gates")
    if not readiness or not all(bool(v) for v in readiness.values()):
        raise ReconstructionValidationError("I4.1.8C requires every I4.1.8B isolation readiness gate to pass.")
    stability = _mapping(b_selected.get("target_bootstrap_stability"), name="I4.1.8B target stability")
    if not bool(stability.get("stable")):
        raise ReconstructionValidationError("I4.1.8C requires a stable isolated-support target.")

    for summary_name, authority in (
        ("I4.1.7B", i417b.get("authority")),
        ("I4.1.8", i418.get("authority")),
        ("I4.1.8B", i418b.get("authority")),
    ):
        auth = _mapping(authority, name=f"{summary_name} authority")
        for forbidden in (
            "metric_scale_claimed", "measurement_authoritative", "surface_mesh_generated",
            "dense_consistency_threshold_relaxation_authorized",
        ):
            if bool(auth.get(forbidden)):
                raise ReconstructionValidationError(f"I4.1.8C refuses authority drift: {summary_name} {forbidden}=True.")

    qualified = _poses(qualified_camera_matrices, name="qualified-path")
    bound = _poses(production_bound_camera_matrices, name="production-bound")
    qualified_sources = _sources(qualified_source_sequence_indices, name="qualified-path", count=len(qualified))
    bound_sources = _sources(production_bound_source_sequence_indices, name="production-bound", count=len(bound))
    qualified_index = {int(source): index for index, source in enumerate(qualified_sources.tolist())}
    if any(int(v) not in qualified_index for v in bound_sources.tolist()):
        raise ReconstructionValidationError("I4.1.8C bound production sources must be a subset of the qualified path.")
    if int(bound_sources[0]) != int(qualified_sources[0]) or int(bound_sources[-1]) != int(qualified_sources[-1]):
        raise ReconstructionValidationError("I4.1.8C bound production path must preserve qualified-path endpoints.")
    expected_bound = np.asarray([qualified[qualified_index[int(source)]] for source in bound_sources], dtype=np.float64)
    if not np.allclose(bound, expected_bound, atol=1e-8, rtol=1e-8):
        raise ReconstructionValidationError(
            "I4.1.8C bound production poses do not exactly reproduce their qualified-path source poses."
        )

    points = np.asarray(isolated_core_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 64 or not np.isfinite(points).all():
        raise ReconstructionValidationError("I4.1.8C isolated retained-Core support must be finite Nx3 with N>=64.")
    voxel_size = float(retained_core_voxel_size)
    if not np.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ReconstructionValidationError("I4.1.8C retained-Core voxel size must be finite and positive.")
    target = _geometric_median(points)
    saved_target = np.asarray(b_selected.get("target_reconstruction"), dtype=np.float64).reshape(-1)
    if saved_target.shape != (3,) or not np.isfinite(saved_target).all():
        raise ReconstructionValidationError("I4.1.8C saved isolated-support target is invalid.")
    target_reproduction_drift_voxels = float(np.linalg.norm(target - saved_target) / voxel_size)
    target_reproduced_exactly = bool(target_reproduction_drift_voxels <= 1e-6)
    if not target_reproduced_exactly:
        raise ReconstructionValidationError(
            f"I4.1.8C isolated-support target does not reproduce exactly ({target_reproduction_drift_voxels:.6g} vox drift)."
        )

    p = policy or I4OrbitQualificationPolicy()
    qualified_eval = _orbit_evaluation(qualified, target, policy=p)
    bound_eval = _orbit_evaluation(bound, target, policy=p)
    if bool(bound_eval["pass"]):
        classification = "final_object_orbit_qualified"
        final_target_authority_ready = True
        final_i4_1_automated_pass = True
        manual_review_required = True
        recapture_required = False
        binding_correction_required = False
    elif bool(qualified_eval["pass"]):
        classification = "production_binding_lost_object_angular_coverage"
        final_target_authority_ready = False
        final_i4_1_automated_pass = False
        manual_review_required = False
        recapture_required = False
        binding_correction_required = True
    else:
        classification = "qualified_path_object_orbit_insufficient"
        final_target_authority_ready = False
        final_i4_1_automated_pass = False
        manual_review_required = False
        recapture_required = True
        binding_correction_required = False

    return {
        "phase": "I4.1.8C",
        "schema_version": I4_1_8C_SCHEMA_VERSION,
        "status": "ready_for_final_manual_review" if final_i4_1_automated_pass else "not_qualified",
        "policy": p.as_dict(),
        "target": {
            "mode": "isolated_retained_core_geometric_median",
            "target_reconstruction": [float(v) for v in target],
            "source": "I4.1.8B independently isolated retained-Core object support",
            "isolated_support_points": int(len(points)),
            "saved_target_reproduction_drift_voxels": target_reproduction_drift_voxels,
            "saved_target_reproduced_exactly": target_reproduced_exactly,
            "production_target_authority_ready": bool(final_target_authority_ready),
        },
        "pose_sets": {
            "qualified_path": {
                "view_count": int(len(qualified)),
                "source_sequence_indices": [int(v) for v in qualified_sources],
                **qualified_eval,
            },
            "production_bound": {
                "view_count": int(len(bound)),
                "source_sequence_indices": [int(v) for v in bound_sources],
                **bound_eval,
            },
        },
        "causality": {
            "classification": classification,
            "qualified_path_pass": bool(qualified_eval["pass"]),
            "production_bound_pass": bool(bound_eval["pass"]),
            "production_binding_lost_object_angular_coverage": bool(binding_correction_required),
            "reconstructed_capture_object_orbit_insufficient": bool(recapture_required),
            "orbit_metrics_used_to_select_object_support": False,
            "orbit_metrics_used_to_select_target": False,
            "camera_subset_search_performed": False,
        },
        "authority": {
            "single_authoritative_sfm_frame": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "dense_pair_rerun": False,
            "fusion_rerun": False,
            "filtering_rerun": False,
            "persistence_rerun": False,
            "production_binding_rerun": False,
            "orbit_thresholds_changed": False,
            "dense_consistency_threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "final_object_target_binding_authorized": bool(final_target_authority_ready),
            "production_sampling_integration_authorized": bool(final_i4_1_automated_pass),
            "i4_2_detail_preservation_authorized": False,
        },
        "decision": {
            "final_i4_1_automated_qualification_pass": bool(final_i4_1_automated_pass),
            "manual_orbit_review_required": bool(manual_review_required),
            "recapture_required": bool(recapture_required),
            "production_binding_correction_required": bool(binding_correction_required),
            "i4_1_closed": False,
            "i4_2_detail_preservation_authorized": False,
            "next_action": (
                "Perform final manual object-orbit review; only a clean review may close I4.1 and authorize I4.2."
                if final_i4_1_automated_pass
                else (
                    "Correct only production view binding to preserve object-centric angular coverage, then rerun the bounded production chain."
                    if binding_correction_required
                    else "Record one better-controlled object-centric orbit capture and rerun qualification; do not change locked orbit thresholds."
                )
            ),
        },
    }


def apply_i4_1_8c_manual_closure(result: Mapping[str, Any], *, manual_review_confirmed: bool) -> dict[str, Any]:
    """Close I4.1 only after automated qualification and explicit manual review."""
    payload = dict(result)
    if str(payload.get("phase")) != "I4.1.8C":
        raise ReconstructionValidationError("I4.1.8C manual closure requires an I4.1.8C result.")
    decision = dict(_mapping(payload.get("decision"), name="manual-closure decision"))
    authority = dict(_mapping(payload.get("authority"), name="manual-closure authority"))
    if not bool(manual_review_confirmed):
        return payload
    if not bool(decision.get("final_i4_1_automated_qualification_pass")):
        raise ReconstructionValidationError("I4.1.8C cannot close I4.1 when automated qualification did not pass.")
    if not bool(decision.get("manual_orbit_review_required")):
        raise ReconstructionValidationError("I4.1.8C manual closure requires a pending manual orbit review.")
    decision.update({
        "manual_orbit_review_confirmed": True,
        "manual_orbit_review_required": False,
        "i4_1_closed": True,
        "i4_2_detail_preservation_authorized": True,
        "next_action": "I4.1 is closed. Proceed to I4.2 thin-structure / connector detail-preservation qualification.",
    })
    authority.update({
        "final_object_target_binding_authorized": True,
        "production_sampling_integration_authorized": True,
        "i4_2_detail_preservation_authorized": True,
    })
    payload["decision"] = decision
    payload["authority"] = authority
    payload["status"] = "i4_1_closed"
    return payload


__all__ = [
    "I4_1_8C_SCHEMA_VERSION",
    "build_i4_1_8c_final_object_orbit_causality",
    "apply_i4_1_8c_manual_closure",
]
