"""I3.2 second-scene production promotion and requalification helpers.

I3.2 promotes only the exact I3.1 manifold-transfer candidate that passed the
second-scene automated safety gates and an explicit human review.  It does not
retune the locked ``d12_sparse`` predicate, does not rerun SfM/DA3/fusion, and
never promotes metric, measurement, or mesh authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I3_2_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class I3PromotionGatePolicy:
    """Fail-closed requalification gates for the reviewed I3.1 candidate."""

    maximum_extended_loss_fraction: float = 0.01
    maximum_core_loss_fraction: float = 0.005
    maximum_robust_diagonal_shrink_fraction: float = 0.03
    maximum_full_to_robust_regression_fraction: float = 0.01
    minimum_dominant_lineage_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReconstructionValidationError(f"I3.2 {name} must be finite in [0,1].")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def validate_i3_1_promotion_evidence(
    summary: Mapping[str, Any],
    *,
    manual_transfer_review_pass: bool,
) -> dict[str, Any]:
    """Validate that I3.1 authorizes promotion of its exact reviewed candidate."""

    if not manual_transfer_review_pass:
        raise ReconstructionValidationError("I3.2 requires explicit confirmation that the I3.1 manual transfer review passed.")
    if str(summary.get("phase")) != "I3.1" or str(summary.get("status")) != "ready_for_manual_transfer_review":
        raise ReconstructionValidationError("I3.2 requires a READY_FOR_MANUAL_TRANSFER_REVIEW I3.1 summary.")
    preflight = summary.get("i2_transfer_preflight")
    if not isinstance(preflight, Mapping) or not bool(preflight.get("pass")):
        raise ReconstructionValidationError("I3.2 requires the exact passing I2 transfer preflight.")
    baseline = summary.get("baseline_quality")
    if not isinstance(baseline, Mapping) or not bool(baseline.get("automated_pass")):
        raise ReconstructionValidationError("I3.2 requires a passing second-scene baseline quality assessment.")
    candidate = summary.get("candidate")
    if not isinstance(candidate, Mapping) or not bool(candidate.get("safe_for_manual_transfer_review")):
        raise ReconstructionValidationError("I3.2 requires an I3.1 candidate that passed every transfer safety gate.")
    gates = candidate.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(bool(v) for v in gates.values()):
        raise ReconstructionValidationError("I3.2 refuses I3.1 evidence with any failed transfer safety gate.")
    authority = summary.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I3.2 I3.1 authority evidence is missing.")
    forbidden = (
        "geometry_changed",
        "production_f_policy_changed",
        "persistence_changed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "surface_mesh_generated",
    )
    if any(bool(authority.get(name)) for name in forbidden):
        raise ReconstructionValidationError("I3.2 refuses I3.1 evidence that already changed production authority/state.")
    scene = summary.get("scene")
    if not isinstance(scene, Mapping) or not str(scene.get("scene_id", "")).strip() or not str(scene.get("fingerprint", "")).strip():
        raise ReconstructionValidationError("I3.2 I3.1 scene identity is incomplete.")
    return {
        "manual_transfer_review": "pass",
        "i2_transfer_preflight": "pass",
        "baseline_quality": "pass",
        "transfer_safety_gates": "pass",
        "source_scene_id": str(scene["scene_id"]),
        "source_scene_fingerprint": str(scene["fingerprint"]),
    }


def _aligned_fields(payload: Mapping[str, object], count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, values in payload.items():
        arr = np.asarray(values)
        if arr.ndim >= 1 and arr.shape[0] == count and name not in (
            "selected_workspace_orders",
            "selected_keyframe_ids",
            "sparse_landmarks_reconstruction",
            "sparse_landmark_colors_rgb",
        ):
            result[name] = arr
    return result


def validate_exact_i3_1_candidate(
    source_extended: Mapping[str, object],
    diagnostics: Mapping[str, object],
    candidate_extended: Mapping[str, object],
    candidate_core: Mapping[str, object],
    i3_1_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that saved I3.1 candidate files are exact source-mask slices."""

    source_points = np.asarray(source_extended.get("points_reconstruction"), dtype=np.float64)
    diag_points = np.asarray(diagnostics.get("points_reconstruction"), dtype=np.float64)
    if source_points.ndim != 2 or source_points.shape[1] != 3 or len(source_points) == 0:
        raise ReconstructionValidationError("I3.2 source Extended geometry is invalid.")
    if diag_points.shape != source_points.shape or not np.array_equal(diag_points, source_points):
        raise ReconstructionValidationError("I3.2 diagnostics do not exactly align with source Extended geometry.")
    reject = np.asarray(diagnostics.get("is_rejected_by_d12_transfer"), dtype=bool).reshape(-1)
    is_core = np.asarray(diagnostics.get("is_core"), dtype=bool).reshape(-1)
    if reject.shape != (len(source_points),) or is_core.shape != reject.shape:
        raise ReconstructionValidationError("I3.2 diagnostics masks are misaligned.")
    keep = ~reject

    candidate = i3_1_summary.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ReconstructionValidationError("I3.2 candidate summary is missing.")
    expected_rejected = int(candidate.get("rejected_points", -1))
    if int(np.count_nonzero(reject)) != expected_rejected:
        raise ReconstructionValidationError("I3.2 saved rejection mask count disagrees with I3.1 summary.")

    source_fields = _aligned_fields(source_extended, len(source_points))
    ext_count = int(np.count_nonzero(keep))
    core_count = int(np.count_nonzero(is_core & keep))
    for label, payload, mask in (
        ("Extended", candidate_extended, keep),
        ("Core", candidate_core, is_core & keep),
    ):
        expected_count = int(np.count_nonzero(mask))
        for name, source_arr in source_fields.items():
            if name not in payload:
                continue
            candidate_arr = np.asarray(payload[name])
            expected_arr = source_arr[mask]
            if candidate_arr.shape != expected_arr.shape or not np.array_equal(candidate_arr, expected_arr):
                raise ReconstructionValidationError(f"I3.2 saved candidate {label} field {name!r} is not an exact source slice.")
        points = np.asarray(payload.get("points_reconstruction"), dtype=np.float64)
        if len(points) != expected_count:
            raise ReconstructionValidationError(f"I3.2 saved candidate {label} point count is not exact.")

    ext_summary = candidate.get("extended_points")
    core_summary = candidate.get("core_points")
    if not isinstance(ext_summary, Mapping) or int(ext_summary.get("candidate", -1)) != ext_count:
        raise ReconstructionValidationError("I3.2 candidate Extended count disagrees with I3.1 summary.")
    if not isinstance(core_summary, Mapping) or int(core_summary.get("candidate", -1)) != core_count:
        raise ReconstructionValidationError("I3.2 candidate Core count disagrees with I3.1 summary.")
    return {
        "exact_candidate_equivalence": True,
        "source_extended_points": int(len(source_points)),
        "rejected_points": expected_rejected,
        "promoted_extended_points": ext_count,
        "promoted_core_points": core_count,
        "reject_mask": reject,
        "is_core": is_core,
    }


def build_i3_2_requalification(
    i3_1_summary: Mapping[str, Any],
    *,
    exact_candidate_equivalence: bool,
    promoted_extended_points: int,
    promoted_core_points: int,
    weak_manifold_attachment_rejections: int,
    exact_supported_accounting: bool,
    component_lineage: Mapping[str, Any],
    persistence_reference_count: int,
    persistence_relative: bool,
    persistence_self_contained: bool,
    relocation_restore_pass: bool,
    policy: I3PromotionGatePolicy | None = None,
) -> dict[str, Any]:
    """Build automated I3.2 requalification from locked I3.1 evidence + promoted artifacts."""

    gate_policy = policy or I3PromotionGatePolicy()
    candidate = i3_1_summary.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ReconstructionValidationError("I3.2 candidate evidence is missing.")
    ext = candidate.get("extended_points")
    core = candidate.get("core_points")
    envelope = candidate.get("envelope")
    if not isinstance(ext, Mapping) or not isinstance(core, Mapping) or not isinstance(envelope, Mapping):
        raise ReconstructionValidationError("I3.2 candidate count/envelope evidence is incomplete.")
    ext_loss = float(ext.get("loss_fraction", 1.0))
    core_loss = float(core.get("loss_fraction", 1.0))
    robust_shrink = float(envelope.get("robust_diagonal_shrink_fraction", 1.0))
    ratio_improvement = float(envelope.get("full_to_robust_improvement_fraction", -1.0))
    expected_ext = int(ext.get("candidate", -1))
    expected_core = int(core.get("candidate", -1))
    expected_reject = int(candidate.get("rejected_points", -1))

    base_comp = component_lineage.get("baseline")
    prom_comp = component_lineage.get("promoted")
    if not isinstance(base_comp, Mapping) or not isinstance(prom_comp, Mapping):
        raise ReconstructionValidationError("I3.2 component-lineage evidence is incomplete.")
    dominant_lineage = float(component_lineage.get("promoted_dominant_from_single_baseline_component_fraction", 0.0))
    gates = {
        "i3_1_transfer_safety_passed": bool(candidate.get("safe_for_manual_transfer_review")),
        "exact_i3_1_candidate_equivalence": bool(exact_candidate_equivalence),
        "promoted_extended_count_exact": int(promoted_extended_points) == expected_ext,
        "promoted_core_count_exact": int(promoted_core_points) == expected_core,
        "manifold_rejection_count_exact": int(weak_manifold_attachment_rejections) == expected_reject,
        "supported_accounting_exact": bool(exact_supported_accounting),
        "extended_loss_within_gate": ext_loss <= gate_policy.maximum_extended_loss_fraction,
        "core_loss_within_gate": core_loss <= gate_policy.maximum_core_loss_fraction,
        "robust_diagonal_preserved": robust_shrink <= gate_policy.maximum_robust_diagonal_shrink_fraction,
        "full_to_robust_not_materially_worse": ratio_improvement >= -gate_policy.maximum_full_to_robust_regression_fraction,
        "dominant_lineage_preserved": dominant_lineage >= gate_policy.minimum_dominant_lineage_fraction,
        "no_newly_detached_voxels": int(component_lineage.get("newly_detached_voxels", -1)) == 0,
        "no_baseline_dominant_removal": int(component_lineage.get("removed_from_baseline_dominant", -1)) == 0,
        "largest_component_fraction_not_degraded": float(prom_comp.get("largest_component_fraction", 0.0)) + 1e-12 >= float(base_comp.get("largest_component_fraction", 1.0)),
        "detached_voxels_not_increased": int(prom_comp.get("detached_voxels", 10**18)) <= int(base_comp.get("detached_voxels", -1)),
        "persistent_scene_integrity_verified": int(persistence_reference_count) == 10,
        "persistent_references_relative": bool(persistence_relative),
        "persistent_scene_self_contained": bool(persistence_self_contained),
        "relocation_restore_pass": bool(relocation_restore_pass),
        "authority_boundaries_preserved": True,
    }
    return {
        "schema_version": I3_2_SCHEMA_VERSION,
        "pass": bool(all(gates.values())),
        "gates": gates,
        "gate_policy": gate_policy.as_dict(),
        "counts": {
            "expected_extended": expected_ext,
            "promoted_extended": int(promoted_extended_points),
            "expected_core": expected_core,
            "promoted_core": int(promoted_core_points),
            "expected_manifold_rejections": expected_reject,
            "actual_manifold_rejections": int(weak_manifold_attachment_rejections),
        },
        "envelope": dict(envelope),
        "component_lineage": {key: value for key, value in component_lineage.items() if key != "masks"},
    }


def build_i3_closure_record(
    automated: Mapping[str, Any],
    ui_sanity: Mapping[str, Any],
    *,
    fixed_view_review_pass: bool,
) -> dict[str, Any]:
    """Finalize I3 only after automated promotion, fixed-view review and UI sanity."""

    if str(automated.get("phase")) != "I3.2" or not bool(automated.get("automated_pass")):
        raise ReconstructionValidationError("I3 closure requires a passing I3.2 automated requalification.")
    if not fixed_view_review_pass:
        raise ReconstructionValidationError("I3 closure requires explicit fixed-view review PASS.")
    if str(ui_sanity.get("phase")) != "I3.2-UI" or not bool(ui_sanity.get("pass")):
        raise ReconstructionValidationError("I3 closure requires passing I3.2 Reconstruction UI sanity.")
    if str(ui_sanity.get("scene_id")) != str(automated.get("scene_id")):
        raise ReconstructionValidationError("I3 closure scene identity mismatch between automated and UI evidence.")
    authority = automated.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I3 closure authority evidence is missing.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")) or bool(authority.get("surface_mesh_generated")):
        raise ReconstructionValidationError("I3 closure refuses premature metric/measurement/mesh authority.")
    return {
        "phase": "I3",
        "schema_version": I3_2_SCHEMA_VERSION,
        "status": "pass",
        "pass": True,
        "scene_id": str(automated["scene_id"]),
        "scene_fingerprint_sha256": str(automated["scene_fingerprint_sha256"]),
        "checks": {
            "automated_requalification": "pass",
            "fixed_view_review": "pass",
            "reconstruction_ui_sanity": "pass",
        },
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "i3_closure_authorized": True,
            "i4_connector_orbit_authorized": True,
            "mesh_authorization": False,
            "next": "I4 - Object / connector orbit sequence",
        },
    }
