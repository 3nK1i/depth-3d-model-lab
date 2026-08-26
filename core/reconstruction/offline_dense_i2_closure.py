"""Final I2 full-video quality requalification helpers.

I2.7 is validation/closure only. It consumes the promoted I2.6 formal F/G
product, recomputes spatial/component evidence, compares it with the original
I2 baseline, and keeps metric/measurement/mesh authority unpromoted.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

I2_CLOSURE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class I2ClosurePolicy:
    """Conservative automated gates before final fixed-view/UI review."""

    maximum_extended_loss_fraction: float = 0.01
    maximum_core_loss_fraction: float = 0.005
    minimum_full_to_robust_improvement_fraction: float = 0.03
    minimum_full_diagonal_shrink_fraction: float = 0.05
    maximum_robust_diagonal_shrink_fraction: float = 0.03
    minimum_far_peripheral_reduction_fraction: float = 0.15
    minimum_largest_component_fraction_of_extended: float = 0.98
    minimum_driver_rejection_fraction: float = 0.50
    minimum_stubborn_rejection_fraction: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "maximum_extended_loss_fraction",
            "maximum_core_loss_fraction",
            "minimum_full_to_robust_improvement_fraction",
            "minimum_full_diagonal_shrink_fraction",
            "maximum_robust_diagonal_shrink_fraction",
            "minimum_far_peripheral_reduction_fraction",
            "minimum_largest_component_fraction_of_extended",
            "minimum_driver_rejection_fraction",
            "minimum_stubborn_rejection_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise ReconstructionValidationError(f"{name} must be finite in [0, 1].")
            object.__setattr__(self, name, value)

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def voxel_component_profile(points: object, voxel_size: float) -> dict[str, Any]:
    """Recompute 26-neighbour connected components from retained voxel centers."""

    xyz = np.asarray(points, dtype=np.float64)
    voxel = float(voxel_size)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ReconstructionValidationError("I2.7 component analysis requires non-empty Nx3 geometry.")
    if not isfinite(voxel) or voxel <= 0.0 or not np.isfinite(xyz).all():
        raise ReconstructionValidationError("I2.7 component analysis received invalid geometry/voxel size.")
    keys_arr = np.floor(xyz / voxel).astype(np.int64)
    keys = [tuple(map(int, row)) for row in keys_arr]
    if len(set(keys)) != len(keys):
        raise ReconstructionValidationError("I2.7 expects exactly one retained point per voxel key.")

    key_set = set(keys)
    unseen = set(keys)
    sizes: list[int] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        size = 0
        while stack:
            x, y, z = stack.pop()
            size += 1
            for dx, dy, dz in offsets:
                candidate = (x + dx, y + dy, z + dz)
                if candidate in unseen:
                    unseen.remove(candidate)
                    stack.append(candidate)
        sizes.append(size)
    sizes.sort(reverse=True)
    largest = sizes[0]
    return {
        "component_count": len(sizes),
        "largest_component_voxels": int(largest),
        "largest_component_fraction_of_extended": float(largest / len(keys)),
        "top_component_sizes": [int(v) for v in sizes[:10]],
        "retained_voxels": len(keys),
        "connectivity": 26,
    }



def baseline_aware_component_topology_gates(
    qualification: Mapping[str, Any],
    *,
    expected_scene_id: str,
    expected_scene_fingerprint: str,
    manual_lineage_review_pass: bool,
) -> dict[str, bool]:
    """Validate I2.7.1 evidence for a baseline-relative topology gate.

    The original absolute 98% dominant-component fraction remains useful as a
    diagnostic threshold, but I2.7.1 proved that the qualified baseline scene
    itself does not satisfy it under the same fresh 26-neighbour computation.
    For this requalification path we therefore require strict *non-regression*
    of the dominant topology instead of lowering that absolute threshold.
    """

    if not isinstance(qualification, Mapping):
        raise TypeError("qualification must be a mapping.")
    if str(qualification.get("phase")) != "I2.7.1" or str(qualification.get("status")) != "diagnosis_ready":
        raise ReconstructionValidationError("Baseline-aware I2.7 topology gating requires a DIAGNOSIS_READY I2.7.1 summary.")

    source_scene = qualification.get("source_scene", {})
    comparison = qualification.get("component_comparison", {})
    baseline = comparison.get("baseline", {})
    promoted = comparison.get("promoted", {})
    promotion_cut = qualification.get("promotion_cut_relationship", {})
    equivalence = qualification.get("surviving_evidence_equivalence", {})

    required_numeric = {
        "baseline_largest": baseline.get("largest_component_voxels"),
        "promoted_largest": promoted.get("largest_component_voxels"),
        "baseline_fraction": baseline.get("largest_component_fraction"),
        "promoted_fraction": promoted.get("largest_component_fraction"),
        "baseline_detached": baseline.get("detached_voxels"),
        "promoted_detached": promoted.get("detached_voxels"),
        "dominant_lineage_fraction": comparison.get("promoted_dominant_from_single_baseline_component_fraction"),
        "newly_detached_voxels": comparison.get("newly_detached_voxels"),
        "removed_from_baseline_dominant": comparison.get("removed_from_baseline_dominant"),
        "newly_detached_components": promotion_cut.get("newly_detached_component_count"),
    }
    if any(value is None for value in required_numeric.values()):
        raise ReconstructionValidationError("I2.7.1 topology evidence is incomplete.")

    baseline_fraction = float(required_numeric["baseline_fraction"])
    promoted_fraction = float(required_numeric["promoted_fraction"])
    lineage_fraction = float(required_numeric["dominant_lineage_fraction"])
    for name, value in (
        ("baseline largest-component fraction", baseline_fraction),
        ("promoted largest-component fraction", promoted_fraction),
        ("dominant lineage fraction", lineage_fraction),
    ):
        if not isfinite(value) or value < 0.0 or value > 1.0:
            raise ReconstructionValidationError(f"{name} must be finite in [0, 1].")

    evidence_equivalent = bool(equivalence) and all(bool(value) for value in equivalence.values())
    gates = {
        "i2_7_1_diagnosis_ready": True,
        "scene_identity_matches": (
            str(source_scene.get("scene_id")) == str(expected_scene_id)
            and str(source_scene.get("fingerprint")) == str(expected_scene_fingerprint)
        ),
        "preexisting_baseline_fragmentation_confirmed": str(qualification.get("failure_classification")) == "preexisting_baseline_fragmentation",
        "surviving_evidence_equivalence": evidence_equivalent,
        "dominant_component_size_not_degraded": int(required_numeric["promoted_largest"]) >= int(required_numeric["baseline_largest"]),
        "dominant_component_lineage_exact": bool(np.isclose(lineage_fraction, 1.0, atol=0.0, rtol=0.0)),
        "no_newly_detached_voxels": int(required_numeric["newly_detached_voxels"]) == 0,
        "no_newly_detached_components": int(required_numeric["newly_detached_components"]) == 0,
        "no_removed_baseline_dominant_voxels": int(required_numeric["removed_from_baseline_dominant"]) == 0,
        "largest_component_fraction_not_degraded": promoted_fraction >= baseline_fraction,
        "detached_voxel_count_not_increased": int(required_numeric["promoted_detached"]) <= int(required_numeric["baseline_detached"]),
        "manual_lineage_review_pass": bool(manual_lineage_review_pass),
    }
    return gates

def closure_quality_gates(
    *,
    policy: I2ClosurePolicy,
    i2_6_pass: bool,
    exact_counterfactual_equivalence: bool,
    weak_manifold_attachment_rejections: int,
    expected_manifold_rejections: int,
    extended_loss_fraction: float,
    core_loss_fraction: float,
    full_to_robust_improvement_fraction: float,
    full_diagonal_shrink_fraction: float,
    robust_diagonal_shrink_fraction: float,
    far_peripheral_reduction_fraction: float,
    largest_component_fraction_of_extended: float,
    driver_rejection_fraction: float,
    stubborn_rejection_fraction: float,
    upstream_quality_gates_pass: bool,
    persistent_integrity_pass: bool,
    authority_boundaries_preserved: bool,
    baseline_component_topology_preserved: bool | None = None,
) -> dict[str, bool]:
    if not isinstance(policy, I2ClosurePolicy):
        raise TypeError("policy must be I2ClosurePolicy.")
    values = (
        extended_loss_fraction,
        core_loss_fraction,
        full_to_robust_improvement_fraction,
        full_diagonal_shrink_fraction,
        robust_diagonal_shrink_fraction,
        far_peripheral_reduction_fraction,
        largest_component_fraction_of_extended,
        driver_rejection_fraction,
        stubborn_rejection_fraction,
    )
    if any((not isfinite(float(v))) for v in values):
        raise ReconstructionValidationError("I2.7 gate metrics must be finite.")
    return {
        "i2_6_pass": bool(i2_6_pass),
        "exact_counterfactual_equivalence": bool(exact_counterfactual_equivalence),
        "manifold_rejection_count_exact": int(weak_manifold_attachment_rejections) == int(expected_manifold_rejections),
        "extended_loss_within_gate": float(extended_loss_fraction) <= policy.maximum_extended_loss_fraction,
        "core_loss_within_gate": float(core_loss_fraction) <= policy.maximum_core_loss_fraction,
        "full_to_robust_improvement": float(full_to_robust_improvement_fraction) >= policy.minimum_full_to_robust_improvement_fraction,
        "full_diagonal_shrink": float(full_diagonal_shrink_fraction) >= policy.minimum_full_diagonal_shrink_fraction,
        "robust_diagonal_preservation": float(robust_diagonal_shrink_fraction) <= policy.maximum_robust_diagonal_shrink_fraction,
        "far_peripheral_reduction": float(far_peripheral_reduction_fraction) >= policy.minimum_far_peripheral_reduction_fraction,
        "largest_component_preserved": (
            bool(baseline_component_topology_preserved)
            if baseline_component_topology_preserved is not None
            else float(largest_component_fraction_of_extended) >= policy.minimum_largest_component_fraction_of_extended
        ),
        "driver_rejection_qualified": float(driver_rejection_fraction) >= policy.minimum_driver_rejection_fraction,
        "stubborn_rejection_qualified": float(stubborn_rejection_fraction) >= policy.minimum_stubborn_rejection_fraction,
        "upstream_quality_gates_pass": bool(upstream_quality_gates_pass),
        "persistent_integrity_pass": bool(persistent_integrity_pass),
        "authority_boundaries_preserved": bool(authority_boundaries_preserved),
    }


def build_i2_closure_record(
    automated_summary: Mapping[str, Any],
    ui_sanity: Mapping[str, Any],
    *,
    fixed_view_review_pass: bool,
) -> dict[str, Any]:
    """Return the final closure record; fail closed on missing human/UI evidence."""

    if str(automated_summary.get("phase")) != "I2.7" or not bool(automated_summary.get("automated_pass")):
        raise ReconstructionValidationError("I2.7 closure requires a passing automated requalification summary.")
    if str(ui_sanity.get("phase")) != "I2.7-UI" or not bool(ui_sanity.get("pass")):
        raise ReconstructionValidationError("I2.7 closure requires a passing promoted-scene UI sanity result.")
    if str(ui_sanity.get("scene_fingerprint_sha256")) != str(automated_summary.get("scene", {}).get("fingerprint")):
        raise ReconstructionValidationError("I2.7 UI sanity scene fingerprint does not match requalification scene.")
    if not bool(fixed_view_review_pass):
        raise ReconstructionValidationError("I2.7 closure requires explicit final fixed-view visual review PASS.")
    return {
        "phase": "I2.7-CLOSURE",
        "schema_version": I2_CLOSURE_SCHEMA_VERSION,
        "status": "pass",
        "i2_complete": True,
        "scene_id": str(automated_summary["scene"]["scene_id"]),
        "scene_fingerprint_sha256": str(automated_summary["scene"]["fingerprint"]),
        "evidence": {
            "automated_requalification": "pass",
            "fixed_view_visual_review": "pass",
            "reconstruction_ui_sanity": "pass",
        },
        "decision": {
            "authorizes_i2_closure": True,
            "authorizes_i3_second_scene_campaign": True,
            "authorizes_surface_mesh_reconstruction": False,
            "i_campaign_mesh_readiness": False,
            "next_step": "I3 - Second indoor/static sequence",
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
    }
