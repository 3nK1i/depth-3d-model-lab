"""I4.3F3 production validation and adaptive-selection architecture closure."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.reconstruction.dense_provider_architecture import I43A_AUTHORITATIVE_FRAME_OWNER
from core.reconstruction.errors import ReconstructionValidationError

I43F3_SCHEMA_VERSION = 1
I43F3_PHASE = "I4.3F3"


def _require_pass(snapshot: Mapping[str, Any], *, phase: str) -> None:
    if not isinstance(snapshot, Mapping):
        raise TypeError(f"{phase} snapshot must be a mapping.")
    if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
        raise ReconstructionValidationError(f"I4.3F3 requires a passing {phase} snapshot.")


def validate_i4_3f3_predecessors(
    *,
    i4_3f1_snapshot: Mapping[str, Any],
    i4_3f2_snapshot: Mapping[str, Any],
) -> None:
    _require_pass(i4_3f1_snapshot, phase="I4.3F1")
    _require_pass(i4_3f2_snapshot, phase="I4.3F2")
    f1_decision = i4_3f1_snapshot.get("decision")
    f2_decision = i4_3f2_snapshot.get("decision")
    f2_authority = i4_3f2_snapshot.get("authority")
    if not isinstance(f1_decision, Mapping) or f1_decision.get("i4_3f2_production_wiring_authorized") is not True:
        raise ReconstructionValidationError("I4.3F3 requires F1 production-wiring authorization evidence.")
    if not isinstance(f2_decision, Mapping):
        raise ReconstructionValidationError("I4.3F3 requires F2 decision evidence.")
    if f2_decision.get("i4_3f2_production_wiring_closed") is not True:
        raise ReconstructionValidationError("I4.3F3 requires F2 production wiring to be closed.")
    if f2_decision.get("i4_3f3_production_validation_authorized") is not True:
        raise ReconstructionValidationError("I4.3F3 requires explicit F2 production-validation authorization.")
    if f1_decision.get("permanent_global_view_cap") is not None or f2_decision.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3F3 refuses any predecessor permanent global view cap.")
    if not isinstance(f2_authority, Mapping):
        raise ReconstructionValidationError("I4.3F3 requires F2 authority evidence.")
    if f2_authority.get("camera_pose_authority_owner") != I43A_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError("I4.3F3 requires PyCOLMAP SfM camera-pose authority.")
    forbidden = (
        "authoritative_sfm_modified",
        "auto_policy_implemented",
        "hybrid_authorized",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "provider_promoted",
        "provider_selected",
        "surface_mesh_generated",
    )
    drifted = [name for name in forbidden if bool(f2_authority.get(name, False))]
    if drifted:
        raise ReconstructionValidationError(f"I4.3F3 predecessor authority drift: {drifted}")


def build_i4_3f3_closure_snapshot(
    *,
    i4_3f1_snapshot: Mapping[str, Any],
    i4_3f2_snapshot: Mapping[str, Any],
    source_contract_checks: Mapping[str, bool],
    scenario_results: Sequence[Mapping[str, Any]],
    validated_provider_implementations: Sequence[str],
) -> dict[str, Any]:
    validate_i4_3f3_predecessors(
        i4_3f1_snapshot=i4_3f1_snapshot,
        i4_3f2_snapshot=i4_3f2_snapshot,
    )
    checks = {str(k): bool(v) for k, v in source_contract_checks.items()}
    failed = [name for name, value in checks.items() if not value]
    if failed:
        raise ReconstructionValidationError(f"I4.3F3 production source-contract checks failed: {failed}")
    scenarios = [dict(item) for item in scenario_results]
    if not scenarios or any(item.get("pass") is not True for item in scenarios):
        raise ReconstructionValidationError("I4.3F3 requires every production-validation scenario to pass.")
    providers = tuple(str(item).strip() for item in validated_provider_implementations)
    if len(providers) != 10 or len(set(providers)) != 10 or any(not item for item in providers):
        raise ReconstructionValidationError("I4.3F3 requires validation coverage for all 10 provider implementations.")
    return {
        "schema_version": I43F3_SCHEMA_VERSION,
        "phase": I43F3_PHASE,
        "status": "pass",
        "pass": True,
        "scope": {
            "production_validation_complete": True,
            "pre_sfm_resource_scaling_validated": True,
            "post_sfm_all_provider_wiring_validated": True,
            "temporal_source_order_validated": True,
            "engineer_override_validated": True,
            "deterministic_planning_validated": True,
            "authoritative_sfm_immutability_validated": True,
            "production_default_no_fixed_keyframe_cap": True,
        },
        "source_contract_checks": checks,
        "scenario_results": scenarios,
        "validated_provider_implementations": list(providers),
        "authority": {
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i4_3f_closed": True,
            "i4_3g_cross_provider_qualification_authorized": True,
            "provider_selected": False,
            "auto_policy_implemented": False,
            "permanent_global_view_cap": None,
        },
        "next": "I4.3G controlled cross-provider scenario qualification.",
    }


__all__ = [
    "I43F3_SCHEMA_VERSION",
    "I43F3_PHASE",
    "validate_i4_3f3_predecessors",
    "build_i4_3f3_closure_snapshot",
]
