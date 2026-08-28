"""I4.3D3 temporal-provider architecture closure.

This module is closure-only. It validates the already-produced I4.3D1 Video
Depth Anything and I4.3D2 RollingDepth binding snapshots, freezes their shared
canonical temporal-provider contract while preserving implementation-specific
temporal semantics, and authorizes I4.3E. It does not execute a backend, select
or promote a provider, establish physical metric scale, modify the authoritative
PyCOLMAP SfM frame, create measurement/metrology authority, or generate a mesh.
"""
from __future__ import annotations

from typing import Any, Mapping

from core.reconstruction.errors import ReconstructionValidationError

I43D3_SCHEMA_VERSION = 1
I43D3_PHASE = "I4.3D3"
I43D3_AUTHORITATIVE_FRAME_OWNER = "pycolmap_sfm"
I43D3_TEMPORAL_FAMILY = "temporal_depth"
I43D3_FRAME_BINDING = "align_to_authoritative_sfm"
I43D3_NATIVE_OUTPUT = "depth_evidence"
I43D3_IMPLEMENTATIONS = ("video_depth_anything", "rolling_depth")


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3D3 {label} is missing or malformed.")
    return value


def _require_false(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if mapping.get(key) is not False:
        raise ReconstructionValidationError(f"I4.3D3 requires {label}.{key}=False.")


def _require_true(mapping: Mapping[str, Any], key: str, *, label: str) -> None:
    if mapping.get(key) is not True:
        raise ReconstructionValidationError(f"I4.3D3 requires {label}.{key}=True.")


def _validate_common_binding(snapshot: Mapping[str, Any], *, phase: str, implementation: str) -> dict[str, Any]:
    if snapshot.get("phase") != phase or snapshot.get("status") != "pass" or snapshot.get("pass") is not True:
        raise ReconstructionValidationError(f"I4.3D3 requires passing {phase} binding evidence.")

    provider = _mapping(snapshot.get("provider"), label=f"{phase} provider")
    if provider.get("implementation") != implementation:
        raise ReconstructionValidationError(f"I4.3D3 {phase} implementation identity drifted.")
    if provider.get("family") != I43D3_TEMPORAL_FAMILY:
        raise ReconstructionValidationError(f"I4.3D3 {phase} must remain in temporal_depth family.")
    if provider.get("qualification_state") != "candidate" or provider.get("production_qualified") is not False:
        raise ReconstructionValidationError(f"I4.3D3 {phase} must remain candidate and not production-qualified.")
    if provider.get("available") is not True:
        raise ReconstructionValidationError(f"I4.3D3 requires an available {phase} runtime binding.")

    contract = _mapping(provider.get("implementation_contract"), label=f"{phase} implementation_contract")
    if contract.get("family") != I43D3_TEMPORAL_FAMILY:
        raise ReconstructionValidationError(f"I4.3D3 {phase} implementation family contract drifted.")
    if contract.get("frame_binding") != I43D3_FRAME_BINDING or contract.get("native_output") != I43D3_NATIVE_OUTPUT:
        raise ReconstructionValidationError(f"I4.3D3 {phase} frame/native-output contract drifted.")
    if contract.get("integration_stage") != "I4.3D":
        raise ReconstructionValidationError(f"I4.3D3 {phase} integration-stage contract drifted.")

    architecture = _mapping(snapshot.get("architecture"), label=f"{phase} architecture")
    if architecture.get("camera_pose_authority_owner") != I43D3_AUTHORITATIVE_FRAME_OWNER:
        raise ReconstructionValidationError(f"I4.3D3 {phase} changed camera-pose authority.")
    if architecture.get("frame_binding") != I43D3_FRAME_BINDING or architecture.get("native_output") != I43D3_NATIVE_OUTPUT:
        raise ReconstructionValidationError(f"I4.3D3 {phase} canonical frame/native-output contract drifted.")
    if architecture.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError(f"I4.3D3 {phase} introduced a permanent global view cap.")
    if architecture.get("adaptive_view_selection_deferred_to") != "I4.3F":
        raise ReconstructionValidationError(f"I4.3D3 {phase} moved adaptive view selection away from I4.3F.")
    _require_true(architecture, "progressive_status_supported", label=f"{phase} architecture")

    authority = _mapping(snapshot.get("authority"), label=f"{phase} authority")
    for key in (
        "backend_started",
        "provider_selected",
        "provider_promotion_performed",
        "metric_scale_claimed",
        "measurement_authoritative",
        "certified_metrology",
        "surface_mesh_generated",
    ):
        _require_false(authority, key, label=f"{phase} authority")

    qualification = _mapping(snapshot.get("qualification"), label=f"{phase} qualification")
    if qualification.get("qualification_state") != "candidate":
        raise ReconstructionValidationError(f"I4.3D3 {phase} qualification must remain candidate.")
    _require_true(qualification, "temporal_cross_provider_comparison_pending_i4_3g_i5", label=f"{phase} qualification")
    _require_false(qualification, "universal_provider_winner_assumed", label=f"{phase} qualification")

    return {
        "provider_id": str(provider.get("provider_id", "")),
        "implementation": implementation,
        "backend_version": str(provider.get("backend_version", "")),
        "temporal_mode": str(architecture.get("temporal_mode", "")),
    }


def build_i4_3d3_temporal_provider_closure(
    *,
    i4_3d1_snapshot: Mapping[str, Any],
    i4_3d2_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate D1/D2 bindings and freeze the I4.3D temporal-provider closure."""

    if not isinstance(i4_3d1_snapshot, Mapping) or not isinstance(i4_3d2_snapshot, Mapping):
        raise TypeError("i4_3d1_snapshot and i4_3d2_snapshot must be mappings.")

    vda = _validate_common_binding(
        i4_3d1_snapshot,
        phase="I4.3D1",
        implementation="video_depth_anything",
    )
    rolling = _validate_common_binding(
        i4_3d2_snapshot,
        phase="I4.3D2",
        implementation="rolling_depth",
    )

    d1_arch = _mapping(i4_3d1_snapshot.get("architecture"), label="I4.3D1 architecture")
    d1_auth = _mapping(i4_3d1_snapshot.get("authority"), label="I4.3D1 authority")
    if d1_arch.get("temporal_mode") != "official_streaming_cache" or d1_arch.get("stateful_sequence") is not True:
        raise ReconstructionValidationError("I4.3D3 requires VDA official streaming-cache temporal semantics.")
    _require_false(
        d1_auth,
        "estimated_model_intrinsics_consumed_for_canonical_projection",
        label="I4.3D1 authority",
    )
    _require_false(
        d1_auth,
        "model_metric_depth_promoted_to_canonical_metric_scale",
        label="I4.3D1 authority",
    )

    d2_arch = _mapping(i4_3d2_snapshot.get("architecture"), label="I4.3D2 architecture")
    d2_auth = _mapping(i4_3d2_snapshot.get("authority"), label="I4.3D2 authority")
    d2_qual = _mapping(i4_3d2_snapshot.get("qualification"), label="I4.3D2 qualification")
    if d2_arch.get("temporal_mode") != "multi_dilation_snippet_coalignment":
        raise ReconstructionValidationError("I4.3D3 requires RollingDepth multi-dilation snippet co-alignment semantics.")
    if d2_arch.get("sequence_level_alignment") != "global_affine_to_authoritative_sfm":
        raise ReconstructionValidationError("I4.3D3 requires one sequence-level RollingDepth alignment to SfM.")
    _require_false(d2_arch, "per_frame_alignment", label="I4.3D2 architecture")
    _require_false(d2_auth, "rolling_depth_pose_authority_promoted", label="I4.3D2 authority")
    _require_false(d2_auth, "sequence_affine_establishes_physical_metric_scale", label="I4.3D2 authority")
    _require_true(d2_qual, "video_depth_anything_binding_retained", label="I4.3D2 qualification")
    _require_true(d2_qual, "temporal_provider_closure_pending_i4_3d3", label="I4.3D2 qualification")

    return {
        "schema_version": I43D3_SCHEMA_VERSION,
        "phase": I43D3_PHASE,
        "status": "pass",
        "pass": True,
        "decision": {
            "i4_3d_temporal_provider_architecture_closed": True,
            "i4_3e_authorized": True,
            "provider_selected": False,
            "provider_promoted": False,
            "production_provider_winner_selected": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "temporal_provider_contract": {
            "family": I43D3_TEMPORAL_FAMILY,
            "implementations": list(I43D3_IMPLEMENTATIONS),
            "native_output": I43D3_NATIVE_OUTPUT,
            "frame_binding": I43D3_FRAME_BINDING,
            "camera_pose_authority_owner": I43D3_AUTHORITATIVE_FRAME_OWNER,
            "single_authoritative_sfm_frame_required": True,
            "permanent_global_view_cap": None,
            "adaptive_view_selection_deferred_to": "I4.3F",
            "progressive_status_required": True,
            "sequence_order_source": "sfm_workspace.source_sequence_index",
            "provider_specific_temporal_semantics_preserved": True,
        },
        "providers": {
            "video_depth_anything": {
                **vda,
                "stateful_sequence": True,
                "canonical_projection_pose_source": "pycolmap_sfm",
                "estimated_model_intrinsics_are_canonical_authority": False,
                "model_metric_depth_is_canonical_metric_authority": False,
            },
            "rolling_depth": {
                **rolling,
                "stateful_sequence": True,
                "sequence_level_alignment": "global_affine_to_authoritative_sfm",
                "per_frame_alignment": False,
                "canonical_projection_pose_source": "pycolmap_sfm",
                "sequence_affine_establishes_physical_metric_scale": False,
            },
        },
        "authority": {
            "backend_started": False,
            "provider_selected": False,
            "provider_promotion_performed": False,
            "camera_pose_authority_owner": I43D3_AUTHORITATIVE_FRAME_OWNER,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "qualification": {
            "video_depth_anything_state": "candidate",
            "rolling_depth_state": "candidate",
            "temporal_cross_provider_comparison_pending_i4_3g_i5": True,
            "scenario_specific_qualification_required": True,
            "universal_provider_winner_assumed": False,
            "i4_3g_temporal_video_ab_required": True,
        },
        "roadmap": {
            "i4_3e": "remaining_installed_single_image_depth_provider_adapters",
            "i4_3f": "adaptive_frame_and_view_selection",
            "i4_3g": "controlled_cross_provider_qualification",
            "i4_3h": "provider_architecture_closure",
        },
        "next": "I4.3E remaining installed depth-model provider adapters. No temporal-provider promotion before I4.3G/I5 evidence.",
    }


__all__ = [
    "I43D3_SCHEMA_VERSION",
    "I43D3_PHASE",
    "I43D3_IMPLEMENTATIONS",
    "build_i4_3d3_temporal_provider_closure",
]
