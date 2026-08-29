"""I4.3G4A single-image source/runtime/evidence binding preflight.

G4A is execution-free. It binds the three frozen G4 scenarios to the already
qualified G2 source/SfM contexts, verifies formal G3C authorization, probes the
six single-image production runtimes, reconciles G1's older run-status schema
with the regression-protected G3 terminal quality rejection semantic, and
fails closed on evidence/runtime/storage drift.

No SfM or dense provider inference is started here. No provider is selected or
promoted and no metric/measurement/mesh authority is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.cross_provider_qualification import ComparisonMode
from core.reconstruction.errors import ReconstructionValidationError

I43G4A_SCHEMA_VERSION = 1
I43G4A_PHASE = "I4.3G4A"
I43G3C_FROZEN_CLOSURE_SHA256 = "18f1c54d421277cf4dd1cf81a96e0bb74b5b1846a9d38f099fdd0222f5f350bd"

I43G4A_SCENARIOS = (
    "g4_single_image_complete_room",
    "g4_single_image_low_texture",
    "g4_single_image_weak_overlap",
)
I43G4A_G2_SCENARIO_MAP = {
    "g4_single_image_complete_room": "g2_complete_room_mvs_da3",
    "g4_single_image_low_texture": "g2_low_texture_mvs_da3",
    "g4_single_image_weak_overlap": "g2_weak_overlap_mvs_da3",
}
I43G4A_PROVIDER_IMPLEMENTATIONS = (
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)
I43G4A_MANUAL_REVIEW_DIMENSIONS = (
    "global_completeness",
    "local_shape_plausibility",
    "edge_definition",
    "small_detail_preservation",
    "texture_sensitivity",
    "cross_view_stability_after_sfm_alignment",
    "false_geometry_and_floaters",
)
I43G4A_G1_STATUS_VALUES = ("pass", "degraded", "failed", "cancelled", "unavailable")
I43G4A_QUALITY_STATUS_VALUES = ("pass", "degraded", "rejected")
I43G4A_INFRASTRUCTURE_STATUS_VALUES = ("failed", "cancelled", "unavailable")
_REQUIRED_AUTHORITY_FALSE = (
    "authoritative_sfm_modified",
    "provider_selected",
    "provider_promoted",
    "auto_policy_implemented",
    "hybrid_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "certified_metrology",
    "surface_mesh_generated",
)


@dataclass(frozen=True, slots=True)
class SingleImageScenarioBinding:
    scenario_id: str
    predecessor_g2_scenario_id: str
    source_directory: str
    source_image_count: int
    source_asset_set_sha256: str
    source_manifest_path: str
    source_manifest_sha256: str
    authoritative_sfm_snapshot_path: str
    authoritative_sfm_snapshot_sha256: str
    authoritative_sfm_frame_id: str
    registered_view_count: int
    landmark_count: int
    g2b_production_snapshot_path: str
    g2b_production_snapshot_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "predecessor_g2_scenario_id": self.predecessor_g2_scenario_id,
            "source_directory": self.source_directory,
            "source_image_count": self.source_image_count,
            "source_asset_set_sha256": self.source_asset_set_sha256,
            "source_manifest_path": self.source_manifest_path,
            "source_manifest_sha256": self.source_manifest_sha256,
            "authoritative_sfm_snapshot_path": self.authoritative_sfm_snapshot_path,
            "authoritative_sfm_snapshot_sha256": self.authoritative_sfm_snapshot_sha256,
            "authoritative_sfm_frame_id": self.authoritative_sfm_frame_id,
            "registered_view_count": self.registered_view_count,
            "landmark_count": self.landmark_count,
            "g2b_production_snapshot_path": self.g2b_production_snapshot_path,
            "g2b_production_snapshot_sha256": self.g2b_production_snapshot_sha256,
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3G4A requires mapping section: {label}")
    return value


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value.lower())


def validate_i4_3g4a_predecessors(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g3c_closure: Mapping[str, Any],
) -> None:
    if i4_3g1_snapshot.get("phase") != "I4.3G1" or i4_3g1_snapshot.get("status") != "pass" or i4_3g1_snapshot.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4A requires the passing frozen I4.3G1 framework snapshot.")

    coverage = _mapping(i4_3g1_snapshot.get("coverage"), label="G1 coverage")
    if coverage.get("g4_scenarios") != 3 or coverage.get("provider_implementation_count") != 10 or coverage.get("metric_count") != 21:
        raise ReconstructionValidationError("I4.3G4A refuses G1 coverage/registry drift.")

    scenarios = i4_3g1_snapshot.get("scenario_matrix")
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise ReconstructionValidationError("I4.3G4A requires the frozen G1 scenario matrix.")
    g4 = [item for item in scenarios if isinstance(item, Mapping) and item.get("substep") == "I4.3G4"]
    if len(g4) != 3:
        raise ReconstructionValidationError("I4.3G4A requires exactly three frozen G4 scenarios.")
    by_id = {str(item.get("scenario_id")): item for item in g4}
    if tuple(sid for sid in I43G4A_SCENARIOS if sid in by_id) != I43G4A_SCENARIOS or len(by_id) != 3:
        raise ReconstructionValidationError("I4.3G4A frozen G4 scenario IDs/order drifted.")
    expected_modes = (ComparisonMode.PRODUCTION_NATIVE.value, ComparisonMode.MATCHED_VIEW_DIAGNOSTIC.value)
    for sid in I43G4A_SCENARIOS:
        item = by_id[sid]
        if tuple(item.get("provider_implementations", ())) != I43G4A_PROVIDER_IMPLEMENTATIONS:
            raise ReconstructionValidationError(f"I4.3G4A provider-set/order drift for {sid}.")
        if tuple(item.get("comparison_modes", ())) != expected_modes:
            raise ReconstructionValidationError(f"I4.3G4A comparison-mode drift for {sid}.")
        if item.get("new_real_execution_required") is not True or item.get("manual_review_required") is not True:
            raise ReconstructionValidationError(f"I4.3G4A qualification obligations drift for {sid}.")
        if item.get("temporal_order_required") is not False:
            raise ReconstructionValidationError(f"I4.3G4A single-image scenario incorrectly requires temporal order: {sid}.")
        if tuple(item.get("manual_review_dimensions", ())) != I43G4A_MANUAL_REVIEW_DIMENSIONS:
            raise ReconstructionValidationError(f"I4.3G4A manual-review dimensions drift for {sid}.")

    schema = _mapping(i4_3g1_snapshot.get("run_evidence_schema"), label="G1 run-evidence schema")
    if tuple(schema.get("status_values", ())) != I43G4A_G1_STATUS_VALUES:
        raise ReconstructionValidationError("I4.3G4A expected the frozen G1 v1 run-status vocabulary before refinement.")

    fairness = _mapping(i4_3g1_snapshot.get("fairness"), label="G1 fairness")
    for key in (
        "production_native_comparison_primary",
        "matched_view_diagnostic_secondary",
        "same_authoritative_sfm_within_static_scenario",
        "same_source_identity_within_scenario",
        "provider_specific_adaptive_plans_persisted",
        "manual_review_same_camera_and_framing",
        "independent_visual_auto_fit_forbidden",
    ):
        if fairness.get(key) is not True:
            raise ReconstructionValidationError(f"I4.3G4A G1 fairness drift: {key}")

    authority = _mapping(i4_3g1_snapshot.get("authority"), label="G1 authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4A requires unchanged PyCOLMAP authority and no permanent view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4A refuses G1 authority drift: {drifted}")

    if i4_3g3c_closure.get("phase") != "I4.3G3C" or i4_3g3c_closure.get("status") != "formally_closed" or i4_3g3c_closure.get("pass") is not True:
        raise ReconstructionValidationError("I4.3G4A requires the formally closed passing I4.3G3C closure.")
    if i4_3g3c_closure.get("i4_3g3_closed") is not True or i4_3g3c_closure.get("i4_3g4_authorized") is not True:
        raise ReconstructionValidationError("I4.3G4A requires explicit G3 closure and G4 authorization.")
    if i4_3g3c_closure.get("all_required_temporal_scenarios_accounted") is not True or i4_3g3c_closure.get("all_required_manual_reviews_complete") is not True:
        raise ReconstructionValidationError("I4.3G4A requires complete G3 production/manual-review accounting.")
    g3_authority = _mapping(i4_3g3c_closure.get("authority"), label="G3C authority")
    if g3_authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or g3_authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError("I4.3G4A refuses G3C camera-authority/view-cap drift.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if g3_authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"I4.3G4A refuses G3C authority drift: {drifted}")
    decision = _mapping(i4_3g3c_closure.get("decision"), label="G3C decision")
    if decision.get("quality_failures_preserved") is not True or decision.get("scenario_specific_verdicts_preserved") is not True:
        raise ReconstructionValidationError("I4.3G4A requires G3's preserved terminal quality-outcome semantics.")
    motion = _mapping(_mapping(i4_3g3c_closure.get("scenario_synthesis"), label="G3C synthesis").get("g3_temporal_motion"), label="G3C motion synthesis")
    statuses = _mapping(motion.get("production_statuses"), label="G3C motion production statuses")
    if "rejected" not in set(str(value) for value in statuses.values()):
        raise ReconstructionValidationError("I4.3G4A requires frozen proof that 'rejected' is a legitimate terminal provider-quality status.")


def build_g4a_scenario_binding(
    *,
    scenario_id: str,
    g2b_snapshot: Mapping[str, Any],
    g2b_snapshot_path: str,
    g2b_snapshot_sha256: str,
) -> SingleImageScenarioBinding:
    if scenario_id not in I43G4A_G2_SCENARIO_MAP:
        raise ReconstructionValidationError(f"Unsupported I4.3G4A scenario: {scenario_id}")
    expected_g2 = I43G4A_G2_SCENARIO_MAP[scenario_id]
    if g2b_snapshot.get("phase") != "I4.3G2B" or g2b_snapshot.get("status") != "production_native_complete" or g2b_snapshot.get("pass") is not True:
        raise ReconstructionValidationError(f"{scenario_id} requires a complete passing I4.3G2B production snapshot.")
    scenario = _mapping(g2b_snapshot.get("scenario"), label=f"{scenario_id} G2B scenario")
    if scenario.get("scenario_id") != expected_g2:
        raise ReconstructionValidationError(f"{scenario_id} must bind the matching {expected_g2} G2 context.")
    authority = _mapping(g2b_snapshot.get("authority"), label=f"{scenario_id} G2B authority")
    if authority.get("camera_pose_authority_owner") != "pycolmap_sfm" or authority.get("same_authoritative_sfm_shared_by_both_providers") is not True:
        raise ReconstructionValidationError(f"{scenario_id} G2B context does not preserve one authoritative PyCOLMAP SfM.")
    if authority.get("permanent_global_view_cap") is not None:
        raise ReconstructionValidationError(f"{scenario_id} G2B context introduced a permanent global view cap.")
    drifted = [key for key in _REQUIRED_AUTHORITY_FALSE if authority.get(key) is not False]
    if drifted:
        raise ReconstructionValidationError(f"{scenario_id} G2B authority drift: {drifted}")

    source_identity = _mapping(g2b_snapshot.get("source_identity"), label=f"{scenario_id} G2B source identity")
    sfm = _mapping(g2b_snapshot.get("authoritative_sfm"), label=f"{scenario_id} authoritative SfM")
    sfm_snapshot = _mapping(sfm.get("snapshot"), label=f"{scenario_id} SfM snapshot artifact")
    source_dir = str(scenario.get("source_directory") or "").strip()
    source_count = int(scenario.get("source_image_count") or 0)
    source_asset_sha = str(scenario.get("source_asset_set_sha256") or "").strip()
    source_manifest_path = str(source_identity.get("path") or "").strip()
    source_manifest_sha = str(source_identity.get("sha256") or "").strip()
    sfm_path = str(sfm_snapshot.get("path") or "").strip()
    sfm_sha = str(sfm_snapshot.get("sha256") or "").strip()
    frame_id = str(sfm.get("frame_id") or "").strip()
    registered = int(sfm.get("registered_view_count") or 0)
    landmarks = int(sfm.get("landmark_count") or 0)
    if not source_dir or source_count < 3 or not _valid_sha256(source_asset_sha):
        raise ReconstructionValidationError(f"{scenario_id} G2B source identity is incomplete.")
    if not source_manifest_path or not _valid_sha256(source_manifest_sha):
        raise ReconstructionValidationError(f"{scenario_id} G2B source-manifest artifact identity is incomplete.")
    if not sfm_path or not _valid_sha256(sfm_sha) or not frame_id or registered < 3 or landmarks <= 0:
        raise ReconstructionValidationError(f"{scenario_id} authoritative SfM identity is incomplete.")
    if not str(g2b_snapshot_path).strip() or not _valid_sha256(g2b_snapshot_sha256):
        raise ReconstructionValidationError(f"{scenario_id} G2B production snapshot identity is incomplete.")
    return SingleImageScenarioBinding(
        scenario_id=scenario_id,
        predecessor_g2_scenario_id=expected_g2,
        source_directory=source_dir,
        source_image_count=source_count,
        source_asset_set_sha256=source_asset_sha,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=source_manifest_sha,
        authoritative_sfm_snapshot_path=sfm_path,
        authoritative_sfm_snapshot_sha256=sfm_sha,
        authoritative_sfm_frame_id=frame_id,
        registered_view_count=registered,
        landmark_count=landmarks,
        g2b_production_snapshot_path=str(g2b_snapshot_path),
        g2b_production_snapshot_sha256=g2b_snapshot_sha256,
    )


def validate_runtime_probe(runtime: Mapping[str, Any], *, implementation: str) -> bool:
    if implementation not in I43G4A_PROVIDER_IMPLEMENTATIONS:
        raise ReconstructionValidationError(f"Unsupported I4.3G4A provider runtime: {implementation}")
    if runtime.get("available") is not True or runtime.get("adapter_importable") is not True:
        return False
    if runtime.get("torch_available") is not True or runtime.get("cuda_available") is not True:
        return False
    if runtime.get("checkpoint_ready") is not True:
        return False
    if implementation == "metric3d_v2":
        return bool(runtime.get("source_ready") is True and str(runtime.get("source_revision") or "").strip())
    if implementation == "unidepth_v2":
        return bool(
            runtime.get("source_ready") is True
            and runtime.get("source_checkout_matches_marker") is True
            and runtime.get("official_runtime_importable") is True
            and runtime.get("official_runtime_source_matches_checkout") is True
            and str(runtime.get("source_revision") or "").strip()
        )
    if implementation == "moge_2":
        return bool(
            runtime.get("source_revision_pinned") is True
            and runtime.get("official_runtime_importable") is True
            and runtime.get("official_runtime_source_matches_distribution") is True
            and str(runtime.get("checkpoint_revision") or "").strip()
        )
    if implementation == "depth_pro":
        # E4 deliberately distinguishes the exact installed official VCS commit
        # from whether the project requirement line itself is commit-pinned.
        # Fresh-install reproducibility is recorded provenance, not a G4 runtime
        # readiness condition for the already-qualified workstation state.
        return bool(
            runtime.get("source_revision_resolved") is True
            and runtime.get("official_runtime_importable") is True
            and runtime.get("official_runtime_source_matches_distribution") is True
            and _valid_sha256(runtime.get("checkpoint_sha256"))
        )
    if implementation == "depth_anything_v2":
        # E5 records that the installer/checkpoint requested refs may be `main`,
        # while binding the actual local source checkout to its compatibility
        # marker and hashing the exact local checkpoint. Those requested-ref
        # pinning flags are therefore audit metadata rather than readiness gates.
        return bool(
            runtime.get("source_checkout_matches_marker") is True
            and runtime.get("official_runtime_importable") is True
            and runtime.get("official_runtime_source_matches_checkout") is True
            and _valid_sha256(runtime.get("checkpoint_sha256"))
        )
    if implementation == "marigold":
        # E6 explicitly records a version-ranged Diffusers requirement. Runtime
        # readiness is instead bound to the installed distribution plus the
        # locally resolved, commit-pinned Hugging Face snapshot and its hashes.
        return bool(
            runtime.get("official_runtime_importable") is True
            and runtime.get("official_runtime_source_matches_distribution") is True
            and runtime.get("checkpoint_revision_pinned") is True
            and _valid_sha256(runtime.get("checkpoint_marker_sha256"))
            and _valid_sha256(runtime.get("model_index_sha256"))
        )
    return False


def build_i4_3g4a_preflight_snapshot(
    *,
    i4_3g1_snapshot: Mapping[str, Any],
    i4_3g3c_closure: Mapping[str, Any],
    scenario_bindings: Sequence[SingleImageScenarioBinding],
    runtime_probes: Mapping[str, Mapping[str, Any]],
    free_disk_bytes: int,
    minimum_free_disk_bytes: int,
) -> dict[str, Any]:
    validate_i4_3g4a_predecessors(i4_3g1_snapshot=i4_3g1_snapshot, i4_3g3c_closure=i4_3g3c_closure)
    bindings = tuple(scenario_bindings)
    if len(bindings) != 3 or any(not isinstance(item, SingleImageScenarioBinding) for item in bindings):
        raise ReconstructionValidationError("I4.3G4A requires exactly three explicit single-image scenario bindings.")
    by_id = {item.scenario_id: item for item in bindings}
    if tuple(sid for sid in I43G4A_SCENARIOS if sid in by_id) != I43G4A_SCENARIOS or len(by_id) != 3:
        raise ReconstructionValidationError("I4.3G4A scenario binding coverage is incomplete or duplicated.")
    if set(runtime_probes) != set(I43G4A_PROVIDER_IMPLEMENTATIONS):
        missing = sorted(set(I43G4A_PROVIDER_IMPLEMENTATIONS) - set(runtime_probes))
        extra = sorted(set(runtime_probes) - set(I43G4A_PROVIDER_IMPLEMENTATIONS))
        raise ReconstructionValidationError(f"I4.3G4A runtime-probe coverage mismatch: missing={missing} extra={extra}")
    if isinstance(free_disk_bytes, bool) or int(free_disk_bytes) < 0:
        raise ReconstructionValidationError("I4.3G4A free_disk_bytes must be a non-negative integer.")
    if isinstance(minimum_free_disk_bytes, bool) or int(minimum_free_disk_bytes) <= 0:
        raise ReconstructionValidationError("I4.3G4A minimum_free_disk_bytes must be a positive integer.")

    runtime_payload: dict[str, Any] = {}
    all_runtimes_ready = True
    for implementation in I43G4A_PROVIDER_IMPLEMENTATIONS:
        payload = dict(runtime_probes[implementation])
        ready = validate_runtime_probe(payload, implementation=implementation)
        payload["g4a_ready"] = ready
        runtime_payload[implementation] = payload
        all_runtimes_ready = all_runtimes_ready and ready

    storage_ready = int(free_disk_bytes) >= int(minimum_free_disk_bytes)
    all_sources_bound = len(bindings) == 3
    ready = bool(all_sources_bound and all_runtimes_ready and storage_ready)
    return {
        "schema_version": I43G4A_SCHEMA_VERSION,
        "phase": I43G4A_PHASE,
        "status": "ready" if ready else "not_ready",
        "pass": ready,
        "scope": {
            "execution_free_preflight": True,
            "sfm_started": False,
            "single_image_provider_inference_started": False,
            "matched_view_diagnostic_started": False,
            "g2_source_sfm_context_reused": True,
            "new_sfm_variability_introduced": False,
        },
        "scenario_bindings": [by_id[sid].as_dict() for sid in I43G4A_SCENARIOS],
        "binding_summary": {
            "required_scenario_count": 3,
            "bound_scenario_count": len(bindings),
            "all_g4_sources_bound": all_sources_bound,
            "g2_context_reuse_map": dict(I43G4A_G2_SCENARIO_MAP),
        },
        "runtime": runtime_payload,
        "runtime_summary": {
            "required_provider_count": 6,
            "ready_provider_count": sum(bool(runtime_payload[item]["g4a_ready"]) for item in I43G4A_PROVIDER_IMPLEMENTATIONS),
            "all_single_image_runtimes_ready": all_runtimes_ready,
        },
        "runtime_provenance_policy": {
            "installed_runtime_identity_is_authoritative_for_g4a": True,
            "fresh_install_reproducibility_flags_recorded_not_readiness_gates": True,
            "depth_pro_requirement_commit_pinned_recorded": runtime_payload["depth_pro"].get("requirement_commit_pinned"),
            "depth_anything_v2_installer_ref_commit_pinned_recorded": runtime_payload["depth_anything_v2"].get("installer_ref_commit_pinned"),
            "depth_anything_v2_checkpoint_requested_ref_commit_pinned_recorded": runtime_payload["depth_anything_v2"].get("checkpoint_revision_commit_pinned"),
            "marigold_runtime_requirement_exact_pinned_recorded": runtime_payload["marigold"].get("runtime_requirement_exact_pinned"),
            "source_checkpoint_integrity_still_required": True,
        },
        "storage": {
            "free_disk_bytes": int(free_disk_bytes),
            "minimum_free_disk_bytes": int(minimum_free_disk_bytes),
            "free_disk_gib": int(free_disk_bytes) / 1024**3,
            "minimum_free_disk_gib": int(minimum_free_disk_bytes) / 1024**3,
            "storage_ready_for_g4b": storage_ready,
            "threshold_is_preflight_safety_floor_not_quality_policy": True,
        },
        "status_contract": {
            "g1_snapshot_unchanged": True,
            "g1_declared_status_values": list(I43G4A_G1_STATUS_VALUES),
            "g4_quality_terminal_status_values": list(I43G4A_QUALITY_STATUS_VALUES),
            "infrastructure_terminal_status_values": list(I43G4A_INFRASTRUCTURE_STATUS_VALUES),
            "rejected_is_terminal_provider_quality_evidence": True,
            "rejected_is_not_automatically_software_failure": True,
            "refinement_source": "I4.3G3B/G3C regression-protected production rejection semantics",
            "threshold_weakening_to_avoid_rejection_forbidden": True,
        },
        "manual_review_contract": {
            "required_for_all_three_scenarios": True,
            "dimensions": list(I43G4A_MANUAL_REVIEW_DIMENSIONS),
            "not_assessable_allowed_with_explicit_reason": True,
            "manual_review_cannot_override_production_rejection": True,
            "independent_provider_auto_fit_forbidden": True,
            "deferred_to": "I4.3G4C",
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
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
            "i4_3g4a_preflight_closed": ready,
            "i4_3g4b_production_native_execution_authorized": ready,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "global_scalar_score_forbidden": True,
            "universal_provider_winner_assumed": False,
        },
        "next": "I4.3G4B real single-image production-native qualification." if ready else "Resolve G4A preflight readiness failures without changing quality thresholds.",
    }


__all__ = [
    "I43G4A_SCHEMA_VERSION",
    "I43G4A_PHASE",
    "I43G3C_FROZEN_CLOSURE_SHA256",
    "I43G4A_SCENARIOS",
    "I43G4A_G2_SCENARIO_MAP",
    "I43G4A_PROVIDER_IMPLEMENTATIONS",
    "I43G4A_MANUAL_REVIEW_DIMENSIONS",
    "I43G4A_G1_STATUS_VALUES",
    "I43G4A_QUALITY_STATUS_VALUES",
    "I43G4A_INFRASTRUCTURE_STATUS_VALUES",
    "SingleImageScenarioBinding",
    "validate_i4_3g4a_predecessors",
    "build_g4a_scenario_binding",
    "validate_runtime_probe",
    "build_i4_3g4a_preflight_snapshot",
]
