"""I5.6A MVS robustness scenario admission and envelope design.

I5.6A is admission-only. It reuses frozen authoritative PyCOLMAP SfM reports
from I5.2/I5.3/I5.5, verifies that the exact production MVS inputs are still
present and hash-consistent, records upstream-blocked exclusions, and freezes
the existing adaptive COLMAP MVS production path for I5.6B.

No SfM, MVS, provider comparison, threshold tuning, Auto, Hybrid, MP-SfM,
metric-scale promotion, or production-default modification is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.dense_provider_architecture import (
    DenseProviderExecutionResult,
    DenseProviderImplementation,
    DenseProviderRunStatus,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.mvs_provider import (
    I43B_PROVIDER_ID,
    I43B_DEFAULT_PATCH_MATCH_CACHE_GIB,
    I43B_DEFAULT_SOURCE_NEIGHBORS,
    I43B_DEFAULT_STEREO_FUSION_CACHE_GIB,
    MvsProductionConfig,
    _resolve_registered_sfm_inputs,
)
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict

I56A_SCHEMA_VERSION = 1
I56A_PHASE = "I5.6A"
I56A_RECEIPT_PHASE = "I5.6A_ADMISSION_RECEIPT"
I56A_SCENARIO_IDS = (
    "castle_control",
    "castle_low_overlap",
    "castle_weak_parallax",
    "castle_overexposure",
    "castle_symmetry",
    "castle_dynamic_proxy",
)

_SCENARIO_EXPECTATIONS: dict[str, tuple[int, int]] = {
    "castle_control": (19, 19),
    "castle_low_overlap": (7, 6),
    "castle_weak_parallax": (7, 7),
    "castle_overexposure": (19, 19),
    "castle_symmetry": (19, 12),
    "castle_dynamic_proxy": (19, 19),
}


@dataclass(frozen=True, slots=True)
class I56AAdmissionPolicy:
    scenario_ids: tuple[str, ...] = I56A_SCENARIO_IDS
    provider_id: str = I43B_PROVIDER_ID
    implementation: str = DenseProviderImplementation.COLMAP_MVS.value
    adaptive_selection: bool = True
    engineer_override: str | None = None
    source_neighbors: int = I43B_DEFAULT_SOURCE_NEIGHBORS
    patch_match_cache_gib: float = I43B_DEFAULT_PATCH_MATCH_CACHE_GIB
    stereo_fusion_cache_gib: float = I43B_DEFAULT_STEREO_FUSION_CACHE_GIB
    max_image_size: int = -1
    geometric_consistency: bool = True
    sequential_execution: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I56A_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.6A scenario matrix is frozen to the six admitted Castle-P19 cases.")
        if self.provider_id != I43B_PROVIDER_ID or self.implementation != DenseProviderImplementation.COLMAP_MVS.value:
            raise ReconstructionValidationError("I5.6A must bind the existing COLMAP MVS production provider.")
        if self.adaptive_selection is not True or self.engineer_override is not None:
            raise ReconstructionValidationError("I5.6A freezes production adaptive selection with no engineer override.")
        if self.geometric_consistency is not True:
            raise ReconstructionValidationError("I5.6A refuses to weaken MVS geometric consistency.")
        if self.sequential_execution is not True:
            raise ReconstructionValidationError("I5.6A formal replay is sequential by design.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.6A cannot introduce a permanent global view cap.")
        if self.result_driven_retuning is not False:
            raise ReconstructionValidationError("I5.6A forbids result-driven retuning.")
        # Reuse the production config validator so admission cannot silently drift.
        MvsProductionConfig(
            source_neighbors=self.source_neighbors,
            patch_match_cache_gib=self.patch_match_cache_gib,
            stereo_fusion_cache_gib=self.stereo_fusion_cache_gib,
            max_image_size=self.max_image_size,
            geometric_consistency=self.geometric_consistency,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "provider_id": self.provider_id,
            "implementation": self.implementation,
            "production_adapter": "MvsProductionProviderAdapter",
            "adaptive_preparation": "prepare_adaptive_provider_execution",
            "adaptive_selection": self.adaptive_selection,
            "engineer_override": self.engineer_override,
            "source_neighbors": self.source_neighbors,
            "patch_match_cache_gib": self.patch_match_cache_gib,
            "stereo_fusion_cache_gib": self.stereo_fusion_cache_gib,
            "max_image_size": self.max_image_size,
            "geometric_consistency": self.geometric_consistency,
            "sequential_execution": self.sequential_execution,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _require(node: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if node.get(key) != expected:
        raise ReconstructionValidationError(f"{label}.{key} must be {expected!r}; found {node.get(key)!r}.")


def _valid_sha(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _validate_binding(binding: I51EvidenceBinding, *, label: str) -> None:
    if not str(binding.path).strip() or not _valid_sha(binding.sha256):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_pairs(pairs: tuple[tuple[str, str], ...]) -> str:
    h = hashlib.sha256()
    for name, digest in pairs:
        h.update(name.encode("utf-8")); h.update(b"\0"); h.update(digest.encode("ascii")); h.update(b"\n")
    return h.hexdigest()


def validate_i5_5c_for_i5_6(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.5C closure")
    _require(payload, "phase", "I5.5C", label="I5.5C closure")
    _require(payload, "status", "dynamic_scene_handling_robustness_qualified_and_closed", label="I5.5C closure")
    _require(payload, "pass", True, label="I5.5C closure")
    scope = _mapping(payload.get("scope"), label="I5.5C scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.5C scope")
    for key in ("sfm_rerun", "dense_provider_rerun", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.5C scope")
    decision = _mapping(payload.get("decision"), label="I5.5C decision")
    for key in ("i5_5_complete", "i5_6_authorized"):
        _require(decision, key, True, label="I5.5C decision")
    for key in ("i5_6_started", "provider_specific_robustness_started", "auto_policy_implemented", "hybrid_authorized"):
        _require(decision, key, False, label="I5.5C decision")
    authority = _mapping(payload.get("authority"), label="I5.5C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.5C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.5C authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology"):
        _require(authority, key, False, label="I5.5C authority")
    return deepcopy(dict(payload))


def validate_i5_6a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.6A receipt")
    _require(payload, "phase", I56A_RECEIPT_PHASE, label="I5.6A receipt")
    _require(payload, "status", "pass", label="I5.6A receipt")
    _require(payload, "pass", True, label="I5.6A receipt")
    scope = _mapping(payload.get("scope"), label="I5.6A receipt scope")
    for key in ("mvs_execution", "sfm_rerun", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.6A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.6A receipt counts")
    if int(counts.get("combined_exercised_tests", 0)) <= 0:
        raise ReconstructionValidationError("I5.6A receipt must record exercised tests.")
    return deepcopy(dict(payload))


def analyze_i5_6a_sfm_report(
    report: Mapping[str, Any], *, scenario_id: str, expected_input: int, expected_registered: int,
) -> dict[str, Any]:
    if scenario_id not in I56A_SCENARIO_IDS:
        raise ReconstructionValidationError(f"I5.6A unknown scenario: {scenario_id}")
    _require(report, "phase", "8.3D", label=f"I5.6A {scenario_id} report")
    _require(report, "real_sfm_executed", True, label=f"I5.6A {scenario_id} report")
    quality = _mapping(report.get("quality_assessment"), label=f"I5.6A {scenario_id} quality")
    status = str(quality.get("status", ""))
    if status not in {"accepted", "degraded"}:
        raise ReconstructionValidationError(f"I5.6A {scenario_id} requires reusable non-rejected authoritative SfM.")
    sfm = offline_sfm_result_from_dict(_mapping(report.get("sfm_result"), label=f"I5.6A {scenario_id} sfm_result"))
    if sfm.backend_id != PYCOLMAP_BACKEND_ID:
        raise ReconstructionValidationError(f"I5.6A {scenario_id} must reuse authoritative PyCOLMAP SfM.")
    if sfm.statistics.input_keyframes != int(expected_input) or sfm.statistics.registered_keyframes != int(expected_registered):
        raise ReconstructionValidationError(
            f"I5.6A {scenario_id} registration does not match frozen expectation: "
            f"{sfm.statistics.registered_keyframes}/{sfm.statistics.input_keyframes}"
        )
    if sfm.gauge.unit_to_metre is not None or sfm.gauge.scale_status.value != "arbitrary":
        raise ReconstructionValidationError(f"I5.6A {scenario_id} SfM gauge must remain arbitrary-scale.")
    inputs = _resolve_registered_sfm_inputs(sfm)
    required = [name for name, _ in inputs.sparse_sha256 if name in {"cameras.bin", "images.bin", "points3D.bin"}]
    if set(required) != {"cameras.bin", "images.bin", "points3D.bin"}:
        raise ReconstructionValidationError(f"I5.6A {scenario_id} sparse authority is incomplete.")
    return {
        "scenario_id": scenario_id,
        "input_images": sfm.statistics.input_keyframes,
        "registered_images": sfm.statistics.registered_keyframes,
        "registration_ratio": sfm.statistics.registered_keyframes / sfm.statistics.input_keyframes,
        "quality_status": status,
        "statistics": sfm.statistics.as_dict(),
        "workspace_root": str(inputs.workspace_root),
        "selected_model_id": inputs.selected_model_id,
        "registered_image_count": len(inputs.image_names),
        "registered_image_names": list(inputs.image_names),
        "registered_images_sha256": _hash_pairs(inputs.image_sha256),
        "selected_sparse_sha256": _hash_pairs(inputs.sparse_sha256),
        "required_sparse_files_verified": sorted(required),
        "mvs_input_integrity_verified": True,
        "sfm_rehydrated_without_rerun": True,
        "camera_pose_authority_owner": "pycolmap_sfm",
        "gauge": "arbitrary",
        "unit_to_metre": None,
    }


def build_i5_6a_historical_mvs_anchor(i4_2d_doc: Path, i4_2f_doc: Path) -> dict[str, Any]:
    d = Path(i4_2d_doc).expanduser().resolve(); f = Path(i4_2f_doc).expanduser().resolve()
    if not d.is_file() or not f.is_file():
        raise ReconstructionValidationError("I5.6A TempleRing historical anchor documents are missing.")
    dt = d.read_text(encoding="utf-8").lower(); ft = f.read_text(encoding="utf-8").lower()
    for token in ("47/47", "geometric depth maps", "geometric normal maps", "geometric consistency", "fused.ply", "no permanent global view cap"):
        if token not in dt:
            raise ReconstructionValidationError(f"I5.6A I4.2D historical anchor is missing expected contract text: {token}")
    if "closure-only" not in ft or "templering" not in ft or "universal" not in ft:
        raise ReconstructionValidationError("I5.6A I4.2F historical anchor scope is incomplete.")
    return {
        "scenario_id": "historical_temple_ring_positive_anchor",
        "dataset": "Middlebury TempleRing",
        "reference_views": 47,
        "all_reference_views_required": True,
        "all_geometric_depth_maps_required": True,
        "all_geometric_normal_maps_required": True,
        "geometric_consistency": True,
        "fused_ply_required": True,
        "evidence_only": True,
        "rerun_authorized": False,
        "universal_mvs_robustness_claimed": False,
        "i4_2d_doc": {"path": str(d), "sha256": _sha256(d)},
        "i4_2f_doc": {"path": str(f), "sha256": _sha256(f)},
    }


def _validate_parent_replays(i52: Mapping[str, Any], i53: Mapping[str, Any], i54: Mapping[str, Any], i55: Mapping[str, Any]) -> str:
    _require(i52, "phase", "I5.2B", label="I5.2B replay")
    _require(i52, "status", "low_overlap_weak_parallax_replay_evidence_complete", label="I5.2B replay")
    _require(i52, "pass", True, label="I5.2B replay")
    _require(i53, "phase", "I5.3B", label="I5.3B replay")
    _require(i53, "status", "blur_exposure_clutter_symmetry_replay_evidence_complete", label="I5.3B replay")
    _require(i53, "pass", True, label="I5.3B replay")
    _require(i55, "phase", "I5.5B", label="I5.5B replay")
    _require(i55, "status", "current_frozen_foundation_dynamic_scene_replayed", label="I5.5B replay")
    _require(i55, "pass", True, label="I5.5B replay")
    _require(i54, "phase", "I5.4C", label="I5.4C closure")
    _require(i54, "status", "forward_motion_kitti_robustness_qualified_and_closed", label="I5.4C closure")
    _require(i54, "pass", True, label="I5.4C closure")
    source_hashes=[]
    for payload,label in ((i52,"I5.2B"),(i53,"I5.3B"),(i55,"I5.5B")):
        prov=_mapping(payload.get("provenance"),label=f"{label} provenance")
        value=str(prov.get("frozen_source_asset_set_sha256","")).lower()
        if not _valid_sha(value):
            raise ReconstructionValidationError(f"{label} frozen source SHA is invalid.")
        source_hashes.append(value)
    if len(set(source_hashes)) != 1:
        raise ReconstructionValidationError("I5.6A Castle-P19 parent evidence does not share one frozen source identity.")
    return source_hashes[0]


def _scenario_parent(i52: Mapping[str, Any], i53: Mapping[str, Any], i55: Mapping[str, Any], sid: str) -> Mapping[str, Any]:
    if sid == "castle_control": return _mapping(_mapping(i52.get("scenarios"),label="I5.2B scenarios").get("control"),label=sid)
    if sid == "castle_low_overlap": return _mapping(_mapping(i52.get("scenarios"),label="I5.2B scenarios").get("low_overlap"),label=sid)
    if sid == "castle_weak_parallax": return _mapping(_mapping(i52.get("scenarios"),label="I5.2B scenarios").get("weak_parallax"),label=sid)
    if sid == "castle_overexposure": return _mapping(_mapping(i53.get("scenarios"),label="I5.3B scenarios").get("overexposure"),label=sid)
    if sid == "castle_symmetry": return _mapping(_mapping(i53.get("scenarios"),label="I5.3B scenarios").get("symmetry"),label=sid)
    if sid == "castle_dynamic_proxy": return _mapping(_mapping(i55.get("scenarios"),label="I5.5B scenarios").get("dynamic_proxy"),label=sid)
    raise ReconstructionValidationError(f"I5.6A no parent scenario mapping for {sid}")


def _match_report_parent(report: Mapping[str, Any], parent: Mapping[str, Any], *, sid: str) -> None:
    for rk, pk in (("input_images","input_images"),("registered_images","registered_images"),("quality_status","quality_status")):
        if report.get(rk) != parent.get(pk):
            raise ReconstructionValidationError(f"I5.6A {sid} report/parent mismatch for {rk}.")
    rs=_mapping(report.get("statistics"),label=f"{sid} report statistics")
    ps=_mapping(parent.get("statistics"),label=f"{sid} parent statistics")
    for key in ("input_keyframes","registered_keyframes","landmarks","observations","mean_reprojection_error_px"):
        if rs.get(key) != ps.get(key):
            raise ReconstructionValidationError(f"I5.6A {sid} report/parent statistics mismatch for {key}.")


def build_i5_6a_mvs_robustness_admission(
    *, i5_5c_result: Mapping[str, Any], i5_5c_binding: I51EvidenceBinding,
    i5_2b_result: Mapping[str, Any], i5_2b_binding: I51EvidenceBinding,
    i5_3b_result: Mapping[str, Any], i5_3b_binding: I51EvidenceBinding,
    i5_4c_result: Mapping[str, Any], i5_4c_binding: I51EvidenceBinding,
    i5_5b_result: Mapping[str, Any], i5_5b_binding: I51EvidenceBinding,
    scenario_reports: Mapping[str, Mapping[str, Any]], historical_anchor: Mapping[str, Any],
    admission_receipt: Mapping[str, Any], admission_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    closure=validate_i5_5c_for_i5_6(i5_5c_result)
    receipt=validate_i5_6a_receipt(admission_receipt)
    for b,label in ((i5_5c_binding,"I5.5C"),(i5_2b_binding,"I5.2B"),(i5_3b_binding,"I5.3B"),(i5_4c_binding,"I5.4C"),(i5_5b_binding,"I5.5B"),(admission_receipt_binding,"I5.6A receipt")):
        _validate_binding(b,label=label)
    source_sha=_validate_parent_replays(i5_2b_result,i5_3b_result,i5_4c_result,i5_5b_result)
    if set(scenario_reports) != set(I56A_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.6A requires exactly the six frozen scenario reports.")
    matrix: dict[str, Any]={}
    for sid in I56A_SCENARIO_IDS:
        report=deepcopy(dict(_mapping(scenario_reports[sid],label=f"I5.6A {sid} report analysis")))
        expected=_SCENARIO_EXPECTATIONS[sid]
        if (report.get("input_images"),report.get("registered_images")) != expected:
            raise ReconstructionValidationError(f"I5.6A {sid} does not match frozen registration expectation {expected}.")
        rb=_mapping(report.get("report_binding"),label=f"I5.6A {sid} report binding")
        if not _valid_sha(rb.get("sha256")) or not str(rb.get("path","")).strip():
            raise ReconstructionValidationError(f"I5.6A {sid} report binding is malformed.")
        _match_report_parent(report,_scenario_parent(i5_2b_result,i5_3b_result,i5_5b_result,sid),sid=sid)
        matrix[sid]=report

    s53=_mapping(i5_3b_result.get("scenarios"),label="I5.3B scenarios")
    exclusions={}
    for sid,key in (("castle_blur","blur"),("castle_underexposure","underexposure")):
        node=_mapping(s53.get(key),label=f"I5.3B {key}")
        if node.get("registered_images") != 0 or node.get("pre_sfm_ingestion_rejected") is not True or node.get("rejection_stage") != "ingestion":
            raise ReconstructionValidationError(f"I5.6A {sid} upstream-block evidence is inconsistent.")
        exclusions[sid]={"classification":"blocked_upstream","reason":"pre_sfm_ingestion_rejection_no_authoritative_camera_solution","mvs_failure":False}
    clutter_revalidation=_mapping(_mapping(i5_3b_result.get("image_domain_revalidation"),label="I5.3B image revalidation").get("clutter"),label="I5.3B clutter revalidation")
    if clutter_revalidation.get("semantic_stress_revalidated") is not False:
        raise ReconstructionValidationError("I5.6A clutter exclusion requires preserved inconclusive semantic revalidation.")
    exclusions["castle_clutter"]={"classification":"not_formally_admitted","reason":"semantic_clutter_proxy_not_revalidated","mvs_failure":False}
    q54=_mapping(i5_4c_result.get("qualification"),label="I5.4C qualification")
    if q54.get("registered_images") != 0 or q54.get("primary_bottleneck") != "incremental_mapping_no_sparse_model":
        raise ReconstructionValidationError("I5.6A KITTI upstream-block evidence is inconsistent.")
    exclusions["kitti_forward_motion"]={"classification":"blocked_upstream","reason":"incremental_mapping_no_sparse_model_no_authoritative_camera_solution","mvs_failure":False}

    anchor=deepcopy(dict(_mapping(historical_anchor,label="I5.6A historical anchor")))
    if anchor.get("reference_views") != 47 or anchor.get("evidence_only") is not True or anchor.get("rerun_authorized") is not False:
        raise ReconstructionValidationError("I5.6A TempleRing historical anchor is malformed.")
    policy=I56AAdmissionPolicy()
    return {
        "schema_version":I56A_SCHEMA_VERSION,"phase":I56A_PHASE,
        "status":"mvs_robustness_scenarios_admitted","pass":True,
        "scope":{
            "scenario_admission_only":True,"mvs_execution":False,"sfm_rerun":False,"dense_provider_rerun":False,
            "provider_comparison_started":False,"da3_rerun":False,"mp_sfm_started":False,"auto_policy_work_started":False,
            "hybrid_experiment_started":False,"thresholds_modified":False,"production_default_modified":False,
        },
        "evidence_bindings":{
            "i5_5c_closure":i5_5c_binding.as_dict(),"i5_2b_replay":i5_2b_binding.as_dict(),"i5_3b_replay":i5_3b_binding.as_dict(),
            "i5_4c_closure":i5_4c_binding.as_dict(),"i5_5b_replay":i5_5b_binding.as_dict(),"i5_6a_receipt":admission_receipt_binding.as_dict(),
        },
        "source_family":{"label":"Castle-P19","frozen_source_asset_set_sha256":source_sha,"matched_source_family_across_admitted_cases":True},
        "execution_policy":policy.as_dict(),
        "scenario_matrix":matrix,
        "exclusions":exclusions,
        "historical_positive_anchor":anchor,
        "classification_policy":{
            "pass":"production MVS completes with valid native artifacts on admitted authoritative SfM input",
            "degraded":"MVS completes but native support/coverage is directionally materially poorer than matched control; no post-hoc scalar cutoff",
            "rejected":"valid authoritative SfM exists but current frozen production MVS cannot complete structurally/runtime",
            "blocked_upstream":"no authoritative reusable SfM exists; not attributable to MVS",
            "classification_threshold_used":None,"global_scalar_score_created":False,"retry_until_pass":False,
        },
        "authority":{
            "camera_pose_authority_owner":"pycolmap_sfm","authoritative_sfm_modified":False,"provider_selected_by_auto":False,
            "provider_promoted":False,"metric_scale_claimed":False,"measurement_authoritative":False,"certified_metrology":False,
            "surface_mesh_generated":False,"permanent_global_view_cap":None,
        },
        "regression_summary":{
            "focused_i5_6a_tests":receipt["counts"].get("focused_tests"),
            "predecessor_tests":receipt["counts"].get("predecessor_total_tests"),
            "combined_exercised_tests":receipt["counts"].get("combined_exercised_tests"),
            "all_required_suites_passed":True,"test_counts_are_not_runtime_quality_scores":True,
        },
        "decision":{
            "i5_5_complete":True,"i5_6_started":True,"i5_6a_complete":True,"i5_6b_authorized":True,"i5_6b_started":False,
            "provider_selected":False,"provider_promoted":False,"provider_ranking_created":False,"universal_provider_winner_declared":False,
            "global_scalar_score_created":False,"auto_policy_implemented":False,"hybrid_authorized":False,"mp_sfm_authorized":False,
            "threshold_relaxation_authorized":False,"production_default_modified":False,
        },
        "roadmap_guard":{"i5_6b_real_mvs_replay_authorized_not_started":True,"i5_7_not_authorized_until_i5_6_closure":True,"mp_sfm_deferred_to_i5_10":True,"hybrid_deferred_to_i5_12_if_justified":True,"auto_deferred_to_i5_13":True},
        "next":"I5.6B current frozen-foundation MVS robustness replay.",
    }


# ---------------------------------------------------------------------------
# I5.6B current frozen-foundation MVS robustness replay
# ---------------------------------------------------------------------------

from statistics import median
import struct

I56B_SCHEMA_VERSION = 1
I56B_PHASE = "I5.6B"
I56B_RECEIPT_PHASE = "I5.6B_REPLAY_RECEIPT"
I56B_SCENARIO_IDS = I56A_SCENARIO_IDS


@dataclass(frozen=True, slots=True)
class I56BReplayPolicy:
    """Frozen production-native MVS execution policy for the six admitted cases."""

    scenario_ids: tuple[str, ...] = I56B_SCENARIO_IDS
    provider_id: str = I43B_PROVIDER_ID
    implementation: str = DenseProviderImplementation.COLMAP_MVS.value
    production_adapter: str = "MvsProductionProviderAdapter"
    adaptive_preparation: str = "prepare_adaptive_provider_execution"
    adaptive_selection: bool = True
    engineer_override: str | None = None
    source_neighbors: int = I43B_DEFAULT_SOURCE_NEIGHBORS
    patch_match_cache_gib: float = I43B_DEFAULT_PATCH_MATCH_CACHE_GIB
    stereo_fusion_cache_gib: float = I43B_DEFAULT_STEREO_FUSION_CACHE_GIB
    max_image_size: int = -1
    geometric_consistency: bool = True
    sequential_execution: bool = True
    permanent_global_view_cap: int | None = None
    result_driven_retuning: bool = False

    def __post_init__(self) -> None:
        if tuple(self.scenario_ids) != I56B_SCENARIO_IDS:
            raise ReconstructionValidationError("I5.6B must execute exactly the six I5.6A-admitted Castle-P19 scenarios.")
        if self.provider_id != I43B_PROVIDER_ID or self.implementation != DenseProviderImplementation.COLMAP_MVS.value:
            raise ReconstructionValidationError("I5.6B must use the existing COLMAP MVS production provider.")
        if self.production_adapter != "MvsProductionProviderAdapter" or self.adaptive_preparation != "prepare_adaptive_provider_execution":
            raise ReconstructionValidationError("I5.6B cannot substitute a parallel MVS execution architecture.")
        if self.adaptive_selection is not True or self.engineer_override is not None:
            raise ReconstructionValidationError("I5.6B requires production adaptive selection without engineer override.")
        if self.geometric_consistency is not True:
            raise ReconstructionValidationError("I5.6B refuses to disable geometric consistency.")
        if self.sequential_execution is not True:
            raise ReconstructionValidationError("I5.6B formal execution is sequential.")
        if self.permanent_global_view_cap is not None:
            raise ReconstructionValidationError("I5.6B cannot introduce a permanent global view cap.")
        if self.result_driven_retuning is not False:
            raise ReconstructionValidationError("I5.6B forbids result-driven retuning.")
        MvsProductionConfig(
            source_neighbors=self.source_neighbors,
            patch_match_cache_gib=self.patch_match_cache_gib,
            stereo_fusion_cache_gib=self.stereo_fusion_cache_gib,
            max_image_size=self.max_image_size,
            geometric_consistency=self.geometric_consistency,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_ids": list(self.scenario_ids),
            "provider_id": self.provider_id,
            "implementation": self.implementation,
            "production_adapter": self.production_adapter,
            "adaptive_preparation": self.adaptive_preparation,
            "adaptive_selection": self.adaptive_selection,
            "engineer_override": self.engineer_override,
            "source_neighbors": self.source_neighbors,
            "patch_match_cache_gib": self.patch_match_cache_gib,
            "stereo_fusion_cache_gib": self.stereo_fusion_cache_gib,
            "max_image_size": self.max_image_size,
            "geometric_consistency": self.geometric_consistency,
            "sequential_execution": self.sequential_execution,
            "permanent_global_view_cap": self.permanent_global_view_cap,
            "result_driven_retuning": self.result_driven_retuning,
        }


def _as_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ratio(numerator: Any, denominator: Any) -> float | None:
    a = _as_float_or_none(numerator)
    b = _as_float_or_none(denominator)
    if a is None or b is None or b == 0.0:
        return None
    return a / b


def validate_i5_6a_for_i5_6b(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that an I5.6A artifact authorizes exactly the frozen B replay."""

    _require(payload, "schema_version", 1, label="I5.6A admission")
    _require(payload, "phase", I56A_PHASE, label="I5.6A admission")
    _require(payload, "status", "mvs_robustness_scenarios_admitted", label="I5.6A admission")
    _require(payload, "pass", True, label="I5.6A admission")
    scope = _mapping(payload.get("scope"), label="I5.6A scope")
    _require(scope, "scenario_admission_only", True, label="I5.6A scope")
    for key in ("mvs_execution", "sfm_rerun", "dense_provider_rerun", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.6A scope")

    policy = _mapping(payload.get("execution_policy"), label="I5.6A execution policy")
    frozen = I56BReplayPolicy().as_dict()
    for key in (
        "scenario_ids", "provider_id", "implementation", "production_adapter", "adaptive_preparation",
        "adaptive_selection", "engineer_override", "source_neighbors", "patch_match_cache_gib",
        "stereo_fusion_cache_gib", "max_image_size", "geometric_consistency", "sequential_execution",
        "permanent_global_view_cap", "result_driven_retuning",
    ):
        _require(policy, key, frozen[key], label="I5.6A execution policy")

    matrix = _mapping(payload.get("scenario_matrix"), label="I5.6A scenario matrix")
    # I5.6A formal evidence is serialized with sort_keys=True, so JSON object key
    # order is not an execution-order contract. The authoritative replay order is
    # already frozen in execution_policy.scenario_ids above. Require exactly the
    # same six scenario identities here without depending on serialized mapping order.
    if len(matrix) != len(I56B_SCENARIO_IDS) or set(matrix) != set(I56B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.6B requires exactly the six-scenario I5.6A matrix.")
    for sid in I56B_SCENARIO_IDS:
        node = _mapping(matrix.get(sid), label=f"I5.6A {sid}")
        _require(node, "scenario_id", sid, label=f"I5.6A {sid}")
        _require(node, "mvs_input_integrity_verified", True, label=f"I5.6A {sid}")
        rb = _mapping(node.get("report_binding"), label=f"I5.6A {sid} report binding")
        if not str(rb.get("path", "")).strip() or not _valid_sha(rb.get("sha256")):
            raise ReconstructionValidationError(f"I5.6A {sid} report binding is malformed.")

    anchor = _mapping(payload.get("historical_positive_anchor"), label="I5.6A historical anchor")
    _require(anchor, "reference_views", 47, label="I5.6A historical anchor")
    _require(anchor, "evidence_only", True, label="I5.6A historical anchor")
    _require(anchor, "rerun_authorized", False, label="I5.6A historical anchor")

    classification = _mapping(payload.get("classification_policy"), label="I5.6A classification policy")
    _require(classification, "classification_threshold_used", None, label="I5.6A classification policy")
    _require(classification, "global_scalar_score_created", False, label="I5.6A classification policy")
    _require(classification, "retry_until_pass", False, label="I5.6A classification policy")

    authority = _mapping(payload.get("authority"), label="I5.6A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6A authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.6A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.6A authority")
    for key in ("provider_selected_by_auto", "provider_promoted", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.6A authority")

    decision = _mapping(payload.get("decision"), label="I5.6A decision")
    _require(decision, "i5_6a_complete", True, label="I5.6A decision")
    _require(decision, "i5_6b_authorized", True, label="I5.6A decision")
    _require(decision, "i5_6b_started", False, label="I5.6A decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.6A decision")
    return deepcopy(dict(payload))


def validate_i5_6b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.6B receipt")
    _require(payload, "phase", I56B_RECEIPT_PHASE, label="I5.6B receipt")
    _require(payload, "status", "pass", label="I5.6B receipt")
    _require(payload, "pass", True, label="I5.6B receipt")
    scope = _mapping(payload.get("scope"), label="I5.6B receipt scope")
    for key in ("real_mvs_executed_by_test_receipt", "sfm_rerun", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.6B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.6B receipt counts")
    if int(counts.get("combined_exercised_tests", 0)) <= 0:
        raise ReconstructionValidationError("I5.6B receipt must record exercised tests.")
    return deepcopy(dict(payload))


def read_i5_6b_visibility_support(path: Path) -> dict[str, Any]:
    """Read COLMAP fused.ply.vis support counts without importing PyCOLMAP."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I5.6B native visibility sidecar is missing: {source}")
    supports: list[int] = []
    unique: set[int] = set()
    with source.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ReconstructionValidationError("I5.6B native visibility header is truncated.")
        point_count = int(struct.unpack("<Q", raw)[0])
        for _ in range(point_count):
            raw = handle.read(4)
            if len(raw) != 4:
                raise ReconstructionValidationError("I5.6B native visibility record is truncated.")
            count = int(struct.unpack("<I", raw)[0])
            ids_raw = handle.read(4 * count)
            if len(ids_raw) != 4 * count:
                raise ReconstructionValidationError("I5.6B native visibility image-index list is truncated.")
            ids = struct.unpack("<" + "I" * count, ids_raw) if count else ()
            supports.append(count)
            unique.update(int(v) for v in ids)
        if handle.read(1):
            raise ReconstructionValidationError("I5.6B native visibility sidecar contains unexpected trailing bytes.")
    if point_count <= 0 or len(supports) != point_count:
        raise ReconstructionValidationError("I5.6B native visibility sidecar contains no fused-point support records.")
    ordered = sorted(supports)
    p90_index = min(len(ordered) - 1, max(0, int(round(0.90 * (len(ordered) - 1)))))
    return {
        "point_count": point_count,
        "support_min": ordered[0],
        "support_p50": float(median(ordered)),
        "support_p90": ordered[p90_index],
        "support_max": ordered[-1],
        "support_mean": sum(ordered) / float(len(ordered)),
        "unique_contributing_image_indices": sorted(unique),
        "unique_contributing_image_count": len(unique),
    }


def _peak_worker_vram_mib(worker: Mapping[str, Any]) -> float | None:
    snapshots = worker.get("resource_snapshots")
    if not isinstance(snapshots, Mapping):
        return None
    values: list[float] = []
    for snap in snapshots.values():
        if not isinstance(snap, Mapping):
            continue
        gpus = snap.get("gpus")
        if not isinstance(gpus, list):
            continue
        for gpu in gpus:
            if isinstance(gpu, Mapping) and gpu.get("used_vram_gib") is not None:
                try:
                    values.append(max(0.0, float(gpu.get("used_vram_gib", 0.0))) * 1024.0)
                except (TypeError, ValueError):
                    continue
    return max(values) if values else None


def analyze_i5_6b_provider_summary(summary: Mapping[str, Any], *, wall_time_s: float) -> dict[str, Any]:
    """Extract native MVS runtime evidence without introducing a new quality score."""

    _require(summary, "schema_version", 1, label="I5.6B MVS summary")
    _require(summary, "phase", "I4.3B", label="I5.6B MVS summary")
    resolved = _mapping(summary.get("resolved_config"), label="I5.6B resolved MVS config")
    _require(resolved, "geometric_consistency", True, label="I5.6B resolved MVS config")
    _require(resolved, "quality_thresholds_relaxed", False, label="I5.6B resolved MVS config")
    _require(resolved, "permanent_global_view_cap", None, label="I5.6B resolved MVS config")
    selected = int(resolved.get("selected_view_count", 0))
    if selected < 2:
        raise ReconstructionValidationError("I5.6B resolved MVS selected-view count is invalid.")
    authority = _mapping(summary.get("authority"), label="I5.6B MVS authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6B MVS authority")
    _require(authority, "single_authoritative_sfm_frame_preserved", True, label="I5.6B MVS authority")
    for key in ("sfm_rerun", "bundle_adjustment_rerun", "pose_refinement", "production_filter_policy_changed", "provider_promotion_performed", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.6B MVS authority")
    integrity = _mapping(summary.get("integrity"), label="I5.6B MVS integrity")
    _require(integrity, "authoritative_sparse_unchanged", True, label="I5.6B MVS integrity")
    _require(integrity, "registered_images_unchanged", True, label="I5.6B MVS integrity")

    worker = _mapping(summary.get("worker", {}), label="I5.6B worker evidence")
    passed = summary.get("status") == "pass" and summary.get("pass") is True
    metrics: dict[str, Any] = {
        "wall_time_s": float(wall_time_s),
        "selected_view_count": selected,
        "geometric_depth_maps": 0,
        "geometric_normal_maps": 0,
        "fused_point_count": None,
        "visibility_point_count": None,
        "visibility_support_min": None,
        "visibility_support_p50": None,
        "visibility_support_p90": None,
        "visibility_support_max": None,
        "visibility_support_mean": None,
        "unique_contributing_image_count": None,
        "process_peak_rss_gib": _as_float_or_none(worker.get("process_peak_rss_gib")),
        "peak_vram_mib": _peak_worker_vram_mib(worker),
        "stage_seconds": deepcopy(dict(worker.get("stage_seconds", {}))) if isinstance(worker.get("stage_seconds"), Mapping) else {},
    }
    failure: dict[str, Any] | None = None
    if passed:
        _require(worker, "status", "pass", label="I5.6B worker")
        patch = _mapping(worker.get("patch_match_outputs"), label="I5.6B PatchMatch outputs")
        depth = int(patch.get("geometric_depth_maps", -1))
        normal = int(patch.get("geometric_normal_maps", -1))
        expected = int(patch.get("expected_reference_views", -1))
        if expected != selected or depth != selected or normal != selected:
            raise ReconstructionValidationError("I5.6B PatchMatch did not preserve all selected reference views.")
        fusion = _mapping(worker.get("fusion"), label="I5.6B fusion evidence")
        points = int(fusion.get("ply_vertex_count", 0))
        vis_points = int(fusion.get("native_visibility_point_count", -1))
        if points <= 0 or vis_points != points:
            raise ReconstructionValidationError("I5.6B fused PLY/native visibility correspondence is invalid.")
        vis_path = Path(str(fusion.get("native_visibility", ""))).expanduser().resolve()
        support = read_i5_6b_visibility_support(vis_path)
        if support["point_count"] != points:
            raise ReconstructionValidationError("I5.6B native visibility support count drifted from fused PLY count.")
        metrics.update({
            "geometric_depth_maps": depth,
            "geometric_normal_maps": normal,
            "fused_point_count": points,
            "visibility_point_count": vis_points,
            "visibility_support_min": support["support_min"],
            "visibility_support_p50": support["support_p50"],
            "visibility_support_p90": support["support_p90"],
            "visibility_support_max": support["support_max"],
            "visibility_support_mean": support["support_mean"],
            "unique_contributing_image_count": support["unique_contributing_image_count"],
        })
    else:
        failure = {
            "reason": str(_mapping(summary.get("execution_result", {}), label="I5.6B failed execution result").get("failure_reason", "")).strip()
                      or str(worker.get("error", "")).strip() or "mvs_execution_failed",
            "worker_status": str(worker.get("status", "fail")),
            "worker_error": str(worker.get("error", "")),
            "heavy_mvs_started": bool(worker.get("heavy_mvs_started", False)),
            "image_undistortion_started": bool(worker.get("image_undistortion_started", False)),
            "patch_match_started": bool(worker.get("patch_match_started", False)),
            "stereo_fusion_started": bool(worker.get("stereo_fusion_started", False)),
        }
    return {"runtime_status": "pass" if passed else "rejected", "native_metrics": metrics, "failure": failure}


# ---------------------------------------------------------------------------
# I5.6B interrupted-parent/native-worker recovery finalization
# ---------------------------------------------------------------------------

I56B_RECOVERY_SCHEMA_VERSION = 1
I56B_RECOVERY_PHASE = "I5.6B_RECOVERY_FINALIZATION"
I56B_RECOVERY_SCENARIO_ID = "castle_dynamic_proxy"


def _i56b_ply_vertex_count(path: Path) -> int:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I5.6B recovered fused PLY is missing: {source}")
    with source.open("rb") as handle:
        header = handle.read(1024 * 1024)
    marker = b"end_header"
    if marker not in header:
        raise ReconstructionValidationError("I5.6B recovered fused PLY header is incomplete.")
    text = header.split(marker, 1)[0].decode("ascii", errors="strict")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element" and parts[1] == "vertex":
            count = int(parts[2])
            if count <= 0:
                raise ReconstructionValidationError("I5.6B recovered fused PLY contains no vertices.")
            return count
    raise ReconstructionValidationError("I5.6B recovered fused PLY has no vertex declaration.")


def _i56b_verify_hash_map(directory: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, str]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.6B recovery {label} directory is missing: {root}")
    actual: dict[str, str] = {}
    for raw_name, raw_digest in expected.items():
        name = str(raw_name)
        expected_digest = str(raw_digest).strip().lower()
        if not _valid_sha(expected_digest):
            raise ReconstructionValidationError(f"I5.6B recovery {label} SHA256 is malformed for {name}.")
        path = root / name
        if not path.is_file():
            raise ReconstructionValidationError(f"I5.6B recovery {label} file is missing: {path}")
        digest = _sha256(path)
        if digest != expected_digest:
            raise ReconstructionValidationError(
                f"I5.6B recovery {label} checksum mismatch for {name}: expected {expected_digest}, found {digest}"
            )
        actual[name] = digest
    if not actual:
        raise ReconstructionValidationError(f"I5.6B recovery {label} hash map is empty.")
    return actual


def validate_i5_6b_existing_scenario_record(payload: Mapping[str, Any], *, scenario_id: str) -> dict[str, Any]:
    if scenario_id not in I56B_SCENARIO_IDS:
        raise ReconstructionValidationError(f"I5.6B recovery unknown existing scenario: {scenario_id}")
    node = deepcopy(dict(_mapping(payload, label=f"I5.6B existing {scenario_id} record")))
    _require(node, "scenario_id", scenario_id, label=f"I5.6B existing {scenario_id} record")
    if node.get("runtime_status") not in {"pass", "rejected"}:
        raise ReconstructionValidationError(f"I5.6B existing {scenario_id} runtime_status is invalid.")
    authority = _mapping(node.get("authority"), label=f"I5.6B existing {scenario_id} authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.6B existing {scenario_id} authority")
    _require(authority, "authoritative_sfm_modified", False, label=f"I5.6B existing {scenario_id} authority")
    _require(authority, "permanent_global_view_cap", None, label=f"I5.6B existing {scenario_id} authority")
    for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology"):
        _require(authority, key, False, label=f"I5.6B existing {scenario_id} authority")
    return node


def recover_i5_6b_interrupted_native_scenario(
    *,
    admission_node: Mapping[str, Any],
    adaptive_plan: Mapping[str, Any],
    worker_request: Mapping[str, Any],
    worker_state: Mapping[str, Any],
    fused_ply: Path,
    native_visibility: Path,
    depth_maps_dir: Path,
    normal_maps_dir: Path,
    artifact_bindings: Mapping[str, Any],
    resource_probe_after: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recover the single interrupted I5.6B scenario from immutable native artifacts.

    This is intentionally narrow: it never launches SfM, PatchMatch, StereoFusion,
    or the production adapter. It accepts only the already-admitted dynamic-proxy
    scenario whose native fusion artifacts survived an interrupted parent/finalizer.
    """

    sid = I56B_RECOVERY_SCENARIO_ID
    admission = _mapping(admission_node, label="I5.6B recovery admission node")
    _require(admission, "scenario_id", sid, label="I5.6B recovery admission node")
    _require(admission, "mvs_input_integrity_verified", True, label="I5.6B recovery admission node")
    source_views = int(admission.get("registered_images", 0))
    if source_views != 19:
        raise ReconstructionValidationError("I5.6B recovery is frozen to the 19-view Castle dynamic proxy.")

    plan = _mapping(adaptive_plan, label="I5.6B recovery adaptive plan")
    _require(plan, "schema_version", 1, label="I5.6B recovery adaptive plan")
    if int(plan.get("source_view_count", 0)) != source_views or int(plan.get("selected_view_count", 0)) != source_views:
        raise ReconstructionValidationError("I5.6B recovery adaptive plan must preserve all 19 admitted views.")
    selection_request = _mapping(plan.get("selection_request"), label="I5.6B recovery selection request")
    _require(selection_request, "authoritative_sfm_available", True, label="I5.6B recovery selection request")
    if int(selection_request.get("available_view_count", 0)) != source_views:
        raise ReconstructionValidationError("I5.6B recovery selection request view count drifted.")
    selection_decision = _mapping(plan.get("selection_decision"), label="I5.6B recovery selection decision")
    _require(selection_decision, "selected_provider_id", I43B_PROVIDER_ID, label="I5.6B recovery selection decision")
    _require(selection_decision, "selected_implementation", DenseProviderImplementation.COLMAP_MVS.value, label="I5.6B recovery selection decision")
    _require(selection_decision, "automatic", False, label="I5.6B recovery selection decision")
    _require(selection_decision, "engineer_override_applied", False, label="I5.6B recovery selection decision")

    request = _mapping(worker_request, label="I5.6B recovery worker request")
    _require(request, "schema_version", 1, label="I5.6B recovery worker request")
    _require(request, "phase", "I4.3B", label="I5.6B recovery worker request")
    _require(request, "action", "colmap_mvs_dense_provider", label="I5.6B recovery worker request")
    _require(request, "reconstruction_frame_id", plan.get("reconstruction_frame_id"), label="I5.6B recovery worker request")
    if int(request.get("expected_registered_views", 0)) != source_views:
        raise ReconstructionValidationError("I5.6B recovery worker request view count drifted.")
    image_names = [str(item) for item in request.get("image_names", [])]
    if len(image_names) != source_views or len(set(image_names)) != source_views:
        raise ReconstructionValidationError("I5.6B recovery worker request image set is incomplete or duplicated.")
    config = _mapping(request.get("config"), label="I5.6B recovery MVS config")
    expected_config = MvsProductionConfig(gpu_index="0").resolved(selected_view_count=source_views)
    if dict(config) != expected_config:
        raise ReconstructionValidationError("I5.6B recovery refuses MVS config drift from the frozen production path.")
    authority = _mapping(request.get("authority"), label="I5.6B recovery worker authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6B recovery worker authority")
    _require(authority, "single_authoritative_sfm_frame", True, label="I5.6B recovery worker authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.6B recovery worker authority")
    for key in (
        "sfm_rerun", "bundle_adjustment_rerun", "pose_refinement", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
        "production_filter_policy_changed", "provider_promotion_performed",
    ):
        _require(authority, key, False, label="I5.6B recovery worker authority")

    state = _mapping(worker_state, label="I5.6B recovery worker state")
    _require(state, "schema_version", 1, label="I5.6B recovery worker state")
    _require(state, "phase", "I4.3B", label="I5.6B recovery worker state")
    _require(state, "stage", "stereo_fusion", label="I5.6B recovery worker state")
    _require(state, "stage_started", True, label="I5.6B recovery worker state")
    _require(state, "stage_completed", False, label="I5.6B recovery worker state")
    for key in ("heavy_mvs_started", "image_undistortion_started", "patch_match_started", "stereo_fusion_started"):
        _require(state, key, True, label="I5.6B recovery worker state")

    input_images = Path(str(request.get("input_images", ""))).expanduser().resolve()
    input_sparse = Path(str(request.get("input_sparse", ""))).expanduser().resolve()
    image_hashes = _mapping(request.get("image_sha256"), label="I5.6B recovery registered-image hashes")
    sparse_hashes = _mapping(request.get("input_sparse_sha256"), label="I5.6B recovery sparse hashes")
    _i56b_verify_hash_map(input_images, image_hashes, label="registered image")
    _i56b_verify_hash_map(input_sparse, sparse_hashes, label="authoritative sparse")

    fused = Path(fused_ply).expanduser().resolve()
    vis = Path(native_visibility).expanduser().resolve()
    if Path(str(request.get("fused_ply", ""))).expanduser().resolve() != fused:
        raise ReconstructionValidationError("I5.6B recovery fused PLY path does not match the frozen worker request.")
    vertices = _i56b_ply_vertex_count(fused)
    support = read_i5_6b_visibility_support(vis)
    if int(support["point_count"]) != vertices:
        raise ReconstructionValidationError("I5.6B recovery fused PLY/native visibility correspondence is invalid.")

    depth_dir = Path(depth_maps_dir).expanduser().resolve()
    normal_dir = Path(normal_maps_dir).expanduser().resolve()
    expected_depth = {f"{name}.geometric.bin" for name in image_names}
    depth_names = {path.name for path in depth_dir.glob("*.geometric.bin")} if depth_dir.is_dir() else set()
    normal_names = {path.name for path in normal_dir.glob("*.geometric.bin")} if normal_dir.is_dir() else set()
    if depth_names != expected_depth or normal_names != expected_depth:
        raise ReconstructionValidationError(
            f"I5.6B recovery requires all 19 geometric depth/normal maps exactly; depth={len(depth_names)} normal={len(normal_names)}"
        )

    frame_id = str(request.get("reconstruction_frame_id", "")).strip()
    recovered_execution = DenseProviderExecutionResult(
        provider_id=I43B_PROVIDER_ID,
        implementation=DenseProviderImplementation.COLMAP_MVS,
        status=DenseProviderRunStatus.COMPLETED,
        source_view_count=source_views,
        selected_view_count=source_views,
        adaptive_view_selection=True,
        view_selection_reason=";".join(str(item) for item in _mapping(plan.get("selection_plan"), label="I5.6B recovery selection plan").get("reason_codes", [])),
        permanent_global_view_cap=None,
        canonical_geometry_ref=str(fused),
        canonical_frame_id=frame_id,
        native_artifact_refs=(str(fused), str(vis), str(fused.parent)),
        progress_event_count=0,
        final_preview_revision=0,
        canonical_geometry_authoritative=True,
        single_authoritative_sfm_frame_preserved=True,
        camera_pose_authority_owner="pycolmap_sfm",
        metric_scale_claimed=False,
        measurement_authoritative=False,
        certified_metrology=False,
        surface_mesh_generated=False,
        production_filter_policy_changed=False,
        provider_promotion_performed=False,
    ).as_dict()

    state_resources = state.get("resources") if isinstance(state.get("resources"), Mapping) else {}
    last_rss = _as_float_or_none(state_resources.get("process_peak_rss_gib")) if isinstance(state_resources, Mapping) else None
    metrics = {
        "wall_time_s": None,
        "selected_view_count": source_views,
        "geometric_depth_maps": source_views,
        "geometric_normal_maps": source_views,
        "fused_point_count": vertices,
        "visibility_point_count": int(support["point_count"]),
        "visibility_support_min": support["support_min"],
        "visibility_support_p50": support["support_p50"],
        "visibility_support_p90": support["support_p90"],
        "visibility_support_max": support["support_max"],
        "visibility_support_mean": support["support_mean"],
        "unique_contributing_image_count": support["unique_contributing_image_count"],
        "process_peak_rss_gib": None,
        "peak_vram_mib": None,
        "stage_seconds": {},
        "recovery_last_recorded_process_peak_rss_gib": last_rss,
        "wall_time_unavailable_due_interrupted_parent_finalization": True,
        "peak_resource_metrics_incomplete_due_missing_worker_result": True,
    }
    recovery = {
        "schema_version": I56B_RECOVERY_SCHEMA_VERSION,
        "phase": I56B_RECOVERY_PHASE,
        "scenario_id": sid,
        "recovery_required": True,
        "native_worker_finalization_incomplete": True,
        "worker_result_present": False,
        "worker_state_stale_at_stereo_fusion": True,
        "native_patchmatch_outputs_recovered": True,
        "native_stereo_fusion_artifacts_recovered": True,
        "native_fusion_structurally_validated": True,
        "authoritative_inputs_revalidated": True,
        "provider_adapter_returned": False,
        "execution_result_reconstructed_from_native_evidence": True,
        "worker_termination_cause": "unresolved_abrupt_termination_after_native_fusion_artifact_write",
        "sfm_rerun": False,
        "patch_match_rerun": False,
        "stereo_fusion_rerun": False,
        "mvs_rerun": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "result_driven_retuning": False,
        "retry_until_pass": False,
    }
    record = {
        "scenario_id": sid,
        "runtime_status": "pass",
        "provider_execution_completed": True,
        "provider_adapter_returned": False,
        "recovered_from_existing_native_artifacts": True,
        "source_registered_views": source_views,
        "selected_views": source_views,
        "selection_fraction": 1.0,
        "adaptive_selection": True,
        "plan_reason_codes": list(_mapping(plan.get("selection_plan"), label="I5.6B recovery selection plan").get("reason_codes", [])),
        "native_metrics": metrics,
        "failure": None,
        "execution_result": recovered_execution,
        "execution_result_provenance": "reconstructed_from_existing_native_artifacts_not_original_adapter_return",
        "artifact_bindings": deepcopy(dict(artifact_bindings)),
        "resource_probe_before": deepcopy(dict(state_resources)) if isinstance(state_resources, Mapping) else {},
        "resource_probe_after": deepcopy(dict(resource_probe_after or {})),
        "input_integrity_revalidated_after": True,
        "recovery_provenance": recovery,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "provider_promotion_performed": False,
            "permanent_global_view_cap": None,
        },
    }
    recovery_summary = {
        "schema_version": I56B_RECOVERY_SCHEMA_VERSION,
        "phase": I56B_RECOVERY_PHASE,
        "status": "native_mvs_recovered_after_interrupted_finalization",
        "pass": True,
        "scenario_id": sid,
        "resolved_config": deepcopy(dict(config)),
        "native_metrics": deepcopy(metrics),
        "recovery_provenance": deepcopy(recovery),
        "artifact_bindings": deepcopy(dict(artifact_bindings)),
        "authority": deepcopy(record["authority"]),
    }
    return record, recovery_summary


def compare_i5_6b_scenarios(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(records) != I56B_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.6B comparison requires the exact ordered six-scenario record set.")
    control = _mapping(records.get("castle_control"), label="I5.6B control record")
    control_metrics = _mapping(control.get("native_metrics"), label="I5.6B control metrics")
    control_pass = control.get("runtime_status") == "pass"
    out: dict[str, Any] = {}
    for sid in I56B_SCENARIO_IDS:
        node = _mapping(records[sid], label=f"I5.6B {sid} record")
        metrics = _mapping(node.get("native_metrics"), label=f"I5.6B {sid} metrics")
        runtime_pass = node.get("runtime_status") == "pass"
        fused_ratio = _ratio(metrics.get("fused_point_count"), control_metrics.get("fused_point_count")) if control_pass and runtime_pass else None
        support_ratio = _ratio(metrics.get("visibility_support_mean"), control_metrics.get("visibility_support_mean")) if control_pass and runtime_pass else None
        wall_ratio = _ratio(metrics.get("wall_time_s"), control_metrics.get("wall_time_s")) if control_pass and runtime_pass else None
        peak_vram_ratio = _ratio(metrics.get("peak_vram_mib"), control_metrics.get("peak_vram_mib")) if control_pass and runtime_pass else None
        reduction = None if not (control_pass and runtime_pass) else bool(
            (fused_ratio is not None and fused_ratio < 1.0) or (support_ratio is not None and support_ratio < 1.0)
        )
        out[sid] = {
            "runtime_status": node.get("runtime_status"),
            "selected_view_ratio_to_control": _ratio(node.get("selected_views"), control.get("selected_views")) if control_pass and runtime_pass else None,
            "fused_point_ratio_to_control": fused_ratio,
            "visibility_support_mean_ratio_to_control": support_ratio,
            "wall_time_ratio_to_control": wall_ratio,
            "peak_vram_ratio_to_control": peak_vram_ratio,
            "directional_native_support_reduction_observed": reduction,
            "cross_run_absolute_geometry_comparison": False,
            "cross_run_scale_normalization_performed": False,
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
        }
    return out


def build_i5_6b_mvs_robustness_replay(
    *,
    i5_6a_admission: Mapping[str, Any],
    i5_6a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    runtime_preflight: Mapping[str, Any],
    scenario_records: Mapping[str, Mapping[str, Any]],
    recovery_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    admission = validate_i5_6a_for_i5_6b(i5_6a_admission)
    receipt = validate_i5_6b_receipt(replay_receipt)
    _validate_binding(i5_6a_binding, label="I5.6A")
    _validate_binding(replay_receipt_binding, label="I5.6B receipt")
    if runtime_preflight.get("mvs_runtime_ready") is not True or int(runtime_preflight.get("cuda_devices", 0)) < 1:
        raise ReconstructionValidationError("I5.6B formal replay requires a qualified CUDA PyCOLMAP MVS runtime.")
    if tuple(scenario_records) != I56B_SCENARIO_IDS:
        raise ReconstructionValidationError("I5.6B requires exactly six completed structured scenario records.")

    records: dict[str, Any] = {}
    for sid in I56B_SCENARIO_IDS:
        node = deepcopy(dict(_mapping(scenario_records[sid], label=f"I5.6B {sid} record")))
        _require(node, "scenario_id", sid, label=f"I5.6B {sid} record")
        if node.get("runtime_status") not in {"pass", "rejected"}:
            raise ReconstructionValidationError(f"I5.6B {sid} runtime_status must be pass or rejected.")
        authority = _mapping(node.get("authority"), label=f"I5.6B {sid} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.6B {sid} authority")
        _require(authority, "authoritative_sfm_modified", False, label=f"I5.6B {sid} authority")
        _require(authority, "permanent_global_view_cap", None, label=f"I5.6B {sid} authority")
        for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology"):
            _require(authority, key, False, label=f"I5.6B {sid} authority")
        records[sid] = node

    control_status = records["castle_control"]["runtime_status"]
    stress_statuses = [records[sid]["runtime_status"] for sid in I56B_SCENARIO_IDS if sid != "castle_control"]
    if control_status != "pass":
        quality = "rejected"
    elif any(status != "pass" for status in stress_statuses):
        quality = "degraded"
    else:
        quality = "pass"
    comparisons = compare_i5_6b_scenarios(records)
    directional_reductions = [
        sid for sid, item in comparisons.items()
        if sid != "castle_control" and item.get("directional_native_support_reduction_observed") is True
    ]
    return {
        "schema_version": I56B_SCHEMA_VERSION,
        "phase": I56B_PHASE,
        "status": "current_frozen_foundation_mvs_robustness_replayed",
        "pass": True,
        "quality_status": quality,
        "scope": {
            "real_production_mvs_replay": True,
            "six_admitted_castle_scenarios_attempted": True,
            "sequential_execution": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "provider_comparison_started": False,
            "da3_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "interrupted_parent_recovery_finalization": recovery_provenance is not None,
            "mvs_rerun_during_recovery": False,
        },
        "evidence_bindings": {
            "i5_6a_admission": i5_6a_binding.as_dict(),
            "i5_6b_receipt": replay_receipt_binding.as_dict(),
        },
        "runtime_preflight": deepcopy(dict(runtime_preflight)),
        "recovery_provenance": deepcopy(dict(recovery_provenance)) if recovery_provenance is not None else None,
        "execution_policy": I56BReplayPolicy().as_dict(),
        "scenario_records": records,
        "control_relative_comparisons": comparisons,
        "observations": {
            "directional_native_support_reduction_scenarios": directional_reductions,
            "directional_native_support_reduction_observed": bool(directional_reductions),
            "cross_run_absolute_geometry_comparison": False,
            "cross_run_scale_normalization_performed": False,
            "ground_truth_surface_accuracy_available": False,
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "quality_penalty_interpretation_deferred_to_i5_6c": True,
        },
        "upstream_exclusions_preserved": deepcopy(dict(_mapping(admission.get("exclusions"), label="I5.6A exclusions"))),
        "historical_positive_anchor_preserved": deepcopy(dict(_mapping(admission.get("historical_positive_anchor"), label="I5.6A historical anchor"))),
        "regression_summary": {
            "focused_i5_6b_tests": receipt["counts"].get("focused_tests"),
            "predecessor_tests": receipt["counts"].get("predecessor_total_tests"),
            "combined_exercised_tests": receipt["counts"].get("combined_exercised_tests"),
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_6_started": True,
            "i5_6a_complete": True,
            "i5_6b_complete": True,
            "i5_6c_authorized": True,
            "i5_6c_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_6c_evidence_only_qualification_authorized": True,
            "i5_7_not_authorized_until_i5_6c_closure": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_deferred_to_i5_13": True,
        },
        "next": "I5.6C MVS robustness qualification and closure.",
    }



# ---------------------------------------------------------------------------
# I5.6C - MVS robustness qualification + closure
# ---------------------------------------------------------------------------

I56C_SCHEMA_VERSION = 1
I56C_PHASE = "I5.6C"
I56C_RECEIPT_PHASE = "I5.6C_QUALIFICATION_RECEIPT"


def _i56c_nonnegative_number(value: Any, *, label: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReconstructionValidationError(f"{label} must be a non-negative number.")
    number = float(value)
    if number < 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise ReconstructionValidationError(f"{label} must be a finite non-negative number.")
    return number


def _i56c_validate_recovery_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    recovery = deepcopy(dict(_mapping(payload, label="I5.6B recovery provenance")))
    _require(recovery, "schema_version", I56B_RECOVERY_SCHEMA_VERSION, label="I5.6B recovery provenance")
    _require(recovery, "phase", I56B_RECOVERY_PHASE, label="I5.6B recovery provenance")
    _require(recovery, "recovery_required", True, label="I5.6B recovery provenance")
    _require(recovery, "interrupted_parent_exit_code", 130, label="I5.6B recovery provenance")
    _require(recovery, "recovered_scenario_id", I56B_RECOVERY_SCENARIO_ID, label="I5.6B recovery provenance")
    for key in (
        "scenarios_1_through_5_reused_without_rerun",
        "scenario_6_native_artifacts_reused_without_rerun",
        "worker_result_missing",
        "normal_provider_summary_missing",
        "native_fusion_structurally_validated",
        "authoritative_inputs_revalidated",
    ):
        _require(recovery, key, True, label="I5.6B recovery provenance")
    for key in (
        "sfm_rerun", "patch_match_rerun", "stereo_fusion_rerun", "mvs_rerun",
        "thresholds_modified", "production_default_modified", "retry_until_pass",
    ):
        _require(recovery, key, False, label="I5.6B recovery provenance")
    cause = str(recovery.get("worker_termination_cause", "")).strip()
    if not cause:
        raise ReconstructionValidationError("I5.6C requires the interrupted-worker termination cause provenance to be preserved.")
    bindings = _mapping(recovery.get("artifact_bindings"), label="I5.6B recovery artifact bindings")
    for key in (
        "runtime_preflight", "original_replay_receipt", "recovery_validation_receipt",
        "recovered_scenario_record", "recovery_provider_summary", "recovered_native_manifest",
    ):
        binding = _mapping(bindings.get(key), label=f"I5.6B recovery binding {key}")
        if not str(binding.get("path", "")).strip() or not _valid_sha(binding.get("sha256")):
            raise ReconstructionValidationError(f"I5.6C recovery binding {key} is malformed.")
    return recovery


def validate_i5_6b_for_i5_6c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete I5.6B replay being closed by I5.6C."""

    _require(payload, "schema_version", I56B_SCHEMA_VERSION, label="I5.6B replay")
    _require(payload, "phase", I56B_PHASE, label="I5.6B replay")
    _require(payload, "status", "current_frozen_foundation_mvs_robustness_replayed", label="I5.6B replay")
    _require(payload, "pass", True, label="I5.6B replay")
    quality_status = str(payload.get("quality_status", ""))
    if quality_status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.6C I5.6B quality status is invalid.")

    scope = _mapping(payload.get("scope"), label="I5.6B scope")
    for key in ("real_production_mvs_replay", "six_admitted_castle_scenarios_attempted", "sequential_execution"):
        _require(scope, key, True, label="I5.6B scope")
    for key in (
        "sfm_rerun", "bundle_adjustment_rerun", "source_rematerialized", "provider_comparison_started",
        "da3_rerun", "thresholds_modified", "production_default_modified", "result_driven_retuning",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
        "mvs_rerun_during_recovery",
    ):
        _require(scope, key, False, label="I5.6B scope")
    recovery_used = scope.get("interrupted_parent_recovery_finalization")
    if not isinstance(recovery_used, bool):
        raise ReconstructionValidationError("I5.6C I5.6B recovery-finalization flag must be boolean.")
    recovery_raw = payload.get("recovery_provenance")
    recovery = None
    if recovery_used:
        recovery = _i56c_validate_recovery_provenance(_mapping(recovery_raw, label="I5.6B recovery provenance"))
    elif recovery_raw is not None:
        raise ReconstructionValidationError("I5.6C I5.6B non-recovery replay cannot carry recovery provenance.")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.6B evidence bindings")
    for key in ("i5_6a_admission", "i5_6b_receipt"):
        binding = _mapping(bindings.get(key), label=f"I5.6B binding {key}")
        if not str(binding.get("path", "")).strip() or not _valid_sha(binding.get("sha256")):
            raise ReconstructionValidationError(f"I5.6C I5.6B binding {key} is malformed.")

    preflight = _mapping(payload.get("runtime_preflight"), label="I5.6B runtime preflight")
    _require(preflight, "mvs_runtime_ready", True, label="I5.6B runtime preflight")
    if int(preflight.get("cuda_devices", 0)) < 1:
        raise ReconstructionValidationError("I5.6C I5.6B runtime preflight must preserve a CUDA-capable MVS runtime.")

    policy = _mapping(payload.get("execution_policy"), label="I5.6B execution policy")
    frozen_policy = I56BReplayPolicy().as_dict()
    for key, expected in frozen_policy.items():
        _require(policy, key, expected, label="I5.6B execution policy")

    records = _mapping(payload.get("scenario_records"), label="I5.6B scenario records")
    if len(records) != len(I56B_SCENARIO_IDS) or set(records) != set(I56B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.6C requires exactly the six admitted I5.6B scenario records.")
    runtime_statuses: dict[str, str] = {}
    for sid in I56B_SCENARIO_IDS:
        record = _mapping(records.get(sid), label=f"I5.6B {sid} record")
        _require(record, "scenario_id", sid, label=f"I5.6B {sid} record")
        status = str(record.get("runtime_status", ""))
        if status not in {"pass", "rejected"}:
            raise ReconstructionValidationError(f"I5.6C {sid} runtime status is invalid.")
        runtime_statuses[sid] = status
        source_views = record.get("source_registered_views")
        selected_views = record.get("selected_views")
        if not isinstance(source_views, int) or source_views < 2 or not isinstance(selected_views, int) or not (2 <= selected_views <= source_views):
            raise ReconstructionValidationError(f"I5.6C {sid} selected/source view counts are invalid.")
        _require(record, "adaptive_selection", True, label=f"I5.6B {sid} record")
        authority = _mapping(record.get("authority"), label=f"I5.6B {sid} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.6B {sid} authority")
        _require(authority, "authoritative_sfm_modified", False, label=f"I5.6B {sid} authority")
        _require(authority, "permanent_global_view_cap", None, label=f"I5.6B {sid} authority")
        for key in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology"):
            _require(authority, key, False, label=f"I5.6B {sid} authority")
        metrics = _mapping(record.get("native_metrics"), label=f"I5.6B {sid} native metrics")
        if status == "pass":
            if int(metrics.get("geometric_depth_maps", -1)) != selected_views or int(metrics.get("geometric_normal_maps", -1)) != selected_views:
                raise ReconstructionValidationError(f"I5.6C {sid} must preserve one geometric depth/normal map per selected view.")
            points = metrics.get("fused_point_count")
            visibility = metrics.get("visibility_point_count")
            if not isinstance(points, int) or points <= 0 or visibility != points:
                raise ReconstructionValidationError(f"I5.6C {sid} fused point/visibility evidence is invalid.")
            support = _i56c_nonnegative_number(metrics.get("visibility_support_mean"), label=f"I5.6C {sid} visibility support")
            if support is None or support <= 0.0:
                raise ReconstructionValidationError(f"I5.6C {sid} visibility support must be positive.")
        recovered = record.get("recovered_from_existing_native_artifacts") is True
        if recovered:
            if sid != I56B_RECOVERY_SCENARIO_ID or not recovery_used:
                raise ReconstructionValidationError("I5.6C recovery provenance is only admitted for the frozen dynamic-proxy scenario.")
            _require(record, "provider_adapter_returned", False, label=f"I5.6B {sid} recovery record")
            rp = _mapping(record.get("recovery_provenance"), label=f"I5.6B {sid} recovery provenance")
            for key in ("native_patchmatch_outputs_recovered", "native_stereo_fusion_artifacts_recovered", "native_fusion_structurally_validated", "authoritative_inputs_revalidated"):
                _require(rp, key, True, label=f"I5.6B {sid} recovery provenance")
            for key in ("sfm_rerun", "patch_match_rerun", "stereo_fusion_rerun", "mvs_rerun", "thresholds_modified", "production_default_modified", "retry_until_pass"):
                _require(rp, key, False, label=f"I5.6B {sid} recovery provenance")
            _require(metrics, "wall_time_unavailable_due_interrupted_parent_finalization", True, label=f"I5.6B {sid} native metrics")
            _require(metrics, "peak_resource_metrics_incomplete_due_missing_worker_result", True, label=f"I5.6B {sid} native metrics")
            if metrics.get("wall_time_s") is not None:
                raise ReconstructionValidationError("I5.6C recovered scenario cannot invent the interrupted parent wall time.")

    expected_quality = (
        "rejected" if runtime_statuses["castle_control"] != "pass"
        else "degraded" if any(runtime_statuses[sid] != "pass" for sid in I56B_SCENARIO_IDS if sid != "castle_control")
        else "pass"
    )
    if quality_status != expected_quality:
        raise ReconstructionValidationError("I5.6C I5.6B quality status does not match the six preserved runtime outcomes.")

    comparisons = _mapping(payload.get("control_relative_comparisons"), label="I5.6B comparisons")
    if len(comparisons) != len(I56B_SCENARIO_IDS) or set(comparisons) != set(I56B_SCENARIO_IDS):
        raise ReconstructionValidationError("I5.6C requires the complete I5.6B control-relative comparison set.")
    independently_reduced: list[str] = []
    for sid in I56B_SCENARIO_IDS:
        item = _mapping(comparisons.get(sid), label=f"I5.6B {sid} comparison")
        _require(item, "runtime_status", runtime_statuses[sid], label=f"I5.6B {sid} comparison")
        _require(item, "cross_run_absolute_geometry_comparison", False, label=f"I5.6B {sid} comparison")
        _require(item, "cross_run_scale_normalization_performed", False, label=f"I5.6B {sid} comparison")
        _require(item, "classification_threshold_used", None, label=f"I5.6B {sid} comparison")
        _require(item, "global_scalar_score_created", False, label=f"I5.6B {sid} comparison")
        if sid != "castle_control" and item.get("directional_native_support_reduction_observed") is True:
            independently_reduced.append(sid)

    observations = _mapping(payload.get("observations"), label="I5.6B observations")
    recorded_reduced = observations.get("directional_native_support_reduction_scenarios")
    if not isinstance(recorded_reduced, list) or recorded_reduced != independently_reduced:
        raise ReconstructionValidationError("I5.6C I5.6B directional support-reduction scenario list drifted.")
    _require(observations, "directional_native_support_reduction_observed", bool(independently_reduced), label="I5.6B observations")
    _require(observations, "cross_run_absolute_geometry_comparison", False, label="I5.6B observations")
    _require(observations, "cross_run_scale_normalization_performed", False, label="I5.6B observations")
    _require(observations, "ground_truth_surface_accuracy_available", False, label="I5.6B observations")
    _require(observations, "classification_threshold_used", None, label="I5.6B observations")
    _require(observations, "global_scalar_score_created", False, label="I5.6B observations")
    _require(observations, "quality_penalty_interpretation_deferred_to_i5_6c", True, label="I5.6B observations")

    anchor = _mapping(payload.get("historical_positive_anchor_preserved"), label="I5.6B historical anchor")
    _require(anchor, "reference_views", 47, label="I5.6B historical anchor")
    _require(anchor, "evidence_only", True, label="I5.6B historical anchor")
    _require(anchor, "rerun_authorized", False, label="I5.6B historical anchor")

    authority = _mapping(payload.get("authority"), label="I5.6B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.6B authority")
    _require(authority, "authoritative_sfm_modified", False, label="I5.6B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.6B authority")
    for key in ("provider_selected", "provider_promoted", "provider_ranking_created", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.6B authority")

    decision = _mapping(payload.get("decision"), label="I5.6B decision")
    for key in ("i5_6_started", "i5_6a_complete", "i5_6b_complete", "i5_6c_authorized"):
        _require(decision, key, True, label="I5.6B decision")
    _require(decision, "i5_6c_started", False, label="I5.6B decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized", "mp_sfm_authorized",
        "threshold_relaxation_authorized", "production_default_modified",
    ):
        _require(decision, key, False, label="I5.6B decision")
    roadmap = _mapping(payload.get("roadmap_guard"), label="I5.6B roadmap guard")
    _require(roadmap, "i5_6c_evidence_only_qualification_authorized", True, label="I5.6B roadmap guard")
    _require(roadmap, "i5_7_not_authorized_until_i5_6c_closure", True, label="I5.6B roadmap guard")
    _require(roadmap, "mp_sfm_deferred_to_i5_10", True, label="I5.6B roadmap guard")
    _require(roadmap, "hybrid_deferred_to_i5_12_if_justified", True, label="I5.6B roadmap guard")
    _require(roadmap, "auto_deferred_to_i5_13", True, label="I5.6B roadmap guard")
    if str(payload.get("next", "")) != "I5.6C MVS robustness qualification and closure.":
        raise ReconstructionValidationError("I5.6C predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_i5_6c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I56C_SCHEMA_VERSION, label="I5.6C receipt")
    _require(payload, "phase", I56C_RECEIPT_PHASE, label="I5.6C receipt")
    _require(payload, "status", "pass", label="I5.6C receipt")
    _require(payload, "pass", True, label="I5.6C receipt")
    scope = _mapping(payload.get("scope"), label="I5.6C receipt scope")
    for key in (
        "focused_i5_6c_tests", "predecessor_i5_6b_regression", "predecessor_i5_6a_regression",
        "evidence_only_closure_contract_regression", "interrupted_recovery_provenance_regression",
        "directional_native_support_preservation_regression", "authority_guard_regression",
        "roadmap_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.6C receipt scope")
    for key in (
        "sfm_rerun", "mvs_rerun", "patch_match_rerun", "stereo_fusion_rerun",
        "provider_comparison_started", "da3_rerun", "thresholds_modified", "production_default_modified",
        "retry_until_pass", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.6C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.6C receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.6C receipt test counts must be non-negative integers.")
    if focused < 15 or predecessor < 40 or combined < 60 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.6C receipt coverage is below the required closure floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.6C receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.6C receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _i56c_directional_native_support(payload: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = _mapping(payload.get("control_relative_comparisons"), label="I5.6B comparisons")
    scenarios: dict[str, Any] = {}
    reductions: list[str] = []
    native_complete = True
    performance_complete = True
    resource_complete = True
    for sid in I56B_SCENARIO_IDS:
        item = _mapping(comparisons.get(sid), label=f"I5.6B {sid} comparison")
        row = {
            "runtime_status": item.get("runtime_status"),
            "selected_view_ratio_to_control": item.get("selected_view_ratio_to_control"),
            "fused_point_ratio_to_control": item.get("fused_point_ratio_to_control"),
            "visibility_support_mean_ratio_to_control": item.get("visibility_support_mean_ratio_to_control"),
            "wall_time_ratio_to_control": item.get("wall_time_ratio_to_control"),
            "peak_vram_ratio_to_control": item.get("peak_vram_ratio_to_control"),
            "directional_native_support_reduction_observed": item.get("directional_native_support_reduction_observed"),
        }
        scenarios[sid] = row
        if sid != "castle_control" and row["directional_native_support_reduction_observed"] is True:
            reductions.append(sid)
        if row["runtime_status"] == "pass":
            if row["fused_point_ratio_to_control"] is None or row["visibility_support_mean_ratio_to_control"] is None:
                native_complete = False
            if row["wall_time_ratio_to_control"] is None:
                performance_complete = False
            if row["peak_vram_ratio_to_control"] is None:
                resource_complete = False
    return {
        "scenario_comparisons": scenarios,
        "directional_native_support_reduction_scenarios": reductions,
        "directional_native_support_reduction_observed": bool(reductions),
        "native_support_comparison_complete_for_passing_scenarios": native_complete,
        "runtime_performance_comparison_complete_for_passing_scenarios": performance_complete,
        "resource_comparison_complete_for_passing_scenarios": resource_complete,
        "classification_threshold_used": None,
        "global_scalar_score_created": False,
        "directional_comparison_only": True,
        "cross_run_absolute_geometry_comparison": False,
        "cross_run_scale_normalization_performed": False,
    }


def build_i5_6c_mvs_robustness_closure(
    *,
    i5_6a_result: Mapping[str, Any],
    i5_6a_binding: I51EvidenceBinding,
    i5_6b_result: Mapping[str, Any],
    i5_6b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_6a_for_i5_6b(i5_6a_result)
    replay = validate_i5_6b_for_i5_6c(i5_6b_result)
    receipt = validate_i5_6c_receipt(qualification_receipt)
    for binding, label in (
        (i5_6a_binding, "I5.6C I5.6A"),
        (i5_6b_binding, "I5.6C I5.6B"),
        (qualification_receipt_binding, "I5.6C receipt"),
    ):
        _validate_binding(binding, label=label)

    replay_bindings = _mapping(replay.get("evidence_bindings"), label="I5.6B evidence bindings")
    replay_a = _mapping(replay_bindings.get("i5_6a_admission"), label="I5.6B I5.6A binding")
    if str(replay_a.get("sha256", "")).lower() != i5_6a_binding.sha256.lower():
        raise ReconstructionValidationError("I5.6C I5.6B predecessor SHA does not match the independently verified I5.6A artifact.")
    if dict(_mapping(replay.get("execution_policy"), label="I5.6B execution policy")) != dict(_mapping(admission.get("execution_policy"), label="I5.6A execution policy")):
        raise ReconstructionValidationError("I5.6C I5.6A/I5.6B frozen MVS execution policy drifted.")

    quality_status = str(replay["quality_status"])
    records = _mapping(replay.get("scenario_records"), label="I5.6B scenario records")
    passed = [sid for sid in I56B_SCENARIO_IDS if records[sid].get("runtime_status") == "pass"]
    rejected = [sid for sid in I56B_SCENARIO_IDS if records[sid].get("runtime_status") != "pass"]
    directional = _i56c_directional_native_support(replay)
    reduction = directional["directional_native_support_reduction_observed"]

    if quality_status == "pass":
        operating_envelope = (
            "qualified_on_all_six_admitted_castle_mvs_scenarios_with_directional_native_support_reductions"
            if reduction else "qualified_on_all_six_admitted_castle_mvs_scenarios"
        )
    elif quality_status == "degraded":
        operating_envelope = "degraded_on_admitted_castle_mvs_envelope"
    else:
        operating_envelope = "rejected_on_admitted_castle_mvs_envelope"

    recovery_used = replay["scope"]["interrupted_parent_recovery_finalization"] is True
    recovery = deepcopy(dict(replay["recovery_provenance"])) if recovery_used else None
    recovery_cause = str(recovery.get("worker_termination_cause")) if recovery is not None else None

    return {
        "schema_version": I56C_SCHEMA_VERSION,
        "phase": I56C_PHASE,
        "status": "mvs_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": quality_status,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_6a_admission_reused": True,
            "i5_6b_real_replay_reused": True,
            "interrupted_recovery_evidence_reused": recovery_used,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "mvs_rerun": False,
            "patch_match_rerun": False,
            "stereo_fusion_rerun": False,
            "source_rematerialized": False,
            "provider_comparison_started": False,
            "da3_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "mp_sfm_started": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "evidence_bindings": {
            "i5_6a_admission": i5_6a_binding.as_dict(),
            "i5_6b_real_replay": i5_6b_binding.as_dict(),
            "i5_6c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "provenance": {
            "i5_6b_quality_status_preserved": True,
            "i5_6b_scenario_runtime_outcomes_preserved": True,
            "i5_6b_control_relative_native_metrics_preserved": True,
            "i5_6b_recovery_provenance_preserved": recovery_used,
            "historical_result_overwritten": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "worker_termination_cause": recovery_cause,
            "worker_termination_cause_resolved": False if recovery_used and "unresolved" in (recovery_cause or "") else None,
        },
        "qualification": {
            "quality_status": quality_status,
            "operating_envelope": operating_envelope,
            "admitted_scenario_count": len(I56B_SCENARIO_IDS),
            "runtime_pass_scenario_count": len(passed),
            "runtime_rejected_scenario_count": len(rejected),
            "runtime_pass_scenarios": passed,
            "runtime_rejected_scenarios": rejected,
            "all_six_admitted_scenarios_runtime_pass": len(passed) == len(I56B_SCENARIO_IDS),
            "current_production_colmap_mvs_path_qualified_within_admitted_envelope": quality_status == "pass",
            "directional_native_support_reduction_observed": reduction,
            "directional_native_support_reduction_scenarios": list(directional["directional_native_support_reduction_scenarios"]),
            "native_support_comparison_complete_for_passing_scenarios": directional["native_support_comparison_complete_for_passing_scenarios"],
            "runtime_performance_qualification_claimed": False,
            "resource_robustness_qualification_claimed": False,
            "ground_truth_surface_accuracy_available": False,
            "ground_truth_surface_accuracy_claimed": False,
            "cross_run_absolute_geometry_comparison_performed": False,
            "cross_run_scale_normalization_performed": False,
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "universal_mvs_robustness_claimed": False,
            "universal_provider_winner_claimed": False,
            "provider_ranking_claimed": False,
            "metric_scale_claimed": False,
        },
        "directional_native_support_evidence": directional,
        "recovery_provenance": recovery,
        "limitations": {
            "recovered_dynamic_proxy_parent_wall_time_unavailable": bool(
                recovery_used and records[I56B_RECOVERY_SCENARIO_ID]["native_metrics"].get("wall_time_s") is None
            ),
            "recovered_dynamic_proxy_peak_resource_metrics_incomplete": bool(
                recovery_used and records[I56B_RECOVERY_SCENARIO_ID]["native_metrics"].get("peak_resource_metrics_incomplete_due_missing_worker_result") is True
            ),
            "runtime_performance_envelope_complete": directional["runtime_performance_comparison_complete_for_passing_scenarios"],
            "resource_envelope_complete": directional["resource_comparison_complete_for_passing_scenarios"],
            "upstream_excluded_scenarios_are_not_reclassified_as_mvs_failures": True,
            "historical_temple_ring_anchor_is_evidence_only": True,
            "castle_suite_is_not_universal_mvs_ground_truth": True,
            "arbitrary_sfm_gauge_preserved": True,
        },
        "upstream_exclusions_preserved": deepcopy(dict(_mapping(replay.get("upstream_exclusions_preserved"), label="I5.6B exclusions"))),
        "historical_positive_anchor_preserved": deepcopy(dict(_mapping(replay.get("historical_positive_anchor_preserved"), label="I5.6B historical anchor"))),
        "regression_summary": {
            "focused_i5_6c_tests": receipt["counts"].get("focused_tests"),
            "predecessor_tests": receipt["counts"].get("predecessor_total_tests"),
            "combined_exercised_tests": receipt["counts"].get("combined_exercised_tests"),
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_6_complete": True,
            "i5_7_authorized": True,
            "i5_7_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
        "roadmap_guard": {
            "i5_7_rollingdepth_authorized": True,
            "i5_8_video_depth_anything_not_started": True,
            "i5_10_mp_sfm_deferred": True,
            "i5_12_hybrid_deferred_if_justified": True,
            "i5_13_auto_deferred": True,
        },
        "next": "I5.7 RollingDepth integration and robustness envelope.",
    }

__all__ = [
    "I56A_SCHEMA_VERSION", "I56A_PHASE", "I56A_RECEIPT_PHASE", "I56A_SCENARIO_IDS", "I56AAdmissionPolicy",
    "validate_i5_5c_for_i5_6", "validate_i5_6a_receipt", "analyze_i5_6a_sfm_report",
    "build_i5_6a_historical_mvs_anchor", "build_i5_6a_mvs_robustness_admission",
    "I56B_SCHEMA_VERSION", "I56B_PHASE", "I56B_RECEIPT_PHASE", "I56B_SCENARIO_IDS", "I56BReplayPolicy",
    "validate_i5_6a_for_i5_6b", "validate_i5_6b_receipt", "read_i5_6b_visibility_support",
    "analyze_i5_6b_provider_summary", "validate_i5_6b_existing_scenario_record",
    "recover_i5_6b_interrupted_native_scenario", "compare_i5_6b_scenarios", "build_i5_6b_mvs_robustness_replay",
    "I56B_RECOVERY_SCHEMA_VERSION", "I56B_RECOVERY_PHASE", "I56B_RECOVERY_SCENARIO_ID",
    "I56C_SCHEMA_VERSION", "I56C_PHASE", "I56C_RECEIPT_PHASE",
    "validate_i5_6b_for_i5_6c", "validate_i5_6c_receipt", "build_i5_6c_mvs_robustness_closure",
]
