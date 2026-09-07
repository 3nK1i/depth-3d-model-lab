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

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
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


__all__ = [
    "I56A_SCHEMA_VERSION","I56A_PHASE","I56A_RECEIPT_PHASE","I56A_SCENARIO_IDS","I56AAdmissionPolicy",
    "validate_i5_5c_for_i5_6","validate_i5_6a_receipt","analyze_i5_6a_sfm_report",
    "build_i5_6a_historical_mvs_anchor","build_i5_6a_mvs_robustness_admission",
]
