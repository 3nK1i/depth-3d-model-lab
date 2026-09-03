"""I5.2 low-overlap / weak-parallax robustness contracts.

I5.2A is an execution-free admission boundary.  It binds the frozen I5.1C
handoff, admits only the already-qualified I4.3 weak-overlap history, content-
addresses one ordered static multiview source, and freezes deterministic
low-overlap and weak-parallax *stress proxies* for the later I5.2B replay.

The source-order proxies are deliberately not claimed as authoritative overlap
or parallax measurements.  I5.2B must revalidate the resulting subsets with
PyCOLMAP camera/covisibility geometry under one like-for-like matching policy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.sources import natural_path_key

I52A_SCHEMA_VERSION = 1
I52A_PHASE = "I5.2A"
I52A_RECEIPT_PHASE = "I5.2A_ADMISSION_RECEIPT"

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"})
_AUTHORITY_FALSE = (
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
class I52AStressPlanPolicy:
    minimum_source_images: int = 12
    minimum_subset_images: int = 6
    maximum_subset_images: int = 24
    maximum_low_overlap_spacing_factor: int = 8

    def __post_init__(self) -> None:
        if int(self.minimum_source_images) < 12:
            raise ReconstructionValidationError("I5.2A minimum_source_images must remain >= 12.")
        if int(self.minimum_subset_images) < 6:
            raise ReconstructionValidationError("I5.2A minimum_subset_images must remain >= 6.")
        if int(self.maximum_subset_images) < int(self.minimum_subset_images):
            raise ReconstructionValidationError("I5.2A maximum_subset_images must be >= minimum_subset_images.")
        if int(self.maximum_low_overlap_spacing_factor) < 2:
            raise ReconstructionValidationError("I5.2A maximum low-overlap spacing factor must be >= 2.")

    def as_dict(self) -> dict[str, int]:
        return {
            "minimum_source_images": int(self.minimum_source_images),
            "minimum_subset_images": int(self.minimum_subset_images),
            "maximum_subset_images": int(self.maximum_subset_images),
            "maximum_low_overlap_spacing_factor": int(self.maximum_low_overlap_spacing_factor),
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"{label}.{key} drifted: expected={expected!r} actual={mapping.get(key)!r}"
        )


def _valid_sha256(value: Any) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_i5_1c_for_i5_2(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I5.1C" or payload.get("status") != "difficult_handheld_robustness_qualified" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2A requires the passing frozen I5.1C closure.")
    decision = _mapping(payload.get("decision"), label="I5.1C decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_authorized"):
        _require(decision, key, True, label="I5.1C decision")
    _require(decision, "i5_2_started", False, label="I5.1C decision")
    for key in (
        "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized",
        "dense_consistency_threshold_relaxation_authorized", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.1C decision")
    authority = _mapping(payload.get("authority"), label="I5.1C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.1C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.1C authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="I5.1C authority")
    if payload.get("next") != "I5.2 low overlap / weak parallax.":
        raise ReconstructionValidationError("I5.2A predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_historical_weak_overlap_evidence(
    *,
    g2_final: Mapping[str, Any],
    g2a3_doc: str,
    g2b_doc: str,
    g2c_doc: str,
) -> dict[str, Any]:
    if g2_final.get("phase") != "I4.3G2" or g2_final.get("status") != "formally_closed":
        raise ReconstructionValidationError("I5.2A requires the formally closed I4.3G2 result.")
    for key in (
        "all_required_g2_scenarios_accounted", "all_required_manual_reviews_complete",
        "g2a_source_binding_complete", "g2b_production_native_complete",
        "g2c_manual_qualification_complete", "i4_3g2_closed",
    ):
        _require(g2_final, key, True, label="I4.3G2 closure")
    scenarios = _mapping(g2_final.get("scenario_synthesis"), label="I4.3G2 scenario_synthesis")
    weak = _mapping(scenarios.get("g2_weak_overlap_mvs_da3"), label="I4.3G2 weak-overlap scenario")
    _require(weak, "controlled_reveal_complete", True, label="I4.3G2 weak-overlap scenario")
    _require(weak, "evidence_mode", "fresh_production_plus_matched_view", label="I4.3G2 weak-overlap scenario")
    _require(weak, "full_scenario_claim_permitted", False, label="I4.3G2 weak-overlap scenario")
    _require(weak, "global_provider_winner_declared", False, label="I4.3G2 weak-overlap scenario")
    _require(weak, "manual_review_complete", True, label="I4.3G2 weak-overlap scenario")
    _require(weak, "partial_registration_preserved", True, label="I4.3G2 weak-overlap scenario")
    _require(weak, "review_scope", "authoritative_registered_views", label="I4.3G2 weak-overlap scenario")
    synthesis = _mapping(g2_final.get("synthesis"), label="I4.3G2 synthesis")
    _require(synthesis, "camera_pose_authority_owner", "pycolmap_sfm", label="I4.3G2 synthesis")
    for key in (
        "provider_selected", "provider_promoted", "universal_provider_winner_declared",
        "auto_provider_selection_implemented", "hybrid_provider_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(synthesis, key, False, label="I4.3G2 synthesis")

    docs = {
        "I4.3G2A.3": (g2a3_doc, (
            "ordered, increased-spacing subset",
            "authoritative overlap measurement",
            "authoritative PyCOLMAP SfM/covisibility evidence",
            "does not weaken SfM gates",
        )),
        "I4.3G2B": (g2b_doc, (
            "authoritative PyCOLMAP co-visibility connectivity",
            "thresholds are not relaxed",
        )),
        "I4.3G2C": (g2c_doc, (
            "connected five-view authoritative registered scope",
            "authoritative_registered_views",
            "No provider or SfM rerun is authorized",
        )),
    }
    for label, (text, needles) in docs.items():
        if not isinstance(text, str) or not text.strip():
            raise ReconstructionValidationError(f"I5.2A {label} phase history is empty.")
        normalized = " ".join(text.split())
        missing = [needle for needle in needles if needle not in normalized]
        if missing:
            raise ReconstructionValidationError(f"I5.2A {label} phase history lost decisive weak-overlap claims: {missing}")

    return {
        "evidence_level": "frozen_i4_3g2_closure_plus_committed_phase_history",
        "scenario_id": "g2_weak_overlap_mvs_da3",
        "historical_authoritative_registered_view_scope": 5,
        "historical_review_scope": "authoritative_registered_views",
        "historical_partial_registration_preserved": True,
        "historical_full_scenario_claim_permitted": False,
        "historical_weak_overlap_proxy_was_not_authoritative_measurement": True,
        "historical_authoritative_pycolmap_revalidation_required": True,
        "historical_threshold_relaxation_performed": False,
        "historical_provider_or_sfm_rerun_by_i5_2a": False,
    }


def build_ordered_source_identity(
    directory: str | Path,
    *,
    policy: I52AStressPlanPolicy = I52AStressPlanPolicy(),
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.2A ordered source directory does not exist: {root}")
    images = tuple(sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=natural_path_key,
    ))
    if len(images) < int(policy.minimum_source_images):
        raise ReconstructionValidationError(
            f"I5.2A ordered source requires at least {policy.minimum_source_images} images; found {len(images)}"
        )
    names = tuple(path.name for path in images)
    if len(set(names)) != len(names):
        raise ReconstructionValidationError("I5.2A ordered source contains duplicate image names.")
    aggregate = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(images):
        sha = _sha256(path)
        size = int(path.stat().st_size)
        record = {"order": index, "name": path.name, "size_bytes": size, "sha256": sha}
        records.append(record)
        aggregate.update(str(index).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(path.name.encode("utf-8")); aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii")); aggregate.update(b"\n")
    return {
        "directory": str(root),
        "image_count": len(records),
        "ordered_names": list(names),
        "asset_set_sha256": aggregate.hexdigest(),
        "files": records,
        "natural_source_order_used": True,
        "recursive_discovery": False,
    }


def validate_ordered_source_identity(
    identity: Mapping[str, Any],
    *,
    policy: I52AStressPlanPolicy = I52AStressPlanPolicy(),
) -> dict[str, Any]:
    count = identity.get("image_count")
    if not isinstance(count, int) or count < int(policy.minimum_source_images):
        raise ReconstructionValidationError("I5.2A source identity image_count is below the required minimum.")
    names = identity.get("ordered_names")
    files = identity.get("files")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != count:
        raise ReconstructionValidationError("I5.2A ordered_names do not match image_count.")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or len(files) != count:
        raise ReconstructionValidationError("I5.2A file records do not match image_count.")
    if len(set(str(name) for name in names)) != count:
        raise ReconstructionValidationError("I5.2A ordered_names contain duplicates.")
    if identity.get("natural_source_order_used") is not True or identity.get("recursive_discovery") is not False:
        raise ReconstructionValidationError("I5.2A source ordering contract drifted.")
    if not _valid_sha256(identity.get("asset_set_sha256")):
        raise ReconstructionValidationError("I5.2A source asset-set SHA256 is invalid.")
    for index, (name, raw) in enumerate(zip(names, files)):
        record = _mapping(raw, label=f"I5.2A source file {index}")
        _require(record, "order", index, label=f"I5.2A source file {index}")
        _require(record, "name", str(name), label=f"I5.2A source file {index}")
        if not isinstance(record.get("size_bytes"), int) or int(record["size_bytes"]) <= 0:
            raise ReconstructionValidationError("I5.2A source file size must be positive.")
        if not _valid_sha256(record.get("sha256")):
            raise ReconstructionValidationError("I5.2A source file SHA256 is invalid.")
    return deepcopy(dict(identity))


def _bounded_indices(indices: Sequence[int], *, maximum: int) -> list[int]:
    values = list(indices)
    if len(values) <= maximum:
        return values
    if maximum < 2:
        return values[:maximum]
    positions = [round(i * (len(values) - 1) / (maximum - 1)) for i in range(maximum)]
    chosen = [values[int(pos)] for pos in positions]
    return list(dict.fromkeys(chosen))


def _slice_record(names: Sequence[str], indices: Sequence[int], *, kind: str, proxy_fields: Mapping[str, Any]) -> dict[str, Any]:
    selected = list(int(index) for index in indices)
    if len(selected) < 2:
        raise ReconstructionValidationError("I5.2A stress slice requires at least two images.")
    denominator = max(1, len(names) - 1)
    return {
        "kind": kind,
        "indices": selected,
        "names": [str(names[index]) for index in selected],
        "image_count": len(selected),
        "first_source_index": selected[0],
        "last_source_index": selected[-1],
        "source_span_fraction": float((selected[-1] - selected[0]) / denominator),
        **dict(proxy_fields),
    }


def build_stress_plan(
    source_identity: Mapping[str, Any],
    *,
    policy: I52AStressPlanPolicy = I52AStressPlanPolicy(),
) -> dict[str, Any]:
    identity = validate_ordered_source_identity(source_identity, policy=policy)
    names = tuple(str(name) for name in identity["ordered_names"])
    count = len(names)

    max_stride_from_minimum = max(2, count // int(policy.minimum_subset_images))
    stride = min(int(policy.maximum_low_overlap_spacing_factor), max_stride_from_minimum)
    low_indices = list(range(0, count, stride))
    if low_indices[-1] != count - 1:
        low_indices.append(count - 1)
    low_indices = _bounded_indices(low_indices, maximum=int(policy.maximum_subset_images))
    if len(low_indices) < int(policy.minimum_subset_images):
        raise ReconstructionValidationError("I5.2A cannot form the minimum low-overlap proxy slice.")

    weak_count = min(
        int(policy.maximum_subset_images),
        max(int(policy.minimum_subset_images), len(low_indices)),
        count,
    )
    weak_start = (count - weak_count) // 2
    weak_indices = list(range(weak_start, weak_start + weak_count))

    control = _slice_record(
        names, range(count), kind="full_source_control",
        proxy_fields={
            "stress_proxy": False,
            "production_default": False,
        },
    )
    low = _slice_record(
        names, low_indices, kind="controlled_low_overlap_source_spacing_proxy",
        proxy_fields={
            "stress_proxy": True,
            "capture_spacing_factor": stride,
            "source_order_preserved": True,
            "full_span_requested": True,
            "proxy_is_authoritative_overlap_measurement": False,
            "requires_pycolmap_covisibility_revalidation": True,
        },
    )
    weak = _slice_record(
        names, weak_indices, kind="controlled_weak_parallax_narrow_span_proxy",
        proxy_fields={
            "stress_proxy": True,
            "capture_spacing_factor": 1,
            "contiguous_source_window": True,
            "source_order_preserved": True,
            "proxy_is_authoritative_parallax_measurement": False,
            "requires_pycolmap_camera_geometry_revalidation": True,
        },
    )
    if low["source_span_fraction"] < 0.999999:
        raise ReconstructionValidationError("I5.2A low-overlap proxy must preserve full source span.")
    if weak["source_span_fraction"] >= low["source_span_fraction"]:
        raise ReconstructionValidationError("I5.2A weak-parallax proxy must have a narrower source span than low-overlap proxy.")

    return {
        "policy": policy.as_dict(),
        "source_asset_set_sha256": identity["asset_set_sha256"],
        "control": control,
        "low_overlap_proxy": low,
        "weak_parallax_proxy": weak,
        "interpretation_guard": {
            "source_index_spacing_is_overlap_ground_truth": False,
            "source_index_span_is_parallax_ground_truth": False,
            "semantic_stress_labels_require_authoritative_i5_2b_revalidation": True,
            "failed_authoritative_revalidation_is_legitimate_scenario_evidence": True,
            "quality_threshold_relaxation_authorized": False,
        },
        "execution_contract": {
            "matching_strategy_locked_for_like_for_like_replay": "exhaustive",
            "same_pycolmap_authority_for_control_and_stress_slices": True,
            "provider_execution_required_for_i5_2a": False,
            "production_sampling_policy_changed": False,
            "permanent_global_view_cap": None,
        },
    }


def validate_i5_2a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I52A_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2A requires a passing admission regression receipt.")
    scope = _mapping(payload.get("scope"), label="I5.2A receipt scope")
    for key in (
        "focused_i5_2a_tests", "predecessor_i5_1c_regression", "weak_overlap_methodology_regression",
        "source_identity_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.2A receipt scope")
    for key in (
        "provider_execution_started", "sfm_rerun", "dense_provider_rerun", "thresholds_modified",
        "production_default_modified", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.2A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.2A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.2A receipt test counts must be non-negative integers.")
    if focused < 15 or predecessor < 30 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.2A receipt coverage is below the required admission floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.2A receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.2A receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_2a_low_overlap_weak_parallax_admission(
    *,
    i5_1c_result: Mapping[str, Any],
    i5_1c_binding: I51EvidenceBinding,
    i4_3g2_final_result: Mapping[str, Any],
    i4_3g2_final_binding: I51EvidenceBinding,
    i4_3g2a3_doc: str,
    i4_3g2a3_binding: I51EvidenceBinding,
    i4_3g2b_doc: str,
    i4_3g2b_binding: I51EvidenceBinding,
    i4_3g2c_doc: str,
    i4_3g2c_binding: I51EvidenceBinding,
    source_identity: Mapping[str, Any],
    semantic_confirmation: bool,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
    policy: I52AStressPlanPolicy = I52AStressPlanPolicy(),
) -> dict[str, Any]:
    predecessor = validate_i5_1c_for_i5_2(i5_1c_result)
    historical = validate_historical_weak_overlap_evidence(
        g2_final=i4_3g2_final_result,
        g2a3_doc=i4_3g2a3_doc,
        g2b_doc=i4_3g2b_doc,
        g2c_doc=i4_3g2c_doc,
    )
    identity = validate_ordered_source_identity(source_identity, policy=policy)
    receipt = validate_i5_2a_receipt(admission_receipt)
    if semantic_confirmation is not True:
        raise ReconstructionValidationError(
            "I5.2A requires explicit confirmation that the bound source is an ordered static multiview capture suitable for controlled stress slicing."
        )
    plan = build_stress_plan(identity, policy=policy)

    authority = deepcopy(dict(_mapping(predecessor.get("authority"), label="I5.1C authority")))
    return {
        "schema_version": I52A_SCHEMA_VERSION,
        "phase": I52A_PHASE,
        "status": "low_overlap_weak_parallax_scenario_admitted",
        "pass": True,
        "scope": {
            "historical_baseline_admission": True,
            "fresh_source_binding": True,
            "deterministic_stress_plan_only": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "current_low_overlap_replay_started": False,
            "current_weak_parallax_replay_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "provider_specific_robustness_campaign_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_1c_closure": i5_1c_binding.as_dict(),
            "i4_3g2_final_closure": i4_3g2_final_binding.as_dict(),
            "i4_3g2a3_phase_history": i4_3g2a3_binding.as_dict(),
            "i4_3g2b_phase_history": i4_3g2b_binding.as_dict(),
            "i4_3g2c_phase_history": i4_3g2c_binding.as_dict(),
            "i5_2a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "provenance": {
            "i5_1c_reverified": True,
            "historical_i4_3g2_weak_overlap_reverified": True,
            "committed_weak_overlap_methodology_sha256_bound": True,
            "fresh_source_asset_set_sha256": identity["asset_set_sha256"],
            "fresh_source_per_file_sha256_bound": True,
            "semantic_label_inferred_from_directory_name": False,
            "semantic_confirmation_explicit": True,
            "source_order_proxy_tuned_from_provider_results": False,
            "synthetic_runtime_quality_result_created": False,
        },
        "historical_baseline": historical,
        "source_identity": identity,
        "source_semantics": {
            "ordered_static_multiview_confirmed": True,
            "path_name_used_as_semantic_evidence": False,
            "low_overlap_label_authoritatively_proven_in_i5_2a": False,
            "weak_parallax_label_authoritatively_proven_in_i5_2a": False,
        },
        "stress_plan": plan,
        "regression_summary": {
            "focused_i5_2a_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": authority,
        "decision": {
            "i4_closed": True,
            "i5_authorized": True,
            "i5_1_complete": True,
            "i5_2_started": True,
            "i5_2a_complete": True,
            "i5_2b_real_replay_authorized": True,
            "i5_2_complete": False,
            "i5_3_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_2b_real_low_overlap_weak_parallax_replay_authorized_not_started": True,
            "i5_3_blur_exposure_clutter_symmetry_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_evaluation_not_started": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.2B current frozen-foundation low-overlap / weak-parallax replay.",
    }


__all__ = [
    "I52A_SCHEMA_VERSION",
    "I52A_PHASE",
    "I52A_RECEIPT_PHASE",
    "I52AStressPlanPolicy",
    "build_ordered_source_identity",
    "validate_ordered_source_identity",
    "build_stress_plan",
    "validate_i5_1c_for_i5_2",
    "validate_historical_weak_overlap_evidence",
    "validate_i5_2a_receipt",
    "build_i5_2a_low_overlap_weak_parallax_admission",
]
