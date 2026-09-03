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

# ---------------------------------------------------------------------------
# I5.2B — current frozen-foundation real replay
# ---------------------------------------------------------------------------

import numpy as np

I52B_SCHEMA_VERSION = 1
I52B_PHASE = "I5.2B"
I52B_RECEIPT_PHASE = "I5.2B_REPLAY_RECEIPT"


def _float_quantiles(values: object) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
    }


def validate_i5_2a_for_i5_2b(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != "I5.2A" or payload.get("status") != "low_overlap_weak_parallax_scenario_admitted" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2B requires the passing frozen I5.2A admission result.")
    decision = _mapping(payload.get("decision"), label="I5.2A decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_started", "i5_2a_complete", "i5_2b_real_replay_authorized"):
        _require(decision, key, True, label="I5.2A decision")
    for key in (
        "i5_2_complete", "i5_3_authorized", "provider_selected", "provider_promoted",
        "provider_ranking_created", "universal_provider_winner_declared", "global_scalar_score_created",
        "auto_policy_implemented", "hybrid_authorized", "threshold_relaxation_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.2A decision")
    scope = _mapping(payload.get("scope"), label="I5.2A scope")
    for key in ("current_low_overlap_replay_started", "current_weak_parallax_replay_started", "thresholds_modified", "production_default_modified"):
        _require(scope, key, False, label="I5.2A scope")
    authority = _mapping(payload.get("authority"), label="I5.2A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.2A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.2A authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="I5.2A authority")
    plan = _mapping(payload.get("stress_plan"), label="I5.2A stress_plan")
    execution = _mapping(plan.get("execution_contract"), label="I5.2A execution_contract")
    _require(execution, "matching_strategy_locked_for_like_for_like_replay", "exhaustive", label="I5.2A execution_contract")
    _require(execution, "same_pycolmap_authority_for_control_and_stress_slices", True, label="I5.2A execution_contract")
    _require(execution, "production_sampling_policy_changed", False, label="I5.2A execution_contract")
    _require(execution, "permanent_global_view_cap", None, label="I5.2A execution_contract")
    for key in ("control", "low_overlap_proxy", "weak_parallax_proxy"):
        scenario = _mapping(plan.get(key), label=f"I5.2A {key}")
        names = scenario.get("names")
        indices = scenario.get("indices")
        count = scenario.get("image_count")
        if not isinstance(count, int) or count < 3:
            raise ReconstructionValidationError(f"I5.2B {key} image_count is invalid.")
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != count:
            raise ReconstructionValidationError(f"I5.2B {key} names do not match image_count.")
        if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)) or len(indices) != count:
            raise ReconstructionValidationError(f"I5.2B {key} indices do not match image_count.")
    if payload.get("next") != "I5.2B current frozen-foundation low-overlap / weak-parallax replay.":
        raise ReconstructionValidationError("I5.2B predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_i5_2b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I52B_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2B requires a passing replay regression receipt.")
    scope = _mapping(payload.get("scope"), label="I5.2B receipt scope")
    for key in (
        "focused_i5_2b_tests", "predecessor_i5_2a_regression", "production_native_sfm_contract_regression",
        "scale_invariant_geometry_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.2B receipt scope")
    for key in (
        "real_sfm_executed_by_test_receipt", "dense_provider_rerun", "thresholds_modified",
        "production_default_modified", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.2B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.2B receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.2B receipt test counts must be non-negative integers.")
    if focused < 15 or predecessor < 30 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.2B receipt coverage is below the required replay floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.2B receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.2B receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _scenario_plan_identity(plan: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names_raw = plan.get("names")
    indices_raw = plan.get("indices")
    count = plan.get("image_count")
    if not isinstance(count, int) or count < 3:
        raise ReconstructionValidationError("I5.2B scenario plan image_count must be at least three.")
    if not isinstance(names_raw, Sequence) or isinstance(names_raw, (str, bytes)) or len(names_raw) != count:
        raise ReconstructionValidationError("I5.2B scenario names do not match image_count.")
    if not isinstance(indices_raw, Sequence) or isinstance(indices_raw, (str, bytes)) or len(indices_raw) != count:
        raise ReconstructionValidationError("I5.2B scenario indices do not match image_count.")
    names = tuple(str(value) for value in names_raw)
    indices = tuple(int(value) for value in indices_raw)
    if len(set(names)) != count or len(set(indices)) != count or list(indices) != sorted(indices):
        raise ReconstructionValidationError("I5.2B scenario plan identity is not unique/ordered.")
    return names, indices


def _registered_components(registered_ids: Sequence[str], pair_common: Mapping[tuple[str, str], int]) -> tuple[int, bool]:
    nodes = tuple(str(value) for value in registered_ids)
    if not nodes:
        return 0, False
    adjacency = {node: set() for node in nodes}
    for (a, b), common in pair_common.items():
        if int(common) > 0:
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
    seen: set[str] = set()
    largest = 0
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        size = 0
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            size += 1
            stack.extend(adjacency.get(current, ()))
        largest = max(largest, size)
    return largest, largest == len(nodes)


def analyze_i5_2b_sfm_report(
    report: Mapping[str, Any],
    *,
    scenario_plan: Mapping[str, Any],
) -> dict[str, Any]:
    if report.get("phase") != "8.3D" or report.get("real_sfm_executed") is not True:
        raise ReconstructionValidationError("I5.2B requires a real Phase 8.3D PyCOLMAP report.")
    names, source_indices = _scenario_plan_identity(scenario_plan)
    result = _mapping(report.get("sfm_result"), label="I5.2B sfm_result")
    if result.get("backend_id") != "pycolmap-offline-sfm":
        raise ReconstructionValidationError("I5.2B report is not from the authoritative production-native PyCOLMAP backend.")
    gauge = _mapping(result.get("gauge"), label="I5.2B gauge")
    _require(gauge, "scale_status", "arbitrary", label="I5.2B gauge")
    _require(gauge, "unit_to_metre", None, label="I5.2B gauge")
    metadata = _mapping(result.get("metadata"), label="I5.2B metadata")
    _require(metadata, "real_sfm", True, label="I5.2B metadata")
    _require(metadata, "matching_strategy", "exhaustive", label="I5.2B metadata")
    workspace = _mapping(result.get("workspace"), label="I5.2B workspace")
    workspace_images = workspace.get("images")
    if not isinstance(workspace_images, Sequence) or isinstance(workspace_images, (str, bytes)) or len(workspace_images) != len(names):
        raise ReconstructionValidationError("I5.2B workspace image count does not match the frozen scenario slice.")

    keyframe_to_name: dict[str, str] = {}
    planned_names = set(names)
    for raw in workspace_images:
        item = _mapping(raw, label="I5.2B workspace image")
        keyframe_id = str(item.get("keyframe_id", ""))
        source_locator = str(item.get("source_locator", ""))
        staged = str(item.get("staged_image_path", ""))
        candidate = Path(source_locator).name or Path(staged).name
        if candidate not in planned_names:
            staged_name = Path(staged).name
            if staged_name in planned_names:
                candidate = staged_name
        if candidate not in planned_names or not keyframe_id:
            raise ReconstructionValidationError("I5.2B workspace/source identity drifted from the frozen scenario plan.")
        keyframe_to_name[keyframe_id] = candidate
    if set(keyframe_to_name.values()) != planned_names:
        raise ReconstructionValidationError("I5.2B materialized workspace does not contain the exact frozen scenario names.")

    cameras_raw = result.get("cameras")
    landmarks_raw = result.get("landmarks")
    statistics = _mapping(result.get("statistics"), label="I5.2B statistics")
    if not isinstance(cameras_raw, Sequence) or isinstance(cameras_raw, (str, bytes)):
        raise ReconstructionValidationError("I5.2B cameras must be a sequence.")
    if not isinstance(landmarks_raw, Sequence) or isinstance(landmarks_raw, (str, bytes)):
        raise ReconstructionValidationError("I5.2B landmarks must be a sequence.")
    registered_count = int(statistics.get("registered_keyframes", -1))
    if registered_count != len(cameras_raw) or int(statistics.get("input_keyframes", -1)) != len(names):
        raise ReconstructionValidationError("I5.2B SfM statistics do not match scenario/camera counts.")

    source_index_by_name = dict(zip(names, source_indices))
    camera_centers: list[list[float]] = []
    registered_keyframes: list[str] = []
    registered_names: list[str] = []
    for raw in cameras_raw:
        camera = _mapping(raw, label="I5.2B camera")
        keyframe_id = str(camera.get("keyframe_id", ""))
        if keyframe_id not in keyframe_to_name:
            raise ReconstructionValidationError("I5.2B registered camera references an unknown workspace keyframe.")
        matrix = np.asarray(camera.get("T_reconstruction_from_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ReconstructionValidationError("I5.2B camera pose must be a finite homogeneous 4x4 matrix.")
        registered_keyframes.append(keyframe_id)
        registered_names.append(keyframe_to_name[keyframe_id])
        camera_centers.append(matrix[:3, 3].tolist())
    if len(registered_keyframes) != len(set(registered_keyframes)):
        raise ReconstructionValidationError("I5.2B registered keyframes are not unique.")

    registered_set = set(registered_keyframes)
    visible_landmarks: dict[str, set[str]] = {key: set() for key in registered_keyframes}
    metric_points: list[list[float]] = []
    metric_support_rows: list[list[bool]] = []
    for raw in landmarks_raw:
        landmark = _mapping(raw, label="I5.2B landmark")
        landmark_id = str(landmark.get("landmark_id", ""))
        point = np.asarray(landmark.get("position_reconstruction"), dtype=np.float64)
        observations = landmark.get("observations")
        if not landmark_id or point.shape != (3,) or not np.isfinite(point).all():
            raise ReconstructionValidationError("I5.2B sparse landmark is invalid.")
        if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
            raise ReconstructionValidationError("I5.2B sparse landmark observations are invalid.")
        observed = set()
        for raw_obs in observations:
            obs = _mapping(raw_obs, label="I5.2B observation")
            keyframe_id = str(obs.get("keyframe_id", ""))
            if keyframe_id in registered_set:
                observed.add(keyframe_id)
                visible_landmarks[keyframe_id].add(landmark_id)
        if len(observed) >= 2:
            metric_points.append(point.tolist())
            metric_support_rows.append([key in observed for key in registered_keyframes])

    pair_common: dict[tuple[str, str], int] = {}
    pair_records: list[dict[str, Any]] = []
    for i in range(len(registered_keyframes) - 1):
        a = registered_keyframes[i]
        a_name = keyframe_to_name[a]
        set_a = visible_landmarks[a]
        for j in range(i + 1, len(registered_keyframes)):
            b = registered_keyframes[j]
            b_name = keyframe_to_name[b]
            set_b = visible_landmarks[b]
            common = len(set_a & set_b)
            union = len(set_a | set_b)
            minimum_visible = min(len(set_a), len(set_b))
            normalized = common / minimum_visible if minimum_visible else 0.0
            jaccard = common / union if union else 0.0
            pair_common[(a, b)] = common
            pair_records.append({
                "name_a": a_name,
                "name_b": b_name,
                "source_index_a": int(source_index_by_name[a_name]),
                "source_index_b": int(source_index_by_name[b_name]),
                "source_index_separation": abs(int(source_index_by_name[b_name]) - int(source_index_by_name[a_name])),
                "common_landmarks": common,
                "minimum_visible_landmarks": minimum_visible,
                "normalized_common_over_min_visible": float(normalized),
                "jaccard": float(jaccard),
            })

    registered_name_set = set(registered_names)
    adjacent_records: list[dict[str, Any]] = []
    for a_name, b_name in zip(names[:-1], names[1:]):
        if a_name not in registered_name_set or b_name not in registered_name_set:
            continue
        record = next((item for item in pair_records if {item["name_a"], item["name_b"]} == {a_name, b_name}), None)
        if record is not None:
            adjacent_records.append(record)
    largest_component, connected = _registered_components(registered_keyframes, pair_common)

    parallax_payload: dict[str, Any]
    if len(registered_keyframes) >= 2 and metric_points:
        matrix = np.asarray(metric_support_rows, dtype=bool).T
        centers = np.asarray(camera_centers, dtype=np.float64)
        orders = [int(source_index_by_name[name]) for name in registered_names]
        # The established I2.4 formulation measures maximum supporting-pair
        # parallax and baseline/range per sparse landmark.  Reimplementing the
        # small numeric kernel here keeps I5.2B independent of retained-dense
        # voxel assumptions while preserving the exact scale-invariant meaning.
        xyz = np.asarray(metric_points, dtype=np.float64)
        max_angle = np.zeros(len(xyz), dtype=np.float64)
        max_ratio = np.zeros(len(xyz), dtype=np.float64)
        for i in range(len(centers) - 1):
            for j in range(i + 1, len(centers)):
                mask = matrix[i] & matrix[j]
                ids = np.flatnonzero(mask)
                if len(ids) == 0:
                    continue
                vi = xyz[ids] - centers[i]
                vj = xyz[ids] - centers[j]
                ri = np.linalg.norm(vi, axis=1)
                rj = np.linalg.norm(vj, axis=1)
                valid = (ri > 1e-12) & (rj > 1e-12)
                if not np.any(valid):
                    continue
                ids = ids[valid]
                vi = vi[valid]; vj = vj[valid]; ri = ri[valid]; rj = rj[valid]
                cosine = np.einsum("ij,ij->i", vi, vj) / (ri * rj)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                baseline = float(np.linalg.norm(centers[i] - centers[j]))
                ratio = baseline / np.maximum((ri + rj) * 0.5, 1e-12)
                max_angle[ids] = np.maximum(max_angle[ids], angle)
                max_ratio[ids] = np.maximum(max_ratio[ids], ratio)
        parallax_payload = {
            "landmarks_with_two_or_more_registered_observations": int(len(xyz)),
            "maximum_pair_parallax_degrees_quantiles": _float_quantiles(max_angle),
            "maximum_baseline_to_range_ratio_quantiles": _float_quantiles(max_ratio),
            "scale_invariant": True,
            "absolute_baseline_cross_scenario_comparison_permitted": False,
            "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
        }
    else:
        parallax_payload = {
            "landmarks_with_two_or_more_registered_observations": 0,
            "maximum_pair_parallax_degrees_quantiles": _float_quantiles([]),
            "maximum_baseline_to_range_ratio_quantiles": _float_quantiles([]),
            "scale_invariant": True,
            "absolute_baseline_cross_scenario_comparison_permitted": False,
            "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
        }

    quality = _mapping(report.get("quality_assessment", metadata.get("quality_assessment", {})), label="I5.2B quality_assessment")
    quality_status = str(quality.get("status", ""))
    if quality_status not in {"accepted", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.2B SfM quality status is invalid.")
    adjacent_norm = [float(item["normalized_common_over_min_visible"]) for item in adjacent_records]
    adjacent_common = [float(item["common_landmarks"]) for item in adjacent_records]
    all_norm = [float(item["normalized_common_over_min_visible"]) for item in pair_records]
    all_common = [float(item["common_landmarks"]) for item in pair_records]
    return {
        "scenario_kind": str(scenario_plan.get("kind", "")),
        "input_images": len(names),
        "registered_images": registered_count,
        "registration_ratio": float(registered_count / len(names)),
        "full_input_scope_registered": registered_count == len(names),
        "partial_registration_preserved": 0 < registered_count < len(names),
        "registered_names": registered_names,
        "quality_status": quality_status,
        "quality_assessment": deepcopy(dict(quality)),
        "statistics": deepcopy(dict(statistics)),
        "geometry": {
            "covisibility": {
                "pair_count": len(pair_records),
                "adjacent_selected_pair_count": len(adjacent_records),
                "registered_graph_largest_component": int(largest_component),
                "registered_graph_connected": bool(connected),
                "all_pair_common_landmarks_quantiles": _float_quantiles(all_common),
                "all_pair_normalized_overlap_quantiles": _float_quantiles(all_norm),
                "adjacent_pair_common_landmarks_quantiles": _float_quantiles(adjacent_common),
                "adjacent_pair_normalized_overlap_quantiles": _float_quantiles(adjacent_norm),
                "normalization": "common_landmarks / min(visible_landmarks_a, visible_landmarks_b)",
                "authoritative_source": "PyCOLMAP sparse landmark co-visibility",
            },
            "parallax": parallax_payload,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "per_scenario_gauge": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
    }


def _p50(payload: Mapping[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if current is None:
        return None
    value = float(current)
    return value if isfinite(value) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def compare_i5_2b_stress_geometry(
    *,
    control: Mapping[str, Any],
    low_overlap: Mapping[str, Any],
    weak_parallax: Mapping[str, Any],
) -> dict[str, Any]:
    control_cov = _p50(control, "geometry", "covisibility", "adjacent_pair_normalized_overlap_quantiles", "p50")
    low_cov = _p50(low_overlap, "geometry", "covisibility", "adjacent_pair_normalized_overlap_quantiles", "p50")
    control_common = _p50(control, "geometry", "covisibility", "adjacent_pair_common_landmarks_quantiles", "p50")
    low_common = _p50(low_overlap, "geometry", "covisibility", "adjacent_pair_common_landmarks_quantiles", "p50")
    control_angle = _p50(control, "geometry", "parallax", "maximum_pair_parallax_degrees_quantiles", "p50")
    weak_angle = _p50(weak_parallax, "geometry", "parallax", "maximum_pair_parallax_degrees_quantiles", "p50")
    control_ratio = _p50(control, "geometry", "parallax", "maximum_baseline_to_range_ratio_quantiles", "p50")
    weak_ratio = _p50(weak_parallax, "geometry", "parallax", "maximum_baseline_to_range_ratio_quantiles", "p50")

    low_confirmed = control_common is not None and low_common is not None and low_common < control_common
    weak_angle_confirmed = control_angle is not None and weak_angle is not None and weak_angle < control_angle
    weak_ratio_confirmed = control_ratio is not None and weak_ratio is not None and weak_ratio < control_ratio
    usable = {
        "control": control.get("quality_status") != "rejected",
        "low_overlap": low_overlap.get("quality_status") != "rejected",
        "weak_parallax": weak_parallax.get("quality_status") != "rejected",
    }
    return {
        "comparison_gauge": "per_scenario_arbitrary_sfm_gauge",
        "scale_invariant_metrics_only": True,
        "cross_scenario_absolute_baseline_comparison_permitted": False,
        "low_overlap": {
            "control_adjacent_common_landmarks_p50": control_common,
            "stress_adjacent_common_landmarks_p50": low_common,
            "common_landmark_stress_to_control_ratio": _ratio(low_common, control_common),
            "control_adjacent_normalized_covisibility_p50": control_cov,
            "stress_adjacent_normalized_covisibility_p50": low_cov,
            "normalized_overlap_stress_to_control_ratio": _ratio(low_cov, control_cov),
            "authoritative_covisibility_reduction_confirmed": bool(low_confirmed),
            "semantic_confirmation_metric": "median adjacent PyCOLMAP co-visible sparse landmark count",
            "registered_graph_connected": bool(_mapping(low_overlap.get("geometry"), label="low geometry").get("covisibility", {}).get("registered_graph_connected", False)),
        },
        "weak_parallax": {
            "control_max_pair_parallax_degrees_p50": control_angle,
            "stress_max_pair_parallax_degrees_p50": weak_angle,
            "parallax_stress_to_control_ratio": _ratio(weak_angle, control_angle),
            "authoritative_parallax_reduction_confirmed": bool(weak_angle_confirmed),
            "control_baseline_to_range_ratio_p50": control_ratio,
            "stress_baseline_to_range_ratio_p50": weak_ratio,
            "baseline_ratio_stress_to_control_ratio": _ratio(weak_ratio, control_ratio),
            "authoritative_baseline_to_range_reduction_confirmed": bool(weak_ratio_confirmed),
        },
        "scenario_usable": usable,
        "semantic_stress_revalidation_complete": bool(low_confirmed and weak_angle_confirmed and weak_ratio_confirmed),
        "threshold_used_for_semantic_confirmation": None,
        "directional_comparison_only": True,
    }


def build_i5_2b_low_overlap_weak_parallax_replay(
    *,
    i5_2a_result: Mapping[str, Any],
    i5_2a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    control_analysis: Mapping[str, Any],
    low_overlap_analysis: Mapping[str, Any],
    weak_parallax_analysis: Mapping[str, Any],
    source_reverified_before: bool,
    source_reverified_after: bool,
) -> dict[str, Any]:
    predecessor = validate_i5_2a_for_i5_2b(i5_2a_result)
    receipt = validate_i5_2b_receipt(replay_receipt)
    if source_reverified_before is not True or source_reverified_after is not True:
        raise ReconstructionValidationError("I5.2B source asset identity must be reverified before and after the real replay.")
    comparison = compare_i5_2b_stress_geometry(
        control=control_analysis,
        low_overlap=low_overlap_analysis,
        weak_parallax=weak_parallax_analysis,
    )
    statuses = [str(item.get("quality_status", "")) for item in (control_analysis, low_overlap_analysis, weak_parallax_analysis)]
    if any(status not in {"accepted", "degraded", "rejected"} for status in statuses):
        raise ReconstructionValidationError("I5.2B scenario analysis contains an invalid quality status.")
    if statuses[0] == "rejected":
        quality_status = "rejected"
    elif comparison["semantic_stress_revalidation_complete"] and all(status == "accepted" for status in statuses):
        quality_status = "pass"
    else:
        quality_status = "degraded"

    authority = deepcopy(dict(_mapping(predecessor.get("authority"), label="I5.2A authority")))
    return {
        "schema_version": I52B_SCHEMA_VERSION,
        "phase": I52B_PHASE,
        "status": "low_overlap_weak_parallax_replay_evidence_complete",
        "pass": True,
        "quality_status": quality_status,
        "scope": {
            "current_frozen_foundation_real_replay": True,
            "control_sfm_rerun": True,
            "low_overlap_sfm_rerun": True,
            "weak_parallax_sfm_rerun": True,
            "dense_provider_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_2a_admission": i5_2a_binding.as_dict(),
            "i5_2b_replay_receipt": replay_receipt_binding.as_dict(),
        },
        "provenance": {
            "frozen_source_asset_set_sha256": predecessor["source_identity"]["asset_set_sha256"],
            "source_reverified_before_replay": True,
            "source_reverified_after_replay": True,
            "same_matching_policy_all_scenarios": "exhaustive",
            "same_production_native_sfm_runner_all_scenarios": True,
            "stress_subset_membership_reused_exactly_from_i5_2a": True,
            "result_driven_subset_tuning": False,
            "retry_until_quality_pass": False,
            "historical_result_overwritten": False,
        },
        "scenarios": {
            "control": deepcopy(dict(control_analysis)),
            "low_overlap": deepcopy(dict(low_overlap_analysis)),
            "weak_parallax": deepcopy(dict(weak_parallax_analysis)),
        },
        "comparison": comparison,
        "regression_summary": {
            "focused_i5_2b_tests": receipt["counts"]["focused_tests"],
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
            "i5_2b_complete": True,
            "i5_2c_qualification_closure_authorized": True,
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
            "i5_2c_qualification_closure_authorized_not_started": True,
            "i5_3_blur_exposure_clutter_symmetry_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_evaluation_not_started": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.2C low-overlap / weak-parallax robustness qualification and closure.",
    }


__all__ += [
    "I52B_SCHEMA_VERSION",
    "I52B_PHASE",
    "I52B_RECEIPT_PHASE",
    "validate_i5_2a_for_i5_2b",
    "validate_i5_2b_receipt",
    "analyze_i5_2b_sfm_report",
    "compare_i5_2b_stress_geometry",
    "build_i5_2b_low_overlap_weak_parallax_replay",
]

# ---------------------------------------------------------------------------
# I5.2C — low-overlap / weak-parallax robustness qualification + closure
# ---------------------------------------------------------------------------

I52C_SCHEMA_VERSION = 1
I52C_PHASE = "I5.2C"
I52C_RECEIPT_PHASE = "I5.2C_QUALIFICATION_RECEIPT"


def _i52c_registration_string(scenario: Mapping[str, Any], *, label: str) -> str:
    input_images = scenario.get("input_images")
    registered_images = scenario.get("registered_images")
    if not isinstance(input_images, int) or input_images <= 0:
        raise ReconstructionValidationError(f"I5.2C {label} input_images is invalid.")
    if not isinstance(registered_images, int) or registered_images < 0 or registered_images > input_images:
        raise ReconstructionValidationError(f"I5.2C {label} registered_images is invalid.")
    ratio = scenario.get("registration_ratio")
    expected_ratio = registered_images / input_images
    if not isinstance(ratio, (int, float)) or not isfinite(float(ratio)) or abs(float(ratio) - expected_ratio) > 1e-9:
        raise ReconstructionValidationError(f"I5.2C {label} registration ratio is inconsistent with image counts.")
    full = scenario.get("full_input_scope_registered")
    partial = scenario.get("partial_registration_preserved")
    if full is not (registered_images == input_images):
        raise ReconstructionValidationError(f"I5.2C {label} full-input registration flag is inconsistent.")
    if partial is not (0 < registered_images < input_images):
        raise ReconstructionValidationError(f"I5.2C {label} partial-registration flag is inconsistent.")
    return f"{registered_images}/{input_images}"


def validate_i5_2b_for_i5_2c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete I5.2B replay being closed by I5.2C."""
    if payload.get("phase") != I52B_PHASE or payload.get("status") != "low_overlap_weak_parallax_replay_evidence_complete" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2C requires the passing evidence-complete I5.2B replay record.")
    quality_status = str(payload.get("quality_status", ""))
    if quality_status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.2C I5.2B quality status is invalid.")

    scope = _mapping(payload.get("scope"), label="I5.2B scope")
    for key in ("current_frozen_foundation_real_replay", "control_sfm_rerun", "low_overlap_sfm_rerun", "weak_parallax_sfm_rerun"):
        _require(scope, key, True, label="I5.2B scope")
    for key in (
        "dense_provider_rerun", "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started",
        "thresholds_modified", "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.2B scope")

    provenance = _mapping(payload.get("provenance"), label="I5.2B provenance")
    for key in (
        "source_reverified_before_replay", "source_reverified_after_replay",
        "same_production_native_sfm_runner_all_scenarios", "stress_subset_membership_reused_exactly_from_i5_2a",
    ):
        _require(provenance, key, True, label="I5.2B provenance")
    _require(provenance, "same_matching_policy_all_scenarios", "exhaustive", label="I5.2B provenance")
    for key in ("result_driven_subset_tuning", "retry_until_quality_pass", "historical_result_overwritten"):
        _require(provenance, key, False, label="I5.2B provenance")
    if not _valid_sha256(provenance.get("frozen_source_asset_set_sha256")):
        raise ReconstructionValidationError("I5.2C I5.2B source asset-set SHA256 is malformed.")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.2B evidence_bindings")
    a_binding = _mapping(bindings.get("i5_2a_admission"), label="I5.2B I5.2A binding")
    if not _valid_sha256(a_binding.get("sha256")):
        raise ReconstructionValidationError("I5.2C I5.2B predecessor I5.2A SHA256 binding is malformed.")

    scenarios = _mapping(payload.get("scenarios"), label="I5.2B scenarios")
    scenario_quality: dict[str, str] = {}
    for key in ("control", "low_overlap", "weak_parallax"):
        scenario = _mapping(scenarios.get(key), label=f"I5.2B {key} scenario")
        _i52c_registration_string(scenario, label=key)
        status = str(scenario.get("quality_status", ""))
        if status not in {"accepted", "degraded", "rejected"}:
            raise ReconstructionValidationError(f"I5.2C {key} scenario quality status is invalid.")
        scenario_quality[key] = status
        authority = _mapping(scenario.get("authority"), label=f"I5.2B {key} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.2B {key} authority")
        _require(authority, "per_scenario_gauge", "arbitrary", label=f"I5.2B {key} authority")
        for flag in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
            _require(authority, flag, False, label=f"I5.2B {key} authority")

    comparison = _mapping(payload.get("comparison"), label="I5.2B comparison")
    _require(comparison, "scale_invariant_metrics_only", True, label="I5.2B comparison")
    _require(comparison, "cross_scenario_absolute_baseline_comparison_permitted", False, label="I5.2B comparison")
    _require(comparison, "threshold_used_for_semantic_confirmation", None, label="I5.2B comparison")
    _require(comparison, "directional_comparison_only", True, label="I5.2B comparison")
    low = _mapping(comparison.get("low_overlap"), label="I5.2B low-overlap comparison")
    weak = _mapping(comparison.get("weak_parallax"), label="I5.2B weak-parallax comparison")
    semantic_complete = comparison.get("semantic_stress_revalidation_complete") is True
    if quality_status == "pass":
        if not semantic_complete:
            raise ReconstructionValidationError("I5.2C PASS replay lost authoritative stress revalidation.")
        for key in ("authoritative_covisibility_reduction_confirmed",):
            _require(low, key, True, label="I5.2B low-overlap comparison")
        for key in ("authoritative_parallax_reduction_confirmed", "authoritative_baseline_to_range_reduction_confirmed"):
            _require(weak, key, True, label="I5.2B weak-parallax comparison")
        if any(status != "accepted" for status in scenario_quality.values()):
            raise ReconstructionValidationError("I5.2C PASS replay must retain accepted scenario quality states.")
    elif quality_status in {"degraded", "rejected"}:
        # Legitimate non-PASS evidence remains closable.  I5.2C must not turn a
        # degraded/rejected replay into a reason to retune or rerun until PASS.
        pass

    authority = _mapping(payload.get("authority"), label="I5.2B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.2B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.2B authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="I5.2B authority")

    decision = _mapping(payload.get("decision"), label="I5.2B decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_started", "i5_2a_complete", "i5_2b_complete", "i5_2c_qualification_closure_authorized"):
        _require(decision, key, True, label="I5.2B decision")
    for key in (
        "i5_2_complete", "i5_3_authorized", "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
        "threshold_relaxation_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.2B decision")
    if payload.get("next") != "I5.2C low-overlap / weak-parallax robustness qualification and closure.":
        raise ReconstructionValidationError("I5.2C predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_i5_2c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("phase") != I52C_RECEIPT_PHASE or payload.get("status") != "pass" or payload.get("pass") is not True:
        raise ReconstructionValidationError("I5.2C requires a passing qualification regression receipt.")
    scope = _mapping(payload.get("scope"), label="I5.2C receipt scope")
    for key in (
        "focused_i5_2c_tests", "predecessor_i5_2_regression", "real_replay_contract_regression",
        "stress_geometry_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.2C receipt scope")
    for key in (
        "sfm_rerun", "dense_provider_rerun", "quality_replay_rerun", "thresholds_modified",
        "production_default_modified", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.2C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.2C receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.2C receipt test counts must be non-negative integers.")
    if focused < 10 or predecessor < 40 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.2C receipt coverage is below the required closure floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.2C receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.2C receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_2c_low_overlap_weak_parallax_closure(
    *,
    i5_2a_result: Mapping[str, Any],
    i5_2a_binding: I51EvidenceBinding,
    i5_2b_result: Mapping[str, Any],
    i5_2b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_2a_for_i5_2b(i5_2a_result)
    replay = validate_i5_2b_for_i5_2c(i5_2b_result)
    receipt = validate_i5_2c_receipt(qualification_receipt)

    replay_bindings = _mapping(replay.get("evidence_bindings"), label="I5.2B evidence_bindings")
    replay_a_binding = _mapping(replay_bindings.get("i5_2a_admission"), label="I5.2B I5.2A binding")
    if str(replay_a_binding.get("sha256", "")).lower() != str(i5_2a_binding.sha256).lower():
        raise ReconstructionValidationError("I5.2C I5.2B predecessor SHA does not match the independently verified I5.2A artifact.")
    source = _mapping(admission.get("source_identity"), label="I5.2A source_identity")
    provenance = _mapping(replay.get("provenance"), label="I5.2B provenance")
    if provenance.get("frozen_source_asset_set_sha256") != source.get("asset_set_sha256"):
        raise ReconstructionValidationError("I5.2C I5.2A/I5.2B source asset identity drifted.")
    if replay.get("authority") != admission.get("authority"):
        raise ReconstructionValidationError("I5.2C authority drifted between I5.2A and I5.2B.")

    scenarios = _mapping(replay.get("scenarios"), label="I5.2B scenarios")
    control = _mapping(scenarios.get("control"), label="I5.2B control")
    low = _mapping(scenarios.get("low_overlap"), label="I5.2B low overlap")
    weak = _mapping(scenarios.get("weak_parallax"), label="I5.2B weak parallax")
    control_reg = _i52c_registration_string(control, label="control")
    low_reg = _i52c_registration_string(low, label="low_overlap")
    weak_reg = _i52c_registration_string(weak, label="weak_parallax")
    comparison = _mapping(replay.get("comparison"), label="I5.2B comparison")
    low_evidence = deepcopy(dict(_mapping(comparison.get("low_overlap"), label="I5.2B low-overlap comparison")))
    weak_evidence = deepcopy(dict(_mapping(comparison.get("weak_parallax"), label="I5.2B weak-parallax comparison")))
    quality_status = str(replay.get("quality_status"))
    if quality_status == "pass":
        operating_envelope = "qualified_on_admitted_castle_p19_sparse_sfm_stress_cases"
    elif quality_status == "degraded":
        operating_envelope = "degraded_on_admitted_castle_p19_sparse_sfm_stress_cases"
    else:
        operating_envelope = "rejected_on_admitted_castle_p19_sparse_sfm_stress_cases"

    authority = deepcopy(dict(_mapping(replay.get("authority"), label="I5.2B authority")))
    return {
        "schema_version": I52C_SCHEMA_VERSION,
        "phase": I52C_PHASE,
        "status": "low_overlap_weak_parallax_robustness_qualified",
        "pass": True,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_2a_admission_reused": True,
            "i5_2b_real_replay_reused": True,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "quality_replay_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_2a_admission": i5_2a_binding.as_dict(),
            "i5_2b_replay": i5_2b_binding.as_dict(),
            "i5_2c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "qualification": {
            "quality_status": quality_status,
            "operating_envelope": operating_envelope,
            "source_asset_set_sha256": source["asset_set_sha256"],
            "control_registration": control_reg,
            "low_overlap_registration": low_reg,
            "weak_parallax_registration": weak_reg,
            "partial_registration_preserved": bool(low.get("partial_registration_preserved") or weak.get("partial_registration_preserved") or control.get("partial_registration_preserved")),
            "low_overlap_authoritative_revalidation": bool(low_evidence.get("authoritative_covisibility_reduction_confirmed")),
            "weak_parallax_authoritative_revalidation": bool(
                weak_evidence.get("authoritative_parallax_reduction_confirmed")
                and weak_evidence.get("authoritative_baseline_to_range_reduction_confirmed")
            ),
            "semantic_stress_revalidation_complete": bool(comparison.get("semantic_stress_revalidation_complete")),
            "retry_until_quality_pass": False,
            "result_driven_subset_tuning": False,
            "universal_low_overlap_robustness_claimed": False,
            "universal_weak_parallax_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
        },
        "stress_evidence": {
            "comparison_gauge": comparison.get("comparison_gauge"),
            "scale_invariant_metrics_only": True,
            "cross_scenario_absolute_baseline_comparison_permitted": False,
            "low_overlap": low_evidence,
            "weak_parallax": weak_evidence,
        },
        "regression_summary": {
            "focused_i5_2c_tests": receipt["counts"]["focused_tests"],
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
            "i5_2b_complete": True,
            "i5_2c_complete": True,
            "i5_2_complete": True,
            "i5_2_quality_status": quality_status,
            "i5_3_authorized": True,
            "i5_3_started": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_3_blur_exposure_clutter_symmetry_authorized_not_started": True,
            "i5_4_forward_motion_kitti_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.3 blur / exposure / clutter / symmetry.",
    }


__all__ += [
    "I52C_SCHEMA_VERSION",
    "I52C_PHASE",
    "I52C_RECEIPT_PHASE",
    "validate_i5_2b_for_i5_2c",
    "validate_i5_2c_receipt",
    "build_i5_2c_low_overlap_weak_parallax_closure",
]
