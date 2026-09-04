"""I5.4A forward-motion / KITTI-style scenario admission contracts.

I5.4A is deliberately execution-free. It binds the frozen I5.3C closure,
binds an operator-confirmed ordered KITTI-style forward-motion image sequence,
and freezes a deterministic contiguous replay window plus the geometry evidence
that I5.4B must obtain from the existing PyCOLMAP-authoritative sparse pipeline.

The operator confirmation is provenance only. It is *not* authoritative camera
motion evidence: I5.4B must revalidate forward-axis motion from reconstructed
camera geometry before a forward-motion robustness conclusion can be made.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from math import isfinite
import numpy as np
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding

I54A_SCHEMA_VERSION = 1
I54A_PHASE = "I5.4A"
I54A_RECEIPT_PHASE = "I5.4A_ADMISSION_RECEIPT"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class I54AAdmissionPolicy:
    """Frozen result-independent source/window policy for I5.4B."""

    minimum_source_images: int = 60
    window_start_index: int = 0
    admitted_window_images: int = 60
    source_stride: int = 1
    nominal_image_sequence_fps: float = 10.0
    matching_policy: str = "sequential"
    sequential_overlap: int = 10

    def __post_init__(self) -> None:
        for name in ("minimum_source_images", "window_start_index", "admitted_window_images", "source_stride", "sequential_overlap"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ReconstructionValidationError(f"I5.4A {name} must be an integer.")
        if self.minimum_source_images < 60:
            raise ReconstructionValidationError("I5.4A minimum_source_images must be at least 60.")
        if self.window_start_index < 0:
            raise ReconstructionValidationError("I5.4A window_start_index must be non-negative.")
        if self.admitted_window_images < 60:
            raise ReconstructionValidationError("I5.4A admitted_window_images must be at least 60.")
        if self.source_stride != 1:
            raise ReconstructionValidationError("I5.4A source_stride is frozen to contiguous native-order sampling (1).")
        fps = float(self.nominal_image_sequence_fps)
        if not isfinite(fps) or fps <= 0.0:
            raise ReconstructionValidationError("I5.4A nominal_image_sequence_fps must be finite and positive.")
        object.__setattr__(self, "nominal_image_sequence_fps", fps)
        if self.matching_policy != "sequential":
            raise ReconstructionValidationError("I5.4A matching_policy is frozen to sequential for ordered KITTI-style motion.")
        if self.sequential_overlap < 2:
            raise ReconstructionValidationError("I5.4A sequential_overlap must be at least 2.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_source_images": self.minimum_source_images,
            "window_start_index": self.window_start_index,
            "admitted_window_images": self.admitted_window_images,
            "source_stride": self.source_stride,
            "nominal_image_sequence_fps": self.nominal_image_sequence_fps,
            "matching_policy": self.matching_policy,
            "sequential_overlap": self.sequential_overlap,
        }


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{label} must be an object.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(f"{label}.{key} must be {expected!r}; found {mapping.get(key)!r}.")


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> str:
    aggregate = hashlib.sha256()
    for index, raw in enumerate(records):
        name = str(raw["name"])
        size = int(raw["size_bytes"])
        sha = str(raw["sha256"]).lower()
        aggregate.update(str(index).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(name.encode("utf-8")); aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(sha.encode("ascii")); aggregate.update(b"\n")
    return aggregate.hexdigest()


def validate_i5_3c_for_i5_4(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.3C closure")
    _require(payload, "phase", "I5.3C", label="I5.3C closure")
    _require(payload, "status", "blur_exposure_clutter_symmetry_robustness_qualified", label="I5.3C closure")
    _require(payload, "pass", True, label="I5.3C closure")

    scope = _mapping(payload.get("scope"), label="I5.3C scope")
    _require(scope, "evidence_only_qualification_and_closure", True, label="I5.3C scope")
    for key in (
        "sfm_rerun", "stress_images_rematerialized", "dense_provider_rerun", "quality_replay_rerun",
        "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started", "thresholds_modified",
        "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.3C scope")

    qualification = _mapping(payload.get("qualification"), label="I5.3C qualification")
    quality_status = str(qualification.get("quality_status", ""))
    if quality_status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.4A requires a recognized I5.3C quality_status.")
    _require(qualification, "retry_until_quality_pass", False, label="I5.3C qualification")
    _require(qualification, "result_driven_recipe_tuning", False, label="I5.3C qualification")

    authority = _mapping(payload.get("authority"), label="I5.3C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.3C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.3C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.3C authority")

    decision = _mapping(payload.get("decision"), label="I5.3C decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete", "i5_4_authorized"):
        _require(decision, key, True, label="I5.3C decision")
    _require(decision, "i5_4_started", False, label="I5.3C decision")
    _require(decision, "i5_5_authorized", False, label="I5.3C decision")
    _require(decision, "mp_sfm_authorized", False, label="I5.3C decision")
    _require(decision, "production_default_modified", False, label="I5.3C decision")
    _require(decision, "threshold_relaxation_authorized", False, label="I5.3C decision")
    if payload.get("next") != "I5.4 forward-motion / KITTI robustness.":
        raise ReconstructionValidationError("I5.3C next-step boundary does not authorize I5.4.")
    return deepcopy(dict(payload))


def build_i5_4a_source_identity(
    directory: str | Path,
    *,
    policy: I54AAdmissionPolicy = I54AAdmissionPolicy(),
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.4A source directory does not exist: {root}")
    images = tuple(sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    ))
    if len(images) < policy.minimum_source_images:
        raise ReconstructionValidationError(
            f"I5.4A ordered source requires at least {policy.minimum_source_images} images; found {len(images)}"
        )
    records: list[dict[str, Any]] = []
    for index, path in enumerate(images):
        records.append({
            "order": index,
            "name": path.name,
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return {
        "source_kind": "ordered_image_sequence",
        "directory": str(root),
        "image_count": len(records),
        "ordered_names": [record["name"] for record in records],
        "asset_set_sha256": _aggregate_records(records),
        "files": records,
        "natural_source_order_used": True,
        "recursive_discovery": False,
        "images_transformed": False,
    }


def validate_i5_4a_source_identity(
    identity: Mapping[str, Any],
    *,
    policy: I54AAdmissionPolicy = I54AAdmissionPolicy(),
) -> dict[str, Any]:
    _require(identity, "source_kind", "ordered_image_sequence", label="I5.4A source identity")
    count = identity.get("image_count")
    if not isinstance(count, int) or count < policy.minimum_source_images:
        raise ReconstructionValidationError("I5.4A source identity image_count is below the frozen minimum.")
    names = identity.get("ordered_names")
    files = identity.get("files")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != count:
        raise ReconstructionValidationError("I5.4A ordered_names do not match image_count.")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or len(files) != count:
        raise ReconstructionValidationError("I5.4A file records do not match image_count.")
    if len(set(str(name) for name in names)) != count:
        raise ReconstructionValidationError("I5.4A ordered_names contain duplicates.")
    _require(identity, "natural_source_order_used", True, label="I5.4A source identity")
    _require(identity, "recursive_discovery", False, label="I5.4A source identity")
    _require(identity, "images_transformed", False, label="I5.4A source identity")
    if not _valid_sha256(identity.get("asset_set_sha256")):
        raise ReconstructionValidationError("I5.4A source asset-set SHA256 is invalid.")

    normalized_records: list[dict[str, Any]] = []
    for index, (name, raw) in enumerate(zip(names, files)):
        record = _mapping(raw, label=f"I5.4A source file {index}")
        _require(record, "order", index, label=f"I5.4A source file {index}")
        _require(record, "name", str(name), label=f"I5.4A source file {index}")
        if not isinstance(record.get("size_bytes"), int) or int(record["size_bytes"]) <= 0:
            raise ReconstructionValidationError("I5.4A source file size must be positive.")
        if not _valid_sha256(record.get("sha256")):
            raise ReconstructionValidationError("I5.4A source file SHA256 is invalid.")
        normalized_records.append({
            "order": index,
            "name": str(name),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]).lower(),
        })
    if _aggregate_records(normalized_records) != str(identity["asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.4A source identity aggregate hash does not match its file records.")
    return deepcopy(dict(identity))


def _window_asset_sha(records: Sequence[Mapping[str, Any]]) -> str:
    normalized = [
        {"name": str(record["name"]), "size_bytes": int(record["size_bytes"]), "sha256": str(record["sha256"]).lower()}
        for record in records
    ]
    return _aggregate_records(normalized)


def build_i5_4a_stress_plan(
    source_identity: Mapping[str, Any],
    *,
    policy: I54AAdmissionPolicy = I54AAdmissionPolicy(),
) -> dict[str, Any]:
    identity = validate_i5_4a_source_identity(source_identity, policy=policy)
    start = policy.window_start_index
    stop = start + policy.admitted_window_images * policy.source_stride
    indices = list(range(start, stop, policy.source_stride))
    if not indices or indices[-1] >= int(identity["image_count"]):
        raise ReconstructionValidationError("I5.4A frozen replay window exceeds the bound source sequence.")
    names = [str(identity["ordered_names"][index]) for index in indices]
    files = [_mapping(identity["files"][index], label=f"I5.4A selected source file {index}") for index in indices]
    return {
        "forward_motion_window": {
            "kind": "operator_confirmed_kitti_style_forward_motion_contiguous_window_proxy",
            "indices": indices,
            "names": names,
            "image_count": len(indices),
            "first_source_index": indices[0],
            "last_source_index": indices[-1],
            "source_stride": policy.source_stride,
            "contiguous": policy.source_stride == 1,
            "source_order_preserved": True,
            "source_asset_set_sha256": identity["asset_set_sha256"],
            "window_asset_set_sha256": _window_asset_sha(files),
            "nominal_image_sequence_fps": policy.nominal_image_sequence_fps,
            "matching_policy": policy.matching_policy,
            "sequential_overlap": policy.sequential_overlap,
            "forward_motion_semantics_authoritative": False,
            "operator_semantic_confirmation_only": True,
            "i5_4b_geometry_revalidation_required": True,
            "required_i5_4b_revalidation": {
                "camera_motion": "pycolmap_consecutive_camera_center_motion_in_camera_coordinates",
                "forward_axis_dominance": "median_forward_component_fraction_must_exceed_median_lateral_component_fraction",
                "triangulation_health": "report_i2_4_parallax_and_baseline_to_range_without_absolute_scale_claim",
                "registration_health": "report_registered_input_ratio_landmarks_observations_reprojection_and_track_length",
                "motion_label_if_geometry_not_confirmed": "inconclusive_forward_motion_proxy_not_revalidated",
                "result_driven_window_tuning": False,
            },
        },
        "policy": policy.as_dict(),
        "result_driven_window_tuning": False,
        "retry_until_quality_pass": False,
        "production_default_modified": False,
    }


def validate_i5_4a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.4A receipt")
    _require(payload, "phase", I54A_RECEIPT_PHASE, label="I5.4A receipt")
    _require(payload, "status", "pass", label="I5.4A receipt")
    _require(payload, "pass", True, label="I5.4A receipt")
    scope = _mapping(payload.get("scope"), label="I5.4A receipt scope")
    for key in (
        "focused_i5_4a_tests", "predecessor_i5_3c_regression", "ordered_source_identity_regression",
        "forward_motion_admission_design_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.4A receipt scope")
    for key in (
        "sfm_rerun", "images_transformed", "dense_provider_rerun", "thresholds_modified",
        "production_default_modified", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.4A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.4A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.4A receipt test counts must be non-negative integers.")
    if focused < 10 or predecessor < 40 or combined < 50 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.4A receipt coverage is below the required admission floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.4A receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.4A receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_4a_forward_motion_kitti_admission(
    *,
    i5_3c_result: Mapping[str, Any],
    i5_3c_binding: I51EvidenceBinding,
    source_identity: Mapping[str, Any],
    source_label: str,
    kitti_style_forward_motion_confirmation: bool,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
    policy: I54AAdmissionPolicy = I54AAdmissionPolicy(),
) -> dict[str, Any]:
    closure = validate_i5_3c_for_i5_4(i5_3c_result)
    identity = validate_i5_4a_source_identity(source_identity, policy=policy)
    receipt = validate_i5_4a_receipt(admission_receipt)
    label = str(source_label or "").strip()
    if not label:
        raise ReconstructionValidationError("I5.4A source_label must be non-empty.")
    if kitti_style_forward_motion_confirmation is not True:
        raise ReconstructionValidationError(
            "I5.4A requires explicit operator confirmation that the bound ordered source is a KITTI-style forward-motion sequence."
        )
    plan = build_i5_4a_stress_plan(identity, policy=policy)
    predecessor_quality = str(_mapping(closure.get("qualification"), label="I5.3C qualification").get("quality_status"))
    authority = deepcopy(dict(_mapping(closure.get("authority"), label="I5.3C authority")))
    return {
        "schema_version": I54A_SCHEMA_VERSION,
        "phase": I54A_PHASE,
        "status": "forward_motion_kitti_scenario_admitted",
        "pass": True,
        "scope": {
            "scenario_admission_only": True,
            "i5_3c_closure_reused": True,
            "ordered_source_bound": True,
            "source_window_frozen": True,
            "images_transformed": False,
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
            "i5_3c_closure": i5_3c_binding.as_dict(),
            "i5_4a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "predecessor": {
            "i5_3_quality_status": predecessor_quality,
            "i5_3_closed": True,
            "i5_4_authorized": True,
        },
        "source_identity": identity,
        "source_semantics": {
            "label": label,
            "operator_kitti_style_forward_motion_confirmation": True,
            "operator_confirmation_is_camera_geometry_ground_truth": False,
            "authoritative_forward_motion_geometry_proven": False,
            "i5_4b_geometry_revalidation_required": True,
        },
        "stress_plan": plan,
        "qualification": {
            "operator_kitti_style_forward_motion_confirmation": True,
            "authoritative_forward_motion_geometry_proven": False,
            "forward_motion_proxy_only_until_i5_4b": True,
            "source_asset_set_sha256": identity["asset_set_sha256"],
            "window_asset_set_sha256": plan["forward_motion_window"]["window_asset_set_sha256"],
            "admitted_window_images": plan["forward_motion_window"]["image_count"],
            "retry_until_quality_pass": False,
            "result_driven_window_tuning": False,
            "universal_forward_motion_robustness_claimed": False,
            "kitti_dataset_universal_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
        },
        "regression_summary": {
            "focused_i5_4a_tests": receipt["counts"]["focused_tests"],
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
            "i5_2_complete": True,
            "i5_3_complete": True,
            "i5_4_started": True,
            "i5_4a_complete": True,
            "i5_4b_real_replay_authorized": True,
            "i5_4b_started": False,
            "i5_4c_complete": False,
            "i5_4_complete": False,
            "i5_5_authorized": False,
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
            "i5_4b_real_replay_authorized_not_started": True,
            "i5_5_dynamic_scene_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.4B current frozen-foundation forward-motion / KITTI replay.",
    }


I54B_SCHEMA_VERSION = 1
I54B_PHASE = "I5.4B"
I54B_RECEIPT_PHASE = "I5.4B_REPLAY_RECEIPT"


def _float_quantiles_i54b(values: object) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    q = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]), "p75": float(q[3]), "p90": float(q[4])}


def validate_i5_4a_for_i5_4b(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.4A admission")
    _require(payload, "phase", "I5.4A", label="I5.4A admission")
    _require(payload, "status", "forward_motion_kitti_scenario_admitted", label="I5.4A admission")
    _require(payload, "pass", True, label="I5.4A admission")

    scope = _mapping(payload.get("scope"), label="I5.4A scope")
    for key in ("scenario_admission_only", "i5_3c_closure_reused", "ordered_source_bound", "source_window_frozen"):
        _require(scope, key, True, label="I5.4A scope")
    for key in (
        "images_transformed", "sfm_rerun", "dense_provider_rerun", "quality_replay_rerun",
        "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started", "thresholds_modified",
        "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.4A scope")

    identity = validate_i5_4a_source_identity(_mapping(payload.get("source_identity"), label="I5.4A source identity"))
    semantics = _mapping(payload.get("source_semantics"), label="I5.4A source semantics")
    _require(semantics, "operator_kitti_style_forward_motion_confirmation", True, label="I5.4A source semantics")
    _require(semantics, "operator_confirmation_is_camera_geometry_ground_truth", False, label="I5.4A source semantics")
    _require(semantics, "authoritative_forward_motion_geometry_proven", False, label="I5.4A source semantics")
    _require(semantics, "i5_4b_geometry_revalidation_required", True, label="I5.4A source semantics")

    plan = _mapping(payload.get("stress_plan"), label="I5.4A stress plan")
    _require(plan, "result_driven_window_tuning", False, label="I5.4A stress plan")
    _require(plan, "retry_until_quality_pass", False, label="I5.4A stress plan")
    _require(plan, "production_default_modified", False, label="I5.4A stress plan")
    window = _mapping(plan.get("forward_motion_window"), label="I5.4A forward-motion window")
    _require(window, "image_count", 60, label="I5.4A forward-motion window")
    _require(window, "first_source_index", 0, label="I5.4A forward-motion window")
    _require(window, "last_source_index", 59, label="I5.4A forward-motion window")
    _require(window, "source_stride", 1, label="I5.4A forward-motion window")
    _require(window, "contiguous", True, label="I5.4A forward-motion window")
    _require(window, "source_order_preserved", True, label="I5.4A forward-motion window")
    _require(window, "matching_policy", "sequential", label="I5.4A forward-motion window")
    _require(window, "sequential_overlap", 10, label="I5.4A forward-motion window")
    _require(window, "forward_motion_semantics_authoritative", False, label="I5.4A forward-motion window")
    _require(window, "operator_semantic_confirmation_only", True, label="I5.4A forward-motion window")
    _require(window, "i5_4b_geometry_revalidation_required", True, label="I5.4A forward-motion window")
    names = window.get("names")
    indices = window.get("indices")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != 60:
        raise ReconstructionValidationError("I5.4B requires exactly 60 frozen window names.")
    if list(indices or []) != list(range(60)):
        raise ReconstructionValidationError("I5.4B requires the frozen contiguous source indices 0..59.")
    if [str(x) for x in names] != [str(x) for x in identity["ordered_names"][:60]]:
        raise ReconstructionValidationError("I5.4B frozen window names drifted from the admitted source identity.")
    files = [_mapping(identity["files"][i], label=f"I5.4A source file {i}") for i in range(60)]
    if _window_asset_sha(files) != str(window.get("window_asset_set_sha256", "")).lower():
        raise ReconstructionValidationError("I5.4B frozen window SHA256 drifted from the admitted source files.")
    if str(window.get("source_asset_set_sha256", "")).lower() != str(identity["asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.4B source asset-set binding drifted inside the frozen window.")

    qualification = _mapping(payload.get("qualification"), label="I5.4A qualification")
    _require(qualification, "authoritative_forward_motion_geometry_proven", False, label="I5.4A qualification")
    _require(qualification, "forward_motion_proxy_only_until_i5_4b", True, label="I5.4A qualification")
    _require(qualification, "retry_until_quality_pass", False, label="I5.4A qualification")
    _require(qualification, "result_driven_window_tuning", False, label="I5.4A qualification")
    if str(qualification.get("source_asset_set_sha256", "")).lower() != str(identity["asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.4B qualification source SHA does not match source identity.")
    if str(qualification.get("window_asset_set_sha256", "")).lower() != str(window["window_asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.4B qualification window SHA does not match frozen window.")

    authority = _mapping(payload.get("authority"), label="I5.4A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.4A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.4A authority")
    for key in ("provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
        _require(authority, key, False, label="I5.4A authority")

    decision = _mapping(payload.get("decision"), label="I5.4A decision")
    for key in ("i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete", "i5_4_started", "i5_4a_complete", "i5_4b_real_replay_authorized"):
        _require(decision, key, True, label="I5.4A decision")
    for key in ("i5_4b_started", "i5_4c_complete", "i5_4_complete", "i5_5_authorized", "mp_sfm_authorized", "production_default_modified", "threshold_relaxation_authorized"):
        _require(decision, key, False, label="I5.4A decision")
    if payload.get("next") != "I5.4B current frozen-foundation forward-motion / KITTI replay.":
        raise ReconstructionValidationError("I5.4A next-step boundary does not authorize I5.4B.")
    return deepcopy(dict(payload))


def validate_i5_4b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.4B receipt")
    _require(payload, "phase", I54B_RECEIPT_PHASE, label="I5.4B receipt")
    _require(payload, "status", "pass", label="I5.4B receipt")
    _require(payload, "pass", True, label="I5.4B receipt")
    scope = _mapping(payload.get("scope"), label="I5.4B receipt scope")
    for key in ("focused_i5_4b_tests", "predecessor_i5_4a_regression", "production_native_sfm_contract_regression", "forward_motion_geometry_regression", "i2_4_scale_invariant_geometry_regression", "import_probe", "runner_help_probe"):
        _require(scope, key, True, label="I5.4B receipt scope")
    for key in ("real_sfm_executed_by_test_receipt", "dense_provider_rerun", "thresholds_modified", "production_default_modified", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized"):
        _require(scope, key, False, label="I5.4B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.4B receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.4B receipt counts must be non-negative integers.")
    if focused < 15 or predecessor < 50 or combined < 65 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.4B receipt coverage is below the required replay floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.4B receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.4B receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _i54b_plan_identity(scenario_plan: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    names = scenario_plan.get("names")
    indices = scenario_plan.get("indices")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ReconstructionValidationError("I5.4B scenario plan names/indices are invalid.")
    names_out = [str(x) for x in names]
    indices_out = [int(x) for x in indices]
    if len(names_out) != 60 or indices_out != list(range(60)):
        raise ReconstructionValidationError("I5.4B scenario plan must preserve the frozen 60-frame window.")
    return names_out, indices_out


def _i54b_registered_components(keys: Sequence[str], pair_common: Mapping[tuple[str, str], int]) -> tuple[int, bool]:
    if not keys:
        return 0, False
    adjacency = {key: set() for key in keys}
    for (a, b), common in pair_common.items():
        if common > 0:
            adjacency[a].add(b); adjacency[b].add(a)
    seen: set[str] = set(); largest = 0
    for key in keys:
        if key in seen:
            continue
        stack = [key]; size = 0; seen.add(key)
        while stack:
            cur = stack.pop(); size += 1
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
        largest = max(largest, size)
    return largest, largest == len(keys)


def analyze_i5_4b_sfm_report(report: Mapping[str, Any], *, scenario_plan: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("phase") != "8.3D" or report.get("real_sfm_executed") is not True:
        raise ReconstructionValidationError("I5.4B requires a real Phase 8.3D PyCOLMAP report.")
    names, source_indices = _i54b_plan_identity(scenario_plan)
    _require(scenario_plan, "matching_policy", "sequential", label="I5.4B scenario plan")
    _require(scenario_plan, "sequential_overlap", 10, label="I5.4B scenario plan")
    result = _mapping(report.get("sfm_result"), label="I5.4B sfm_result")
    _require(result, "backend_id", "pycolmap-offline-sfm", label="I5.4B sfm_result")
    gauge = _mapping(result.get("gauge"), label="I5.4B gauge")
    _require(gauge, "scale_status", "arbitrary", label="I5.4B gauge")
    _require(gauge, "unit_to_metre", None, label="I5.4B gauge")
    metadata = _mapping(result.get("metadata"), label="I5.4B metadata")
    _require(metadata, "real_sfm", True, label="I5.4B metadata")
    _require(metadata, "matching_strategy", "sequential", label="I5.4B metadata")
    resolved = metadata.get("resolved_configuration")
    if isinstance(resolved, Mapping) and "sequential_overlap" in resolved:
        _require(resolved, "sequential_overlap", 10, label="I5.4B resolved configuration")

    workspace = _mapping(result.get("workspace"), label="I5.4B workspace")
    workspace_images = workspace.get("images")
    if not isinstance(workspace_images, Sequence) or isinstance(workspace_images, (str, bytes)) or len(workspace_images) != len(names):
        raise ReconstructionValidationError("I5.4B workspace image count does not match the frozen 60-frame window.")
    planned_names = set(names); keyframe_to_name: dict[str, str] = {}
    for raw in workspace_images:
        item = _mapping(raw, label="I5.4B workspace image")
        keyframe_id = str(item.get("keyframe_id", ""))
        candidate = Path(str(item.get("source_locator", ""))).name or Path(str(item.get("staged_image_path", ""))).name
        if candidate not in planned_names:
            staged_name = Path(str(item.get("staged_image_path", ""))).name
            if staged_name in planned_names:
                candidate = staged_name
        if not keyframe_id or candidate not in planned_names:
            raise ReconstructionValidationError("I5.4B workspace/source identity drifted from the frozen window.")
        keyframe_to_name[keyframe_id] = candidate
    if set(keyframe_to_name.values()) != planned_names:
        raise ReconstructionValidationError("I5.4B workspace does not contain the exact frozen window names.")

    cameras_raw = result.get("cameras"); landmarks_raw = result.get("landmarks")
    if not isinstance(cameras_raw, Sequence) or isinstance(cameras_raw, (str, bytes)):
        raise ReconstructionValidationError("I5.4B cameras must be a sequence.")
    if not isinstance(landmarks_raw, Sequence) or isinstance(landmarks_raw, (str, bytes)):
        raise ReconstructionValidationError("I5.4B landmarks must be a sequence.")
    statistics = _mapping(result.get("statistics"), label="I5.4B statistics")
    registered_count = int(statistics.get("registered_keyframes", -1))
    if registered_count != len(cameras_raw) or int(statistics.get("input_keyframes", -1)) != len(names):
        raise ReconstructionValidationError("I5.4B SfM statistics do not match the frozen window/camera counts.")

    source_index_by_name = dict(zip(names, source_indices))
    camera_by_name: dict[str, dict[str, Any]] = {}
    registered_keyframes: list[str] = []
    for raw in cameras_raw:
        camera = _mapping(raw, label="I5.4B camera")
        keyframe_id = str(camera.get("keyframe_id", ""))
        if keyframe_id not in keyframe_to_name:
            raise ReconstructionValidationError("I5.4B registered camera references an unknown workspace keyframe.")
        matrix = np.asarray(camera.get("T_reconstruction_from_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
            raise ReconstructionValidationError("I5.4B camera pose must be a finite homogeneous 4x4 matrix.")
        name = keyframe_to_name[keyframe_id]
        if name in camera_by_name:
            raise ReconstructionValidationError("I5.4B duplicate registered camera name.")
        camera_by_name[name] = {"keyframe_id": keyframe_id, "matrix": matrix, "source_index": source_index_by_name[name]}
        registered_keyframes.append(keyframe_id)

    motion_records: list[dict[str, Any]] = []
    forward_fractions: list[float] = []; lateral_fractions: list[float] = []; signed_forward: list[float] = []
    for a_name, b_name in zip(names[:-1], names[1:]):
        if a_name not in camera_by_name or b_name not in camera_by_name:
            continue
        a = camera_by_name[a_name]; b = camera_by_name[b_name]
        Ta = a["matrix"]; Tb = b["matrix"]
        delta_recon = Tb[:3, 3] - Ta[:3, 3]
        norm = float(np.linalg.norm(delta_recon))
        if not isfinite(norm) or norm <= 1e-12:
            continue
        delta_camera_a = Ta[:3, :3].T @ delta_recon
        forward = abs(float(delta_camera_a[2])) / norm
        lateral = float(np.linalg.norm(delta_camera_a[:2])) / norm
        signed = float(delta_camera_a[2]) / norm
        forward_fractions.append(forward); lateral_fractions.append(lateral); signed_forward.append(signed)
        motion_records.append({
            "name_a": a_name, "name_b": b_name,
            "source_index_a": int(a["source_index"]), "source_index_b": int(b["source_index"]),
            "delta_reconstruction": delta_recon.tolist(), "delta_camera_a": delta_camera_a.tolist(),
            "motion_norm_arbitrary_gauge": norm, "forward_component_fraction": forward,
            "lateral_component_fraction": lateral, "signed_forward_component_fraction": signed,
        })
    median_forward = float(np.median(forward_fractions)) if forward_fractions else None
    median_lateral = float(np.median(lateral_fractions)) if lateral_fractions else None
    geometry_proven = bool(
        len(motion_records) >= 2 and median_forward is not None and median_lateral is not None and median_forward > median_lateral
    )
    motion_payload = {
        "coordinate_convention": "delta_camera_a = R_reconstruction_from_camera_a^T * (C_b - C_a)",
        "camera_forward_axis": "+Z camera axis; dominance uses absolute forward component to avoid direction-sign overclaim",
        "adjacent_registered_motion_pairs": len(motion_records),
        "records": motion_records,
        "forward_component_fraction_quantiles": _float_quantiles_i54b(forward_fractions),
        "lateral_component_fraction_quantiles": _float_quantiles_i54b(lateral_fractions),
        "signed_forward_component_fraction_quantiles": _float_quantiles_i54b(signed_forward),
        "median_forward_component_fraction": median_forward,
        "median_lateral_component_fraction": median_lateral,
        "authoritative_forward_motion_geometry_proven": geometry_proven,
        "forward_motion_label": "pycolmap_forward_axis_dominant_motion_revalidated" if geometry_proven else "inconclusive_forward_motion_proxy_not_revalidated",
        "absolute_translation_magnitude_cross_run_comparison_permitted": False,
        "arbitrary_gauge": True,
    }

    registered_set = set(registered_keyframes)
    visible_landmarks: dict[str, set[str]] = {key: set() for key in registered_keyframes}
    points: list[list[float]] = []; support_rows: list[list[bool]] = []
    for raw in landmarks_raw:
        lm = _mapping(raw, label="I5.4B landmark"); lm_id = str(lm.get("landmark_id", ""))
        point = np.asarray(lm.get("position_reconstruction"), dtype=np.float64); obs = lm.get("observations")
        if not lm_id or point.shape != (3,) or not np.isfinite(point).all() or not isinstance(obs, Sequence) or isinstance(obs, (str, bytes)):
            raise ReconstructionValidationError("I5.4B sparse landmark is invalid.")
        observed: set[str] = set()
        for raw_obs in obs:
            o = _mapping(raw_obs, label="I5.4B observation"); key = str(o.get("keyframe_id", ""))
            if key in registered_set:
                observed.add(key); visible_landmarks[key].add(lm_id)
        if len(observed) >= 2:
            points.append(point.tolist()); support_rows.append([key in observed for key in registered_keyframes])

    pair_common: dict[tuple[str, str], int] = {}; adjacent_common: list[float] = []
    key_to_name = {v["keyframe_id"]: k for k, v in camera_by_name.items()}
    for i, a in enumerate(registered_keyframes[:-1]):
        for b in registered_keyframes[i + 1:]:
            pair_common[(a, b)] = len(visible_landmarks[a] & visible_landmarks[b])
    registered_name_set = set(camera_by_name)
    for a_name, b_name in zip(names[:-1], names[1:]):
        if a_name in registered_name_set and b_name in registered_name_set:
            a_key = camera_by_name[a_name]["keyframe_id"]; b_key = camera_by_name[b_name]["keyframe_id"]
            adjacent_common.append(float(pair_common.get((a_key, b_key), pair_common.get((b_key, a_key), 0))))
    largest, connected = _i54b_registered_components(registered_keyframes, pair_common)

    if len(registered_keyframes) >= 2 and points:
        centers = np.asarray([camera_by_name[key_to_name[key]]["matrix"][:3, 3] for key in registered_keyframes], dtype=np.float64)
        matrix = np.asarray(support_rows, dtype=bool).T; xyz = np.asarray(points, dtype=np.float64)
        max_angle = np.zeros(len(xyz), dtype=np.float64); max_ratio = np.zeros(len(xyz), dtype=np.float64)
        for i in range(len(centers) - 1):
            for j in range(i + 1, len(centers)):
                mask = matrix[i] & matrix[j]; ids = np.flatnonzero(mask)
                if len(ids) == 0: continue
                vi = xyz[ids] - centers[i]; vj = xyz[ids] - centers[j]
                ri = np.linalg.norm(vi, axis=1); rj = np.linalg.norm(vj, axis=1)
                valid = (ri > 1e-12) & (rj > 1e-12)
                if not np.any(valid): continue
                ids = ids[valid]; vi = vi[valid]; vj = vj[valid]; ri = ri[valid]; rj = rj[valid]
                cosine = np.einsum("ij,ij->i", vi, vj) / (ri * rj)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                baseline = float(np.linalg.norm(centers[i] - centers[j]))
                ratio = baseline / np.maximum((ri + rj) * 0.5, 1e-12)
                max_angle[ids] = np.maximum(max_angle[ids], angle); max_ratio[ids] = np.maximum(max_ratio[ids], ratio)
        parallax = {
            "landmarks_with_two_or_more_registered_observations": int(len(xyz)),
            "maximum_pair_parallax_degrees_quantiles": _float_quantiles_i54b(max_angle),
            "maximum_baseline_to_range_ratio_quantiles": _float_quantiles_i54b(max_ratio),
            "scale_invariant": True, "absolute_baseline_cross_scenario_comparison_permitted": False,
            "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
        }
    else:
        parallax = {
            "landmarks_with_two_or_more_registered_observations": 0,
            "maximum_pair_parallax_degrees_quantiles": _float_quantiles_i54b([]),
            "maximum_baseline_to_range_ratio_quantiles": _float_quantiles_i54b([]),
            "scale_invariant": True, "absolute_baseline_cross_scenario_comparison_permitted": False,
            "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
        }

    quality = _mapping(report.get("quality_assessment", metadata.get("quality_assessment", {})), label="I5.4B quality_assessment")
    quality_status = str(quality.get("status", ""))
    if quality_status not in {"accepted", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.4B SfM quality status is invalid.")
    return {
        "input_images": len(names), "registered_images": registered_count,
        "registration_ratio": float(registered_count / len(names)),
        "full_input_scope_registered": registered_count == len(names),
        "partial_registration_preserved": 0 < registered_count < len(names),
        "registered_names": [name for name in names if name in camera_by_name],
        "quality_status": quality_status, "quality_assessment": deepcopy(dict(quality)), "statistics": deepcopy(dict(statistics)),
        "geometry": {
            "camera_motion": motion_payload,
            "covisibility": {
                "registered_graph_largest_component": int(largest), "registered_graph_connected": bool(connected),
                "adjacent_pair_common_landmarks_quantiles": _float_quantiles_i54b(adjacent_common),
                "adjacent_selected_pair_count": len(adjacent_common),
            },
            "parallax": parallax,
        },
        "pre_sfm_ingestion_rejected": False, "registration_attempted": True, "sfm_executed": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "authoritative_sfm_modified": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False,
            "surface_mesh_generated": False, "permanent_global_view_cap": None,
        },
    }


def analyze_i5_4b_pre_sfm_ingestion_rejection(*, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], reason: str) -> dict[str, Any]:
    names, _ = _i54b_plan_identity(scenario_plan)
    if [str(x) for x in staged_names] != names:
        raise ReconstructionValidationError("I5.4B pre-SfM rejection staged names do not match the frozen window.")
    text = str(reason)
    if "Fewer than three valid keyframes survived ingestion" not in text:
        raise ReconstructionValidationError("I5.4B only structures the recognized production ingestion rejection.")
    return {
        "input_images": len(names), "registered_images": 0, "registration_ratio": 0.0,
        "full_input_scope_registered": False, "partial_registration_preserved": False, "registered_names": [],
        "quality_status": "rejected",
        "quality_assessment": {"status": "rejected", "usable": False, "reasons": [text]},
        "statistics": {"input_keyframes": len(names), "registered_keyframes": 0, "landmarks": 0, "observations": 0, "mean_reprojection_error_px": None},
        "geometry": {
            "camera_motion": {
                "adjacent_registered_motion_pairs": 0, "records": [],
                "forward_component_fraction_quantiles": _float_quantiles_i54b([]),
                "lateral_component_fraction_quantiles": _float_quantiles_i54b([]),
                "signed_forward_component_fraction_quantiles": _float_quantiles_i54b([]),
                "median_forward_component_fraction": None, "median_lateral_component_fraction": None,
                "authoritative_forward_motion_geometry_proven": False,
                "forward_motion_label": "inconclusive_forward_motion_proxy_not_revalidated",
                "absolute_translation_magnitude_cross_run_comparison_permitted": False, "arbitrary_gauge": True,
            },
            "covisibility": {"registered_graph_largest_component": 0, "registered_graph_connected": False, "adjacent_pair_common_landmarks_quantiles": _float_quantiles_i54b([]), "adjacent_selected_pair_count": 0},
            "parallax": {
                "landmarks_with_two_or_more_registered_observations": 0,
                "maximum_pair_parallax_degrees_quantiles": _float_quantiles_i54b([]),
                "maximum_baseline_to_range_ratio_quantiles": _float_quantiles_i54b([]),
                "scale_invariant": True, "absolute_baseline_cross_scenario_comparison_permitted": False,
                "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
            },
        },
        "pre_sfm_ingestion_rejected": True, "rejection_stage": "ingestion", "rejection_reason": text,
        "registration_attempted": False, "sfm_executed": False,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "authoritative_sfm_modified": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False,
            "surface_mesh_generated": False, "permanent_global_view_cap": None,
        },
    }


def analyze_i5_4b_mapping_no_sparse_model_rejection(*, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], reason: str) -> dict[str, Any]:
    """Structure the known PyCOLMAP incremental-mapping no-model outcome as rejected evidence.

    This is distinct from pre-SfM ingestion rejection: feature extraction and matching
    completed, sparse reconstruction was attempted, but incremental mapping emitted no
    authoritative sparse model. The admitted window is not changed or retried.
    """
    names, _ = _i54b_plan_identity(scenario_plan)
    if [str(x) for x in staged_names] != names:
        raise ReconstructionValidationError("I5.4B mapping rejection staged names do not match the frozen window.")
    text = str(reason)
    if "PyCOLMAP incremental mapping produced no sparse model" not in text:
        raise ReconstructionValidationError("I5.4B only structures the recognized incremental-mapping no-sparse-model rejection.")
    return {
        "input_images": len(names), "registered_images": 0, "registration_ratio": 0.0,
        "full_input_scope_registered": False, "partial_registration_preserved": False, "registered_names": [],
        "quality_status": "rejected",
        "quality_assessment": {"status": "rejected", "usable": False, "reasons": [text]},
        "statistics": {"input_keyframes": len(names), "registered_keyframes": 0, "landmarks": 0, "observations": 0, "mean_reprojection_error_px": None},
        "geometry": {
            "camera_motion": {
                "adjacent_registered_motion_pairs": 0, "records": [],
                "forward_component_fraction_quantiles": _float_quantiles_i54b([]),
                "lateral_component_fraction_quantiles": _float_quantiles_i54b([]),
                "signed_forward_component_fraction_quantiles": _float_quantiles_i54b([]),
                "median_forward_component_fraction": None, "median_lateral_component_fraction": None,
                "authoritative_forward_motion_geometry_proven": False,
                "forward_motion_label": "inconclusive_forward_motion_proxy_not_revalidated",
                "absolute_translation_magnitude_cross_run_comparison_permitted": False, "arbitrary_gauge": True,
            },
            "covisibility": {"registered_graph_largest_component": 0, "registered_graph_connected": False, "adjacent_pair_common_landmarks_quantiles": _float_quantiles_i54b([]), "adjacent_selected_pair_count": 0},
            "parallax": {
                "landmarks_with_two_or_more_registered_observations": 0,
                "maximum_pair_parallax_degrees_quantiles": _float_quantiles_i54b([]),
                "maximum_baseline_to_range_ratio_quantiles": _float_quantiles_i54b([]),
                "scale_invariant": True, "absolute_baseline_cross_scenario_comparison_permitted": False,
                "methodology_lineage": "I2.4 maximum-supporting-pair parallax and baseline/range formulation",
            },
        },
        "pre_sfm_ingestion_rejected": False,
        "rejection_stage": "incremental_mapping",
        "rejection_reason": text,
        "mapping_produced_sparse_model": False,
        "registration_attempted": True,
        "sfm_executed": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm", "authoritative_sfm_modified": False,
            "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False,
            "surface_mesh_generated": False, "permanent_global_view_cap": None,
        },
    }


def build_i5_4b_forward_motion_kitti_replay(
    *, i5_4a_result: Mapping[str, Any], i5_4a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any], replay_receipt_binding: I51EvidenceBinding,
    sfm_analysis: Mapping[str, Any], source_reverified_before: bool, source_window_reverified_after: bool,
    staged_inputs_reverified: bool,
) -> dict[str, Any]:
    admission = validate_i5_4a_for_i5_4b(i5_4a_result); receipt = validate_i5_4b_receipt(replay_receipt)
    if not (source_reverified_before and source_window_reverified_after and staged_inputs_reverified):
        raise ReconstructionValidationError("I5.4B requires source-before, selected-window-after, and staged-input reverification.")
    analysis = deepcopy(dict(_mapping(sfm_analysis, label="I5.4B SfM analysis")))
    if int(analysis.get("input_images", -1)) != 60:
        raise ReconstructionValidationError("I5.4B SfM analysis must cover the frozen 60-frame input window.")
    motion = _mapping(_mapping(analysis.get("geometry"), label="I5.4B geometry").get("camera_motion"), label="I5.4B camera motion")
    geometry_proven = bool(motion.get("authoritative_forward_motion_geometry_proven"))
    sfm_quality = str(analysis.get("quality_status", ""))
    if sfm_quality not in {"accepted", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.4B analysis quality_status is invalid.")
    full = bool(analysis.get("full_input_scope_registered"))
    if sfm_quality == "rejected":
        quality_status = "rejected"
        conclusion = "current_frozen_foundation_rejected_on_admitted_forward_motion_case" if geometry_proven else "rejected_before_forward_motion_geometry_revalidation"
    elif not geometry_proven:
        quality_status = "degraded"
        conclusion = "inconclusive_forward_motion_proxy_not_revalidated"
    elif full and sfm_quality == "accepted":
        quality_status = "pass"
        conclusion = "survived_admitted_forward_motion_case"
    else:
        quality_status = "degraded"
        conclusion = "degraded_on_admitted_forward_motion_case"
    authority = deepcopy(dict(_mapping(admission.get("authority"), label="I5.4A authority")))
    window = _mapping(_mapping(admission.get("stress_plan"), label="I5.4A stress plan").get("forward_motion_window"), label="I5.4A window")
    return {
        "schema_version": I54B_SCHEMA_VERSION, "phase": I54B_PHASE,
        "status": "current_frozen_foundation_forward_motion_kitti_replayed", "pass": True, "quality_status": quality_status,
        "scope": {
            "current_frozen_foundation_real_replay": True, "i5_4a_admission_reused": True, "frozen_60_frame_window_reused": True,
            "production_native_pycolmap_replay": True, "sequential_matching_reused": True, "source_images_transformed": False,
            "window_retuned": False, "retry_until_quality_pass": False, "dense_provider_rerun": False,
            "provider_specific_robustness_started": False, "mp_sfm_strategy_evaluation_started": False, "thresholds_modified": False,
            "production_default_modified": False, "auto_policy_work_started": False, "hybrid_experiment_started": False,
        },
        "evidence_bindings": {"i5_4a_admission": i5_4a_binding.as_dict(), "i5_4b_replay_receipt": replay_receipt_binding.as_dict()},
        "source_binding": {
            "source_asset_set_sha256": admission["source_identity"]["asset_set_sha256"],
            "window_asset_set_sha256": window["window_asset_set_sha256"], "window_images": 60,
            "source_reverified_before": True, "source_window_reverified_after": True, "staged_inputs_reverified": True,
        },
        "replay": analysis,
        "forward_motion_revalidation": deepcopy(dict(motion)),
        "qualification": {
            "quality_status": quality_status, "forward_motion_robustness_conclusion": conclusion,
            "authoritative_forward_motion_geometry_proven": geometry_proven,
            "registration": f"{analysis.get('registered_images')}/{analysis.get('input_images')}",
            "retry_until_quality_pass": False, "result_driven_window_tuning": False,
            "universal_forward_motion_robustness_claimed": False, "kitti_dataset_universal_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False, "provider_robustness_claimed": False,
        },
        "regression_summary": {
            "focused_i5_4b_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True, "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": authority,
        "decision": {
            "i4_closed": True, "i5_authorized": True, "i5_1_complete": True, "i5_2_complete": True, "i5_3_complete": True,
            "i5_4_started": True, "i5_4a_complete": True, "i5_4b_complete": True, "i5_4c_qualification_closure_authorized": True,
            "i5_4c_complete": False, "i5_4_complete": False, "i5_5_authorized": False,
            "provider_specific_robustness_started": False, "mp_sfm_authorized": False, "provider_selected": False,
            "provider_promoted": False, "provider_ranking_created": False, "universal_provider_winner_declared": False,
            "global_scalar_score_created": False, "auto_policy_implemented": False, "hybrid_authorized": False,
            "threshold_relaxation_authorized": False, "production_default_modified": False, "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_4c_qualification_closure_authorized_not_started": True, "i5_5_dynamic_scene_not_started": True,
            "provider_specific_robustness_not_started": True, "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True, "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.4C forward-motion / KITTI robustness qualification and closure.",
    }


I54C_SCHEMA_VERSION = 1
I54C_PHASE = "I5.4C"
I54C_RECEIPT_PHASE = "I5.4C_CLOSURE_RECEIPT"


def _validate_i54c_binding_mapping(raw: Any, *, label: str) -> dict[str, Any]:
    item = _mapping(raw, label=label)
    path = str(item.get("path", "")).strip()
    sha = str(item.get("sha256", "")).lower()
    if not path:
        raise ReconstructionValidationError(f"{label}.path must be non-empty.")
    if not _valid_sha256(sha):
        raise ReconstructionValidationError(f"{label}.sha256 must be a valid SHA256 digest.")
    return {"path": path, "sha256": sha}


def validate_i5_4b_for_i5_4c(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.4B replay")
    _require(payload, "phase", "I5.4B", label="I5.4B replay")
    _require(payload, "status", "current_frozen_foundation_forward_motion_kitti_replayed", label="I5.4B replay")
    _require(payload, "pass", True, label="I5.4B replay")
    _require(payload, "quality_status", "rejected", label="I5.4B replay")

    scope = _mapping(payload.get("scope"), label="I5.4B scope")
    for key in (
        "current_frozen_foundation_real_replay", "i5_4a_admission_reused", "frozen_60_frame_window_reused",
        "production_native_pycolmap_replay", "sequential_matching_reused",
    ):
        _require(scope, key, True, label="I5.4B scope")
    for key in (
        "source_images_transformed", "window_retuned", "retry_until_quality_pass", "dense_provider_rerun",
        "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started", "thresholds_modified",
        "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.4B scope")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.4B evidence bindings")
    _validate_i54c_binding_mapping(bindings.get("i5_4a_admission"), label="I5.4B I5.4A binding")
    _validate_i54c_binding_mapping(bindings.get("i5_4b_replay_receipt"), label="I5.4B replay receipt binding")

    source = _mapping(payload.get("source_binding"), label="I5.4B source binding")
    _require(source, "window_images", 60, label="I5.4B source binding")
    for key in ("source_reverified_before", "source_window_reverified_after", "staged_inputs_reverified"):
        _require(source, key, True, label="I5.4B source binding")
    for key in ("source_asset_set_sha256", "window_asset_set_sha256"):
        if not _valid_sha256(source.get(key)):
            raise ReconstructionValidationError(f"I5.4B source binding {key} is invalid.")

    replay = _mapping(payload.get("replay"), label="I5.4B replay evidence")
    _require(replay, "input_images", 60, label="I5.4B replay evidence")
    _require(replay, "registered_images", 0, label="I5.4B replay evidence")
    _require(replay, "registration_ratio", 0.0, label="I5.4B replay evidence")
    _require(replay, "full_input_scope_registered", False, label="I5.4B replay evidence")
    _require(replay, "quality_status", "rejected", label="I5.4B replay evidence")
    _require(replay, "pre_sfm_ingestion_rejected", False, label="I5.4B replay evidence")
    _require(replay, "rejection_stage", "incremental_mapping", label="I5.4B replay evidence")
    _require(replay, "mapping_produced_sparse_model", False, label="I5.4B replay evidence")
    _require(replay, "registration_attempted", True, label="I5.4B replay evidence")
    _require(replay, "sfm_executed", True, label="I5.4B replay evidence")
    reason = str(replay.get("rejection_reason", ""))
    if "incremental mapping produced no sparse model" not in reason.lower():
        raise ReconstructionValidationError("I5.4C requires the frozen I5.4B no-sparse-model rejection reason.")

    stats = _mapping(replay.get("statistics"), label="I5.4B replay statistics")
    _require(stats, "input_keyframes", 60, label="I5.4B replay statistics")
    _require(stats, "registered_keyframes", 0, label="I5.4B replay statistics")
    _require(stats, "landmarks", 0, label="I5.4B replay statistics")
    _require(stats, "observations", 0, label="I5.4B replay statistics")

    geometry = _mapping(replay.get("geometry"), label="I5.4B replay geometry")
    motion = _mapping(geometry.get("camera_motion"), label="I5.4B replay camera motion")
    _require(motion, "adjacent_registered_motion_pairs", 0, label="I5.4B replay camera motion")
    _require(motion, "authoritative_forward_motion_geometry_proven", False, label="I5.4B replay camera motion")
    _require(motion, "forward_motion_label", "inconclusive_forward_motion_proxy_not_revalidated", label="I5.4B replay camera motion")
    _require(motion, "absolute_translation_magnitude_cross_run_comparison_permitted", False, label="I5.4B replay camera motion")
    _require(motion, "arbitrary_gauge", True, label="I5.4B replay camera motion")

    parallax = _mapping(geometry.get("parallax"), label="I5.4B replay parallax")
    _require(parallax, "scale_invariant", True, label="I5.4B replay parallax")
    _require(parallax, "absolute_baseline_cross_scenario_comparison_permitted", False, label="I5.4B replay parallax")

    forward = _mapping(payload.get("forward_motion_revalidation"), label="I5.4B forward-motion revalidation")
    _require(forward, "adjacent_registered_motion_pairs", 0, label="I5.4B forward-motion revalidation")
    _require(forward, "authoritative_forward_motion_geometry_proven", False, label="I5.4B forward-motion revalidation")

    qualification = _mapping(payload.get("qualification"), label="I5.4B qualification")
    _require(qualification, "quality_status", "rejected", label="I5.4B qualification")
    _require(qualification, "forward_motion_robustness_conclusion", "rejected_before_forward_motion_geometry_revalidation", label="I5.4B qualification")
    _require(qualification, "authoritative_forward_motion_geometry_proven", False, label="I5.4B qualification")
    _require(qualification, "registration", "0/60", label="I5.4B qualification")
    _require(qualification, "retry_until_quality_pass", False, label="I5.4B qualification")
    _require(qualification, "result_driven_window_tuning", False, label="I5.4B qualification")
    for key in (
        "universal_forward_motion_robustness_claimed", "kitti_dataset_universal_robustness_claimed",
        "dense_reconstruction_robustness_claimed", "provider_robustness_claimed",
    ):
        _require(qualification, key, False, label="I5.4B qualification")

    authority = _mapping(payload.get("authority"), label="I5.4B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.4B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.4B authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.4B authority")

    decision = _mapping(payload.get("decision"), label="I5.4B decision")
    for key in (
        "i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete",
        "i5_4_started", "i5_4a_complete", "i5_4b_complete", "i5_4c_qualification_closure_authorized",
    ):
        _require(decision, key, True, label="I5.4B decision")
    for key in (
        "i5_4c_complete", "i5_4_complete", "i5_5_authorized", "provider_specific_robustness_started",
        "mp_sfm_authorized", "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented",
        "hybrid_authorized", "threshold_relaxation_authorized", "production_default_modified",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.4B decision")
    if payload.get("next") != "I5.4C forward-motion / KITTI robustness qualification and closure.":
        raise ReconstructionValidationError("I5.4B next-step boundary does not authorize I5.4C.")
    return deepcopy(dict(payload))


def validate_i5_4c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.4C receipt")
    _require(payload, "phase", I54C_RECEIPT_PHASE, label="I5.4C receipt")
    _require(payload, "status", "pass", label="I5.4C receipt")
    _require(payload, "pass", True, label="I5.4C receipt")
    scope = _mapping(payload.get("scope"), label="I5.4C receipt scope")
    for key in (
        "focused_i5_4c_tests", "predecessor_i5_4b_regression", "predecessor_i5_4a_regression",
        "evidence_only_closure_contract_regression", "negative_quality_preservation_regression",
        "authority_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.4C receipt scope")
    for key in (
        "real_sfm_executed_by_test_receipt", "source_rematerialized", "dense_provider_rerun",
        "thresholds_modified", "production_default_modified", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.4C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.4C receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.4C receipt counts must be non-negative integers.")
    if focused < 12 or predecessor < 20 or combined < 32 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.4C receipt coverage is below the required closure floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.4C receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.4C receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_4c_forward_motion_kitti_closure(
    *, i5_4b_result: Mapping[str, Any], i5_4b_binding: I51EvidenceBinding,
    closure_receipt: Mapping[str, Any], closure_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    replay = validate_i5_4b_for_i5_4c(i5_4b_result)
    receipt = validate_i5_4c_receipt(closure_receipt)
    if not _valid_sha256(i5_4b_binding.sha256) or not str(i5_4b_binding.path).strip():
        raise ReconstructionValidationError("I5.4C requires a valid frozen I5.4B evidence binding.")
    if not _valid_sha256(closure_receipt_binding.sha256) or not str(closure_receipt_binding.path).strip():
        raise ReconstructionValidationError("I5.4C requires a valid closure-receipt binding.")

    authority = deepcopy(dict(_mapping(replay.get("authority"), label="I5.4B authority")))
    return {
        "schema_version": I54C_SCHEMA_VERSION,
        "phase": I54C_PHASE,
        "status": "forward_motion_kitti_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": "rejected",
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_4b_real_evidence_reused": True,
            "negative_quality_result_preserved": True,
            "sfm_rerun": False,
            "source_rematerialized": False,
            "window_retuned": False,
            "quality_replay_rerun": False,
            "dense_provider_rerun": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_4b_real_replay": i5_4b_binding.as_dict(),
            "i5_4c_closure_receipt": closure_receipt_binding.as_dict(),
        },
        "qualification": {
            "quality_status": "rejected",
            "operating_envelope": "outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case",
            "primary_bottleneck": "incremental_mapping_no_sparse_model",
            "rejection_stage": "incremental_mapping",
            "registered_images": 0,
            "input_images": 60,
            "mapping_produced_sparse_model": False,
            "registration_attempted": True,
            "sfm_executed": True,
            "forward_motion_geometry_revalidation_status": "unavailable_no_sparse_model",
            "authoritative_forward_motion_geometry_proven": False,
            "i2_4_geometry_available": False,
            "retry_until_quality_pass": False,
            "result_driven_window_tuning": False,
            "universal_forward_motion_robustness_claimed": False,
            "kitti_dataset_universal_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
            "metric_scale_claimed": False,
            "closure_interpretation": (
                "The current frozen production-native PyCOLMAP sparse-SfM foundation did not produce a sparse model "
                "on the admitted 60-frame KITTI-style source case. Because zero cameras registered, reconstructed "
                "camera geometry could not revalidate the forward-motion proxy; the negative sparse-SfM operating-envelope "
                "result is preserved without upgrading it to a universal forward-motion geometry claim."
            ),
        },
        "regression_summary": {
            "focused_i5_4c_tests": receipt["counts"]["focused_tests"],
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
            "i5_2_complete": True,
            "i5_3_complete": True,
            "i5_4_started": True,
            "i5_4a_complete": True,
            "i5_4b_complete": True,
            "i5_4c_complete": True,
            "i5_4_complete": True,
            "i5_5_authorized": True,
            "i5_5_started": False,
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
            "i5_5_dynamic_scene_authorized_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.5 dynamic-scene handling robustness.",
    }


__all__ = [
    "I54A_SCHEMA_VERSION",
    "I54A_PHASE",
    "I54A_RECEIPT_PHASE",
    "I54B_SCHEMA_VERSION",
    "I54B_PHASE",
    "I54B_RECEIPT_PHASE",
    "I54C_SCHEMA_VERSION",
    "I54C_PHASE",
    "I54C_RECEIPT_PHASE",
    "I54AAdmissionPolicy",
    "validate_i5_3c_for_i5_4",
    "build_i5_4a_source_identity",
    "validate_i5_4a_source_identity",
    "build_i5_4a_stress_plan",
    "validate_i5_4a_receipt",
    "build_i5_4a_forward_motion_kitti_admission",
    "validate_i5_4a_for_i5_4b",
    "validate_i5_4b_receipt",
    "analyze_i5_4b_sfm_report",
    "analyze_i5_4b_pre_sfm_ingestion_rejection",
    "analyze_i5_4b_mapping_no_sparse_model_rejection",
    "build_i5_4b_forward_motion_kitti_replay",
    "validate_i5_4b_for_i5_4c",
    "validate_i5_4c_receipt",
    "build_i5_4c_forward_motion_kitti_closure",
]
