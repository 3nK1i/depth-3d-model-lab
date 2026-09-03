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


__all__ = [
    "I54A_SCHEMA_VERSION",
    "I54A_PHASE",
    "I54A_RECEIPT_PHASE",
    "I54AAdmissionPolicy",
    "validate_i5_3c_for_i5_4",
    "build_i5_4a_source_identity",
    "validate_i5_4a_source_identity",
    "build_i5_4a_stress_plan",
    "validate_i5_4a_receipt",
    "build_i5_4a_forward_motion_kitti_admission",
]
