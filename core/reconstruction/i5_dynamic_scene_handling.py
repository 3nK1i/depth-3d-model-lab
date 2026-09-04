"""I5.5A controlled dynamic-scene scenario admission contracts.

I5.5A is deliberately execution-free with respect to reconstruction. It binds
frozen I5.4C closure evidence, binds a known static multiview source, and freezes
one paired control/stress experiment for I5.5B:

* static control: the first 19 naturally ordered source images;
* dynamic stress: those same decoded source pixels plus a deterministic,
  source-derived foreground patch that translates independently across views.

The dynamic stress is a *controlled proxy*, not natural dynamic-scene ground
truth. I5.5B must materialize and revalidate the frozen recipe before running the
existing production-native PyCOLMAP sparse-SfM path with identical control/stress
settings. No dynamic masking, semantic segmentation, MP-SfM, Auto, Hybrid,
threshold relaxation, or production-default modification is introduced here.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding

I55A_SCHEMA_VERSION = 1
I55A_PHASE = "I5.5A"
I55A_RECEIPT_PHASE = "I5.5A_ADMISSION_RECEIPT"

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True, slots=True)
class I55AAdmissionPolicy:
    """Frozen result-independent policy for the I5.5B paired replay."""

    minimum_source_images: int = 19
    admitted_images: int = 19
    source_start_index: int = 0
    donor_source_index: int = 9
    donor_x_fraction: float = 0.38
    donor_y_fraction: float = 0.36
    patch_width_fraction: float = 0.24
    patch_height_fraction: float = 0.22
    target_x_start_fraction: float = 0.03
    target_x_end_fraction: float = 0.73
    target_y_fraction: float = 0.58
    matching_policy: str = "exhaustive"
    max_keyframes: int = 19
    output_encoding: str = "png_lossless"

    def __post_init__(self) -> None:
        for name in ("minimum_source_images", "admitted_images", "source_start_index", "donor_source_index", "max_keyframes"):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise ReconstructionValidationError(f"I5.5A {name} must be an integer.")
        if self.minimum_source_images < 19:
            raise ReconstructionValidationError("I5.5A minimum_source_images must be at least 19.")
        if self.admitted_images != 19:
            raise ReconstructionValidationError("I5.5A admitted_images is frozen to 19 for the paired Castle-P19 control/proxy experiment.")
        if self.source_start_index != 0:
            raise ReconstructionValidationError("I5.5A source_start_index is frozen to 0.")
        if not 0 <= self.donor_source_index < self.admitted_images:
            raise ReconstructionValidationError("I5.5A donor_source_index must lie inside the frozen admitted scope.")
        if self.max_keyframes != self.admitted_images:
            raise ReconstructionValidationError("I5.5A max_keyframes must equal the frozen admitted image count.")
        if self.matching_policy != "exhaustive":
            raise ReconstructionValidationError("I5.5A matching_policy is frozen to exhaustive for the static multiview source.")
        if self.output_encoding != "png_lossless":
            raise ReconstructionValidationError("I5.5A output_encoding is frozen to png_lossless for paired pixel-controlled materialization.")

        for name in (
            "donor_x_fraction", "donor_y_fraction", "patch_width_fraction", "patch_height_fraction",
            "target_x_start_fraction", "target_x_end_fraction", "target_y_fraction",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ReconstructionValidationError(f"I5.5A {name} must be finite.")
            object.__setattr__(self, name, value)

        if not (0.0 < self.patch_width_fraction < 1.0 and 0.0 < self.patch_height_fraction < 1.0):
            raise ReconstructionValidationError("I5.5A patch dimensions must be strictly inside (0,1).")
        if not 0.0 <= self.donor_x_fraction <= 1.0 - self.patch_width_fraction:
            raise ReconstructionValidationError("I5.5A donor crop exceeds normalized image width.")
        if not 0.0 <= self.donor_y_fraction <= 1.0 - self.patch_height_fraction:
            raise ReconstructionValidationError("I5.5A donor crop exceeds normalized image height.")
        if not 0.0 <= self.target_y_fraction <= 1.0 - self.patch_height_fraction:
            raise ReconstructionValidationError("I5.5A target y position exceeds normalized image height.")
        for value in (self.target_x_start_fraction, self.target_x_end_fraction):
            if not 0.0 <= value <= 1.0 - self.patch_width_fraction:
                raise ReconstructionValidationError("I5.5A target x trajectory exceeds normalized image width.")
        if self.target_x_end_fraction <= self.target_x_start_fraction:
            raise ReconstructionValidationError("I5.5A target x trajectory must move left-to-right by a positive amount.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "minimum_source_images": self.minimum_source_images,
            "admitted_images": self.admitted_images,
            "source_start_index": self.source_start_index,
            "donor_source_index": self.donor_source_index,
            "donor_x_fraction": self.donor_x_fraction,
            "donor_y_fraction": self.donor_y_fraction,
            "patch_width_fraction": self.patch_width_fraction,
            "patch_height_fraction": self.patch_height_fraction,
            "target_x_start_fraction": self.target_x_start_fraction,
            "target_x_end_fraction": self.target_x_end_fraction,
            "target_y_fraction": self.target_y_fraction,
            "matching_policy": self.matching_policy,
            "max_keyframes": self.max_keyframes,
            "output_encoding": self.output_encoding,
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


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> str:
    aggregate = hashlib.sha256()
    for index, raw in enumerate(records):
        aggregate.update(str(index).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(str(raw["name"]).encode("utf-8")); aggregate.update(b"\0")
        aggregate.update(str(int(raw["size_bytes"])).encode("ascii")); aggregate.update(b"\0")
        aggregate.update(str(raw["sha256"]).lower().encode("ascii")); aggregate.update(b"\n")
    return aggregate.hexdigest()


def _validate_binding(binding: I51EvidenceBinding, *, label: str) -> None:
    if not str(binding.path).strip() or not _valid_sha256(binding.sha256):
        raise ReconstructionValidationError(f"{label} evidence binding is malformed.")


def validate_i5_4c_for_i5_5(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact frozen I5.4 closure as the authorization boundary for I5.5."""

    _require(payload, "schema_version", 1, label="I5.4C closure")
    _require(payload, "phase", "I5.4C", label="I5.4C closure")
    _require(payload, "status", "forward_motion_kitti_robustness_qualified_and_closed", label="I5.4C closure")
    _require(payload, "pass", True, label="I5.4C closure")
    _require(payload, "quality_status", "rejected", label="I5.4C closure")

    scope = _mapping(payload.get("scope"), label="I5.4C scope")
    for key in ("evidence_only_qualification_and_closure", "i5_4b_real_evidence_reused", "negative_quality_result_preserved"):
        _require(scope, key, True, label="I5.4C scope")
    for key in (
        "sfm_rerun", "source_rematerialized", "window_retuned", "quality_replay_rerun", "dense_provider_rerun",
        "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started", "thresholds_modified",
        "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.4C scope")

    qualification = _mapping(payload.get("qualification"), label="I5.4C qualification")
    _require(qualification, "quality_status", "rejected", label="I5.4C qualification")
    _require(
        qualification,
        "operating_envelope",
        "outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case",
        label="I5.4C qualification",
    )
    _require(qualification, "primary_bottleneck", "incremental_mapping_no_sparse_model", label="I5.4C qualification")
    _require(qualification, "registered_images", 0, label="I5.4C qualification")
    _require(qualification, "input_images", 60, label="I5.4C qualification")
    _require(qualification, "mapping_produced_sparse_model", False, label="I5.4C qualification")
    _require(qualification, "retry_until_quality_pass", False, label="I5.4C qualification")
    _require(qualification, "result_driven_window_tuning", False, label="I5.4C qualification")
    _require(qualification, "metric_scale_claimed", False, label="I5.4C qualification")

    authority = _mapping(payload.get("authority"), label="I5.4C authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.4C authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.4C authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.4C authority")

    decision = _mapping(payload.get("decision"), label="I5.4C decision")
    for key in (
        "i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete",
        "i5_4_started", "i5_4a_complete", "i5_4b_complete", "i5_4c_complete", "i5_4_complete", "i5_5_authorized",
    ):
        _require(decision, key, True, label="I5.4C decision")
    for key in (
        "i5_5_started", "provider_specific_robustness_started", "mp_sfm_authorized", "provider_selected",
        "provider_promoted", "provider_ranking_created", "universal_provider_winner_declared",
        "global_scalar_score_created", "auto_policy_implemented", "hybrid_authorized",
        "threshold_relaxation_authorized", "production_default_modified", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.4C decision")
    if payload.get("next") != "I5.5 dynamic-scene handling robustness.":
        raise ReconstructionValidationError("I5.4C next-step boundary does not authorize I5.5.")
    return deepcopy(dict(payload))


def build_i5_5a_source_identity(
    directory: str | Path,
    *,
    policy: I55AAdmissionPolicy = I55AAdmissionPolicy(),
) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ReconstructionValidationError(f"I5.5A source directory does not exist: {root}")
    images = tuple(sorted(
        (path for path in root.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    ))
    if len(images) < policy.minimum_source_images:
        raise ReconstructionValidationError(
            f"I5.5A source requires at least {policy.minimum_source_images} images; found {len(images)}"
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
        "source_kind": "ordered_static_multiview_image_set",
        "directory": str(root),
        "image_count": len(records),
        "ordered_names": [record["name"] for record in records],
        "asset_set_sha256": _aggregate_records(records),
        "files": records,
        "natural_source_order_used": True,
        "recursive_discovery": False,
        "non_image_metadata_ignored": True,
        "images_transformed": False,
    }


def validate_i5_5a_source_identity(
    identity: Mapping[str, Any],
    *,
    policy: I55AAdmissionPolicy = I55AAdmissionPolicy(),
) -> dict[str, Any]:
    _require(identity, "source_kind", "ordered_static_multiview_image_set", label="I5.5A source identity")
    count = identity.get("image_count")
    if not isinstance(count, int) or count < policy.minimum_source_images:
        raise ReconstructionValidationError("I5.5A source identity image_count is below the frozen minimum.")
    names = identity.get("ordered_names")
    files = identity.get("files")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != count:
        raise ReconstructionValidationError("I5.5A source ordered_names is malformed.")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)) or len(files) != count:
        raise ReconstructionValidationError("I5.5A source file records are malformed.")
    _require(identity, "natural_source_order_used", True, label="I5.5A source identity")
    _require(identity, "recursive_discovery", False, label="I5.5A source identity")
    _require(identity, "images_transformed", False, label="I5.5A source identity")
    if not _valid_sha256(identity.get("asset_set_sha256")):
        raise ReconstructionValidationError("I5.5A source asset_set_sha256 is invalid.")

    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(files):
        item = _mapping(raw, label=f"I5.5A source file {index}")
        name = str(item.get("name", ""))
        if name != str(names[index]):
            raise ReconstructionValidationError("I5.5A source file/name ordering mismatch.")
        if item.get("order") != index:
            raise ReconstructionValidationError("I5.5A source file order is not contiguous.")
        size = item.get("size_bytes")
        sha = str(item.get("sha256", "")).lower()
        if not isinstance(size, int) or size < 0 or not _valid_sha256(sha):
            raise ReconstructionValidationError("I5.5A source file record is malformed.")
        normalized.append({"name": name, "size_bytes": size, "sha256": sha})
    if _aggregate_records(normalized) != str(identity["asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.5A source identity aggregate hash does not match its file records.")
    return deepcopy(dict(identity))


def _selected_scope(identity: Mapping[str, Any], *, policy: I55AAdmissionPolicy) -> tuple[list[int], list[str], list[Mapping[str, Any]]]:
    start = policy.source_start_index
    stop = start + policy.admitted_images
    if stop > int(identity["image_count"]):
        raise ReconstructionValidationError("I5.5A frozen admitted scope exceeds the source image set.")
    indices = list(range(start, stop))
    names = [str(identity["ordered_names"][i]) for i in indices]
    records = [_mapping(identity["files"][i], label=f"I5.5A selected source file {i}") for i in indices]
    return indices, names, records


def _scope_asset_sha(records: Sequence[Mapping[str, Any]]) -> str:
    return _aggregate_records([
        {"name": str(record["name"]), "size_bytes": int(record["size_bytes"]), "sha256": str(record["sha256"]).lower()}
        for record in records
    ])


def _target_boxes(policy: I55AAdmissionPolicy) -> list[dict[str, float]]:
    count = policy.admitted_images
    boxes: list[dict[str, float]] = []
    for index in range(count):
        t = 0.0 if count == 1 else index / float(count - 1)
        x = policy.target_x_start_fraction + t * (policy.target_x_end_fraction - policy.target_x_start_fraction)
        boxes.append({
            "x": round(x, 12),
            "y": policy.target_y_fraction,
            "width": policy.patch_width_fraction,
            "height": policy.patch_height_fraction,
        })
    return boxes


def build_i5_5a_stress_plan(
    source_identity: Mapping[str, Any],
    *,
    policy: I55AAdmissionPolicy = I55AAdmissionPolicy(),
) -> dict[str, Any]:
    identity = validate_i5_5a_source_identity(source_identity, policy=policy)
    indices, names, records = _selected_scope(identity, policy=policy)
    scope_sha = _scope_asset_sha(records)
    boxes = _target_boxes(policy)
    recipe = {
        "recipe_version": 1,
        "kind": "fixed_source_derived_patch_independent_translation",
        "source_scope_asset_set_sha256": scope_sha,
        "donor_source_index": policy.donor_source_index,
        "donor_source_name": str(identity["ordered_names"][policy.donor_source_index]),
        "donor_box_normalized": {
            "x": policy.donor_x_fraction,
            "y": policy.donor_y_fraction,
            "width": policy.patch_width_fraction,
            "height": policy.patch_height_fraction,
        },
        "trajectory": "linear_left_to_right_in_native_source_order",
        "target_x_start_fraction": policy.target_x_start_fraction,
        "target_x_end_fraction": policy.target_x_end_fraction,
        "target_y_fraction": policy.target_y_fraction,
        "target_boxes_normalized": boxes,
        "paste_mode": "opaque",
        "patch_resize": "none_same_normalized_extent",
        "control_materialization": "decode_source_then_write_lossless_png_without_pixel_edit",
        "stress_materialization": "same_decoded_source_pixels_plus_frozen_patch_then_write_lossless_png",
        "output_encoding": policy.output_encoding,
        "source_file_bytes_modified": False,
    }
    recipe_sha = _canonical_sha256(recipe)
    return {
        "static_control": {
            "kind": "static_multiview_control",
            "indices": indices,
            "names": names,
            "image_count": len(indices),
            "source_asset_set_sha256": scope_sha,
            "matching_policy": policy.matching_policy,
            "max_keyframes": policy.max_keyframes,
            "materialization_in_i5_5a": False,
        },
        "dynamic_foreground_proxy": {
            "kind": "controlled_source_derived_independent_foreground_motion_proxy",
            "indices": indices,
            "names": names,
            "image_count": len(indices),
            "source_asset_set_sha256": scope_sha,
            "matching_policy": policy.matching_policy,
            "max_keyframes": policy.max_keyframes,
            "materialization_in_i5_5a": False,
            "controlled_independent_foreground_motion_proxy": True,
            "natural_dynamic_scene_ground_truth": False,
            "dynamic_proxy_motion_independent_of_source_camera_motion_by_construction": True,
            "horizontal_displacement_fraction": policy.target_x_end_fraction - policy.target_x_start_fraction,
            "patch_area_fraction": policy.patch_width_fraction * policy.patch_height_fraction,
            "target_boxes_normalized": boxes,
            "materialization_recipe": recipe,
            "recipe_sha256": recipe_sha,
        },
        "required_i5_5b_revalidation": {
            "source_identity_reverified_before_and_after": True,
            "same_decoded_source_pixels_for_control_and_stress": True,
            "donor_patch_pixel_identity_preserved": True,
            "target_centroid_trajectory_matches_frozen_recipe": True,
            "changed_pixels_confined_to_frozen_target_boxes": True,
            "dynamic_region_feature_bearing_revalidation": True,
            "same_production_sfm_configuration_for_control_and_stress": True,
            "report_registration_landmarks_observations_reprojection_track_length_and_sparse_component_count": True,
            "control_must_be_production_accepted_before_dynamic_robustness_conclusion": True,
            "natural_dynamic_scene_ground_truth": False,
            "result_driven_recipe_tuning": False,
            "retry_until_quality_pass": False,
        },
        "classification_policy": {
            "control_not_accepted": "scenario_control_not_qualified_for_dynamic_robustness_conclusion",
            "control_accepted_stress_accepted": "pass_on_admitted_controlled_dynamic_foreground_proxy",
            "control_accepted_stress_degraded": "degraded_on_admitted_controlled_dynamic_foreground_proxy",
            "control_accepted_stress_rejected": "rejected_on_admitted_controlled_dynamic_foreground_proxy",
            "natural_dynamic_scene_robustness_claimed": False,
        },
        "policy": policy.as_dict(),
        "result_driven_recipe_tuning": False,
        "retry_until_quality_pass": False,
        "production_default_modified": False,
    }


def validate_i5_5a_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.5A receipt")
    _require(payload, "phase", I55A_RECEIPT_PHASE, label="I5.5A receipt")
    _require(payload, "status", "pass", label="I5.5A receipt")
    _require(payload, "pass", True, label="I5.5A receipt")
    scope = _mapping(payload.get("scope"), label="I5.5A receipt scope")
    for key in (
        "focused_i5_5a_tests", "predecessor_i5_4_regression", "source_identity_regression",
        "controlled_dynamic_proxy_design_regression", "authority_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.5A receipt scope")
    for key in (
        "sfm_rerun", "stress_images_materialized", "dense_provider_rerun", "dynamic_masking_added",
        "semantic_segmentation_added", "thresholds_modified", "production_default_modified", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.5A receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.5A receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.5A receipt counts must be non-negative integers.")
    if focused < 12 or predecessor < 20 or combined < 32 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.5A receipt coverage is below the required admission floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.5A receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.5A receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def build_i5_5a_dynamic_scene_admission(
    *,
    i5_4c_result: Mapping[str, Any],
    i5_4c_binding: I51EvidenceBinding,
    source_identity: Mapping[str, Any],
    source_label: str,
    static_control_confirmation: bool,
    admission_receipt: Mapping[str, Any],
    admission_receipt_binding: I51EvidenceBinding,
    policy: I55AAdmissionPolicy = I55AAdmissionPolicy(),
) -> dict[str, Any]:
    closure = validate_i5_4c_for_i5_5(i5_4c_result)
    identity = validate_i5_5a_source_identity(source_identity, policy=policy)
    receipt = validate_i5_5a_receipt(admission_receipt)
    _validate_binding(i5_4c_binding, label="I5.5A I5.4C")
    _validate_binding(admission_receipt_binding, label="I5.5A receipt")
    label = str(source_label or "").strip()
    if not label:
        raise ReconstructionValidationError("I5.5A source_label must be non-empty.")
    if static_control_confirmation is not True:
        raise ReconstructionValidationError(
            "I5.5A requires explicit operator confirmation that the bound source is the intended static multiview control."
        )
    plan = build_i5_5a_stress_plan(identity, policy=policy)
    authority = deepcopy(dict(_mapping(closure.get("authority"), label="I5.4C authority")))
    stress = _mapping(plan.get("dynamic_foreground_proxy"), label="I5.5A dynamic proxy")
    return {
        "schema_version": I55A_SCHEMA_VERSION,
        "phase": I55A_PHASE,
        "status": "controlled_dynamic_scene_scenario_admitted",
        "pass": True,
        "scope": {
            "scenario_admission_only": True,
            "i5_4c_closure_reused": True,
            "static_multiview_source_bound": True,
            "paired_control_dynamic_proxy_frozen": True,
            "sfm_rerun": False,
            "stress_images_materialized": False,
            "source_images_modified": False,
            "dense_provider_rerun": False,
            "dynamic_masking_added": False,
            "semantic_segmentation_added": False,
            "motion_outlier_filter_added": False,
            "provider_specific_robustness_started": False,
            "mp_sfm_strategy_evaluation_started": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "auto_policy_work_started": False,
            "hybrid_experiment_started": False,
        },
        "evidence_bindings": {
            "i5_4c_closure": i5_4c_binding.as_dict(),
            "i5_5a_admission_receipt": admission_receipt_binding.as_dict(),
        },
        "predecessor": {
            "i5_4_quality_status": str(closure["quality_status"]),
            "i5_4_closed": True,
            "i5_5_authorized": True,
        },
        "source_identity": identity,
        "source_semantics": {
            "label": label,
            "operator_static_control_confirmation": True,
            "operator_confirmation_is_scene_motion_ground_truth": False,
            "controlled_dynamic_proxy_created_only_in_i5_5b": True,
            "natural_dynamic_scene_ground_truth": False,
        },
        "stress_plan": plan,
        "qualification": {
            "controlled_dynamic_foreground_proxy_admitted": True,
            "natural_dynamic_scene_ground_truth": False,
            "source_asset_set_sha256": identity["asset_set_sha256"],
            "admitted_images": policy.admitted_images,
            "dynamic_proxy_recipe_sha256": stress["recipe_sha256"],
            "dynamic_proxy_horizontal_displacement_fraction": stress["horizontal_displacement_fraction"],
            "dynamic_proxy_patch_area_fraction": stress["patch_area_fraction"],
            "dynamic_masking_present": False,
            "semantic_segmentation_present": False,
            "retry_until_quality_pass": False,
            "result_driven_recipe_tuning": False,
            "universal_dynamic_scene_robustness_claimed": False,
            "real_world_dynamic_scene_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
        },
        "regression_summary": {
            "focused_i5_5a_tests": receipt["counts"]["focused_tests"],
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
            "i5_4_complete": True,
            "i5_5_started": True,
            "i5_5a_complete": True,
            "i5_5b_real_replay_authorized": True,
            "i5_5b_started": False,
            "i5_5c_complete": False,
            "i5_5_complete": False,
            "i5_6_authorized": False,
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
            "i5_5b_real_replay_authorized_not_started": True,
            "i5_6_mvs_robustness_not_authorized": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.5B current frozen-foundation dynamic-scene replay.",
    }


__all__ = [
    "I55A_SCHEMA_VERSION",
    "I55A_PHASE",
    "I55A_RECEIPT_PHASE",
    "I55AAdmissionPolicy",
    "validate_i5_4c_for_i5_5",
    "build_i5_5a_source_identity",
    "validate_i5_5a_source_identity",
    "build_i5_5a_stress_plan",
    "validate_i5_5a_receipt",
    "build_i5_5a_dynamic_scene_admission",
]
