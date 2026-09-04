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
import shutil
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

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


I55B_SCHEMA_VERSION = 1
I55B_PHASE = "I5.5B"
I55B_RECEIPT_PHASE = "I5.5B_REPLAY_RECEIPT"


def validate_i5_5a_for_i5_5b(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen I5.5A admission before any real replay/materialization."""

    _require(payload, "schema_version", 1, label="I5.5A admission")
    _require(payload, "phase", "I5.5A", label="I5.5A admission")
    _require(payload, "status", "controlled_dynamic_scene_scenario_admitted", label="I5.5A admission")
    _require(payload, "pass", True, label="I5.5A admission")

    scope = _mapping(payload.get("scope"), label="I5.5A scope")
    for key in (
        "scenario_admission_only", "i5_4c_closure_reused", "static_multiview_source_bound",
        "paired_control_dynamic_proxy_frozen",
    ):
        _require(scope, key, True, label="I5.5A scope")
    for key in (
        "sfm_rerun", "stress_images_materialized", "source_images_modified", "dense_provider_rerun",
        "dynamic_masking_added", "semantic_segmentation_added", "motion_outlier_filter_added",
        "provider_specific_robustness_started", "mp_sfm_strategy_evaluation_started", "thresholds_modified",
        "production_default_modified", "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.5A scope")

    identity = validate_i5_5a_source_identity(_mapping(payload.get("source_identity"), label="I5.5A source identity"))
    if int(identity.get("image_count", -1)) < 19:
        raise ReconstructionValidationError("I5.5B requires the frozen I5.5A source with at least 19 images.")

    semantics = _mapping(payload.get("source_semantics"), label="I5.5A source semantics")
    _require(semantics, "operator_static_control_confirmation", True, label="I5.5A source semantics")
    _require(semantics, "operator_confirmation_is_scene_motion_ground_truth", False, label="I5.5A source semantics")
    _require(semantics, "controlled_dynamic_proxy_created_only_in_i5_5b", True, label="I5.5A source semantics")
    _require(semantics, "natural_dynamic_scene_ground_truth", False, label="I5.5A source semantics")

    plan = _mapping(payload.get("stress_plan"), label="I5.5A stress plan")
    control = _mapping(plan.get("static_control"), label="I5.5A static control")
    stress = _mapping(plan.get("dynamic_foreground_proxy"), label="I5.5A dynamic proxy")
    expected_indices = list(range(19))
    expected_names = [str(value) for value in identity["ordered_names"][:19]]
    for node, label in ((control, "I5.5A static control"), (stress, "I5.5A dynamic proxy")):
        _require(node, "image_count", 19, label=label)
        _require(node, "matching_policy", "exhaustive", label=label)
        _require(node, "max_keyframes", 19, label=label)
        if [int(value) for value in node.get("indices", [])] != expected_indices:
            raise ReconstructionValidationError(f"{label} indices drifted from the frozen 0..18 scope.")
        if [str(value) for value in node.get("names", [])] != expected_names:
            raise ReconstructionValidationError(f"{label} names drifted from the frozen source scope.")
        if str(node.get("source_asset_set_sha256", "")).lower() != _scope_asset_sha(identity["files"][:19]):
            raise ReconstructionValidationError(f"{label} source asset-set SHA256 drifted.")
    _require(control, "kind", "static_multiview_control", label="I5.5A static control")
    _require(control, "materialization_in_i5_5a", False, label="I5.5A static control")
    _require(stress, "kind", "controlled_source_derived_independent_foreground_motion_proxy", label="I5.5A dynamic proxy")
    _require(stress, "materialization_in_i5_5a", False, label="I5.5A dynamic proxy")
    _require(stress, "controlled_independent_foreground_motion_proxy", True, label="I5.5A dynamic proxy")
    _require(stress, "natural_dynamic_scene_ground_truth", False, label="I5.5A dynamic proxy")
    _require(stress, "dynamic_proxy_motion_independent_of_source_camera_motion_by_construction", True, label="I5.5A dynamic proxy")

    recipe = _mapping(stress.get("materialization_recipe"), label="I5.5A dynamic recipe")
    _require(recipe, "recipe_version", 1, label="I5.5A dynamic recipe")
    _require(recipe, "kind", "fixed_source_derived_patch_independent_translation", label="I5.5A dynamic recipe")
    _require(recipe, "paste_mode", "opaque", label="I5.5A dynamic recipe")
    _require(recipe, "patch_resize", "none_same_normalized_extent", label="I5.5A dynamic recipe")
    _require(recipe, "output_encoding", "png_lossless", label="I5.5A dynamic recipe")
    _require(recipe, "source_file_bytes_modified", False, label="I5.5A dynamic recipe")
    expected_recipe_sha = _canonical_sha256(recipe)
    if str(stress.get("recipe_sha256", "")).lower() != expected_recipe_sha:
        raise ReconstructionValidationError("I5.5A dynamic proxy recipe SHA256 drifted.")
    if str(payload.get("qualification", {}).get("dynamic_proxy_recipe_sha256", "")).lower() != expected_recipe_sha:
        raise ReconstructionValidationError("I5.5A qualification recipe SHA256 drifted.")

    required = _mapping(plan.get("required_i5_5b_revalidation"), label="I5.5A I5.5B revalidation contract")
    for key in (
        "source_identity_reverified_before_and_after", "same_decoded_source_pixels_for_control_and_stress",
        "donor_patch_pixel_identity_preserved", "target_centroid_trajectory_matches_frozen_recipe",
        "changed_pixels_confined_to_frozen_target_boxes", "dynamic_region_feature_bearing_revalidation",
        "same_production_sfm_configuration_for_control_and_stress",
        "report_registration_landmarks_observations_reprojection_track_length_and_sparse_component_count",
        "control_must_be_production_accepted_before_dynamic_robustness_conclusion",
    ):
        _require(required, key, True, label="I5.5A I5.5B revalidation contract")
    for key in ("natural_dynamic_scene_ground_truth", "result_driven_recipe_tuning", "retry_until_quality_pass"):
        _require(required, key, False, label="I5.5A I5.5B revalidation contract")

    classification = _mapping(plan.get("classification_policy"), label="I5.5A classification policy")
    _require(classification, "control_not_accepted", "scenario_control_not_qualified_for_dynamic_robustness_conclusion", label="I5.5A classification policy")
    _require(classification, "control_accepted_stress_accepted", "pass_on_admitted_controlled_dynamic_foreground_proxy", label="I5.5A classification policy")
    _require(classification, "control_accepted_stress_degraded", "degraded_on_admitted_controlled_dynamic_foreground_proxy", label="I5.5A classification policy")
    _require(classification, "control_accepted_stress_rejected", "rejected_on_admitted_controlled_dynamic_foreground_proxy", label="I5.5A classification policy")
    _require(classification, "natural_dynamic_scene_robustness_claimed", False, label="I5.5A classification policy")
    _require(plan, "result_driven_recipe_tuning", False, label="I5.5A stress plan")
    _require(plan, "retry_until_quality_pass", False, label="I5.5A stress plan")
    _require(plan, "production_default_modified", False, label="I5.5A stress plan")

    authority = _mapping(payload.get("authority"), label="I5.5A authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.5A authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.5A authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.5A authority")

    decision = _mapping(payload.get("decision"), label="I5.5A decision")
    for key in (
        "i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete", "i5_4_complete",
        "i5_5_started", "i5_5a_complete", "i5_5b_real_replay_authorized",
    ):
        _require(decision, key, True, label="I5.5A decision")
    for key in (
        "i5_5b_started", "i5_5c_complete", "i5_5_complete", "i5_6_authorized", "provider_specific_robustness_started",
        "mp_sfm_authorized", "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented",
        "hybrid_authorized", "threshold_relaxation_authorized", "production_default_modified", "metric_scale_claimed",
        "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.5A decision")
    if payload.get("next") != "I5.5B current frozen-foundation dynamic-scene replay.":
        raise ReconstructionValidationError("I5.5A next-step boundary does not authorize I5.5B.")
    return deepcopy(dict(payload))


def validate_i5_5b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", 1, label="I5.5B receipt")
    _require(payload, "phase", I55B_RECEIPT_PHASE, label="I5.5B receipt")
    _require(payload, "status", "pass", label="I5.5B receipt")
    _require(payload, "pass", True, label="I5.5B receipt")
    scope = _mapping(payload.get("scope"), label="I5.5B receipt scope")
    for key in (
        "focused_i5_5b_tests", "predecessor_i5_5a_regression", "production_native_sfm_contract_regression",
        "dynamic_materialization_regression", "dynamic_comparison_regression", "authority_guard_regression",
        "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.5B receipt scope")
    for key in (
        "real_sfm_executed_by_test_receipt", "real_stress_images_materialized_by_test_receipt", "dense_provider_rerun",
        "dynamic_masking_added", "semantic_segmentation_added", "motion_outlier_filter_added", "thresholds_modified",
        "production_default_modified", "mp_sfm_started", "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.5B receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.5B receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.5B receipt counts must be non-negative integers.")
    if focused < 15 or predecessor < 50 or combined < 65 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.5B receipt coverage is below the required replay floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.5B receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.5B receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _i55b_pixel_box(box: Mapping[str, Any], *, width: int, height: int) -> tuple[int, int, int, int]:
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError("I5.5B image dimensions must be positive.")
    x = float(box.get("x")); y = float(box.get("y")); w = float(box.get("width")); h = float(box.get("height"))
    if not all(isfinite(value) for value in (x, y, w, h)):
        raise ReconstructionValidationError("I5.5B normalized box values must be finite.")
    width_px = max(1, int(np.floor(w * width + 0.5)))
    height_px = max(1, int(np.floor(h * height + 0.5)))
    x0 = int(np.floor(x * width + 0.5)); y0 = int(np.floor(y * height + 0.5))
    if x0 < 0 or y0 < 0 or x0 + width_px > width or y0 + height_px > height:
        raise ReconstructionValidationError("I5.5B normalized box does not fit the decoded image.")
    return x0, y0, width_px, height_px


def _i55b_pixel_sha(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii")); digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode("ascii")); digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _i55b_write_lossless_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ReconstructionValidationError(f"I5.5B failed to write lossless PNG: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape != image.shape or not np.array_equal(decoded, image):
        raise ReconstructionValidationError(f"I5.5B lossless PNG verification failed: {path}")
    return _sha256(path)


def materialize_i5_5b_pair(*, dataset_dir: Path, admission: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    frozen = validate_i5_5a_for_i5_5b(admission)
    dataset = Path(dataset_dir).expanduser().resolve()
    if not dataset.is_dir():
        raise ReconstructionValidationError(f"I5.5B dataset directory missing: {dataset}")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    control_dir = output / "control"
    stress_dir = output / "dynamic_proxy"
    control_dir.mkdir(parents=True, exist_ok=False)
    stress_dir.mkdir(parents=True, exist_ok=False)

    identity = _mapping(frozen["source_identity"], label="I5.5A source identity")
    files = identity["files"]
    plan = _mapping(frozen["stress_plan"], label="I5.5A stress plan")
    control_plan = _mapping(plan["static_control"], label="I5.5A static control")
    stress_plan = _mapping(plan["dynamic_foreground_proxy"], label="I5.5A dynamic proxy")
    recipe = _mapping(stress_plan["materialization_recipe"], label="I5.5A materialization recipe")
    source_names = [str(value) for value in control_plan["names"]]

    decoded: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    for local_index, source_index in enumerate(control_plan["indices"]):
        record = _mapping(files[int(source_index)], label=f"I5.5B source file {source_index}")
        name = source_names[local_index]
        if str(record.get("name")) != name:
            raise ReconstructionValidationError("I5.5B frozen source record/name ordering drifted.")
        path = (dataset / name).resolve()
        try:
            path.relative_to(dataset)
        except ValueError as exc:
            raise ReconstructionValidationError(f"I5.5B source path escapes dataset: {path}") from exc
        if not path.is_file() or _sha256(path) != str(record.get("sha256", "")).lower():
            raise ReconstructionValidationError(f"I5.5B frozen source image missing or SHA256 drifted: {name}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ReconstructionValidationError(f"I5.5B cannot decode frozen source image: {path}")
        decoded.append(image)
        source_rows.append({
            "source_index": int(source_index), "source_name": name, "source_path": str(path),
            "source_sha256": _sha256(path), "source_pixel_sha256": _i55b_pixel_sha(image),
            "shape": [int(value) for value in image.shape],
        })
    shapes = {tuple(image.shape) for image in decoded}
    if len(shapes) != 1:
        raise ReconstructionValidationError("I5.5B frozen no-resize proxy requires constant decoded source dimensions.")
    height, width = decoded[0].shape[:2]

    donor_index = int(recipe["donor_source_index"])
    donor_local_index = source_names.index(str(recipe["donor_source_name"]))
    if int(control_plan["indices"][donor_local_index]) != donor_index:
        raise ReconstructionValidationError("I5.5B donor source index/name binding drifted.")
    donor_box = _i55b_pixel_box(_mapping(recipe["donor_box_normalized"], label="I5.5A donor box"), width=width, height=height)
    dx, dy, dw, dh = donor_box
    donor_patch = decoded[donor_local_index][dy:dy + dh, dx:dx + dw].copy()
    donor_patch_sha = _i55b_pixel_sha(donor_patch)

    target_boxes = recipe.get("target_boxes_normalized")
    if not isinstance(target_boxes, Sequence) or isinstance(target_boxes, (str, bytes)) or len(target_boxes) != len(decoded):
        raise ReconstructionValidationError("I5.5B frozen target-box trajectory is malformed.")

    control_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    for index, (source, source_row, raw_box) in enumerate(zip(decoded, source_rows, target_boxes)):
        staged_name = f"{index:04d}.png"
        control_path = control_dir / staged_name
        control_sha = _i55b_write_lossless_png(control_path, source)
        box = _i55b_pixel_box(_mapping(raw_box, label=f"I5.5A target box {index}"), width=width, height=height)
        x, y, w, h = box
        if (w, h) != (dw, dh):
            raise ReconstructionValidationError("I5.5B frozen no-resize proxy target extent differs from donor patch extent.")
        stress = source.copy()
        stress[y:y + h, x:x + w] = donor_patch
        stress_path = stress_dir / staged_name
        stress_sha = _i55b_write_lossless_png(stress_path, stress)
        control_rows.append({
            **source_row, "staged_name": staged_name, "staged_path": str(control_path),
            "staged_sha256": control_sha, "staged_pixel_sha256": _i55b_pixel_sha(source),
        })
        stress_rows.append({
            **source_row, "staged_name": staged_name, "staged_path": str(stress_path),
            "staged_sha256": stress_sha, "staged_pixel_sha256": _i55b_pixel_sha(stress),
            "target_box_normalized": deepcopy(dict(raw_box)),
            "target_box_pixels": {"x": x, "y": y, "width": w, "height": h},
        })

    return {
        "schema_version": 1,
        "phase": "I5.5B_MATERIALIZATION",
        "recipe_sha256": str(stress_plan["recipe_sha256"]),
        "pixel_box_conversion": "nearest_integer_half_up_for_xy_and_extent",
        "source_dimensions": {"width": width, "height": height, "channels": int(decoded[0].shape[2])},
        "source_dimensions_constant": True,
        "donor": {
            "source_index": donor_index, "source_name": str(recipe["donor_source_name"]),
            "box_normalized": deepcopy(dict(recipe["donor_box_normalized"])),
            "box_pixels": {"x": dx, "y": dy, "width": dw, "height": dh},
            "patch_pixel_sha256": donor_patch_sha,
        },
        "control": {"directory": str(control_dir), "image_count": len(control_rows), "files": control_rows},
        "dynamic_proxy": {"directory": str(stress_dir), "image_count": len(stress_rows), "files": stress_rows},
        "natural_dynamic_scene_ground_truth": False,
        "controlled_independent_foreground_motion_proxy": True,
    }


def _i55b_orb_count(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    orb = cv2.ORB_create(nfeatures=2000, edgeThreshold=5, fastThreshold=8)
    keypoints = orb.detect(gray, None)
    return len(keypoints or [])


def _i55b_quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def analyze_i5_5b_materialized_pair(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require(evidence, "schema_version", 1, label="I5.5B materialization")
    _require(evidence, "phase", "I5.5B_MATERIALIZATION", label="I5.5B materialization")
    _require(evidence, "source_dimensions_constant", True, label="I5.5B materialization")
    _require(evidence, "natural_dynamic_scene_ground_truth", False, label="I5.5B materialization")
    _require(evidence, "controlled_independent_foreground_motion_proxy", True, label="I5.5B materialization")
    donor = _mapping(evidence.get("donor"), label="I5.5B donor")
    control = _mapping(evidence.get("control"), label="I5.5B control materialization")
    stress = _mapping(evidence.get("dynamic_proxy"), label="I5.5B dynamic materialization")
    control_files = control.get("files"); stress_files = stress.get("files")
    if not isinstance(control_files, Sequence) or isinstance(control_files, (str, bytes)) or len(control_files) != 19:
        raise ReconstructionValidationError("I5.5B control materialization must contain 19 files.")
    if not isinstance(stress_files, Sequence) or isinstance(stress_files, (str, bytes)) or len(stress_files) != 19:
        raise ReconstructionValidationError("I5.5B dynamic materialization must contain 19 files.")

    donor_box = _mapping(donor.get("box_pixels"), label="I5.5B donor pixel box")
    donor_source_name = str(donor.get("source_name", ""))
    donor_control_row = next((_mapping(row, label="I5.5B control row") for row in control_files if str(row.get("source_name")) == donor_source_name), None)
    if donor_control_row is None:
        raise ReconstructionValidationError("I5.5B donor source is absent from control materialization.")
    donor_image = cv2.imread(str(donor_control_row["staged_path"]), cv2.IMREAD_COLOR)
    if donor_image is None:
        raise ReconstructionValidationError("I5.5B cannot decode staged donor control image.")
    dx, dy, dw, dh = (int(donor_box[k]) for k in ("x", "y", "width", "height"))
    donor_patch = donor_image[dy:dy + dh, dx:dx + dw]
    donor_sha_ok = _i55b_pixel_sha(donor_patch) == str(donor.get("patch_pixel_sha256", ""))
    donor_orb = _i55b_orb_count(donor_patch)

    control_preserved = True
    same_decoded = True
    donor_identity = donor_sha_ok
    confined = True
    target_orb_counts: list[float] = []
    changed_fractions: list[float] = []
    centers: list[float] = []
    target_boxes_match_recipe = True
    dims = _mapping(evidence.get("source_dimensions"), label="I5.5B source dimensions")
    width = int(dims["width"]); height = int(dims["height"])

    for c_raw, s_raw in zip(control_files, stress_files):
        c = _mapping(c_raw, label="I5.5B control row"); s = _mapping(s_raw, label="I5.5B stress row")
        if c.get("source_name") != s.get("source_name") or c.get("source_pixel_sha256") != s.get("source_pixel_sha256"):
            same_decoded = False
        control_image = cv2.imread(str(c["staged_path"]), cv2.IMREAD_COLOR)
        stress_image = cv2.imread(str(s["staged_path"]), cv2.IMREAD_COLOR)
        source_image = cv2.imread(str(c["source_path"]), cv2.IMREAD_COLOR)
        if control_image is None or stress_image is None or source_image is None:
            raise ReconstructionValidationError("I5.5B cannot decode staged/source image during materialization revalidation.")
        if control_image.shape != source_image.shape or not np.array_equal(control_image, source_image):
            control_preserved = False
        box = _mapping(s.get("target_box_pixels"), label="I5.5B stress target pixel box")
        x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
        if stress_image.shape != control_image.shape or (w, h) != donor_patch.shape[1::-1]:
            donor_identity = False
            confined = False
            continue
        target = stress_image[y:y + h, x:x + w]
        if not np.array_equal(target, donor_patch):
            donor_identity = False
        diff = np.any(stress_image != control_image, axis=2)
        allowed = np.zeros(diff.shape, dtype=bool); allowed[y:y + h, x:x + w] = True
        if np.any(diff & ~allowed):
            confined = False
        changed_fractions.append(float(np.count_nonzero(diff)) / float(diff.size))
        whole_kp = cv2.ORB_create(nfeatures=3000, edgeThreshold=5, fastThreshold=8).detect(cv2.cvtColor(stress_image, cv2.COLOR_BGR2GRAY), None) or []
        target_orb_counts.append(float(sum(1 for kp in whole_kp if x <= kp.pt[0] < x + w and y <= kp.pt[1] < y + h)))
        centers.append((x + w * 0.5) / width)
        normalized = _mapping(s.get("target_box_normalized"), label="I5.5B stress target normalized box")
        expected_box = _i55b_pixel_box(normalized, width=width, height=height)
        if (x, y, w, h) != expected_box:
            target_boxes_match_recipe = False

    trajectory_ok = bool(
        target_boxes_match_recipe
        and len(centers) == 19
        and all(b > a for a, b in zip(centers[:-1], centers[1:]))
    )
    target_min = min(target_orb_counts) if target_orb_counts else 0.0
    feature_bearing = bool(donor_orb > 0 and target_orb_counts and target_min > 0)
    return {
        "same_decoded_source_pixels_for_control_and_stress": bool(same_decoded),
        "control_pixel_values_preserved": bool(control_preserved),
        "donor_patch_pixel_identity_preserved": bool(donor_identity),
        "target_centroid_trajectory_matches_frozen_recipe": bool(trajectory_ok),
        "changed_pixels_confined_to_frozen_target_boxes": bool(confined),
        "dynamic_region_feature_bearing_revalidated": feature_bearing,
        "source_dimensions_constant": True,
        "donor_patch_orb_keypoints": int(donor_orb),
        "stress_target_orb_keypoints_min": int(target_min),
        "stress_target_orb_keypoints_p50": _i55b_quantile(target_orb_counts, 0.5),
        "changed_pixel_fraction_p50": _i55b_quantile(changed_fractions, 0.5),
        "natural_dynamic_scene_ground_truth": False,
        "controlled_independent_foreground_motion_proxy": True,
    }


def _i55b_plan_for_sfm(scenario_plan: Mapping[str, Any], staged_names: Sequence[str]) -> tuple[dict[str, Any], list[str]]:
    count = scenario_plan.get("image_count")
    source_names = scenario_plan.get("names")
    if count != 19 or not isinstance(source_names, Sequence) or isinstance(source_names, (str, bytes)) or len(source_names) != 19:
        raise ReconstructionValidationError("I5.5B scenario plan must preserve the frozen 19-image scope.")
    _require(scenario_plan, "matching_policy", "exhaustive", label="I5.5B scenario plan")
    _require(scenario_plan, "max_keyframes", 19, label="I5.5B scenario plan")
    materialized = [str(value) for value in staged_names]
    if len(materialized) != 19 or len(set(materialized)) != 19:
        raise ReconstructionValidationError("I5.5B staged names must be 19 unique images.")
    adapted = deepcopy(dict(scenario_plan))
    adapted["names"] = materialized
    adapted["indices"] = list(range(19))
    return adapted, [str(value) for value in source_names]


def analyze_i5_5b_sfm_report(
    report: Mapping[str, Any], *, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], sparse_component_count: int
) -> dict[str, Any]:
    if not isinstance(sparse_component_count, int) or sparse_component_count < 0:
        raise ReconstructionValidationError("I5.5B sparse_component_count must be a non-negative integer.")
    adapted, source_names = _i55b_plan_for_sfm(scenario_plan, staged_names)
    from core.reconstruction.i5_low_overlap_weak_parallax import analyze_i5_2b_sfm_report

    analysis = analyze_i5_2b_sfm_report(report, scenario_plan=adapted)
    analysis["registration_attempted"] = True
    analysis["sfm_executed"] = True
    analysis["pre_sfm_ingestion_rejected"] = False
    analysis["rejection_stage"] = None
    analysis["rejection_reason"] = None
    analysis["mapping_produced_sparse_model"] = True
    analysis["sparse_component_count"] = sparse_component_count
    registered_staged = [str(value) for value in analysis.get("registered_names", [])]
    mapping = dict(zip([str(value) for value in staged_names], source_names))
    analysis["registered_staged_names"] = registered_staged
    analysis["registered_names"] = [mapping[value] for value in registered_staged]
    analysis["source_membership_names"] = source_names
    analysis["staged_membership_names"] = [str(value) for value in staged_names]
    analysis["lossless_staging_name_mapping_preserved"] = True
    return analysis


def _i55b_rejection_common(*, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], reason: str) -> tuple[list[str], list[str], str]:
    _, source_names = _i55b_plan_for_sfm(scenario_plan, staged_names)
    materialized = [str(value) for value in staged_names]
    text = str(reason).strip()
    if not text:
        raise ReconstructionValidationError("I5.5B structured rejection requires a non-empty reason.")
    return source_names, materialized, text


def analyze_i5_5b_pre_sfm_ingestion_rejection(*, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], reason: str) -> dict[str, Any]:
    source_names, materialized, text = _i55b_rejection_common(scenario_plan=scenario_plan, staged_names=staged_names, reason=reason)
    if "Fewer than three valid keyframes survived ingestion" not in text:
        raise ReconstructionValidationError("I5.5B only structures the recognized pre-SfM ingestion rejection.")
    return {
        "input_images": 19, "registered_images": 0, "registration_ratio": 0.0,
        "full_input_scope_registered": False, "partial_registration_preserved": False,
        "registered_names": [], "registered_staged_names": [], "source_membership_names": source_names,
        "staged_membership_names": materialized, "lossless_staging_name_mapping_preserved": True,
        "quality_status": "rejected", "quality_assessment": {"status": "rejected", "usable": False, "reasons": [text]},
        "statistics": {"input_keyframes": 19, "registered_keyframes": 0, "landmarks": 0, "observations": 0, "mean_reprojection_error_px": None},
        "registration_attempted": False, "sfm_executed": False, "pre_sfm_ingestion_rejected": True,
        "rejection_stage": "ingestion", "rejection_reason": text, "mapping_produced_sparse_model": False,
        "sparse_component_count": 0,
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "per_scenario_gauge": "unavailable_no_sfm_model", "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False},
    }


def analyze_i5_5b_mapping_no_sparse_model_rejection(*, scenario_plan: Mapping[str, Any], staged_names: Sequence[str], reason: str) -> dict[str, Any]:
    source_names, materialized, text = _i55b_rejection_common(scenario_plan=scenario_plan, staged_names=staged_names, reason=reason)
    if "PyCOLMAP incremental mapping produced no sparse model" not in text:
        raise ReconstructionValidationError("I5.5B only structures the recognized incremental-mapping no-model rejection.")
    return {
        "input_images": 19, "registered_images": 0, "registration_ratio": 0.0,
        "full_input_scope_registered": False, "partial_registration_preserved": False,
        "registered_names": [], "registered_staged_names": [], "source_membership_names": source_names,
        "staged_membership_names": materialized, "lossless_staging_name_mapping_preserved": True,
        "quality_status": "rejected", "quality_assessment": {"status": "rejected", "usable": False, "reasons": [text]},
        "statistics": {"input_keyframes": 19, "registered_keyframes": 0, "landmarks": 0, "observations": 0, "mean_reprojection_error_px": None},
        "registration_attempted": True, "sfm_executed": True, "pre_sfm_ingestion_rejected": False,
        "rejection_stage": "incremental_mapping", "rejection_reason": text, "mapping_produced_sparse_model": False,
        "sparse_component_count": 0,
        "authority": {"camera_pose_authority_owner": "pycolmap_sfm", "per_scenario_gauge": "unavailable_no_sfm_model", "metric_scale_claimed": False, "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False},
    }


def _i55b_metric(analysis: Mapping[str, Any], key: str) -> float | int | None:
    stats = _mapping(analysis.get("statistics"), label="I5.5B scenario statistics")
    if key == "mean_track_length":
        quality = _mapping(analysis.get("quality_assessment"), label="I5.5B scenario quality")
        value = quality.get("mean_track_length")
    else:
        value = stats.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ReconstructionValidationError(f"I5.5B scenario metric {key} is invalid.")
    return value


def _i55b_ratio(stress: float | int | None, control: float | int | None) -> float | None:
    if stress is None or control is None or float(control) == 0.0:
        return None
    return float(stress) / float(control)


def compare_i5_5b_sfm_outcomes(
    *, control: Mapping[str, Any], dynamic_proxy: Mapping[str, Any], materialization_revalidation: Mapping[str, Any]
) -> dict[str, Any]:
    c = deepcopy(dict(control)); s = deepcopy(dict(dynamic_proxy)); m = deepcopy(dict(materialization_revalidation))
    for node, label in ((c, "control"), (s, "dynamic_proxy")):
        authority = _mapping(node.get("authority"), label=f"I5.5B {label} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.5B {label} authority")
        status = str(node.get("quality_status", ""))
        if status not in {"accepted", "degraded", "rejected"}:
            raise ReconstructionValidationError(f"I5.5B {label} quality status is invalid.")
        components = node.get("sparse_component_count")
        if not isinstance(components, int) or components < 0:
            raise ReconstructionValidationError(f"I5.5B {label} sparse component count is invalid.")
    required_materialization = (
        "same_decoded_source_pixels_for_control_and_stress", "control_pixel_values_preserved",
        "donor_patch_pixel_identity_preserved", "target_centroid_trajectory_matches_frozen_recipe",
        "changed_pixels_confined_to_frozen_target_boxes", "source_dimensions_constant",
        "controlled_independent_foreground_motion_proxy",
    )
    proxy_core_revalidated = all(m.get(key) is True for key in required_materialization)
    proxy_feature_bearing = m.get("dynamic_region_feature_bearing_revalidated") is True
    proxy_revalidated = bool(proxy_core_revalidated and proxy_feature_bearing)
    c_registered = int(c.get("registered_images", -1)); s_registered = int(s.get("registered_images", -1))
    if c_registered < 0 or s_registered < 0:
        raise ReconstructionValidationError("I5.5B registration counts must be non-negative.")
    c_landmarks = _i55b_metric(c, "landmarks"); s_landmarks = _i55b_metric(s, "landmarks")
    c_obs = _i55b_metric(c, "observations"); s_obs = _i55b_metric(s, "observations")
    c_repr = _i55b_metric(c, "mean_reprojection_error_px"); s_repr = _i55b_metric(s, "mean_reprojection_error_px")
    c_track = _i55b_metric(c, "mean_track_length"); s_track = _i55b_metric(s, "mean_track_length")
    return {
        "control_quality_status": str(c["quality_status"]),
        "dynamic_proxy_quality_status": str(s["quality_status"]),
        "controlled_dynamic_proxy_revalidated": proxy_revalidated,
        "dynamic_region_feature_bearing_revalidated": proxy_feature_bearing,
        "registration_delta": s_registered - c_registered,
        "registration_ratio_delta": float(s.get("registration_ratio", 0.0)) - float(c.get("registration_ratio", 0.0)),
        "landmarks_control": c_landmarks, "landmarks_dynamic_proxy": s_landmarks,
        "landmarks_stress_to_control_ratio": _i55b_ratio(s_landmarks, c_landmarks),
        "observations_control": c_obs, "observations_dynamic_proxy": s_obs,
        "observations_stress_to_control_ratio": _i55b_ratio(s_obs, c_obs),
        "mean_reprojection_error_px_control": c_repr, "mean_reprojection_error_px_dynamic_proxy": s_repr,
        "mean_reprojection_error_px_delta": None if c_repr is None or s_repr is None else float(s_repr) - float(c_repr),
        "mean_track_length_control": c_track, "mean_track_length_dynamic_proxy": s_track,
        "mean_track_length_delta": None if c_track is None or s_track is None else float(s_track) - float(c_track),
        "sparse_component_count_control": int(c["sparse_component_count"]),
        "sparse_component_count_dynamic_proxy": int(s["sparse_component_count"]),
        "sparse_component_count_delta": int(s["sparse_component_count"]) - int(c["sparse_component_count"]),
        "cross_run_absolute_geometry_comparison_permitted": False,
        "cross_run_absolute_translation_comparison_permitted": False,
        "per_scenario_sfm_gauge_remains_arbitrary": True,
        "natural_dynamic_scene_ground_truth": False,
        "threshold_relaxation_used": False,
    }


def build_i5_5b_dynamic_scene_replay(
    *,
    i5_5a_result: Mapping[str, Any],
    i5_5a_binding: I51EvidenceBinding,
    replay_receipt: Mapping[str, Any],
    replay_receipt_binding: I51EvidenceBinding,
    materialization_revalidation: Mapping[str, Any],
    control_analysis: Mapping[str, Any],
    dynamic_proxy_analysis: Mapping[str, Any],
    paired_comparison: Mapping[str, Any],
    source_reverified_before: bool,
    source_reverified_after: bool,
    staged_inputs_reverified: bool,
) -> dict[str, Any]:
    admission = validate_i5_5a_for_i5_5b(i5_5a_result)
    receipt = validate_i5_5b_receipt(replay_receipt)
    _validate_binding(i5_5a_binding, label="I5.5B I5.5A")
    _validate_binding(replay_receipt_binding, label="I5.5B receipt")
    if not (source_reverified_before and source_reverified_after and staged_inputs_reverified):
        raise ReconstructionValidationError("I5.5B requires source-before/source-after and staged-input revalidation.")
    control = deepcopy(dict(_mapping(control_analysis, label="I5.5B control analysis")))
    stress = deepcopy(dict(_mapping(dynamic_proxy_analysis, label="I5.5B dynamic analysis")))
    materialization = deepcopy(dict(_mapping(materialization_revalidation, label="I5.5B materialization revalidation")))
    comparison = compare_i5_5b_sfm_outcomes(control=control, dynamic_proxy=stress, materialization_revalidation=materialization)
    if deepcopy(dict(paired_comparison)) != comparison:
        raise ReconstructionValidationError("I5.5B paired comparison is inconsistent with scenario evidence.")
    for node, label in ((control, "control"), (stress, "dynamic proxy")):
        if int(node.get("input_images", -1)) != 19:
            raise ReconstructionValidationError(f"I5.5B {label} analysis must cover 19 inputs.")
        authority = _mapping(node.get("authority"), label=f"I5.5B {label} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.5B {label} authority")

    classification = _mapping(_mapping(admission["stress_plan"], label="I5.5A stress plan")["classification_policy"], label="I5.5A classification policy")
    control_status = str(control.get("quality_status")); stress_status = str(stress.get("quality_status"))
    proxy_ok = bool(comparison["controlled_dynamic_proxy_revalidated"])
    if control_status != "accepted":
        quality_status = "rejected"
        conclusion = str(classification["control_not_accepted"])
    elif not proxy_ok:
        quality_status = "degraded"
        conclusion = "inconclusive_controlled_dynamic_proxy_not_revalidated"
    elif stress_status == "accepted":
        quality_status = "pass"
        conclusion = str(classification["control_accepted_stress_accepted"])
    elif stress_status == "degraded":
        quality_status = "degraded"
        conclusion = str(classification["control_accepted_stress_degraded"])
    else:
        quality_status = "rejected"
        conclusion = str(classification["control_accepted_stress_rejected"])

    authority = deepcopy(dict(_mapping(admission.get("authority"), label="I5.5A authority")))
    recipe_sha = str(admission["stress_plan"]["dynamic_foreground_proxy"]["recipe_sha256"])
    return {
        "schema_version": I55B_SCHEMA_VERSION,
        "phase": I55B_PHASE,
        "status": "current_frozen_foundation_dynamic_scene_replayed",
        "pass": True,
        "quality_status": quality_status,
        "scope": {
            "current_frozen_foundation_real_replay": True,
            "paired_control_dynamic_proxy_replay": True,
            "stress_images_materialized": True,
            "control_replay_attempted": True,
            "dynamic_proxy_replay_attempted": True,
            "control_sfm_rerun": bool(control.get("sfm_executed")),
            "dynamic_proxy_sfm_rerun": bool(stress.get("sfm_executed")),
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
        "evidence_bindings": {"i5_5a_admission": i5_5a_binding.as_dict(), "i5_5b_replay_receipt": replay_receipt_binding.as_dict()},
        "provenance": {
            "frozen_source_asset_set_sha256": admission["source_identity"]["asset_set_sha256"],
            "frozen_dynamic_proxy_recipe_sha256": recipe_sha,
            "source_reverified_before_replay": True,
            "source_reverified_after_replay": True,
            "staged_inputs_reverified_after_replay": True,
            "same_matching_policy_control_and_stress": "exhaustive",
            "same_max_keyframes_control_and_stress": 19,
            "same_production_native_sfm_runner_control_and_stress": True,
            "result_driven_recipe_tuning": False,
            "retry_until_quality_pass": False,
            "historical_result_overwritten": False,
        },
        "materialization_revalidation": materialization,
        "scenarios": {"control": control, "dynamic_proxy": stress},
        "paired_comparison": comparison,
        "qualification": {
            "quality_status": quality_status,
            "dynamic_scene_robustness_conclusion": conclusion,
            "controlled_dynamic_foreground_proxy_revalidated": proxy_ok,
            "control_quality_status": control_status,
            "dynamic_proxy_quality_status": stress_status,
            "natural_dynamic_scene_ground_truth": False,
            "real_world_dynamic_scene_robustness_claimed": False,
            "universal_dynamic_scene_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
            "metric_scale_claimed": False,
        },
        "regression_summary": {
            "focused_i5_5b_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "all_required_suites_passed": True,
            "test_counts_are_not_runtime_quality_scores": True,
        },
        "authority": authority,
        "decision": {
            "i4_closed": True, "i5_authorized": True, "i5_1_complete": True, "i5_2_complete": True,
            "i5_3_complete": True, "i5_4_complete": True, "i5_5_started": True, "i5_5a_complete": True,
            "i5_5b_complete": True, "i5_5c_qualification_closure_authorized": True, "i5_5c_complete": False,
            "i5_5_complete": False, "i5_6_authorized": False, "provider_specific_robustness_started": False,
            "mp_sfm_authorized": False, "provider_selected": False, "provider_promoted": False,
            "provider_ranking_created": False, "universal_provider_winner_declared": False,
            "global_scalar_score_created": False, "auto_policy_implemented": False, "hybrid_authorized": False,
            "threshold_relaxation_authorized": False, "production_default_modified": False, "metric_scale_claimed": False,
            "measurement_authoritative": False, "certified_metrology": False, "surface_mesh_generated": False,
        },
        "roadmap_guard": {
            "i5_5c_qualification_closure_authorized_not_started": True,
            "i5_6_mvs_robustness_not_authorized": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.5C dynamic-scene handling robustness qualification and closure.",
    }


# ---------------------------------------------------------------------------
# I5.5C - dynamic-scene handling robustness qualification + closure
# ---------------------------------------------------------------------------

I55C_SCHEMA_VERSION = 1
I55C_PHASE = "I5.5C"
I55C_RECEIPT_PHASE = "I5.5C_QUALIFICATION_RECEIPT"


def _i55c_registration_string(scenario: Mapping[str, Any], *, label: str) -> str:
    input_images = scenario.get("input_images")
    registered_images = scenario.get("registered_images")
    if not isinstance(input_images, int) or input_images <= 0:
        raise ReconstructionValidationError(f"I5.5C {label} input_images is invalid.")
    if not isinstance(registered_images, int) or registered_images < 0 or registered_images > input_images:
        raise ReconstructionValidationError(f"I5.5C {label} registered_images is invalid.")
    ratio = scenario.get("registration_ratio")
    expected = registered_images / input_images
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not isfinite(float(ratio)) or abs(float(ratio) - expected) > 1e-9:
        raise ReconstructionValidationError(f"I5.5C {label} registration ratio is inconsistent with image counts.")
    if scenario.get("full_input_scope_registered") is not (registered_images == input_images):
        raise ReconstructionValidationError(f"I5.5C {label} full-input registration flag is inconsistent.")
    if scenario.get("partial_registration_preserved") is not (0 < registered_images < input_images):
        raise ReconstructionValidationError(f"I5.5C {label} partial-registration flag is inconsistent.")
    return f"{registered_images}/{input_images}"


def validate_i5_5b_for_i5_5c(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact evidence-complete I5.5B replay being closed by I5.5C."""

    _require(payload, "schema_version", I55B_SCHEMA_VERSION, label="I5.5B replay")
    _require(payload, "phase", I55B_PHASE, label="I5.5B replay")
    _require(payload, "status", "current_frozen_foundation_dynamic_scene_replayed", label="I5.5B replay")
    _require(payload, "pass", True, label="I5.5B replay")
    quality_status = str(payload.get("quality_status", ""))
    if quality_status not in {"pass", "degraded", "rejected"}:
        raise ReconstructionValidationError("I5.5C I5.5B quality status is invalid.")

    scope = _mapping(payload.get("scope"), label="I5.5B scope")
    for key in (
        "current_frozen_foundation_real_replay", "paired_control_dynamic_proxy_replay",
        "stress_images_materialized", "control_replay_attempted", "dynamic_proxy_replay_attempted",
    ):
        _require(scope, key, True, label="I5.5B scope")
    for key in ("control_sfm_rerun", "dynamic_proxy_sfm_rerun"):
        if not isinstance(scope.get(key), bool):
            raise ReconstructionValidationError(f"I5.5B scope {key} must be boolean.")
    for key in (
        "dense_provider_rerun", "dynamic_masking_added", "semantic_segmentation_added",
        "motion_outlier_filter_added", "provider_specific_robustness_started",
        "mp_sfm_strategy_evaluation_started", "thresholds_modified", "production_default_modified",
        "auto_policy_work_started", "hybrid_experiment_started",
    ):
        _require(scope, key, False, label="I5.5B scope")

    bindings = _mapping(payload.get("evidence_bindings"), label="I5.5B evidence_bindings")
    a_binding = _mapping(bindings.get("i5_5a_admission"), label="I5.5B I5.5A binding")
    if not _valid_sha256(a_binding.get("sha256")):
        raise ReconstructionValidationError("I5.5C I5.5B predecessor I5.5A SHA256 binding is malformed.")

    provenance = _mapping(payload.get("provenance"), label="I5.5B provenance")
    for key in (
        "source_reverified_before_replay", "source_reverified_after_replay",
        "staged_inputs_reverified_after_replay", "same_production_native_sfm_runner_control_and_stress",
    ):
        _require(provenance, key, True, label="I5.5B provenance")
    _require(provenance, "same_matching_policy_control_and_stress", "exhaustive", label="I5.5B provenance")
    _require(provenance, "same_max_keyframes_control_and_stress", 19, label="I5.5B provenance")
    for key in ("result_driven_recipe_tuning", "retry_until_quality_pass", "historical_result_overwritten"):
        _require(provenance, key, False, label="I5.5B provenance")
    for key in ("frozen_source_asset_set_sha256", "frozen_dynamic_proxy_recipe_sha256"):
        if not _valid_sha256(provenance.get(key)):
            raise ReconstructionValidationError(f"I5.5C I5.5B provenance {key} is malformed.")

    materialization = _mapping(payload.get("materialization_revalidation"), label="I5.5B materialization revalidation")
    for key in (
        "same_decoded_source_pixels_for_control_and_stress", "control_pixel_values_preserved",
        "donor_patch_pixel_identity_preserved", "target_centroid_trajectory_matches_frozen_recipe",
        "changed_pixels_confined_to_frozen_target_boxes", "source_dimensions_constant",
        "controlled_independent_foreground_motion_proxy",
    ):
        _require(materialization, key, True, label="I5.5B materialization revalidation")
    _require(materialization, "natural_dynamic_scene_ground_truth", False, label="I5.5B materialization revalidation")

    scenarios = _mapping(payload.get("scenarios"), label="I5.5B scenarios")
    scenario_quality: dict[str, str] = {}
    registrations: dict[str, str] = {}
    for key in ("control", "dynamic_proxy"):
        scenario = _mapping(scenarios.get(key), label=f"I5.5B {key} scenario")
        registrations[key] = _i55c_registration_string(scenario, label=key)
        status = str(scenario.get("quality_status", ""))
        if status not in {"accepted", "degraded", "rejected"}:
            raise ReconstructionValidationError(f"I5.5C {key} scenario quality status is invalid.")
        scenario_quality[key] = status
        components = scenario.get("sparse_component_count")
        if not isinstance(components, int) or components < 0:
            raise ReconstructionValidationError(f"I5.5C {key} sparse component count is invalid.")
        authority = _mapping(scenario.get("authority"), label=f"I5.5B {key} authority")
        _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label=f"I5.5B {key} authority")
        _require(authority, "per_scenario_gauge", "arbitrary", label=f"I5.5B {key} authority")
        for flag in ("metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated"):
            _require(authority, flag, False, label=f"I5.5B {key} authority")

    comparison = _mapping(payload.get("paired_comparison"), label="I5.5B paired comparison")
    _require(comparison, "control_quality_status", scenario_quality["control"], label="I5.5B paired comparison")
    _require(comparison, "dynamic_proxy_quality_status", scenario_quality["dynamic_proxy"], label="I5.5B paired comparison")
    _require(comparison, "cross_run_absolute_geometry_comparison_permitted", False, label="I5.5B paired comparison")
    _require(comparison, "cross_run_absolute_translation_comparison_permitted", False, label="I5.5B paired comparison")
    _require(comparison, "per_scenario_sfm_gauge_remains_arbitrary", True, label="I5.5B paired comparison")
    _require(comparison, "natural_dynamic_scene_ground_truth", False, label="I5.5B paired comparison")
    _require(comparison, "threshold_relaxation_used", False, label="I5.5B paired comparison")
    if not isinstance(comparison.get("controlled_dynamic_proxy_revalidated"), bool):
        raise ReconstructionValidationError("I5.5C I5.5B controlled proxy revalidation flag is invalid.")

    qualification = _mapping(payload.get("qualification"), label="I5.5B qualification")
    _require(qualification, "quality_status", quality_status, label="I5.5B qualification")
    _require(qualification, "controlled_dynamic_foreground_proxy_revalidated", comparison["controlled_dynamic_proxy_revalidated"], label="I5.5B qualification")
    _require(qualification, "control_quality_status", scenario_quality["control"], label="I5.5B qualification")
    _require(qualification, "dynamic_proxy_quality_status", scenario_quality["dynamic_proxy"], label="I5.5B qualification")
    for key in (
        "natural_dynamic_scene_ground_truth", "real_world_dynamic_scene_robustness_claimed",
        "universal_dynamic_scene_robustness_claimed", "dense_reconstruction_robustness_claimed",
        "provider_robustness_claimed", "metric_scale_claimed",
    ):
        _require(qualification, key, False, label="I5.5B qualification")
    conclusion = str(qualification.get("dynamic_scene_robustness_conclusion", ""))
    if quality_status == "pass":
        if scenario_quality != {"control": "accepted", "dynamic_proxy": "accepted"}:
            raise ReconstructionValidationError("I5.5C PASS replay must retain accepted control and dynamic-proxy quality states.")
        _require(comparison, "controlled_dynamic_proxy_revalidated", True, label="I5.5B paired comparison")
        if conclusion != "pass_on_admitted_controlled_dynamic_foreground_proxy":
            raise ReconstructionValidationError("I5.5C PASS replay conclusion drifted.")
    elif quality_status == "degraded":
        if conclusion not in {
            "degraded_on_admitted_controlled_dynamic_foreground_proxy",
            "inconclusive_controlled_dynamic_proxy_not_revalidated",
        }:
            raise ReconstructionValidationError("I5.5C degraded replay conclusion drifted.")
    else:
        if conclusion not in {
            "rejected_on_admitted_controlled_dynamic_foreground_proxy",
            "scenario_control_not_qualified_for_dynamic_robustness_conclusion",
        }:
            raise ReconstructionValidationError("I5.5C rejected replay conclusion drifted.")

    authority = _mapping(payload.get("authority"), label="I5.5B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="I5.5B authority")
    _require(authority, "permanent_global_view_cap", None, label="I5.5B authority")
    for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ):
        _require(authority, key, False, label="I5.5B authority")

    decision = _mapping(payload.get("decision"), label="I5.5B decision")
    for key in (
        "i4_closed", "i5_authorized", "i5_1_complete", "i5_2_complete", "i5_3_complete", "i5_4_complete",
        "i5_5_started", "i5_5a_complete", "i5_5b_complete", "i5_5c_qualification_closure_authorized",
    ):
        _require(decision, key, True, label="I5.5B decision")
    for key in (
        "i5_5c_complete", "i5_5_complete", "i5_6_authorized", "provider_specific_robustness_started",
        "mp_sfm_authorized", "provider_selected", "provider_promoted", "provider_ranking_created",
        "universal_provider_winner_declared", "global_scalar_score_created", "auto_policy_implemented",
        "hybrid_authorized", "threshold_relaxation_authorized", "production_default_modified",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        _require(decision, key, False, label="I5.5B decision")
    if payload.get("next") != "I5.5C dynamic-scene handling robustness qualification and closure.":
        raise ReconstructionValidationError("I5.5C predecessor next-boundary contract drifted.")
    return deepcopy(dict(payload))


def validate_i5_5c_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload, "schema_version", I55C_SCHEMA_VERSION, label="I5.5C receipt")
    _require(payload, "phase", I55C_RECEIPT_PHASE, label="I5.5C receipt")
    _require(payload, "status", "pass", label="I5.5C receipt")
    _require(payload, "pass", True, label="I5.5C receipt")
    scope = _mapping(payload.get("scope"), label="I5.5C receipt scope")
    for key in (
        "focused_i5_5c_tests", "predecessor_i5_5b_regression", "predecessor_i5_5a_regression",
        "evidence_only_closure_contract_regression", "dynamic_quality_penalty_preservation_regression",
        "authority_guard_regression", "import_probe", "runner_help_probe",
    ):
        _require(scope, key, True, label="I5.5C receipt scope")
    for key in (
        "sfm_rerun", "source_rematerialized", "stress_images_rematerialized", "dense_provider_rerun",
        "dynamic_masking_added", "semantic_segmentation_added", "motion_outlier_filter_added",
        "thresholds_modified", "production_default_modified", "mp_sfm_started",
        "auto_policy_implemented", "hybrid_authorized",
    ):
        _require(scope, key, False, label="I5.5C receipt scope")
    counts = _mapping(payload.get("counts"), label="I5.5C receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(value, int) and value >= 0 for value in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.5C receipt test counts must be non-negative integers.")
    if focused < 15 or predecessor < 40 or combined < 70 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.5C receipt coverage is below the required closure floor.")
    rcs = _mapping(payload.get("return_codes"), label="I5.5C receipt return_codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.5C receipt {key} return code is not zero.")
    return deepcopy(dict(payload))


def _i55c_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    if isinstance(numerator, bool) or isinstance(denominator, bool):
        return None
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if not isfinite(float(numerator)) or not isfinite(float(denominator)) or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _i55c_directional_quality_evidence(comparison: Mapping[str, Any]) -> dict[str, Any]:
    c_landmarks = comparison.get("landmarks_control")
    s_landmarks = comparison.get("landmarks_dynamic_proxy")
    c_obs = comparison.get("observations_control")
    s_obs = comparison.get("observations_dynamic_proxy")
    c_repr = comparison.get("mean_reprojection_error_px_control")
    s_repr = comparison.get("mean_reprojection_error_px_dynamic_proxy")
    c_track = comparison.get("mean_track_length_control")
    s_track = comparison.get("mean_track_length_dynamic_proxy")

    def less(a: Any, b: Any) -> bool:
        return isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool) and float(a) < float(b)

    def greater(a: Any, b: Any) -> bool:
        return isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool) and float(a) > float(b)

    landmark_reduced = less(s_landmarks, c_landmarks)
    observation_reduced = less(s_obs, c_obs)
    reprojection_increased = greater(s_repr, c_repr)
    track_reduced = less(s_track, c_track)
    return {
        "landmark_support_reduced": landmark_reduced,
        "observation_support_reduced": observation_reduced,
        "reprojection_error_increased": reprojection_increased,
        "mean_track_length_reduced": track_reduced,
        "directional_sparse_quality_penalty_observed": bool(
            landmark_reduced or observation_reduced or reprojection_increased or track_reduced
        ),
        "landmark_retention_ratio": _i55c_ratio(s_landmarks, c_landmarks),
        "observation_retention_ratio": _i55c_ratio(s_obs, c_obs),
        "reprojection_error_stress_to_control_ratio": _i55c_ratio(s_repr, c_repr),
        "mean_track_length_stress_to_control_ratio": _i55c_ratio(s_track, c_track),
        "classification_threshold_used": None,
        "directional_comparison_only": True,
        "cross_run_absolute_geometry_comparison_permitted": False,
    }


def build_i5_5c_dynamic_scene_closure(
    *,
    i5_5a_result: Mapping[str, Any],
    i5_5a_binding: I51EvidenceBinding,
    i5_5b_result: Mapping[str, Any],
    i5_5b_binding: I51EvidenceBinding,
    qualification_receipt: Mapping[str, Any],
    qualification_receipt_binding: I51EvidenceBinding,
) -> dict[str, Any]:
    admission = validate_i5_5a_for_i5_5b(i5_5a_result)
    replay = validate_i5_5b_for_i5_5c(i5_5b_result)
    receipt = validate_i5_5c_receipt(qualification_receipt)
    for binding, label in (
        (i5_5a_binding, "I5.5C I5.5A"),
        (i5_5b_binding, "I5.5C I5.5B"),
        (qualification_receipt_binding, "I5.5C receipt"),
    ):
        _validate_binding(binding, label=label)

    replay_bindings = _mapping(replay.get("evidence_bindings"), label="I5.5B evidence_bindings")
    replay_a_binding = _mapping(replay_bindings.get("i5_5a_admission"), label="I5.5B I5.5A binding")
    if str(replay_a_binding.get("sha256", "")).lower() != i5_5a_binding.sha256.lower():
        raise ReconstructionValidationError("I5.5C I5.5B predecessor SHA does not match the independently verified I5.5A artifact.")
    if str(replay["provenance"]["frozen_source_asset_set_sha256"]).lower() != str(admission["source_identity"]["asset_set_sha256"]).lower():
        raise ReconstructionValidationError("I5.5C I5.5A/I5.5B source asset identity drifted.")
    if str(replay["provenance"]["frozen_dynamic_proxy_recipe_sha256"]).lower() != str(admission["stress_plan"]["dynamic_foreground_proxy"]["recipe_sha256"]).lower():
        raise ReconstructionValidationError("I5.5C I5.5A/I5.5B dynamic proxy recipe identity drifted.")
    if replay["authority"]["camera_pose_authority_owner"] != admission["authority"]["camera_pose_authority_owner"]:
        raise ReconstructionValidationError("I5.5C authority drifted between I5.5A and I5.5B.")

    comparison = deepcopy(dict(_mapping(replay.get("paired_comparison"), label="I5.5B paired comparison")))
    directional = _i55c_directional_quality_evidence(comparison)
    quality_status = str(replay["quality_status"])
    conclusion = str(replay["qualification"]["dynamic_scene_robustness_conclusion"])
    if quality_status == "pass":
        operating_envelope = (
            "qualified_on_admitted_controlled_dynamic_foreground_proxy_with_observed_sparse_quality_penalty"
            if directional["directional_sparse_quality_penalty_observed"]
            else "qualified_on_admitted_controlled_dynamic_foreground_proxy"
        )
    elif quality_status == "degraded":
        operating_envelope = (
            "inconclusive_on_admitted_controlled_dynamic_foreground_proxy"
            if conclusion == "inconclusive_controlled_dynamic_proxy_not_revalidated"
            else "degraded_on_admitted_controlled_dynamic_foreground_proxy"
        )
    else:
        operating_envelope = "rejected_on_admitted_controlled_dynamic_foreground_proxy"

    scenarios = _mapping(replay.get("scenarios"), label="I5.5B scenarios")
    control = _mapping(scenarios.get("control"), label="I5.5B control")
    stress = _mapping(scenarios.get("dynamic_proxy"), label="I5.5B dynamic proxy")
    authority = deepcopy(dict(_mapping(replay.get("authority"), label="I5.5B authority")))

    return {
        "schema_version": I55C_SCHEMA_VERSION,
        "phase": I55C_PHASE,
        "status": "dynamic_scene_handling_robustness_qualified_and_closed",
        "pass": True,
        "quality_status": quality_status,
        "scope": {
            "evidence_only_qualification_and_closure": True,
            "i5_5a_admission_reused": True,
            "i5_5b_real_replay_reused": True,
            "sfm_rerun": False,
            "source_rematerialized": False,
            "stress_images_rematerialized": False,
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
            "i5_5a_admission": i5_5a_binding.as_dict(),
            "i5_5b_real_replay": i5_5b_binding.as_dict(),
            "i5_5c_qualification_receipt": qualification_receipt_binding.as_dict(),
        },
        "provenance": {
            "frozen_source_asset_set_sha256": replay["provenance"]["frozen_source_asset_set_sha256"],
            "frozen_dynamic_proxy_recipe_sha256": replay["provenance"]["frozen_dynamic_proxy_recipe_sha256"],
            "i5_5b_quality_status_preserved": True,
            "i5_5b_dynamic_scene_conclusion_preserved": True,
            "historical_result_overwritten": False,
            "result_driven_recipe_tuning": False,
            "retry_until_quality_pass": False,
        },
        "qualification": {
            "quality_status": quality_status,
            "operating_envelope": operating_envelope,
            "dynamic_scene_robustness_conclusion": conclusion,
            "control_registration": _i55c_registration_string(control, label="control"),
            "dynamic_proxy_registration": _i55c_registration_string(stress, label="dynamic_proxy"),
            "control_sparse_component_count": int(control["sparse_component_count"]),
            "dynamic_proxy_sparse_component_count": int(stress["sparse_component_count"]),
            "controlled_dynamic_foreground_proxy_revalidated": bool(replay["qualification"]["controlled_dynamic_foreground_proxy_revalidated"]),
            "natural_dynamic_scene_ground_truth": False,
            "real_world_dynamic_scene_robustness_claimed": False,
            "universal_dynamic_scene_robustness_claimed": False,
            "dense_reconstruction_robustness_claimed": False,
            "provider_robustness_claimed": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "directional_quality_evidence": directional,
        "paired_comparison": comparison,
        "materialization_revalidation": deepcopy(dict(_mapping(replay.get("materialization_revalidation"), label="I5.5B materialization revalidation"))),
        "regression_summary": {
            "focused_i5_5c_tests": receipt["counts"]["focused_tests"],
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
            "i5_5b_complete": True,
            "i5_5c_complete": True,
            "i5_5_complete": True,
            "i5_6_authorized": True,
            "i5_6_started": False,
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
            "i5_6_mvs_robustness_authorized_not_started": True,
            "provider_specific_robustness_not_started": True,
            "mp_sfm_deferred_to_i5_10": True,
            "hybrid_deferred_to_i5_12_if_justified": True,
            "auto_policy_deferred_to_i5_13": True,
        },
        "next": "I5.6 MVS robustness envelope.",
    }


__all__ = [
    "I55A_SCHEMA_VERSION", "I55A_PHASE", "I55A_RECEIPT_PHASE", "I55AAdmissionPolicy",
    "validate_i5_4c_for_i5_5", "build_i5_5a_source_identity", "validate_i5_5a_source_identity",
    "build_i5_5a_stress_plan", "validate_i5_5a_receipt", "build_i5_5a_dynamic_scene_admission",
    "I55B_SCHEMA_VERSION", "I55B_PHASE", "I55B_RECEIPT_PHASE", "validate_i5_5a_for_i5_5b",
    "validate_i5_5b_receipt", "materialize_i5_5b_pair", "analyze_i5_5b_materialized_pair",
    "analyze_i5_5b_sfm_report", "analyze_i5_5b_pre_sfm_ingestion_rejection",
    "analyze_i5_5b_mapping_no_sparse_model_rejection", "compare_i5_5b_sfm_outcomes",
    "build_i5_5b_dynamic_scene_replay",
    "I55C_SCHEMA_VERSION", "I55C_PHASE", "I55C_RECEIPT_PHASE",
    "validate_i5_5b_for_i5_5c", "validate_i5_5c_receipt", "build_i5_5c_dynamic_scene_closure",
]
