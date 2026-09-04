from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_dynamic_scene_handling import (
    I55AAdmissionPolicy,
    I55A_PHASE,
    build_i5_5a_dynamic_scene_admission,
    build_i5_5a_source_identity,
    build_i5_5a_stress_plan,
    validate_i5_4c_for_i5_5,
    validate_i5_5a_receipt,
)
from tests.i5_5a_helpers import (
    I54C_SHA,
    RECEIPT_55A_SHA,
    binding,
    castle_source_dir,
    clone,
    passing_i5_4c,
    passing_i5_5a_receipt,
)


def _build(tmp_path: Path, *, confirmed: bool = True):
    source = build_i5_5a_source_identity(castle_source_dir(tmp_path))
    return build_i5_5a_dynamic_scene_admission(
        i5_4c_result=passing_i5_4c(),
        i5_4c_binding=binding("/evidence/i5_4c.json", I54C_SHA),
        source_identity=source,
        source_label="Castle-P19 static multiview control",
        static_control_confirmation=confirmed,
        admission_receipt=passing_i5_5a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_5a_receipt.json", RECEIPT_55A_SHA),
    )


def test_i55a_accepts_negative_i54_closure_and_i55_authorization() -> None:
    closure = validate_i5_4c_for_i5_5(passing_i5_4c())
    assert closure["quality_status"] == "rejected"
    assert closure["decision"]["i5_4_complete"] is True
    assert closure["decision"]["i5_5_authorized"] is True
    assert closure["decision"]["i5_5_started"] is False


def test_i55a_rejects_incomplete_i54_predecessor() -> None:
    bad = clone(passing_i5_4c())
    bad["decision"]["i5_4_complete"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4c_for_i5_5(bad)


def test_i55a_source_identity_uses_natural_order_and_ignores_metadata(tmp_path: Path) -> None:
    identity = build_i5_5a_source_identity(castle_source_dir(tmp_path))
    assert identity["image_count"] == 19
    assert identity["ordered_names"][0] == "0000.jpg"
    assert identity["ordered_names"][-1] == "0018.jpg"
    assert "K.txt" not in identity["ordered_names"]
    assert len(identity["asset_set_sha256"]) == 64


def test_i55a_source_requires_nineteen_images(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        build_i5_5a_source_identity(castle_source_dir(tmp_path, count=18))


def test_i55a_stress_plan_pairs_same_source_scope(tmp_path: Path) -> None:
    plan = build_i5_5a_stress_plan(build_i5_5a_source_identity(castle_source_dir(tmp_path)))
    control = plan["static_control"]
    stress = plan["dynamic_foreground_proxy"]
    assert control["indices"] == list(range(19))
    assert stress["indices"] == list(range(19))
    assert control["names"] == stress["names"]
    assert control["image_count"] == stress["image_count"] == 19
    assert control["matching_policy"] == stress["matching_policy"] == "exhaustive"


def test_i55a_dynamic_proxy_is_independent_motion_by_construction(tmp_path: Path) -> None:
    stress = build_i5_5a_stress_plan(build_i5_5a_source_identity(castle_source_dir(tmp_path)))["dynamic_foreground_proxy"]
    recipe = stress["materialization_recipe"]
    assert stress["natural_dynamic_scene_ground_truth"] is False
    assert stress["controlled_independent_foreground_motion_proxy"] is True
    assert recipe["donor_source_index"] == 9
    assert recipe["trajectory"] == "linear_left_to_right_in_native_source_order"
    assert recipe["paste_mode"] == "opaque"
    assert recipe["output_encoding"] == "png_lossless"


def test_i55a_dynamic_proxy_trajectory_is_frozen_and_in_bounds(tmp_path: Path) -> None:
    stress = build_i5_5a_stress_plan(build_i5_5a_source_identity(castle_source_dir(tmp_path)))["dynamic_foreground_proxy"]
    boxes = stress["target_boxes_normalized"]
    assert len(boxes) == 19
    assert boxes[0]["x"] == pytest.approx(0.03)
    assert boxes[-1]["x"] == pytest.approx(0.73)
    assert all(0.0 <= b["x"] and b["x"] + b["width"] <= 1.0 for b in boxes)
    assert all(0.0 <= b["y"] and b["y"] + b["height"] <= 1.0 for b in boxes)
    assert stress["horizontal_displacement_fraction"] == pytest.approx(0.70)


def test_i55a_recipe_hash_is_deterministic(tmp_path: Path) -> None:
    identity = build_i5_5a_source_identity(castle_source_dir(tmp_path))
    a = build_i5_5a_stress_plan(identity)["dynamic_foreground_proxy"]["recipe_sha256"]
    b = build_i5_5a_stress_plan(identity)["dynamic_foreground_proxy"]["recipe_sha256"]
    assert a == b
    assert len(a) == 64


def test_i55a_policy_rejects_out_of_bounds_patch() -> None:
    with pytest.raises(ReconstructionValidationError):
        I55AAdmissionPolicy(patch_width_fraction=0.40, target_x_end_fraction=0.80)
    with pytest.raises(ReconstructionValidationError):
        I55AAdmissionPolicy(donor_source_index=19)


def test_i55a_requires_static_control_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        _build(tmp_path, confirmed=False)


def test_i55a_receipt_requires_execution_free_admission() -> None:
    receipt = validate_i5_5a_receipt(passing_i5_5a_receipt())
    assert receipt["scope"]["stress_images_materialized"] is False
    bad = clone(passing_i5_5a_receipt())
    bad["scope"]["stress_images_materialized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_5a_receipt(bad)


def test_i55a_builds_admission_and_authorizes_only_i55b(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I55A_PHASE == "I5.5A"
    assert result["pass"] is True
    assert result["status"] == "controlled_dynamic_scene_scenario_admitted"
    assert result["decision"]["i5_5_started"] is True
    assert result["decision"]["i5_5a_complete"] is True
    assert result["decision"]["i5_5b_real_replay_authorized"] is True
    assert result["decision"]["i5_5_complete"] is False
    assert result["decision"]["i5_6_authorized"] is False
    assert result["next"] == "I5.5B current frozen-foundation dynamic-scene replay."


def test_i55a_does_not_claim_natural_dynamic_scene_robustness(tmp_path: Path) -> None:
    result = _build(tmp_path)
    q = result["qualification"]
    assert q["controlled_dynamic_foreground_proxy_admitted"] is True
    assert q["natural_dynamic_scene_ground_truth"] is False
    assert q["universal_dynamic_scene_robustness_claimed"] is False
    assert q["real_world_dynamic_scene_robustness_claimed"] is False


def test_i55a_preserves_authority_and_does_not_add_dynamic_masking(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["scope"]["dynamic_masking_added"] is False
    assert result["scope"]["semantic_segmentation_added"] is False
    assert result["scope"]["thresholds_modified"] is False
    assert result["scope"]["production_default_modified"] is False
