from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_forward_motion_kitti import (
    I54AAdmissionPolicy,
    I54A_PHASE,
    build_i5_4a_forward_motion_kitti_admission,
    build_i5_4a_source_identity,
    build_i5_4a_stress_plan,
    validate_i5_3c_for_i5_4,
    validate_i5_4a_receipt,
)
from tests.i5_4a_helpers import (
    I53C_SHA,
    RECEIPT_54A_SHA,
    binding,
    clone,
    kitti_source_dir,
    passing_i5_3c,
    passing_i5_4a_receipt,
)


def _build(tmp_path: Path, *, confirmed: bool = True, count: int = 80):
    source = build_i5_4a_source_identity(kitti_source_dir(tmp_path, count=count))
    return build_i5_4a_forward_motion_kitti_admission(
        i5_3c_result=passing_i5_3c(tmp_path),
        i5_3c_binding=binding("/evidence/i5_3c.json", I53C_SHA),
        source_identity=source,
        source_label="KITTI odometry sequence 00 left camera",
        kitti_style_forward_motion_confirmation=confirmed,
        admission_receipt=passing_i5_4a_receipt(),
        admission_receipt_binding=binding("/evidence/i5_4a_receipt.json", RECEIPT_54A_SHA),
    )


def test_i54a_accepts_degraded_i53c_and_i54_authorization(tmp_path: Path) -> None:
    closure = validate_i5_3c_for_i5_4(passing_i5_3c(tmp_path))
    assert closure["qualification"]["quality_status"] == "degraded"
    assert closure["decision"]["i5_4_authorized"] is True
    assert closure["decision"]["i5_4_started"] is False


def test_i54a_source_identity_uses_natural_order_and_hash(tmp_path: Path) -> None:
    identity = build_i5_4a_source_identity(kitti_source_dir(tmp_path))
    assert identity["image_count"] == 80
    assert identity["ordered_names"][0] == "000000.png"
    assert identity["ordered_names"][-1] == "000079.png"
    assert len(identity["asset_set_sha256"]) == 64


def test_i54a_source_requires_minimum_images(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        build_i5_4a_source_identity(kitti_source_dir(tmp_path, count=59))


def test_i54a_plan_freezes_first_contiguous_sixty_frames(tmp_path: Path) -> None:
    identity = build_i5_4a_source_identity(kitti_source_dir(tmp_path))
    plan = build_i5_4a_stress_plan(identity)
    native = plan["forward_motion_window"]
    assert native["indices"] == list(range(60))
    assert native["image_count"] == 60
    assert native["source_stride"] == 1
    assert native["contiguous"] is True
    assert native["forward_motion_semantics_authoritative"] is False
    assert native["i5_4b_geometry_revalidation_required"] is True


def test_i54a_plan_freezes_geometry_revalidation_contract(tmp_path: Path) -> None:
    plan = build_i5_4a_stress_plan(build_i5_4a_source_identity(kitti_source_dir(tmp_path)))
    revalidation = plan["forward_motion_window"]["required_i5_4b_revalidation"]
    assert revalidation["camera_motion"] == "pycolmap_consecutive_camera_center_motion_in_camera_coordinates"
    assert revalidation["forward_axis_dominance"] == "median_forward_component_fraction_must_exceed_median_lateral_component_fraction"
    assert revalidation["triangulation_health"] == "report_i2_4_parallax_and_baseline_to_range_without_absolute_scale_claim"
    assert revalidation["result_driven_window_tuning"] is False


def test_i54a_policy_is_strict_and_result_independent() -> None:
    with pytest.raises(ReconstructionValidationError):
        I54AAdmissionPolicy(admitted_window_images=20)
    with pytest.raises(ReconstructionValidationError):
        I54AAdmissionPolicy(source_stride=0)
    assert I54AAdmissionPolicy().admitted_window_images == 60


def test_i54a_requires_operator_semantic_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        _build(tmp_path, confirmed=False)


def test_i54a_receipt_requires_execution_free_gate() -> None:
    receipt = validate_i5_4a_receipt(passing_i5_4a_receipt())
    assert receipt["scope"]["sfm_rerun"] is False
    bad = clone(passing_i5_4a_receipt())
    bad["scope"]["sfm_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_4a_receipt(bad)


def test_i54a_builds_admission_and_authorizes_only_i54b(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["phase"] == I54A_PHASE == "I5.4A"
    assert result["pass"] is True
    assert result["status"] == "forward_motion_kitti_scenario_admitted"
    assert result["decision"]["i5_4_started"] is True
    assert result["decision"]["i5_4a_complete"] is True
    assert result["decision"]["i5_4b_real_replay_authorized"] is True
    assert result["decision"]["i5_4_complete"] is False
    assert result["decision"]["i5_5_authorized"] is False
    assert result["next"] == "I5.4B current frozen-foundation forward-motion / KITTI replay."


def test_i54a_does_not_claim_forward_motion_before_real_geometry(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["qualification"]["operator_kitti_style_forward_motion_confirmation"] is True
    assert result["qualification"]["authoritative_forward_motion_geometry_proven"] is False
    assert result["qualification"]["universal_forward_motion_robustness_claimed"] is False


def test_i54a_preserves_authority_and_defaults(tmp_path: Path) -> None:
    result = _build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["scope"]["thresholds_modified"] is False
    assert result["scope"]["production_default_modified"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False
