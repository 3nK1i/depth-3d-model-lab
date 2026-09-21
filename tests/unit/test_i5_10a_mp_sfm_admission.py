from __future__ import annotations

from copy import deepcopy

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mp_sfm_evaluation import (
    I510A_I59C_FROZEN_SHA256,
    I510A_PHASE,
    I510A_PRIMARY_SCENARIO,
    I510A_SUPPORTED_MATCHING_STRATEGIES,
    I510AAdmissionPolicy,
    build_i5_10a_mp_sfm_admission,
    repository_mp_sfm_inventory,
    validate_i5_9c_for_i5_10a,
)
from tests.i5_10a_helpers import (
    i59c_binding,
    receipt_binding,
    synthetic_i510a_receipt,
    synthetic_i59c_closure,
)


def _build():
    return build_i5_10a_mp_sfm_admission(
        i5_9c_closure=synthetic_i59c_closure(),
        i5_9c_binding=i59c_binding(),
        admission_receipt=synthetic_i510a_receipt(),
        admission_receipt_binding=receipt_binding(),
    )


def test_constants_freeze_i510a_boundary():
    assert I510A_PHASE == "I5.10A"
    assert I510A_I59C_FROZEN_SHA256 == "48148d22de7e67cfa3b5891b230dfd64e69888a224e4e930e6d765b1c3383364"
    assert I510A_PRIMARY_SCENARIO == "kitti_forward_motion"
    assert I510A_SUPPORTED_MATCHING_STRATEGIES == ("sequential", "exhaustive")


def test_repository_inventory_uses_only_existing_matching_strategies():
    inv = repository_mp_sfm_inventory()
    assert inv["sfm_backend"] == "pycolmap-offline-sfm"
    assert inv["pycolmap_recommended_version"] == "4.1.1"
    assert tuple(inv["supported_matching_strategies"]) == ("sequential", "exhaustive")
    assert inv["feature_strategy"] == "colmap_sift"
    assert inv["camera_mode"] == "single"
    assert inv["camera_model"] == "SIMPLE_RADIAL"
    assert inv["new_matching_algorithm_introduced"] is False


def test_policy_freezes_single_counterfactual_candidate():
    policy = I510AAdmissionPolicy().as_dict()
    assert policy["primary_scenario_id"] == "kitti_forward_motion"
    assert policy["baseline_matching_strategy"] == "sequential"
    assert policy["baseline_sequential_overlap"] == 10
    assert policy["candidate_matching_strategy"] == "exhaustive"
    assert policy["baseline_rerun_authorized"] is False
    assert policy["candidate_attempt_count"] == 1
    assert policy["input_window_images"] == 60
    assert policy["profile"] == "efficient"
    assert policy["image_sequence_fps"] == 10.0


def test_policy_freezes_existing_quality_heuristics():
    q = I510AAdmissionPolicy().as_dict()["sfm_quality_policy"]
    assert q["minimum_registered_keyframes"] == 3
    assert q["reject_below_registration_ratio"] == 0.30
    assert q["degrade_below_registration_ratio"] == 0.60
    assert q["minimum_landmarks"] == 20
    assert q["reject_above_mean_reprojection_error_px"] == 8.0
    assert q["degrade_above_mean_reprojection_error_px"] == 3.0


def test_i59c_validation_accepts_frozen_closure():
    validated = validate_i5_9c_for_i5_10a(synthetic_i59c_closure())
    assert validated["decision"]["i5_10_authorized"] is True


def test_i59c_validation_rejects_unclosed_i59():
    bad = synthetic_i59c_closure()
    bad["decision"]["i5_9_complete"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9c_for_i5_10a(bad)


def test_i59c_validation_rejects_mp_sfm_already_started():
    bad = synthetic_i59c_closure()
    bad["decision"]["mp_sfm_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_9c_for_i5_10a(bad)


def test_admission_is_execution_free_and_evidence_bound():
    out = _build()
    assert out["phase"] == "I5.10A"
    assert out["pass"] is True
    assert out["scope"]["execution_free"] is True
    assert out["scope"]["mp_sfm_execution"] is False
    assert out["scope"]["sfm_rerun"] is False
    assert out["evidence_bindings"]["i5_9c_closure"]["sha256"] == I510A_I59C_FROZEN_SHA256


def test_admission_targets_frozen_kitti_matching_policy_hypothesis_only():
    out = _build()
    design = out["experiment_design"]
    assert design["primary_scenario"]["scenario_id"] == "kitti_forward_motion"
    assert design["primary_scenario"]["historical_baseline_quality"] == "rejected"
    assert design["primary_scenario"]["historical_baseline_failure"] == "incremental_mapping_produced_no_sparse_model"
    assert design["primary_scenario"]["forward_motion_geometry_revalidated"] is False
    assert design["candidate"]["matching_strategy"] == "exhaustive"
    assert design["candidate"]["reruns_historical_sequential_baseline"] is False


def test_admission_keeps_horse_as_non_target_historical_anchor():
    out = _build()
    horse = out["historical_anchors"]["difficult_handheld_horse"]
    assert horse["historical_matching_strategy"] == "exhaustive"
    assert horse["global_sfm_completed"] is True
    assert horse["primary_bottleneck"] == "local_dense_consistency_connectivity"
    assert horse["real_i5_10_candidate"] is False


def test_admission_forbids_cross_run_absolute_geometry_comparison():
    out = _build()
    compare = out["comparison_contract"]
    assert compare["registration_and_sparse_health_comparison_allowed"] is True
    assert compare["scale_invariant_geometry_diagnostics_allowed"] is True
    assert compare["absolute_pose_or_translation_cross_run_comparison_allowed"] is False
    assert compare["metric_scale_comparison_allowed"] is False
    assert compare["runtime_performance_ranking_authorized"] is False


def test_admission_requires_b_preflight_to_bind_i54_evidence_and_source():
    out = _build()
    req = out["i5_10b_preflight_requirements"]
    assert req["bind_i5_4a_admission"] is True
    assert req["bind_i5_4b_replay"] is True
    assert req["bind_i5_4c_closure"] is True
    assert req["revalidate_exact_60_image_window"] is True
    assert req["probe_pycolmap_4_1_1_and_required_apis"] is True
    assert req["execute_sfm"] is False


def test_authority_remains_pycolmap_and_arbitrary_gauge():
    out = _build()
    auth = out["authority"]
    assert auth["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert auth["candidate_reconstruction_is_evaluation_only"] is True
    assert auth["candidate_can_replace_production_default"] is False
    assert auth["metric_scale_claimed"] is False
    assert auth["measurement_authoritative"] is False
    assert auth["certified_metrology"] is False


def test_no_auto_hybrid_or_later_phase_started():
    out = _build()
    d = out["decision"]
    assert d["i5_10a_complete"] is True
    assert d["i5_10b_authorized"] is True
    assert d["i5_10b_started"] is False
    assert d["i5_10c_started"] is False
    assert d["i5_10d_started"] is False
    assert d["i5_11_started"] is False
    assert d["i5_12_started"] is False
    assert d["i5_13_started"] is False
    assert d["i5_14_started"] is False
    assert d["auto_policy_implemented"] is False
    assert d["hybrid_authorized"] is False


def test_wrong_frozen_i59c_binding_is_rejected():
    bad = i59c_binding()
    bad = type(bad)(bad.path, "b" * 64)
    with pytest.raises(ReconstructionValidationError):
        build_i5_10a_mp_sfm_admission(
            i5_9c_closure=synthetic_i59c_closure(),
            i5_9c_binding=bad,
            admission_receipt=synthetic_i510a_receipt(),
            admission_receipt_binding=receipt_binding(),
        )


def test_receipt_failure_is_rejected():
    receipt = synthetic_i510a_receipt()
    receipt["pass"] = False
    receipt["status"] = "fail"
    with pytest.raises(ReconstructionValidationError):
        build_i5_10a_mp_sfm_admission(
            i5_9c_closure=synthetic_i59c_closure(),
            i5_9c_binding=i59c_binding(),
            admission_receipt=receipt,
            admission_receipt_binding=receipt_binding(),
        )
