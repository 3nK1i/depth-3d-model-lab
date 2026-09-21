from __future__ import annotations

import pytest

from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_auto_policy_engine import (
    I513B_ENGINE_PROFILE,
    I513B_PHASE,
    I513B_PROVIDER_ID_BY_IMPLEMENTATION,
    I513BOfflineAutoContext,
    compile_i5_13b_policy,
    evaluate_i5_13b_auto_policy,
    validate_i5_13a_for_i5_13b,
)
from tests.i5_13b_helpers import admission, clone, context, request


def engine():
    return compile_i5_13b_policy(admission())


def test_constants():
    assert I513B_PHASE == "I5.13B"
    assert I513B_ENGINE_PROFILE == "evidence_bound_conservative_v1"


def test_provider_id_mapping_covers_all_implementations():
    assert set(I513B_PROVIDER_ID_BY_IMPLEMENTATION) == set(DenseProviderImplementation)
    assert len(set(I513B_PROVIDER_ID_BY_IMPLEMENTATION.values())) == len(DenseProviderImplementation)
    assert "auto" not in I513B_PROVIDER_ID_BY_IMPLEMENTATION.values()


def test_admission_validation_accepts_frozen_a():
    out = validate_i5_13a_for_i5_13b(admission())
    assert out["decision"]["i5_13b_authorized"] is True


def test_rejects_a_if_production_auto_already_authorized():
    bad = clone(admission())
    bad["decision"]["production_auto_authorized"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13a_for_i5_13b(bad)


def test_compiled_engine_is_offline_shadow_only():
    policy = engine()
    assert policy.production_auto_authorized is False
    assert policy.provider_execution_authorized is False
    assert policy.profile == I513B_ENGINE_PROFILE


def test_mvs_unique_pass_selects_colmap_mvs():
    result = evaluate_i5_13b_auto_policy(engine(), request(), context())
    assert result.decision_class == "select"
    assert result.decision.selected_provider_id == "colmap-mvs-external"
    assert result.decision.selected_implementation is DenseProviderImplementation.COLMAP_MVS
    assert result.decision.automatic is True
    assert result.evidence_provider_status["colmap-mvs-external"] == "pass"


def test_temporal_current_castle_selects_rolling_depth_over_degraded_vda():
    ctx = context(
        sparse_context_id="dynamic_scene_proxy",
        evidence_context_id="i5_current_castle_temporal_envelope",
        family=DenseProviderFamily.TEMPORAL_DEPTH,
        available_provider_ids=("rolling-depth-temporal", "video-depth-anything-temporal"),
    )
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert result.decision_class == "select"
    assert result.decision.selected_provider_id == "rolling-depth-temporal"
    assert result.evidence_provider_status["rolling-depth-temporal"] == "pass"
    assert result.evidence_provider_status["video-depth-anything-temporal"] == "degraded"
    assert "evidence_status_degraded" in result.excluded_provider_reasons["video-depth-anything-temporal"]


def test_temporal_motion_abstains_zero_clean_primary():
    ctx = context(
        evidence_context_id="g3_temporal_motion",
        family=DenseProviderFamily.TEMPORAL_DEPTH,
        available_provider_ids=("rolling-depth-temporal", "video-depth-anything-temporal"),
    )
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert result.decision_class == "abstain"
    assert result.decision.selected_provider_id == ""
    assert "no_clean_evidence_eligible_provider" in result.decision.reason_codes


def test_single_image_low_texture_abstains_on_multiple_clean_candidates():
    ctx = context(
        evidence_context_id="g4_single_image_low_texture",
        family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        available_provider_ids=tuple(I513B_PROVIDER_ID_BY_IMPLEMENTATION[i] for i in (
            DenseProviderImplementation.METRIC3D_V2,
            DenseProviderImplementation.UNIDEPTH_V2,
            DenseProviderImplementation.MOGE_2,
            DenseProviderImplementation.DEPTH_PRO,
            DenseProviderImplementation.DEPTH_ANYTHING_V2,
            DenseProviderImplementation.MARIGOLD,
        )),
    )
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert result.decision_class == "abstain"
    assert set(result.evidence_eligible_provider_ids) == {
        "metric3d-v2-single-image", "unidepth-v2-single-image", "moge-2-single-image", "depth-pro-single-image"
    }
    assert "multiple_clean_evidence_eligible_providers" in result.decision.reason_codes
    assert result.evidence_provider_status["depth-anything-v2-single-image"] == "rejected"
    assert result.evidence_provider_status["marigold-single-image"] == "rejected"


def test_single_image_complete_room_abstains_even_if_only_one_evidence_pass_provider_is_available():
    ctx = context(
        evidence_context_id="g4_single_image_complete_room",
        family=DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        available_provider_ids=("metric3d-v2-single-image",),
    )
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert result.decision_class == "abstain"
    assert "evidence_not_unique_even_after_availability" in result.decision.reason_codes


def test_forward_motion_hard_abstains_before_dense_choice():
    ctx = context(sparse_context_id="forward_motion")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert result.decision_class == "abstain"
    assert "sparse_context_rejected_hard_abstention" in result.decision.reason_codes


def test_difficult_handheld_degraded_sparse_context_abstains():
    ctx = context(sparse_context_id="difficult_handheld")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert "sparse_context_degraded_auto_abstention" in result.decision.reason_codes


def test_visual_stress_degraded_sparse_context_abstains():
    ctx = context(sparse_context_id="visual_stress")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert "sparse_context_degraded_auto_abstention" in result.decision.reason_codes


def test_unknown_sparse_context_abstains():
    ctx = context(sparse_context_id="future_unknown")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert "sparse_context_unrecognized" in result.decision.reason_codes


def test_unknown_dense_context_abstains():
    ctx = context(evidence_context_id="future_unknown")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert "evidence_context_unrecognized" in result.decision.reason_codes


def test_no_authoritative_sfm_abstains():
    result = evaluate_i5_13b_auto_policy(engine(), request(sfm=False), context())
    assert "authoritative_sfm_unavailable" in result.decision.reason_codes


def test_nonusable_authoritative_sfm_abstains():
    result = evaluate_i5_13b_auto_policy(engine(), request(), context(sfm_usable=False))
    assert "authoritative_sfm_not_production_usable" in result.decision.reason_codes


def test_runtime_integrity_failure_abstains():
    result = evaluate_i5_13b_auto_policy(engine(), request(), context(runtime_clean=False))
    assert "runtime_integrity_not_clean" in result.decision.reason_codes


def test_hybrid_context_abstains():
    ctx = context(family=DenseProviderFamily.HYBRID, evidence_context_id="hybrid")
    result = evaluate_i5_13b_auto_policy(engine(), request(), ctx)
    assert "hybrid_family_forbidden" in result.decision.reason_codes


def test_request_must_be_auto():
    explicit = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=19,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id="sfm-frame-001",
        requested_provider_id="colmap-mvs-external",
        allow_fallback=False,
    )
    with pytest.raises(ReconstructionValidationError):
        evaluate_i5_13b_auto_policy(engine(), explicit, context())


def test_explicit_engineer_override_is_not_automatic_selection():
    ctx = context(available_provider_ids=("colmap-mvs-external",))
    result = evaluate_i5_13b_auto_policy(engine(), request(override="colmap-mvs-external"), ctx)
    assert result.decision_class == "engineer_override"
    assert result.decision.selected_provider_id == "colmap-mvs-external"
    assert result.decision.engineer_override_applied is True
    assert result.decision.automatic is False
    assert "engineer_override_explicit" in result.decision.reason_codes


def test_engineer_override_unknown_provider_abstains():
    result = evaluate_i5_13b_auto_policy(engine(), request(override="not-a-provider"), context())
    assert result.decision_class == "abstain"
    assert "engineer_override_provider_unknown" in result.decision.reason_codes


def test_engineer_override_unavailable_provider_abstains():
    result = evaluate_i5_13b_auto_policy(engine(), request(override="colmap-mvs-external"), context(available_provider_ids=()))
    assert "engineer_override_provider_unavailable" in result.decision.reason_codes


def test_fallback_flag_does_not_enable_retry_or_degraded_selection():
    ctx = context(
        evidence_context_id="g3_temporal_motion",
        family=DenseProviderFamily.TEMPORAL_DEPTH,
        available_provider_ids=("rolling-depth-temporal", "video-depth-anything-temporal"),
    )
    result = evaluate_i5_13b_auto_policy(engine(), request(allow_fallback=True), ctx)
    assert result.decision.selected_provider_id == ""
    assert result.decision.fallback_used is False


def test_same_input_same_decision():
    policy = engine()
    req = request()
    ctx = context()
    a = evaluate_i5_13b_auto_policy(policy, req, ctx).as_dict()
    b = evaluate_i5_13b_auto_policy(policy, req, ctx).as_dict()
    assert a == b


def test_explainability_has_required_provenance():
    result = evaluate_i5_13b_auto_policy(engine(), request(), context())
    data = result.as_dict()
    assert data["matched_evidence_scenario_ids"]
    assert data["evidence_sha256_bindings"]
    assert data["considered_provider_ids"]
    assert data["reason_codes"]
    assert "global_confidence_scalar" not in data


def test_result_never_claims_production_effect():
    result = evaluate_i5_13b_auto_policy(engine(), request(), context())
    assert result.production_effect is False
    assert result.provider_execution is False
    assert result.sfm_execution is False
    assert result.mvs_execution is False


def test_context_requires_unique_nonempty_available_provider_ids():
    with pytest.raises(ReconstructionValidationError):
        I513BOfflineAutoContext(
            sparse_context_id="low_overlap_weak_parallax",
            evidence_context_id="i5_6_current_mvs_envelope",
            provider_family=DenseProviderFamily.MVS,
            authoritative_sfm_production_usable=True,
            runtime_integrity_clean=True,
            available_provider_ids=("colmap-mvs-external", "colmap-mvs-external"),
        )


def test_receipt_validation_and_formal_freeze():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze, validate_i5_13b_receipt
    from tests.i5_13b_helpers import freeze_args, receipt
    assert validate_i5_13b_receipt(receipt())["pass"] is True
    out = build_i5_13b_engine_freeze(**freeze_args())
    assert out["pass"] is True
    assert out["quality_status"] == "offline_shadow_engine_pass"
    assert out["decision"]["i5_13c_authorized"] is True
    assert out["decision"]["production_auto_authorized"] is False


def test_formal_freeze_has_expected_probe_classes():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    out = build_i5_13b_engine_freeze(**freeze_args())
    assert out["probe_summary"]["decision_class_counts"] == {"select": 2, "abstain": 3, "engineer_override": 1}


def test_formal_freeze_is_execution_free():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    scope = build_i5_13b_engine_freeze(**freeze_args())["scope"]
    assert scope["auto_policy_engine_implemented"] is True
    assert scope["offline_shadow_only"] is True
    for key in ("provider_execution", "sfm_execution", "mvs_execution", "production_auto_integration", "scene_classifier_implemented"):
        assert scope[key] is False


def test_formal_freeze_requires_offline_replay_before_production():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    out = build_i5_13b_engine_freeze(**freeze_args())
    assert out["decision"]["i5_13c_authorized"] is True
    assert out["decision"]["i5_13d_authorized"] is False
    assert out["claim_boundaries"]["production_auto_qualified"] is False


def test_receipt_rejects_provider_execution_claim():
    from core.reconstruction.i5_auto_policy_engine import validate_i5_13b_receipt
    from tests.i5_13b_helpers import clone, receipt
    bad = clone(receipt())
    bad["scope"]["provider_execution"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_13b_receipt(bad)


def test_freeze_binds_engine_module_and_source_checkpoint():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    bindings = build_i5_13b_engine_freeze(**freeze_args())["evidence_bindings"]
    assert set(bindings) == {"i5_13a_admission", "post_i5_13a_source_checkpoint", "i5_13b_engine_module", "i5_13b_engine_receipt"}
    assert all(len(item["sha256"]) == 64 for item in bindings.values())


def test_freeze_compiles_five_sparse_contexts_and_six_dense_contexts():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    compiled = build_i5_13b_engine_freeze(**freeze_args())["compiled_policy"]
    assert len(compiled["sparse_context_status"]) == 5
    assert len(compiled["dense_evidence_contexts"]) == 7


def test_freeze_self_probes_are_deterministic():
    from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze
    from tests.i5_13b_helpers import freeze_args
    out = build_i5_13b_engine_freeze(**freeze_args())
    assert out["probe_summary"]["all_same_input_same_decision"] is True
    assert out["probe_summary"]["forward_motion_hard_abstention_verified"] is True
