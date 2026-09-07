from core.reconstruction.i5_mvs_robustness import I56B_PHASE, I56B_SCENARIO_IDS, I56BReplayPolicy


def test_i56b_contract_keeps_six_admitted_scenarios_and_existing_adapter():
    p = I56BReplayPolicy()
    assert I56B_PHASE == "I5.6B"
    assert len(I56B_SCENARIO_IDS) == 6
    assert p.production_adapter == "MvsProductionProviderAdapter"
    assert p.adaptive_preparation == "prepare_adaptive_provider_execution"
    assert p.geometric_consistency is True
    assert p.permanent_global_view_cap is None


def test_i56b_recovery_contract_is_narrow_and_non_rerun():
    from core.reconstruction.i5_mvs_robustness import (
        I56B_RECOVERY_PHASE,
        I56B_RECOVERY_SCENARIO_ID,
        recover_i5_6b_interrupted_native_scenario,
    )
    assert I56B_RECOVERY_PHASE == "I5.6B_RECOVERY_FINALIZATION"
    assert I56B_RECOVERY_SCENARIO_ID == "castle_dynamic_proxy"
    assert callable(recover_i5_6b_interrupted_native_scenario)
