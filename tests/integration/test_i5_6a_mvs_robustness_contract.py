from core.reconstruction.i5_mvs_robustness import I56A_PHASE, I56A_SCENARIO_IDS, I56AAdmissionPolicy
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation


def test_i56a_binds_existing_production_mvs_contract_without_parallel_provider():
    p=I56AAdmissionPolicy()
    assert I56A_PHASE == "I5.6A"
    assert p.provider_id == I43B_PROVIDER_ID
    assert p.implementation == DenseProviderImplementation.COLMAP_MVS.value
    assert p.scenario_ids == I56A_SCENARIO_IDS
    assert p.adaptive_selection is True
    assert p.engineer_override is None
    assert p.permanent_global_view_cap is None
