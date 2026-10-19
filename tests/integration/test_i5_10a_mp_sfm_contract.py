from __future__ import annotations

from core.reconstruction.i5_mp_sfm_evaluation import (
    I510A_REPOSITORY_ANCHORS,
    build_i5_10a_mp_sfm_admission,
    repository_mp_sfm_inventory,
)
from core.reconstruction.sfm_contracts import SfmMatchingStrategy
from tests.i5_10a_helpers import i59c_binding, receipt_binding, synthetic_i510a_receipt, synthetic_i59c_closure


def test_i510a_repository_anchors_preserve_historical_design_evidence():
    assert I510A_REPOSITORY_ANCHORS["core/reconstruction/sfm_pycolmap.py"] == (
        "bdcc8a9cd2a885c51000f795ebe2edbc9a43449441fdbfae42c1d3e2f9455f59"
    )
    assert I510A_REPOSITORY_ANCHORS["scripts/reconstruction_pycolmap_worker.py"] == (
        "75277ced2640d61b7ebadb1541b1b28ae5b6dc8f5ba23183fb20e442cdd97dac"
    )
    assert I510A_REPOSITORY_ANCHORS["scripts/run_reconstruction_pycolmap_smoke.py"] == (
        "4585bc467166f8e9d4d689f97cb04b0fbefddb33c76d27ad5f4457192741180b"
    )
    assert all(
        len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
        for value in I510A_REPOSITORY_ANCHORS.values()
    )


def test_current_sfm_contract_exposes_exactly_sequential_and_exhaustive():
    assert tuple(item.value for item in SfmMatchingStrategy) == ("sequential", "exhaustive")
    assert repository_mp_sfm_inventory()["supported_matching_strategies"] == ["sequential", "exhaustive"]


def test_admission_end_to_end_preserves_single_candidate_and_roadmap_order():
    out = build_i5_10a_mp_sfm_admission(
        i5_9c_closure=synthetic_i59c_closure(),
        i5_9c_binding=i59c_binding(),
        admission_receipt=synthetic_i510a_receipt(),
        admission_receipt_binding=receipt_binding(),
    )
    assert out["experiment_design"]["real_candidate_run_count"] == 1
    assert out["decision"]["i5_10b_authorized"] is True
    assert out["decision"]["i5_11_started"] is False
    assert out["next"] == "I5.10B MP-SfM runtime/evidence preflight."


def test_worker_inventory_does_not_claim_unsupported_pairing_modes():
    inv = repository_mp_sfm_inventory()
    assert inv["required_pycolmap_matching_apis"] == ["match_sequential", "match_exhaustive"]
    assert inv["vocabulary_tree_matching_admitted"] is False
    assert inv["spatial_matching_admitted"] is False
    assert inv["transitive_matching_admitted"] is False
