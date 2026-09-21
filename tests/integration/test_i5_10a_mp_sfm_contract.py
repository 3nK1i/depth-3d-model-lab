from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_mp_sfm_evaluation import (
    I510A_REPOSITORY_ANCHORS,
    build_i5_10a_mp_sfm_admission,
    repository_mp_sfm_inventory,
    verify_i5_10a_repository_anchors,
)
from core.reconstruction.sfm_contracts import SfmMatchingStrategy
from tests.i5_10a_helpers import i59c_binding, receipt_binding, synthetic_i510a_receipt, synthetic_i59c_closure


def test_i510a_repository_anchor_hashes_match_current_tree():
    root = Path(__file__).resolve().parents[2]
    verified = verify_i5_10a_repository_anchors(root)
    assert set(verified) == set(I510A_REPOSITORY_ANCHORS)
    assert all(node["verified"] for node in verified.values())


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
