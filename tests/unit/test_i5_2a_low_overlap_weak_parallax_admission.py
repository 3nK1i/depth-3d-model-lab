from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_low_overlap_weak_parallax import (
    I52A_PHASE,
    I52AStressPlanPolicy,
    build_i5_2a_low_overlap_weak_parallax_admission,
    build_ordered_source_identity,
    build_stress_plan,
    validate_i5_1c_for_i5_2,
    validate_i5_2a_receipt,
)
from tests.i5_2a_helpers import (
    G2A3_DOC_SHA,
    G2_FINAL_SHA,
    G2B_DOC_SHA,
    G2C_DOC_SHA,
    I51C_SHA,
    RECEIPT_52A_SHA,
    clone,
    g2a3_doc,
    g2_final,
    g2b_doc,
    g2c_doc,
    make_ordered_dataset,
    passing_i5_1c,
    passing_receipt,
)


def binding(path: str, sha: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, sha)


def build(tmp_path: Path, **kwargs) -> dict:
    dataset = make_ordered_dataset(tmp_path / "castle-P19")
    args = {
        "i5_1c_result": passing_i5_1c(),
        "i5_1c_binding": binding("/evidence/i5_1c.json", I51C_SHA),
        "i4_3g2_final_result": g2_final(),
        "i4_3g2_final_binding": binding("/evidence/i4_3g2_final.json", G2_FINAL_SHA),
        "i4_3g2a3_doc": g2a3_doc(),
        "i4_3g2a3_binding": binding("/repo/docs/i4_3g2a3.md", G2A3_DOC_SHA),
        "i4_3g2b_doc": g2b_doc(),
        "i4_3g2b_binding": binding("/repo/docs/i4_3g2b.md", G2B_DOC_SHA),
        "i4_3g2c_doc": g2c_doc(),
        "i4_3g2c_binding": binding("/repo/docs/i4_3g2c.md", G2C_DOC_SHA),
        "source_identity": build_ordered_source_identity(dataset),
        "semantic_confirmation": True,
        "admission_receipt": passing_receipt(),
        "admission_receipt_binding": binding("/evidence/i5_2a_receipt.json", RECEIPT_52A_SHA),
    }
    args.update(kwargs)
    return build_i5_2a_low_overlap_weak_parallax_admission(**args)


def test_i52a_starts_i52_without_starting_real_replay(tmp_path: Path) -> None:
    result = build(tmp_path)
    assert result["phase"] == I52A_PHASE == "I5.2A"
    assert result["status"] == "low_overlap_weak_parallax_scenario_admitted"
    assert result["pass"] is True
    assert result["decision"]["i5_2_started"] is True
    assert result["decision"]["i5_2a_complete"] is True
    assert result["decision"]["i5_2b_real_replay_authorized"] is True
    assert result["decision"]["i5_2_complete"] is False
    assert result["decision"]["i5_3_authorized"] is False


def test_i52a_admits_only_documented_historical_weak_overlap_claims(tmp_path: Path) -> None:
    baseline = build(tmp_path)["historical_baseline"]
    assert baseline["scenario_id"] == "g2_weak_overlap_mvs_da3"
    assert baseline["historical_authoritative_registered_view_scope"] == 5
    assert baseline["historical_review_scope"] == "authoritative_registered_views"
    assert baseline["historical_partial_registration_preserved"] is True
    assert baseline["historical_full_scenario_claim_permitted"] is False
    assert baseline["historical_threshold_relaxation_performed"] is False


def test_i52a_source_identity_is_content_addressed_and_ordered(tmp_path: Path) -> None:
    identity = build_ordered_source_identity(make_ordered_dataset(tmp_path / "dataset"))
    assert identity["image_count"] == 19
    assert len(identity["asset_set_sha256"]) == 64
    assert identity["ordered_names"][0] == "castle00.png"
    assert identity["ordered_names"][-1] == "castle18.png"
    assert all(len(item["sha256"]) == 64 for item in identity["files"])


def test_i52a_stress_plan_is_deterministic_for_p19_style_sequence(tmp_path: Path) -> None:
    identity = build_ordered_source_identity(make_ordered_dataset(tmp_path / "dataset"))
    first = build_stress_plan(identity, policy=I52AStressPlanPolicy())
    second = build_stress_plan(identity, policy=I52AStressPlanPolicy())
    assert first == second
    assert first["control"]["image_count"] == 19
    assert first["low_overlap_proxy"]["indices"] == [0, 3, 6, 9, 12, 15, 18]
    assert first["low_overlap_proxy"]["source_span_fraction"] == 1.0
    assert first["weak_parallax_proxy"]["indices"] == [6, 7, 8, 9, 10, 11, 12]
    assert first["weak_parallax_proxy"]["source_span_fraction"] == pytest.approx(1/3)
    assert first["low_overlap_proxy"]["proxy_is_authoritative_overlap_measurement"] is False
    assert first["weak_parallax_proxy"]["proxy_is_authoritative_parallax_measurement"] is False


def test_i52a_requires_explicit_semantic_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        build(tmp_path, semantic_confirmation=False)


def test_i52a_rejects_too_small_source(tmp_path: Path) -> None:
    dataset = make_ordered_dataset(tmp_path / "small", count=11)
    with pytest.raises(ReconstructionValidationError):
        build_ordered_source_identity(dataset)


def test_i52a_validates_frozen_i51c_handoff() -> None:
    validated = validate_i5_1c_for_i5_2(passing_i5_1c())
    assert validated["decision"]["i5_2_authorized"] is True
    assert validated["decision"]["i5_2_started"] is False


@pytest.mark.parametrize("field,value", [("i5_1_complete", False), ("i5_2_authorized", False), ("i5_2_started", True)])
def test_i52a_rejects_invalid_i51c_handoff(field: str, value) -> None:
    payload = clone(passing_i5_1c())
    payload["decision"][field] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1c_for_i5_2(payload)


def test_i52a_preserves_authority_and_defers_later_i5_work(tmp_path: Path) -> None:
    result = build(tmp_path)
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    for key in (
        "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert result["authority"][key] is False
    assert result["scope"]["sfm_rerun"] is False
    assert result["scope"]["dense_provider_rerun"] is False


def test_i52a_receipt_requires_green_execution_free_gate() -> None:
    receipt = validate_i5_2a_receipt(passing_receipt())
    assert receipt["counts"]["combined_exercised_tests"] >= 60
    bad = clone(passing_receipt())
    bad["scope"]["sfm_rerun"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_2a_receipt(bad)


def test_i52a_next_is_real_controlled_replay(tmp_path: Path) -> None:
    assert build(tmp_path)["next"] == "I5.2B current frozen-foundation low-overlap / weak-parallax replay."
