from __future__ import annotations

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import (
    I51A_PHASE,
    I51EvidenceBinding,
    build_i5_1a_difficult_handheld_baseline,
    validate_horse_phase_history,
    validate_i4_4d_for_i5_1,
    validate_i5_1a_receipt,
)
from tests.i5_1a_helpers import (
    clone,
    i4_1_4_doc,
    i4_1_5_doc,
    i4_1_6b_doc,
    i4_1_7_doc,
    passing_i4_4d,
    passing_receipt,
)


def binding(path: str, char: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, char * 64)


def build(**kwargs) -> dict:
    args = {
        "i4_4d_result": passing_i4_4d(),
        "i4_4d_binding": binding("/evidence/i44d.json", "1"),
        "i4_1_4_doc": i4_1_4_doc(),
        "i4_1_4_binding": binding("/repo/docs/i414.md", "2"),
        "i4_1_5_doc": i4_1_5_doc(),
        "i4_1_5_binding": binding("/repo/docs/i415.md", "3"),
        "i4_1_6b_doc": i4_1_6b_doc(),
        "i4_1_6b_binding": binding("/repo/docs/i416b.md", "4"),
        "i4_1_7_doc": i4_1_7_doc(),
        "i4_1_7_binding": binding("/repo/docs/i417.md", "5"),
        "baseline_receipt": passing_receipt(),
        "baseline_receipt_binding": binding("/evidence/i51a_receipt.json", "6"),
    }
    args.update(kwargs)
    return build_i5_1a_difficult_handheld_baseline(**args)


def test_i51a_admits_documented_horse_baseline_without_closing_i51() -> None:
    result = build()
    assert result["phase"] == I51A_PHASE == "I5.1A"
    assert result["status"] == "historical_baseline_admitted"
    assert result["pass"] is True
    assert result["decision"]["i5_1_started"] is True
    assert result["decision"]["i5_1a_complete"] is True
    assert result["decision"]["i5_1_current_replay_authorized"] is True
    assert result["decision"]["i5_1_complete"] is False
    assert result["decision"]["i5_2_authorized"] is False


def test_i51a_preserves_only_committed_horse_history_claims() -> None:
    baseline = build()["historical_baseline"]
    assert baseline["evidence_level"] == "committed_phase_history"
    assert baseline["historical_capture_regime_difficult"] is True
    assert baseline["historical_local_temporal_density_materially_affects_dense_consistency"] is True
    assert baseline["historical_full_span_da3_sfm_alignment"] == "36/40"
    assert baseline["historical_dense_path_collapsed_to_views"] == 2
    assert baseline["historical_decoupled_sampling_materially_improved_real_dense_consistency"] is True
    assert baseline["historical_dense_gate_reached"] is False
    assert baseline["historical_dense_gate_required_connected_views"] == 10


def test_i51a_does_not_synthesize_missing_historical_runtime_json() -> None:
    result = build()
    assert result["historical_baseline"]["historical_runtime_summary_json_reconstructed"] is False
    provenance = result["provenance"]
    assert provenance["committed_horse_phase_history_sha256_bound"] is True
    assert provenance["historical_horse_standalone_summary_json_bound"] is False
    assert provenance["synthetic_historical_summary_json_created"] is False


def test_i51a_scope_is_evidence_only_and_does_not_replay_capture_early() -> None:
    scope = build()["scope"]
    assert scope["historical_evidence_admission_only"] is True
    assert scope["provider_execution_started"] is False
    assert scope["sfm_rerun"] is False
    assert scope["dense_pair_rerun"] is False
    assert scope["current_difficult_handheld_replay_started"] is False


def test_i51a_preserves_authority_and_defers_auto_hybrid() -> None:
    result = build()
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["authority"]["permanent_global_view_cap"] is None
    assert result["authority_audit"]["default_reconstruction_scale_status"] == "arbitrary"
    for key in (
        "provider_selected", "provider_promoted", "auto_policy_implemented", "hybrid_authorized",
        "metric_scale_claimed", "measurement_authoritative", "certified_metrology", "surface_mesh_generated",
    ):
        assert result["authority"][key] is False


def test_i51a_does_not_claim_raw_horse_source_rehash() -> None:
    p = build()["provenance"]
    assert p["historical_raw_video_content_rehashed_by_i5_1a"] is False
    assert p["historical_nested_native_artifact_tree_recursively_rehashed_by_i5_1a"] is False
    assert p["current_replay_must_establish_fresh_source_identity"] is True


def test_i51a_next_is_current_frozen_foundation_replay() -> None:
    assert build()["next"] == "I5.1B current frozen-foundation difficult-handheld replay."


def test_i51a_validates_final_i4_handoff() -> None:
    validated = validate_i4_4d_for_i5_1(passing_i4_4d())
    assert validated["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"


@pytest.mark.parametrize("field", ["i4_closed", "i5_authorized"])
def test_i51a_rejects_missing_i4_to_i5_authorization(field: str) -> None:
    payload = clone(passing_i4_4d())
    payload["decision"][field] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_for_i5_1(payload)


def test_i51a_rejects_i5_already_started_in_frozen_closure() -> None:
    payload = clone(passing_i4_4d())
    payload["decision"]["i5_started"] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_for_i5_1(payload)


def test_i51a_rejects_wrong_first_i5_boundary() -> None:
    payload = clone(passing_i4_4d())
    payload["handoff_to_i5"]["first_boundary"] = "I5.2 low overlap"
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_for_i5_1(payload)


@pytest.mark.parametrize("field", ["provider_selected", "auto_policy_implemented", "hybrid_authorized", "metric_scale_claimed"])
def test_i51a_rejects_authority_promotion(field: str) -> None:
    payload = clone(passing_i4_4d())
    payload["authority"][field] = True
    with pytest.raises(ReconstructionValidationError):
        validate_i4_4d_for_i5_1(payload)


def test_i51a_phase_history_validation_passes() -> None:
    result = validate_horse_phase_history(
        i4_1_4_doc=i4_1_4_doc(), i4_1_5_doc=i4_1_5_doc(),
        i4_1_6b_doc=i4_1_6b_doc(), i4_1_7_doc=i4_1_7_doc(),
    )
    assert result["historical_horse_preserved_for_i5"] is True


@pytest.mark.parametrize(
    ("name", "needle"),
    [
        ("i4_1_4_doc", "36/40"),
        ("i4_1_5_doc", "mixed failure"),
        ("i4_1_6b_doc", "global_anchors_plus_local_neighbors"),
        ("i4_1_7_doc", "10-view production gate"),
    ],
)
def test_i51a_fails_closed_when_committed_horse_history_loses_decisive_claim(name: str, needle: str) -> None:
    docs = {
        "i4_1_4_doc": i4_1_4_doc(),
        "i4_1_5_doc": i4_1_5_doc(),
        "i4_1_6b_doc": i4_1_6b_doc(),
        "i4_1_7_doc": i4_1_7_doc(),
    }
    docs[name] = docs[name].replace(needle, "REMOVED")
    with pytest.raises(ReconstructionValidationError):
        validate_horse_phase_history(**docs)


def test_i51a_receipt_passes() -> None:
    receipt = validate_i5_1a_receipt(passing_receipt())
    assert receipt["counts"]["combined_exercised_tests"] == 72


@pytest.mark.parametrize("field", ["focused", "affected_unit", "affected_integration", "affected_stress", "import_probe", "runner_help_probe"])
def test_i51a_receipt_rejects_nonzero_return_code(field: str) -> None:
    receipt = clone(passing_receipt())
    receipt["return_codes"][field] = 1
    with pytest.raises(ReconstructionValidationError):
        validate_i5_1a_receipt(receipt)
