from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding, build_i5_1a_difficult_handheld_baseline
from tests.i5_1a_helpers import passing_i4_4d, passing_receipt

ROOT = Path(__file__).resolve().parents[2]


def _binding(path: str, ch: str) -> I51EvidenceBinding:
    return I51EvidenceBinding(path, ch * 64)


def _doc(name: str) -> str:
    return (ROOT / "docs/phase_history" / name).read_text(encoding="utf-8")


def _result() -> dict:
    return build_i5_1a_difficult_handheld_baseline(
        i4_4d_result=passing_i4_4d(),
        i4_4d_binding=_binding("/evidence/i44d.json", "1"),
        i4_1_4_doc=_doc("I4_1_4_DECOUPLED_GLOBAL_COVERAGE_LOCAL_DENSITY_SAMPLING_DIAGNOSIS.md"),
        i4_1_4_binding=_binding("/repo/docs/i414.md", "2"),
        i4_1_5_doc=_doc("I4_1_5_DECOUPLED_GLOBAL_LOCAL_SAMPLING_COUNTERFACTUAL.md"),
        i4_1_5_binding=_binding("/repo/docs/i415.md", "3"),
        i4_1_6b_doc=_doc("I4_1_6B_REAL_DECOUPLED_DENSE_AB_VALIDATION.md"),
        i4_1_6b_binding=_binding("/repo/docs/i416b.md", "4"),
        i4_1_7_doc=_doc("I4_1_7_CLEAN_STATIC_OBJECT_ORBIT_QUALIFICATION.md"),
        i4_1_7_binding=_binding("/repo/docs/i417.md", "5"),
        baseline_receipt=passing_receipt(),
        baseline_receipt_binding=_binding("/evidence/receipt.json", "6"),
    )


def test_i51a_actual_committed_phase_history_supports_baseline() -> None:
    result = _result()
    assert result["historical_baseline"]["historical_horse_preserved_for_i5"] is True
    assert result["historical_baseline"]["historical_dense_gate_reached"] is False


def test_i51a_handoff_does_not_authorize_i52_or_auto() -> None:
    result = _result()
    assert result["decision"]["i5_2_authorized"] is False
    assert result["decision"]["auto_policy_implemented"] is False
    assert result["decision"]["hybrid_authorized"] is False


def test_i51a_runner_no_longer_requires_missing_horse_summary_json() -> None:
    text = (ROOT / "scripts/run_i5_1a_difficult_handheld_baseline.py").read_text(encoding="utf-8")
    assert "--horse-i4-1-4-summary" not in text
    assert "--horse-i4-1-6b-summary" not in text
    assert "--i4-1-4-doc" in text
    assert "--i4-1-7-doc" in text


def test_i51a_runner_exists_and_is_execution_free_by_contract() -> None:
    text = (ROOT / "scripts/run_i5_1a_difficult_handheld_baseline.py").read_text(encoding="utf-8")
    assert "provider_execution_started=False" in text
    assert "current_replay_started=False" in text
    assert "I5.1B current frozen-foundation difficult-handheld replay" in text


def test_i51a_doc_explicitly_preserves_horse_and_provenance_gap() -> None:
    text = (ROOT / "docs/phase_history/I5_1A_DIFFICULT_HANDHELD_BASELINE_ADMISSION.md").read_text(encoding="utf-8")
    assert "does not turn the historical Horse failure into a PASS" in text
    assert "must not synthesize replacements or invent hashes" in text
    assert "I5.1B" in text
    assert "I5.13" in text


def test_i51a_historical_counts_are_not_provider_scores() -> None:
    result = _result()
    assert result["historical_baseline"]["historical_counts_are_not_provider_scores"] is True
    assert result["decision"]["provider_ranking_created"] is False
