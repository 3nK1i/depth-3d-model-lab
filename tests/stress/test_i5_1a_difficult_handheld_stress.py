from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding, build_i5_1a_difficult_handheld_baseline
from tests.i5_1a_helpers import i4_1_4_doc, i4_1_5_doc, i4_1_6b_doc, i4_1_7_doc, passing_i4_4d, passing_receipt


def _build(*, i415: str | None = None) -> dict:
    return build_i5_1a_difficult_handheld_baseline(
        i4_4d_result=passing_i4_4d(),
        i4_4d_binding=I51EvidenceBinding("/evidence/i44d.json", "1" * 64),
        i4_1_4_doc=i4_1_4_doc(),
        i4_1_4_binding=I51EvidenceBinding("/repo/i414.md", "2" * 64),
        i4_1_5_doc=i415 if i415 is not None else i4_1_5_doc(),
        i4_1_5_binding=I51EvidenceBinding("/repo/i415.md", "3" * 64),
        i4_1_6b_doc=i4_1_6b_doc(),
        i4_1_6b_binding=I51EvidenceBinding("/repo/i416b.md", "4" * 64),
        i4_1_7_doc=i4_1_7_doc(),
        i4_1_7_binding=I51EvidenceBinding("/repo/i417.md", "5" * 64),
        baseline_receipt=passing_receipt(),
        baseline_receipt_binding=I51EvidenceBinding("/evidence/receipt.json", "6" * 64),
    )


def test_i51a_is_deterministic_under_repeated_builds() -> None:
    first = _build()
    for _ in range(50):
        assert _build() == first


def test_i51a_is_thread_safe_for_evidence_only_builds() -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _build(), range(64)))
    assert all(item == results[0] for item in results)


def test_i51a_fail_closed_on_repeated_phase_history_mutation() -> None:
    for _ in range(20):
        mutated = i4_1_5_doc().replace("mixed failure", "unknown failure")
        with pytest.raises(ReconstructionValidationError):
            _build(i415=mutated)


def test_i51a_never_promotes_historical_failure_under_repetition() -> None:
    for _ in range(50):
        result = _build()
        assert result["historical_baseline"]["historical_failure_reclassified_as_pass"] is False
        assert result["decision"]["provider_selected"] is False
        assert result["decision"]["auto_policy_implemented"] is False
