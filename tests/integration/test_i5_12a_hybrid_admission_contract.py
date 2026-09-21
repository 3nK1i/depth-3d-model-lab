from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_hybrid_admission import build_i5_12a_hybrid_admission_review
from tests.i5_12a_helpers import build_args

ROOT = Path(__file__).resolve().parents[2]


def test_default_evidence_path_returns_valid_nonexecution_decision():
    out = build_i5_12a_hybrid_admission_review(**build_args(justified=False))
    assert out["pass"] is True
    assert out["quality_status"] == "hybrid_not_justified"
    assert out["scope"]["hybrid_execution"] is False
    assert out["decision"]["i5_13_authorized"] is True


def test_justified_branch_still_does_not_authorize_execution():
    out = build_i5_12a_hybrid_admission_review(**build_args(justified=True))
    assert out["decision"]["i5_12b_authorized"] is True
    assert out["decision"]["i5_12c_authorized"] is False
    assert out["decision"]["hybrid_execution_authorized"] is False


def test_runner_and_doc_preserve_evidence_only_boundary():
    runner = (ROOT / "scripts/run_i5_12a_hybrid_admission.py").read_text(encoding="utf-8")
    doc = (ROOT / "docs/phase_history/I5_12A_HYBRID_EVIDENCE_JUSTIFICATION_AND_ADMISSION_REVIEW.md").read_text(encoding="utf-8")
    assert "hybrid_execution=False" in runner
    assert "Global completeness alone is not enough" in doc
    assert "hybrid_execution_authorized=false" in doc
