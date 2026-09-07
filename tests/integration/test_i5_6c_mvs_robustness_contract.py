from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_mvs_robustness import I56C_PHASE, build_i5_6c_mvs_robustness_closure
from tests.i5_6b_helpers import admission
from tests.i5_6c_helpers import (
    I56A_SHA,
    I56B_SHA,
    I56C_RECEIPT_SHA,
    binding,
    passing_i5_6b,
    passing_i5_6c_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _build() -> dict:
    return build_i5_6c_mvs_robustness_closure(
        i5_6a_result=admission(),
        i5_6a_binding=binding("/evidence/i5_6a.json", I56A_SHA),
        i5_6b_result=passing_i5_6b(),
        i5_6b_binding=binding("/evidence/i5_6b.json", I56B_SHA),
        qualification_receipt=passing_i5_6c_receipt(),
        qualification_receipt_binding=binding("/evidence/i5_6c_receipt.json", I56C_RECEIPT_SHA),
    )


def test_i56c_contract_symbols_are_importable() -> None:
    assert I56C_PHASE == "I5.6C"
    assert callable(build_i5_6c_mvs_robustness_closure)


def test_i56c_closure_preserves_roadmap_and_authority() -> None:
    result = _build()
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["decision"]["i5_7_authorized"] is True
    assert result["decision"]["i5_7_started"] is False


def test_i56c_closure_preserves_interrupted_recovery_limitations() -> None:
    result = _build()
    assert result["recovery_provenance"]["worker_termination_cause"].startswith("unresolved_")
    assert result["limitations"]["recovered_dynamic_proxy_parent_wall_time_unavailable"] is True
    assert result["qualification"]["runtime_performance_qualification_claimed"] is False


def test_i56c_runner_is_evidence_only() -> None:
    text = (ROOT / "scripts/run_i5_6c_mvs_robustness_closure.py").read_text(encoding="utf-8")
    assert "--i5-6a-admission" in text
    assert "--i5-6b-replay" in text
    assert "--closure-receipt" in text
    for forbidden in ("--dataset-dir", "--sfm-python", "--mvs-python", "--gpu-index", "--scenario-timeout-minutes"):
        assert f'add_argument("{forbidden}"' not in text


def test_i56c_history_document_preserves_limitations_and_next_boundary() -> None:
    text = (ROOT / "docs/phase_history/I5_6C_MVS_ROBUSTNESS_QUALIFICATION_AND_CLOSURE.md").read_text(encoding="utf-8").lower()
    for token in ("evidence-only", "directional native support", "397,599", "unresolved", "rollingdepth", "not universal"):
        assert token in text
