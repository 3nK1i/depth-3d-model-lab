from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_dynamic_scene_handling import I55C_PHASE, build_i5_5c_dynamic_scene_closure
from tests.i5_5b_helpers import I55A_SHA, passing_i5_5a
from tests.i5_5c_helpers import I55B_SHA, RECEIPT_55C_SHA, i55c_binding, passing_i5_5b, passing_i5_5c_receipt

ROOT = Path(__file__).resolve().parents[2]


def test_i55c_contract_symbols_are_importable() -> None:
    assert I55C_PHASE == "I5.5C"
    assert callable(build_i5_5c_dynamic_scene_closure)


def test_i55c_closure_preserves_roadmap_and_authority(tmp_path: Path) -> None:
    result = build_i5_5c_dynamic_scene_closure(
        i5_5a_result=passing_i5_5a(tmp_path),
        i5_5a_binding=i55c_binding("/evidence/i5_5a.json", I55A_SHA),
        i5_5b_result=passing_i5_5b(tmp_path),
        i5_5b_binding=i55c_binding("/evidence/i5_5b.json", I55B_SHA),
        qualification_receipt=passing_i5_5c_receipt(),
        qualification_receipt_binding=i55c_binding("/evidence/i5_5c_receipt.json", RECEIPT_55C_SHA),
    )
    assert result["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert result["decision"]["i5_6_authorized"] is True
    assert result["decision"]["i5_6_started"] is False


def test_i55c_runner_is_evidence_only() -> None:
    text = (ROOT / "scripts/run_i5_5c_dynamic_scene_closure.py").read_text(encoding="utf-8")
    assert "--i5-5a-admission" in text
    assert "--i5-5b-replay" in text
    assert "--closure-receipt" in text
    for forbidden in ("--dataset-dir", "--sfm-python", "--matching", "--patch-width", "--target-x"):
        assert f'add_argument("{forbidden}"' not in text


def test_i55c_history_document_preserves_limitations() -> None:
    text = (ROOT / "docs/phase_history/I5_5C_DYNAMIC_SCENE_HANDLING_ROBUSTNESS_QUALIFICATION_AND_CLOSURE.md").read_text(encoding="utf-8").lower()
    assert "controlled" in text
    assert "natural dynamic-scene ground truth" in text
    assert "sparse quality" in text
    assert "i5.6" in text
