from __future__ import annotations

from pathlib import Path

from core.reconstruction.i5_dynamic_scene_handling import (
    I55B_PHASE,
    build_i5_5b_dynamic_scene_replay,
    materialize_i5_5b_pair,
)

ROOT = Path(__file__).resolve().parents[2]


def test_i55b_contract_symbols_are_importable() -> None:
    assert I55B_PHASE == "I5.5B"
    assert callable(build_i5_5b_dynamic_scene_replay)
    assert callable(materialize_i5_5b_pair)


def test_i55b_runner_exists_and_has_no_recipe_tuning_cli() -> None:
    text = (ROOT / "scripts/run_i5_5b_dynamic_scene_replay.py").read_text(encoding="utf-8")
    assert "--i5-5a-admission" in text
    assert "--dataset-dir" in text
    assert "--sfm-python" in text
    for forbidden in ("--patch-width", "--patch-height", "--target-x", "--donor-index", "--matching"):
        assert f'add_argument("{forbidden}"' not in text


def test_i55b_test_gate_includes_i55a_and_i54_regression() -> None:
    text = (ROOT / "scripts/run_i5_5b_tests.sh").read_text(encoding="utf-8")
    assert "test_i5_5a_dynamic_scene_admission.py" in text
    assert "test_i5_4c_forward_motion_kitti_closure.py" in text
    assert "real_sfm_executed_by_test_receipt" in text


def test_i55b_history_document_preserves_proxy_scope() -> None:
    text = (ROOT / "docs/phase_history/I5_5B_CURRENT_FROZEN_FOUNDATION_DYNAMIC_SCENE_REPLAY.md").read_text(encoding="utf-8")
    assert "controlled" in text.lower()
    assert "natural dynamic-scene ground truth" in text.lower()
    assert "PyCOLMAP" in text
    assert "I5.5C" in text
