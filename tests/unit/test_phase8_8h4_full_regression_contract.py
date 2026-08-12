from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_phase8_8h4_full_regression.py"
SHELL = ROOT / "scripts/run_phase8_8h4_tests.sh"
DOC = ROOT / "docs/phase_history/PHASE8_8H4_VALIDATION.md"
H3 = ROOT / "scripts/run_phase8_8h3_tests.sh"
H2 = ROOT / "scripts/run_phase8_8h2_tests.sh"
H1 = ROOT / "scripts/run_phase8_8h1_tests.sh"
G6 = ROOT / "scripts/run_phase8_8g6_tests.sh"


def test_h4_is_validation_only_and_declares_no_production_behavior_change() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "adds no production feature" in text
    assert "does not create a release tag" in text
    runner = RUNNER.read_text(encoding="utf-8")
    assert '"production_behavior_changed_by_h4": False' in runner
    assert '"new_feature_work_added_by_h4": False' in runner


def test_h4_runs_complete_automated_test_categories() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        '"tests/unit"',
        '"tests/ui"',
        '"tests/integration"',
        '"tests/stress"',
    ):
        assert token in text


def test_h4_recompiles_application_core_tests_and_scripts() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"compileall"' in text
    for token in ('"app"', '"core"', '"tests"', '"scripts"', '"main.py"'):
        assert token in text


def test_h4_replays_phase8_8_protocol_and_backward_chains() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '["bash", "scripts/run_phase8_8h3_tests.sh", "protocol"]' in text
    assert '["bash", "scripts/run_phase8_8h3_tests.sh", "backward"]' in text


def test_h4_inherited_chain_reaches_h2_h1_and_g6() -> None:
    h3 = H3.read_text(encoding="utf-8")
    h2 = H2.read_text(encoding="utf-8")
    h1 = H1.read_text(encoding="utf-8")
    assert "run_phase8_8h2_tests.sh" in h3
    assert "run_phase8_8h1_tests.sh" in h2
    assert "run_phase8_8g6_tests.sh" in h1


def test_h4_g6_chain_continues_into_completed_a_to_g_regression_ownership() -> None:
    g6 = G6.read_text(encoding="utf-8")
    assert "run_phase8_8g5_tests.sh" in g6
    assert "test_phase8_8b1_reconstruction_page.py" in g6
    assert "test_phase8_8c5_tracking_diagnostics_hud.py" in g6
    assert "test_phase8_8d5_semantic_object_presentation.py" in g6
    assert "test_phase8_8g5_progressive_disclosure_ux.py" in g6


def test_h4_records_fail_closed_machine_readable_summary() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'summary_path = output_dir / "summary.json"' in text
    assert 'if not result["passed"]:' in text
    assert '"pass": passed' in text
    assert "return 1" in text


def test_h4_does_not_claim_hardware_or_heavy_backend_qualification() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"heavy_backend_executed_by_h4": False' in text
    assert '"hardware_profile_claimed_by_h4": False' in text
    assert '"release_tag_created_by_h4": False' in text
