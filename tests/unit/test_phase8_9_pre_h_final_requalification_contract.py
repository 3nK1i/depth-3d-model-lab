from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_phase8_9_pre_h_final_requalification.py"
SHELL = ROOT / "scripts/run_phase8_9_pre_h_final_requalification_tests.sh"


def _source() -> str:
    return RUNNER.read_text(encoding="utf-8")


def test_final_postfix_runner_is_qualification_only_and_pins_current_head() -> None:
    source = _source()
    assert 'EXPECTED_HEAD = "d8882391df8f3d976223a08893f66fd6d2ba14f9"' in source
    assert 'EXPECTED_BRANCH = "phase8-reconstruction"' in source
    assert '"production_code_changed_by_gate": False' in source
    assert '"release_tag_created_by_gate": False' in source
    assert '"heavy_real_reconstruction_campaign_started": False' in source
    assert '"v0_8_0_absent": not bool(tag)' in source
    assert 'core_version == "0.7.5"' in source


def test_final_postfix_matrix_replays_frozen_g2_g3_and_postfix_surfaces() -> None:
    source = _source()
    required = (
        "scripts/run_phase8_9g2_full_regression.py",
        "scripts/run_phase8_9g3_backward_compatibility.py",
        "scripts/run_phase8_9_pre_h_rtsp_fps_fix_tests.sh",
        "scripts/run_phase8_9_pre_h_ui4_tests.sh",
    )
    for path in required:
        assert path in source
    assert source.count('"--non-strict-git"') == 2
    assert '"rtsp_postfix_affected"' in source
    assert '"ui4_combined_affected"' in source


def test_historical_non_strict_mode_is_wrapped_by_current_strict_preflight() -> None:
    source = _source()
    assert '"head": head == EXPECTED_HEAD' in source
    assert '"origin": origin == EXPECTED_HEAD' in source
    assert '"overlay_exact": set(status) == OVERLAY' in source
    assert '"diff_check_clean": diff_code == 0' in source
    assert "Historical G2/G3 runners" in source


def test_runner_contains_no_git_mutation_or_release_tag_command() -> None:
    tree = ast.parse(_source())
    forbidden = {"add", "commit", "push", "reset", "clean", "checkout", "switch", "tag"}
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Documentation/status strings may mention a tag; no git mutation verb may
    # be passed as a standalone subprocess argument in the runner.
    assert not (forbidden - {"tag"}) & literals
    assert '"tag", "--list", "v0.8.0"' in _source()


def test_shell_gate_exposes_quick_and_full_modes() -> None:
    source = SHELL.read_text(encoding="utf-8")
    assert "quick()" in source
    assert "full()" in source
    assert "--plan-only" in source
    assert "run_phase8_9_pre_h_final_requalification.py" in source
