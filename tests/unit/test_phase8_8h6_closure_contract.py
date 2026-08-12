from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_phase8_8h6_closure.py"
SHELL = ROOT / "scripts/run_phase8_8h6_tests.sh"
DOC = ROOT / "docs/phase_history/PHASE8_8H6_VALIDATION.md"


def test_h6_is_closure_only_and_does_not_release_phase8() -> None:
    doc = DOC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "adds no Reconstruction feature" in doc
    assert "It does not mean Phase 8 itself is released" in doc
    assert '"production_behavior_changed_by_h6": False' in runner
    assert '"new_feature_work_added_by_h6": False' in runner
    assert '"phase8_release_created_by_h6": False' in runner


def test_h6_consumes_authoritative_h4_and_h5_machine_readable_evidence() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "/tmp/phase8_8h4_full_regression/summary.json" in text
    assert "/tmp/phase8_8h5_real_ui_acceptance/summary.json" in text
    for token in (
        'h4.get("full_phase8_8_regression_performed") is True',
        'h4.get("phase8_8_protocol_chain_passed") is True',
        'h4.get("phase8_8_backward_chain_passed") is True',
        'h5.get("real_application_executed") is True',
        'h5.get("real_external_pycolmap_executed") is True',
        'h5.get("responsive_1366x768") is True',
        'h5.get("resource_release_after_completion") is True',
    ):
        assert token in text


def test_h6_reuses_focused_h5_supporting_acceptance_instead_of_faking_new_evidence() -> None:
    text = SHELL.read_text(encoding="utf-8")
    assert "run_phase8_8h5_tests.sh quick" in text
    assert "run_phase8_8h5_tests.sh supporting" in text
    assert "run_phase8_8h4_tests.sh full" not in text
    assert "run_phase8_8h5_real_ui_acceptance.py" not in text


def test_h6_requires_exact_closure_only_git_scope() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for path in (
        "docs/phase_history/PHASE8_8H6_VALIDATION.md",
        "scripts/run_phase8_8h6_closure.py",
        "scripts/run_phase8_8h6_tests.sh",
        "tests/unit/test_phase8_8h6_closure_contract.py",
    ):
        assert f'"?? {path}"' in text
    assert "unexpected_status = sorted(porcelain - ALLOWED_H6_STATUS)" in text
    assert "missing_h6_status = sorted(ALLOWED_H6_STATUS - porcelain)" in text


def test_h6_checks_branch_remote_sync_and_preserves_release_metadata() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'EXPECTED_BRANCH = "phase8-reconstruction"' in text
    assert '_git("rev-parse", "origin/phase8-reconstruction")' in text
    assert 'head == origin' in text
    assert '__version__ == "0.7.5"' in text
    assert 'RELEASE_NAME == "Phase 7.5 Stable"' in text
    assert 'float(RELEASE_PHASE) == 7.5' in text


def test_h6_checks_heavy_backend_isolation_without_executing_heavy_backends() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for token in ("pycolmap", "gsplat", "nerfstudio", "stella_vslam", "depth_anything_3"):
        assert f'"{token}"' in text
    assert '"heavy_backend_executed_by_h6": False' in text
    assert '"hardware_profile_claimed_by_h6": False' in text


def test_h6_does_not_create_v0_8_0_tag_and_only_hands_off_to_8_9() -> None:
    doc = DOC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "No `v0.8.0` tag is created by H6" in doc
    assert '_git("tag", "--list", "v0.8.0")' in runner
    assert '"release_tag_created_by_h6": False' in runner
    assert '"handoff_to_phase8_9_ready": passed' in runner


def test_h6_closure_summary_is_fail_closed() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"phase8_8_complete": passed' in text
    assert '"pass": passed' in text
    assert "return 0 if passed else 1" in text
    assert "Required accepted summary is missing" in text
