from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase8_9g3_backward_compatibility.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase8_9g3_backward_compatibility", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g3_matrix_matches_frozen_g1_backward_plan_exactly():
    module = _module()
    matrix = module.build_matrix()
    assert [item["name"] for item in matrix] == list(module.GROUP_NAMES)
    assert len(matrix) == 6


def test_g3_legacy_release_commands_are_exact():
    module = _module()
    by_name = {item["name"]: item for item in module.build_matrix()}
    assert by_name["phase5_smoke"]["command"] == ["bash", "scripts/run_phase5_tests.sh", "smoke"]
    assert by_name["phase6_release_quick"]["command"] == ["bash", "scripts/run_phase6_step7_tests.sh", "quick"]
    assert by_name["phase7_release_quick"]["command"] == ["bash", "scripts/run_phase7_step4_tests.sh", "quick"]
    assert by_name["phase7_5_release_quick"]["command"] == ["bash", "scripts/run_phase7_5_step10_tests.sh", "quick"]
    assert by_name["phase8_8_backward"]["command"] == ["bash", "scripts/run_phase8_8h3_tests.sh", "backward"]


def test_g3_main_window_gate_is_bounded_to_new_integration_contract():
    module = _module()
    by_name = {item["name"]: item for item in module.build_matrix()}
    command = by_name["main_window_startup_shutdown"]["command"]
    assert command[:3] == [module.sys.executable, "-m", "pytest"]
    assert command[-1] == "tests/integration/test_phase8_9g3_backward_compatibility_contract.py"


def test_g3_does_not_duplicate_complete_g2_primary_suites():
    module = _module()
    text = "\n".join(" ".join(str(part) for part in item["command"]) for item in module.build_matrix())
    for suite in ("tests/unit", "tests/ui", "tests/integration", "tests/stress"):
        assert f"pytest -q {suite}\n" not in text
    assert "run_phase8_9g2_full_regression.py" not in text


def test_g3_does_not_invoke_real_reconstruction_campaigns_or_external_setup():
    module = _module()
    text = "\n".join(" ".join(str(part) for part in item["command"]) for item in module.build_matrix())
    forbidden = (
        "run_phase8_9f2_real_geometry_campaign.py",
        "run_phase8_9f3_real_gaussian_campaign.py",
        "run_phase8_9f4_real_nerf_campaign.py",
        "setup_nerfstudio_backend.sh",
        "setup_gaussian_backends.sh",
        "setup_da3",
    )
    assert all(token not in text for token in forbidden)


def test_g3_historical_preflight_retains_old_release_gate_and_isolation():
    module = _module()
    snapshot = module._repository_snapshot(strict_git=False)
    assert snapshot["pass"] is False
    assert snapshot["checks"]["version_0_7_5"] is False
    assert snapshot["checks"]["v0_8_0_absent"] is True
    assert snapshot["checks"]["no_direct_heavy_imports"] is True
    assert snapshot["checks"]["g2_evidence_present"] is True
    assert snapshot["direct_heavy_imports"] == []


def test_g3_pytest_count_parser_sums_multiple_legacy_subruns():
    module = _module()
    counts = module._pytest_counts("10 passed in 1s\n7 passed, 1 skipped in 2s\n2 errors in 1s")
    assert counts == {"passed": 17, "skipped": 1, "errors": 2}


def test_g3_plan_only_remains_frozen_pre_release_evidence_after_v080(tmp_path):
    module = _module()
    payload = module.run_gate(tmp_path / "g3", strict_git=False, plan_only=True)
    assert payload["pass"] is False
    assert payload["execution_started"] is False
    assert payload["groups"] == []
    assert "repository_preflight" in payload["blockers"]
    assert payload["production_code_changed_by_8_9g3"] is False
    assert payload["heavy_real_reconstruction_campaign_started_by_8_9g3"] is False
    assert payload["release_tag_created_by_8_9g3"] is False


def test_g3_overlay_scope_is_qualification_only():
    module = _module()
    assert module.G3_OVERLAY == {
        "docs/phase_history/PHASE8_9G3_VALIDATION.md",
        "scripts/run_phase8_9g3_backward_compatibility.py",
        "scripts/run_phase8_9g3_tests.sh",
        "tests/integration/test_phase8_9g3_backward_compatibility_contract.py",
        "tests/unit/test_phase8_9g3_backward_compatibility.py",
    }
    assert not any(path.startswith(("app/", "core/", "config/", "requirements/")) for path in module.G3_OVERLAY)


def test_g3_failure_policy_and_next_step_are_locked(tmp_path):
    module = _module()
    payload = module.run_gate(tmp_path / "g3", strict_git=False, plan_only=True)
    assert payload["failure_policy"] == "stop_and_diagnose; never weaken tests to obtain green"
    assert payload["truthful_scope"]["g2_full_suite_duplication"] is False
    assert payload["truthful_scope"]["next_step_if_pass"] == "8.9H_documentation_release_preparation"
