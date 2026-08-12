from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase8_9g2_full_regression.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase8_9g2_full_regression", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g2_matrix_matches_frozen_g1_plan_exactly():
    module = _module()
    matrix = module.build_matrix()
    assert [item["name"] for item in matrix] == list(module.GROUP_NAMES)
    assert len(matrix) == 7


def test_g2_primary_suites_are_separate_pytest_process_commands():
    module = _module()
    by_name = {item["name"]: item for item in module.build_matrix()}
    for name, suite in {
        "unit": "tests/unit",
        "ui": "tests/ui",
        "integration": "tests/integration",
        "stress": "tests/stress",
    }.items():
        assert by_name[name]["command"][:3] == [module.sys.executable, "-m", "pytest"]
        assert by_name[name]["command"][-1] == suite


def test_g2_inherited_protocol_gate_is_exact():
    module = _module()
    by_name = {item["name"]: item for item in module.build_matrix()}
    assert by_name["phase8_8_protocol"]["command"] == [
        "bash",
        "scripts/run_phase8_8h3_tests.sh",
        "protocol",
    ]


def test_g2_phase89_quick_chain_is_exact_a1_through_f6():
    module = _module()
    assert len(module.PHASE89_QUICK_RUNNERS) == 12
    assert module.PHASE89_QUICK_RUNNERS[0] == "scripts/run_phase8_9a1_tests.sh"
    assert module.PHASE89_QUICK_RUNNERS[-1] == "scripts/run_phase8_9f6_tests.sh"
    assert len(set(module.PHASE89_QUICK_RUNNERS)) == 12
    for runner in module.PHASE89_QUICK_RUNNERS:
        assert (ROOT / runner).is_file()


def test_g2_matrix_does_not_invoke_real_f_campaigns_or_backend_setup():
    module = _module()
    text = "\n".join(
        " ".join(str(part) for part in entry.get("command", []))
        for entry in module.build_matrix()
    )
    forbidden = (
        "run_phase8_9f2_real_geometry_campaign.py",
        "run_phase8_9f3_real_gaussian_campaign.py",
        "run_phase8_9f4_real_nerf_campaign.py",
        "setup_nerfstudio_backend.sh",
        "setup_gaussian_backends.sh",
    )
    assert all(token not in text for token in forbidden)


def test_g2_non_strict_repository_preflight_preserves_release_and_isolation():
    module = _module()
    snapshot = module._repository_snapshot(strict_git=False)
    assert snapshot["pass"] is True
    assert snapshot["checks"]["version_0_7_5"] is True
    assert snapshot["checks"]["no_direct_heavy_imports"] is True
    assert snapshot["direct_heavy_imports"] == []


def test_g2_pytest_count_parser_handles_pass_skip_and_error_forms():
    module = _module()
    counts = module._pytest_counts("120 passed, 2 skipped, 1 xfailed in 4.2s\n1 error in 0.1s")
    assert counts == {"passed": 120, "skipped": 2, "xfailed": 1, "errors": 1}


def test_g2_plan_only_is_green_without_starting_execution(tmp_path):
    module = _module()
    payload = module.run_regression(tmp_path / "g2", strict_git=False, plan_only=True)
    assert payload["pass"] is True
    assert payload["execution_started"] is False
    assert payload["groups"] == []
    assert payload["heavy_real_training_or_inference_started"] is False


def test_g2_overlay_scope_contains_only_qualification_files():
    module = _module()
    assert module.G2_OVERLAY == {
        "docs/phase_history/PHASE8_9G2_VALIDATION.md",
        "scripts/run_phase8_9g2_full_regression.py",
        "scripts/run_phase8_9g2_tests.sh",
        "tests/integration/test_phase8_9g2_full_regression_contract.py",
        "tests/unit/test_phase8_9g2_full_regression.py",
    }
    assert not any(path.startswith(("app/", "core/", "config/")) for path in module.G2_OVERLAY)


def test_g2_failure_policy_and_release_scope_are_locked():
    module = _module()
    payload = module.run_regression(Path("/tmp/unused-g2-unit"), strict_git=False, plan_only=True)
    assert payload["failure_policy"] == "stop_and_diagnose; never weaken tests to obtain green"
    assert payload["production_code_changed_by_8_9g2"] is False
    assert payload["release_tag_created_by_8_9g2"] is False
