from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase8_9g1_regression_inventory.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase8_9g1_inventory", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_g1_source_inventory_retains_historical_pre_release_gate():
    module = _module()
    payload = module.build_inventory(strict_git=False)
    assert payload["checks"]["release_version_still_0_7_5"] is False
    assert payload["checks"]["no_direct_heavy_imports_in_core_reconstruction"] is True
    assert payload["checks"]["legacy_backward_gate_inputs_present"] is True


def test_g1_primary_test_suites_are_all_present_and_nonempty():
    module = _module()
    counts = module._test_file_counts()
    assert set(counts) == {"unit", "ui", "integration", "stress"}
    assert all(value > 0 for value in counts.values())


def test_g1_phase89_runner_chain_is_exact_and_supports_quick_affected():
    module = _module()
    payload = module.build_inventory(strict_git=False)
    runners = payload["inventory"]["phase8_9_runners"]
    assert len(runners) == 12
    assert set(runners) == set(module.PHASE89_RUNNERS)
    for modes in runners.values():
        assert {"quick", "affected"}.issubset(set(modes))


def test_g1_inherits_complete_h4_regression_groups():
    module = _module()
    payload = module.build_inventory(strict_git=False)
    groups = payload["inventory"]["inherited_h4_groups"]
    assert all(groups.values())
    assert {"compileall", "unit", "ui", "integration", "stress"}.issubset(groups)
    assert groups["phase8_8_protocol"] is True
    assert groups["phase8_8_backward"] is True


def test_g1_backward_inputs_cover_stable_release_layers():
    module = _module()
    payload = module.build_inventory(strict_git=False)
    assert payload["checks"]["legacy_backward_gate_inputs_present"] is True
    legacy = payload["inventory"]["legacy_runners"]
    assert "smoke" in legacy["phase5"]
    assert "quick" in legacy["phase6_release"]
    assert "quick" in legacy["phase7_release"]
    assert "quick" in legacy["phase7_5_release"]
    assert "backward" in legacy["phase8_8_h3"]


def test_g1_historical_runner_observes_current_v080_release_identity():
    module = _module()
    release = module._release_version()
    assert release["core_version"] == "0.8.0"
    assert release["config_version"] == "0.8.0"
    # G1 itself remains frozen as pre-release evidence; the current release
    # identity is qualified by the dedicated 8.9I2 gate instead.


def test_g1_no_direct_heavy_imports_in_core_reconstruction():
    module = _module()
    assert module._direct_heavy_imports() == []


def test_g2_matrix_keeps_four_pytest_suites_in_separate_processes():
    module = _module()
    groups = {item["name"]: item for item in module.G2_MATRIX}
    for name, suite in {
        "unit": "tests/unit",
        "ui": "tests/ui",
        "integration": "tests/integration",
        "stress": "tests/stress",
    }.items():
        assert groups[name]["command"][-1] == suite
        assert groups[name]["command"][:3] == [module.sys.executable, "-m", "pytest"]


def test_g3_matrix_is_targeted_not_a_second_full_suite_copy():
    module = _module()
    payload = module.build_inventory(strict_git=False)
    assert payload["planned_g3"]["full_suite_duplication"] is False
    names = {item["name"] for item in payload["planned_g3"]["groups"]}
    assert {
        "phase5_smoke",
        "phase6_release_quick",
        "phase7_release_quick",
        "phase7_5_release_quick",
        "phase8_8_backward",
        "main_window_startup_shutdown",
    } == names


def test_g1_is_qualification_only():
    module = _module()
    scope = module.build_inventory(strict_git=False)["truthful_scope"]
    assert scope == {
        "production_code_changed_by_8_9g1": False,
        "heavy_backend_started_by_8_9g1": False,
        "new_regression_framework_created": False,
        "release_tag_created": False,
        "g1_is_inventory_and_preflight_only": True,
    }
