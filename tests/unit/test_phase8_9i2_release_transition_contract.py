from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase8_9i2_release_transition.py"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _module():
    spec = importlib.util.spec_from_file_location("phase8_9i2_release_transition", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_release_identity_has_advanced_beyond_phase8_transition() -> None:
    version = _read("core/version.py")
    pyproject = _read("pyproject.toml")
    app_yaml = _read("config/app.yaml")
    assert '__version__ = "0.9.0"' in version
    assert 'RELEASE_NAME = "Phase 9 Digital Twin Stable"' in version
    assert "RELEASE_PHASE = 9.0" in version
    assert 'version = "0.9.0"' in pyproject
    assert 'version: "0.9.0"' in app_yaml
    assert 'release: "Phase 9 Digital Twin Stable"' in app_yaml


def test_readme_and_release_notes_are_finalized_without_weakening_authority() -> None:
    readme = _read("README.md")
    notes = _read("RELEASE_NOTES_v0.8.0.md")
    assert "Version `0.9.0` is the **Phase 9 Digital Twin Stable** release" in readme
    assert "Release status" in notes
    assert "Release-preparation status" not in notes
    for phrase in (
        "canonical geometry is the authoritative scene geometry",
        "semantic geometry supplies semantic/object context",
        "Gaussian and NeRF/Nerfacto outputs are **appearance only**",
        "`certified_metrology=false`",
        "camera intrinsics alone do not establish metric scale",
    ):
        assert phrase in readme
    assert "**Gaussian and NeRF/Nerfacto representations are appearance only.**" in notes


def test_main_and_release_runtime_continue_to_consume_central_metadata() -> None:
    main = _read("main.py")
    release = _read("core/runtime/release.py")
    assert "from core.version import RELEASE_NAME, __version__" in main
    assert "QApplication.setApplicationVersion(__version__)" in main
    assert "0.8.0" not in main
    assert "from core.version import RELEASE_NAME, RELEASE_PHASE, __version__" in release
    assert '"version": __version__' in release
    assert '"release_name": RELEASE_NAME' in release
    assert '"release_phase": RELEASE_PHASE' in release


def test_i1_i2_records_define_release_and_tag_boundary() -> None:
    i1 = _read("docs/phase_history/PHASE8_9I1_FINAL_RELEASE_PREFLIGHT.md")
    i2 = _read("docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md")
    assert "f959c369f8aa27347cab20cc90b469c6674d0840" in i1
    assert "I2 source snapshot SHA-256" in i1
    assert "Final identity" in i2
    assert "Historical Phase 8 G/H runners remain frozen" in i2
    assert "8.9I3" in i2


def test_historical_g_h_runner_files_remain_outside_i2_overlay() -> None:
    module = _module()
    historical = {
        "scripts/run_phase8_9g1_regression_inventory.py",
        "scripts/run_phase8_9g2_full_regression.py",
        "scripts/run_phase8_9g3_backward_compatibility.py",
        "scripts/run_phase8_9h2_release_prep_tests.sh",
        "scripts/run_phase8_9h3_release_candidate_qualification.py",
        "scripts/run_phase8_9h3_release_candidate_tests.sh",
        "docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md",
        "docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md",
        "docs/phase_history/PHASE8_9H3_RELEASE_CANDIDATE_QUALIFICATION.md",
    }
    integration_contracts = {
        "tests/integration/test_phase8_9g1_regression_contract.py",
        "tests/integration/test_phase8_9g2_full_regression_contract.py",
    }
    assert historical.isdisjoint(module.EXPECTED_OVERLAY)
    assert historical <= set(module.PROTECTED_HISTORICAL_HASHES)
    assert integration_contracts <= module.EXPECTED_OVERLAY
    assert integration_contracts == module.POST_TRANSITION_HISTORICAL_CLI_CONTRACTS

    g1 = _read("tests/integration/test_phase8_9g1_regression_contract.py")
    g2 = _read("tests/integration/test_phase8_9g2_full_regression_contract.py")
    assert 'assert "release_version_still_0_7_5" in blockers' in g1
    assert 'assert release["core_version"] == "0.9.0"' in g1
    assert 'assert "repository_preflight:version_0_7_5" in blockers' in g2
    assert 'assert release["core_version"] == "0.9.0"' in g2


def test_i2_runner_cannot_create_release_tag() -> None:
    source = _read("scripts/run_phase8_9i2_release_transition.py")
    shell = _read("scripts/run_phase8_9i2_tests.sh")
    forbidden = (
        "git tag -a",
        "git tag v0.8.0",
        "git push origin v0.8.0",
        '"tag", "-a"',
    )
    for token in forbidden:
        assert token not in source
        assert token not in shell
    assert "v0_8_0_local_absent" in source
    assert "v0_8_0_remote_absent" in source



def test_status_parser_preserves_first_porcelain_path(monkeypatch) -> None:
    module = _module()

    class Result:
        stdout = (
            " M README.md\n"
            " M RELEASE_NOTES_v0.8.0.md\n"
            "?? docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md\n"
        )

    monkeypatch.setattr(module, "_run", lambda *args, **kwargs: Result())

    assert module._status_paths() == [
        "README.md",
        "RELEASE_NOTES_v0.8.0.md",
        "docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md",
    ]

def test_historical_i2_runner_truthfully_rejects_post_v080_current_identity() -> None:
    module = _module()
    payload = module.qualify(strict_git=False)
    assert payload["pass"] is False
    assert "metadata_final_0_8_0" in payload["blockers"]
    assert payload["checks"]["historical_g1_g2_cli_contracts_transitioned"] is False
    assert payload["release_tag_created_by_i2"] is False
    assert payload["production_reconstruction_implementation_changed_by_i2"] is False
    assert payload["heavy_real_backend_campaign_started_by_i2"] is False
