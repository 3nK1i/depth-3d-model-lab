"""Release metadata and report integrity tests."""

from __future__ import annotations

import json
from pathlib import Path

from core.runtime.release import (
    CommandResult,
    ReleaseCheck,
    build_release_report,
    project_root,
    release_failed,
    version_consistency_check,
    write_release_report,
)
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


def test_repository_release_versions_are_consistent() -> None:
    result = version_consistency_check(project_root())
    assert result.passed, result.detail
    assert __version__ == "0.8.0"
    assert RELEASE_NAME == "Phase 8 Reconstruction Stable"
    assert RELEASE_PHASE == 8.0


def test_version_check_reports_config_drift(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0.8.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "config" / "app.yaml").write_text(
        'app:\n  version: "9.9.9"\n',
        encoding="utf-8",
    )

    result = version_consistency_check(tmp_path)

    assert not result.passed
    assert "config=9.9.9" in result.detail


def test_release_report_is_atomic_and_machine_readable(tmp_path: Path) -> None:
    checks = [ReleaseCheck("static", True, "ok")]
    commands = [CommandResult("compile", ("python", "-m", "compileall"), 0, 0.1, "", "")]
    payload = build_release_report(checks=checks, commands=commands, mode="quick")
    output = tmp_path / "release" / "report.json"

    written = write_release_report(output, payload)
    restored = json.loads(written.read_text(encoding="utf-8"))

    assert restored["passed"] is True
    assert restored["application"]["version"] == __version__
    assert restored["commands"][0]["passed"] is True
    assert not list(output.parent.glob("*.tmp"))


def test_required_failure_controls_release_exit() -> None:
    optional = ReleaseCheck("git_clean", False, "dirty", required=False)
    required = ReleaseCheck("tests", False, "failed", required=True)

    assert not release_failed([optional], [])
    assert release_failed([optional, required], [])
