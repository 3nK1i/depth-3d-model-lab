"""Phase 7 release identity and required operational surface tests."""

from __future__ import annotations

from pathlib import Path

from core.runtime.release import collect_static_checks, project_root
from core.version import RELEASE_NAME, RELEASE_PHASE, __version__


def test_phase7_release_identity() -> None:
    assert __version__ == "0.7.0"
    assert RELEASE_NAME == "Phase 7 Stable"
    assert RELEASE_PHASE == 7


def test_phase7_required_files_are_present() -> None:
    checks = collect_static_checks(project_root(), require_model_dependencies=False)
    layout = next(item for item in checks if item.name == "project_layout")
    assert layout.passed, layout.detail


def test_phase7_release_scripts_are_executable() -> None:
    root = Path(project_root())
    for relative in ("scripts/run_phase7_release.sh", "scripts/run_phase7_step4_tests.sh"):
        path = root / relative
        assert path.is_file()
        assert path.stat().st_mode & 0o111
