from __future__ import annotations

from pathlib import Path

from core.version import RELEASE_NAME, __version__


def test_release_identity_stays_on_phase7_until_measured_gate_passes() -> None:
    assert __version__ == "0.7.0"
    assert RELEASE_NAME == "Phase 7 Stable"


def test_step10_does_not_enable_reconstruction() -> None:
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_line = next(line for line in source.splitlines() if line.startswith("_ENABLED_MODES"))
    assert "AppMode.RECONSTRUCTION" not in enabled_line
