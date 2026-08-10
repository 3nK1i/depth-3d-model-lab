from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_c5_is_presentation_only_and_does_not_require_native_protocol_changes() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    diagnostics = (ROOT / "app/reconstruction_diagnostics.py").read_text(encoding="utf-8")
    assert "LiveTrackingDiagnosticsAccumulator" in page
    assert "processing_latency_ns" in diagnostics
    assert "Not reported by Stella" in diagnostics


def test_c5_viewport_toolbar_has_explicit_diagnostics_visibility_control() -> None:
    page = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    icons = (ROOT / "app/icons.py").read_text(encoding="utf-8")
    assert '"diagnostics"' in icons
    assert 'self._viewport_tool_buttons["diagnostics"]' in page
    assert "Show or hide live tracking diagnostics" in page
