from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"


def test_5g_nerf_panel_exposes_resource_performance_and_lifecycle_diagnostics() -> None:
    source = PAGE.read_text(encoding="utf-8")
    for label in ("Resources", "Performance", "Lifecycle"):
        assert f'"{label}"' in source
    assert "External renderer unloaded" in source
    assert "No interactive render measured" in source
    assert "no NeRF worker resident" in source
    assert "canonical view retained" in source.lower()


def test_5g_diagnostics_do_not_claim_total_application_vram_or_quality_certification() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert "external process only, not total application VRAM usage" in source
    assert "does not certify NeRF visual quality" in source
    assert "canonical geometry remains" in source.lower()
