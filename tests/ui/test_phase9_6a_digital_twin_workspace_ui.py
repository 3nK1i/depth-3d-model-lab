from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"


def test_96a_ui_adds_one_compact_unified_workspace_summary_without_replacing_existing_groups() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'self._settings_group("DIGITAL TWIN WORKSPACE")' in page
    for label in ("Workspace", "Active view", "Composition", "Scale", "Authority", "Resources", "Safety"):
        assert f'"{label}"' in page
    for locked_group in (
        'self._settings_group("CANONICAL SCENE")',
        'self._settings_group("SCENE LAYERS")',
        'self._settings_group("RECONSTRUCTION MEASUREMENTS")',
        'self._settings_group("INSPECTION / QC")',
        'self._settings_group("APPEARANCE REPRESENTATIONS")',
        'self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")',
        'self._settings_group("NERF VISUALIZATION")',
        'self._settings_group("AUTHORITY & SAFETY")',
    ):
        assert locked_group in page


def test_96a_ui_uses_passive_composition_and_explicit_authority_wording() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "build_digital_twin_workspace_presentation" in page
    assert "digital_twin_workspace_changed = pyqtSignal(object)" in page
    assert "Canonical geometry • picking + measurement" in (ROOT / "app/reconstruction_digital_twin_workspace.py").read_text(encoding="utf-8")
    assert "Appearance only overlays • QC/reference separate • metrology not certified" in (ROOT / "app/reconstruction_digital_twin_workspace.py").read_text(encoding="utf-8")
    for forbidden in (
        "import gsplat",
        "import nerfstudio",
        "import tinycudann",
        "from depth_anything_3",
    ):
        assert forbidden not in page
