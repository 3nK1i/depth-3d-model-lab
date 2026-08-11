from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
DISCLOSURE = ROOT / "app" / "widgets" / "reconstruction_settings_disclosure.py"
THEME = ROOT / "app" / "theme.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_g5_uses_one_generic_disclosure_widget_for_reconstruction_settings() -> None:
    page = PAGE.read_text(encoding="utf-8")
    widget = DISCLOSURE.read_text(encoding="utf-8")
    assert "class ReconstructionSettingsDisclosureGroup" in widget
    assert "return ReconstructionSettingsDisclosureGroup(title_text)" in page
    assert "reconstructionSettingsGroup" in widget
    assert "reconstructionSettingsGroupTitle" in widget


def test_g5_disclosure_is_presentation_only() -> None:
    widget = DISCLOSURE.read_text(encoding="utf-8").lower()
    forbidden = (
        "reconstructioncontroller",
        "persistent_scene",
        "engine_checkpoint",
        "prepare_resume",
        "start_runtime",
        "worker_restart",
        "json.dump",
    )
    for value in forbidden:
        assert value not in widget


def test_g5_keeps_primary_context_expanded_by_default() -> None:
    page = PAGE.read_text(encoding="utf-8")
    for title in (
        "WORKFLOW",
        "SESSION & PERSISTENCE",
        "INSPECTION TARGET",
        "APPEARANCE REPRESENTATIONS",
        "AUTHORITY & SAFETY",
    ):
        anchor = f'self._settings_group("{title}")'
        assert anchor in page
        line_end = page.index("\n", page.index(anchor))
        following = page[line_end : line_end + 120]
        assert "set_expanded(False)" not in following


def test_g5_collapses_detail_heavy_groups_by_default() -> None:
    page = PAGE.read_text(encoding="utf-8")
    expected = (
        "_reconstruction_measurement_group",
        "_measurement_validity_group",
        "_inspection_qc_group",
        "_reference_comparison_group",
        "_gaussian_visualization_group",
        "_nerf_visualization_group",
        "_appearance_resource_group",
    )
    for name in expected:
        assert f"self.{name}.set_expanded(False)" in page


def test_g5_does_not_hide_existing_dynamic_offline_or_semantic_groups() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'self._offline_inspection_group = self._settings_group("OFFLINE INSPECTION")' in page
    assert 'self._semantic_objects_group = self._settings_group("SEMANTIC / OBJECTS")' in page
    assert "self._offline_inspection_group.setVisible(False)" in page
    assert "self._semantic_objects_group.setVisible(False)" in page
    assert "self._offline_inspection_group.set_expanded(False)" not in page
    assert "self._semantic_objects_group.set_expanded(False)" not in page


def test_g5_preserves_g3_and_g4_session_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('session = self._settings_group("SESSION & PERSISTENCE")')
    end = page.index("layout.addWidget(session)", start)
    group = page[start:end]
    assert 'QPushButton("Load Checkpoint")' in group
    assert 'QPushButton("Prepare Resume")' in group
    assert 'QPushButton("Retry From Source")' in group


def test_g5_styles_disclosure_control_without_replacing_existing_group_style() -> None:
    theme = THEME.read_text(encoding="utf-8")
    assert "QFrame#reconstructionSettingsGroup" in theme
    assert "QLabel#reconstructionSettingsGroupTitle" in theme
    assert "QToolButton#reconstructionSettingsDisclosureToggle" in theme


def test_g5_stays_generic_and_dependency_light() -> None:
    text = DISCLOSURE.read_text(encoding="utf-8").lower()
    for forbidden in (
        'class_name == "connector"',
        'class_name == "pin"',
        'class_name == "gear"',
        'use_case == "pcb"',
        'industry == "automotive"',
    ):
        assert forbidden not in text
    heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    assert not any(any(value.startswith(prefix) for prefix in heavy) for value in _imports(DISCLOSURE))
