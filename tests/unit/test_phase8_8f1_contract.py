from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_appearance_presentation.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_f1_does_not_import_or_launch_heavy_gaussian_nerf_runtimes() -> None:
    imports = _imports(MODULE) + _imports(PAGE)
    forbidden = (
        "gsplat",
        "nerfstudio",
        "gaussian_renderer",
        "diff_gaussian_rasterization",
        "simple_knn",
        "torch",
        "pycolmap",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    combined = MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")
    for call in (
        "run_gaussian_external_worker(",
        "run_nerf_external_worker(",
        "subprocess.run(",
        "subprocess.Popen(",
        ".render_gaussian(",
        ".render_nerf(",
    ):
        assert call not in combined


def test_f1_ui_has_one_metadata_control_group_and_no_render_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("APPEARANCE REPRESENTATIONS")') == 1
    start = page.index('self._appearance_group = self._settings_group("APPEARANCE REPRESENTATIONS")')
    end = page.index('self._gaussian_visualization_group = self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")', start)
    group = page[start:end]
    assert "QComboBox()" in group
    assert "QPushButton(" not in group
    assert "QFileDialog" not in group
    assert "set_gaussian" not in group
    assert "set_nerf" not in group
    assert '"Available"' in group
    assert '"Provider"' in group
    assert '"Artifact"' in group
    assert '"Render"' in group
    assert '"Authority"' in group


def test_f1_is_object_agnostic_and_representation_metadata_driven() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "missing_pin",
        "bent_pin",
        'class_name == "gear"',
        'class_name == "housing"',
        'semantic_role ==',
    ):
        assert token not in combined
    assert "artifact_id" in MODULE.read_text(encoding="utf-8")
    assert "provider_id" in MODULE.read_text(encoding="utf-8")
    assert "render_capable" in MODULE.read_text(encoding="utf-8")


def test_f1_preserves_authority_as_f4_adds_viewport_switching() -> None:
    module = MODULE.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "Appearance only • canonical geometry remains authoritative" in module
    assert "viewport switching available in F4" in module
    assert "F4 reuses this selector as the explicit" in page
    assert "F1 inspects persisted appearance metadata only" in page
    assert "certified metrology" in module.lower()


def test_e5_contract_boundary_stops_before_f1_group() -> None:
    source = (ROOT / "tests" / "unit" / "test_phase8_8e5_contract.py").read_text(encoding="utf-8")
    assert 'self._appearance_group = self._settings_group("APPEARANCE REPRESENTATIONS")' in source
