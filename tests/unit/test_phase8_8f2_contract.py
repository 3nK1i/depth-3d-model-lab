from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_gaussian_visualization.py"
OVERLAY = ROOT / "app" / "widgets" / "reconstruction_appearance_preview.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
F1_CONTRACT = ROOT / "tests" / "unit" / "test_phase8_8f1_contract.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_f2_does_not_import_or_execute_gaussian_backends() -> None:
    imports = _imports(MODULE) + _imports(OVERLAY) + _imports(PAGE)
    forbidden = (
        "gsplat",
        "gaussian_renderer",
        "diff_gaussian_rasterization",
        "simple_knn",
        "torch",
        "nerfstudio",
        "pycolmap",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    combined = MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")
    for call in (
        "run_gaussian_external_worker(",
        "GaussianWorkerProcessSpec(",
        "subprocess.run(",
        "subprocess.Popen(",
        "torch.load(",
        "gsplat.render",
    ):
        assert call not in combined


def test_f2_ui_has_one_gaussian_visualization_group_with_explicit_preview_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")') == 1
    start = page.index(
        'self._gaussian_visualization_group = self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")'
    )
    end = page.index('self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")', start)
    group = page[start:end]
    assert 'QPushButton("Show preview")' in group
    assert 'QPushButton("Hide")' in group
    assert "QFileDialog" not in group
    assert "QComboBox(" not in group
    assert '"Provider"' in group
    assert '"Scene"' in group
    assert '"Preview"' in group
    assert '"Scale"' in group
    assert '"Authority"' in group


def test_f2_uses_backend_authored_preview_not_gaussian_scene_payload() -> None:
    module = MODULE.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "preview.png" in module
    assert "qualification_metrics.json" in module
    assert "scene.pt" not in module
    assert "point_cloud.ply" not in module
    assert "F2 performs no Gaussian rendering" in page
    assert "F4 coordinates switching between qualified appearance views" in PAGE.read_text(encoding="utf-8")


def test_f2_is_object_agnostic_and_preserves_appearance_authority() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "missing_pin",
        "bent_pin",
        'class_name == "gear"',
        'class_name == "housing"',
        "semantic_role ==",
    ):
        assert token not in combined
    assert "Appearance only • canonical geometry remains authoritative" in MODULE.read_text(encoding="utf-8")
    assert "certified metrology" in MODULE.read_text(encoding="utf-8").lower()


def test_f1_contract_boundary_stops_before_f2_group() -> None:
    source = F1_CONTRACT.read_text(encoding="utf-8")
    assert 'self._gaussian_visualization_group = self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")' in source
