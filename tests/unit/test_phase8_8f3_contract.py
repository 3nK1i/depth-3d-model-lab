from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_nerf_visualization.py"
OVERLAY = ROOT / "app" / "widgets" / "reconstruction_appearance_preview.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
F2_CONTRACT = ROOT / "tests" / "unit" / "test_phase8_8f2_contract.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_f3_does_not_import_or_execute_nerf_backends() -> None:
    imports = _imports(MODULE) + _imports(OVERLAY) + _imports(PAGE)
    forbidden = (
        "nerfstudio",
        "tinycudann",
        "torch",
        "gsplat",
        "pycolmap",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    combined = MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")
    for call in (
        "run_nerf_external_worker(",
        "NerfWorkerProcessSpec(",
        "subprocess.run(",
        "subprocess.Popen(",
        "torch.load(",
        ".render_nerf(",
    ):
        assert call not in combined


def test_f3_ui_has_one_nerf_visualization_group_with_explicit_preview_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("NERF VISUALIZATION")') == 1
    start = page.index('self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")')
    end = page.index('authority = self._settings_group("AUTHORITY & SAFETY")', start)
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


def test_f3_uses_backend_authored_images_not_nerf_scene_runtime_payload() -> None:
    module = MODULE.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "preview.png" in module
    assert "e2_runtime_evidence.json" in module
    assert "heldout_render_manifest.json" in module
    assert "config.yml" not in module
    assert ".ckpt" not in module
    assert "F3 performs no NeRF rendering" in page
    assert "F5 reports resource state and canonical fallback" in PAGE.read_text(encoding="utf-8")


def test_f3_is_object_agnostic_and_preserves_appearance_authority() -> None:
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


def test_f2_contract_boundary_stops_before_f3_group() -> None:
    source = F2_CONTRACT.read_text(encoding="utf-8")
    assert 'self._nerf_visualization_group = self._settings_group("NERF VISUALIZATION")' in source
