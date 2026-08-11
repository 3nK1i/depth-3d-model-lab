from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_representation_switching.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
F1 = ROOT / "tests" / "unit" / "test_phase8_8f1_contract.py"
F2 = ROOT / "tests" / "unit" / "test_phase8_8f2_contract.py"
F3 = ROOT / "tests" / "unit" / "test_phase8_8f3_contract.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_f4_switching_layer_does_not_import_or_launch_heavy_backends() -> None:
    imports = _imports(MODULE) + _imports(PAGE)
    forbidden = ("nerfstudio", "tinycudann", "torch", "gsplat", "pycolmap")
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    combined = MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")
    for call in (
        "subprocess.run(",
        "subprocess.Popen(",
        "run_gaussian_external_worker(",
        "run_nerf_external_worker(",
        "torch.load(",
        ".render_gaussian(",
        ".render_nerf(",
    ):
        assert call not in combined


def test_f4_reuses_existing_appearance_selector_without_duplicate_switch_control() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('self._appearance_group = self._settings_group("APPEARANCE REPRESENTATIONS")')
    end = page.index('self._gaussian_visualization_group = self._settings_group("GAUSSIAN SPLATTING VISUALIZATION")', start)
    group = page[start:end]
    assert group.count("QComboBox()") == 1
    assert ".activated.connect(" in group
    assert '"Viewport"' in group
    assert "QPushButton(" not in group
    assert "QFileDialog" not in group


def test_f4_switches_only_existing_f2_f3_previews_and_leaves_f5_policy_deferred() -> None:
    module = MODULE.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "gaussian_presentation" in module
    assert "nerf_presentation" in module
    assert "F5 owns fallback policy" in page
    assert "resource/fallback UX remains F5" in page
    assert "resource_probe" not in module
    assert "fallback_to" not in module


def test_f4_preserves_canonical_authority_and_is_object_agnostic() -> None:
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
    assert "canonical geometry remains authoritative" in combined
    assert "certified metrology" in MODULE.read_text(encoding="utf-8").lower()


def test_prior_f1_f2_f3_contracts_are_updated_only_for_f4_ownership() -> None:
    assert "viewport switching available in F4" in F1.read_text(encoding="utf-8")
    assert "F4 coordinates switching between qualified appearance views" in F2.read_text(encoding="utf-8")
    assert "F4 coordinates switching between qualified appearance views" in F3.read_text(encoding="utf-8")
