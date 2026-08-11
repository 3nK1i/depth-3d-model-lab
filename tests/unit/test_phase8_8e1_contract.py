from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_inspection_selection.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def test_e1_selection_module_is_application_only_and_backend_neutral() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    forbidden = (
        "core.reconstruction.inspection_qc",
        "core.reconstruction.scale_reference",
        "core.measurement",
        "pycolmap",
        "torch",
        "ultralytics",
        "gsplat",
        "nerfstudio",
    )
    assert not any(any(name.startswith(prefix) for prefix in forbidden) for name in imports)
    assert "run_measurement" not in source
    assert "inspect_semantic_geometry(" not in source


def test_e1_production_selection_is_object_agnostic() -> None:
    source = MODULE.read_text(encoding="utf-8").lower()
    for use_case_token in (
        "connector_body",
        "connector_pin",
        "pin_top",
        "pin_bottom",
        "bent_pin",
        "missing_pin",
    ):
        assert use_case_token not in source


def test_e1_does_not_duplicate_d5_object_selector() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert source.count("reconstructionSemanticObjectCombo") == 1
    assert "INSPECTION TARGET" in source
    start = source.index('self._inspection_target_group = self._settings_group("INSPECTION TARGET")')
    end = source.index('self._reconstruction_measurement_group = self._settings_group("RECONSTRUCTION MEASUREMENTS")', start)
    e1_group = source[start:end]
    assert "QComboBox(" not in e1_group
    assert "Prev" not in e1_group
    assert "Next" not in e1_group


def test_e1_authority_copy_defers_later_e_series_responsibilities() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "Selection only • measurement/QC not evaluated" in source
    assert "certified_metrology" in source
    assert "Phase 8.8E1 selection cannot claim certified metrology" in source
