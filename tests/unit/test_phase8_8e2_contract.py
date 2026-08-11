from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_measurement_presentation.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_e2_is_presentation_only_and_reuses_phase75_contract() -> None:
    source = MODULE.read_text(encoding="utf-8")
    imports = _imports(MODULE)
    assert "core.inspection.contracts" in imports
    forbidden = (
        "core.measurement.pipeline",
        "core.object_geometry",
        "core.alignment",
        "core.perception",
        "core.reconstruction.inspection_qc",
        "pycolmap",
        "torch",
        "ultralytics",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    for forbidden_call in (
        "compute_measurements(",
        "fit_object_geometry(",
        "extract_object_point_clouds(",
        "inspect_semantic_geometry(",
    ):
        assert forbidden_call not in source


def test_e2_does_not_present_e3_e4_e5_authority() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "validity/confidence deferred to E3" in source
    assert "Phase 8.8E2 cannot claim certified metrology" in source
    # E2 may ignore these incoming Phase 7.5 fields, but production output must
    # not read or render them before E3/E4.
    assert 'raw.get("reliability"' not in source
    assert 'raw.get("expanded_uncertainty' not in source
    assert 'raw.get("rejection_reasons"' not in source
    assert "PASS" not in source
    assert "FAIL" not in source
    lowered = source.lower()
    assert "reference" in lowered and "comparison" in lowered


def test_e2_page_adds_one_compact_measurement_group_without_new_selector() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert source.count('self._settings_group("RECONSTRUCTION MEASUREMENTS")') == 1
    start = source.index('self._reconstruction_measurement_group = self._settings_group("RECONSTRUCTION MEASUREMENTS")')
    end = source.index('self._measurement_validity_group = self._settings_group("MEASUREMENT VALIDITY")', start)
    group = source[start:end]
    assert "QComboBox(" not in group
    assert "QPushButton(" not in group
    assert "QTableWidget" not in group


def test_e2_production_logic_remains_object_agnostic() -> None:
    source = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "pin_top",
        "pin_bottom",
        "missing_pin",
        "bent_pin",
    ):
        assert token not in source
