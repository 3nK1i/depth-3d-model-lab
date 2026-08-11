from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_measurement_validity_presentation.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
E2 = ROOT / "app" / "reconstruction_measurement_presentation.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_e3_is_presentation_only_and_reuses_phase75_inspection_contracts() -> None:
    source = MODULE.read_text(encoding="utf-8")
    imports = _imports(MODULE)
    assert "core.inspection.contracts" in imports
    forbidden = (
        "core.measurement.pipeline",
        "core.measurement.reliability",
        "core.measurement.uncertainty",
        "core.object_geometry",
        "core.alignment",
        "core.perception",
        "core.reconstruction.inspection_qc",
        "core.reconstruction.scale_reference",
        "pycolmap",
        "torch",
        "ultralytics",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    for forbidden_call in (
        "compute_measurements(",
        "assess_reliability(",
        "estimate_uncertainty(",
        "inspect_semantic_geometry(",
    ):
        assert forbidden_call not in source


def test_e3_requires_exact_e2_target_source_and_observation_provenance() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert "requires existing E2 measurement evidence for the active target" in source
    assert "validity source must match the exact E2 measurement source" in source
    assert "validity observation must match the exact E2 measurement observation" in source
    assert "never guesses correspondence" in source


def test_e3_does_not_claim_e4_qc_e5_reference_or_certified_metrology() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "Phase 8.8E3 cannot claim certified metrology" in source
    assert "not QC decision" in source
    assert "scene scale unchanged" in source
    for token in ("PASS", "FAIL"):
        assert token not in source
    lowered = source.lower()
    assert "reference" in lowered and "comparison" in lowered


def test_e3_does_not_use_semantic_or_detection_confidence_as_measurement_reliability() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert 'raw.get("reliability"' in source
    assert 'raw.get("confidence"' not in source
    assert 'raw.get("detection_confidence"' not in source
    assert "semantic confidence" not in source.lower()


def test_e3_production_logic_remains_object_agnostic_and_ui_has_no_new_selector() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "pin_top",
        "pin_bottom",
        "missing_pin",
        "bent_pin",
    ):
        assert token not in combined

    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("MEASUREMENT VALIDITY")') == 1
    start = page.index('self._measurement_validity_group = self._settings_group("MEASUREMENT VALIDITY")')
    end = page.index('self._inspection_qc_group = self._settings_group("INSPECTION / QC")', start)
    group = page[start:end]
    assert "QComboBox(" not in group
    assert "QPushButton(" not in group
    assert "QTableWidget" not in group


def test_e3_keeps_phase75_volume_unit_compatibility_in_e2_and_e3_presenters() -> None:
    assert '"m3", "m^3"' in E2.read_text(encoding="utf-8")
    assert '"m3", "m^3"' in MODULE.read_text(encoding="utf-8")
