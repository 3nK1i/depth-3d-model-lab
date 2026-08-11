from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_inspection_qc_presentation.py"
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


def test_e4_presentation_does_not_import_or_run_registration_qc_backends() -> None:
    imports = _imports(MODULE)
    source = MODULE.read_text(encoding="utf-8")
    forbidden_imports = (
        "core.reconstruction.inspection_qc",
        "core.reconstruction.pipeline_geometry",
        "scipy",
        "pycolmap",
        "torch",
        "ultralytics",
    )
    assert not any(any(value.startswith(prefix) for prefix in forbidden_imports) for value in imports)
    for forbidden_call in (
        "inspect_semantic_geometry(",
        "build_reference_assembly(",
        "linear_sum_assignment(",
        "cKDTree(",
    ):
        assert forbidden_call not in source


def test_e4_requires_exact_candidate_object_id_and_never_guesses_by_class_role_or_confidence() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert 'item.get("candidate_object_id"' in source
    assert "== selection.object_id" in source
    assert 'item.get("class_name"' not in source
    assert 'item.get("semantic_role"' not in source
    assert 'item.get("confidence"' not in source


def test_e4_does_not_reaggregate_decision_or_infer_qc_from_e2_e3() -> None:
    source = MODULE.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert 'payload.get("decision")' in source
    assert 'item.get("decision")' in source
    assert "measurement_validity" not in source
    assert "reconstruction_measurement" not in source
    assert "semantic confidence" not in source.lower()
    assert "infer PASS/FAIL" in page or "infer PASS/FAIL" in source


def test_e4_remains_object_agnostic_and_defers_reference_workflow_to_e5() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    # No use-case class IDs or defect labels in the new E4 production layer.
    for token in (
        "connector_body",
        "connector_pin",
        "missing_pin",
        "bent_pin",
        "misaligned_pin",
        "gear-01",
        "shaft-01",
    ):
        assert token not in combined
    assert "reference workflow deferred to e5" in combined
    assert "certified metrology" in MODULE.read_text(encoding="utf-8").lower()


def test_e4_ui_has_one_read_only_group_and_no_reference_picker() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("INSPECTION / QC")') == 1
    start = page.index('self._inspection_qc_group = self._settings_group("INSPECTION / QC")')
    end = page.index('self._reference_comparison_group = self._settings_group("REFERENCE COMPARISON")', start)
    group = page[start:end]
    for token in ("QComboBox(", "QFileDialog", "QPushButton(", "QTableWidget"):
        assert token not in group
    assert '"Decision"' in group
    assert '"Findings"' in group
    assert '"Registration"' in group
    assert '"Comparison"' in group
