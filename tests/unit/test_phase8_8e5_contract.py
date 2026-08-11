from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_reference_comparison.py"
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


def test_e5_presentation_does_not_import_or_run_registration_qc_algorithms() -> None:
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


def test_e5_uses_exact_reference_id_and_candidate_object_id_not_class_guessing() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "report_reference_id == descriptor.reference_id" in source
    assert 'feature_result.get("candidate_object_id"' in source
    assert "== selection.object_id" in source
    for token in ('feature_result.get("class_name"', 'feature_result.get("semantic_role"', 'feature_result.get("confidence"'):
        assert token not in source


def test_e5_remains_object_agnostic_and_does_not_add_use_case_specific_logic() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "missing_pin",
        "bent_pin",
        "misaligned_pin",
        'class_name == "gear"',
        'class_name == "housing"',
    ):
        assert token not in combined


def test_e5_ui_has_one_reference_group_and_load_clear_only() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._settings_group("REFERENCE COMPARISON")') == 1
    start = page.index('self._reference_comparison_group = self._settings_group("REFERENCE COMPARISON")')
    end = page.index('authority = self._settings_group("AUTHORITY & SAFETY")', start)
    group = page[start:end]
    assert 'QPushButton("Load JSON")' in group
    assert 'QPushButton("Clear")' in group
    assert 'QPushButton("Compare")' not in group
    assert '"Deviation"' in group
    assert '"Registration"' in group
    assert '"Scale"' in group


def test_e5_does_not_modify_core_or_claim_certified_metrology() -> None:
    source = MODULE.read_text(encoding="utf-8").lower()
    assert "not certified metrology" in source
    assert "cannot claim certified metrology" in source
    assert "does not register/apply tolerances" in source
