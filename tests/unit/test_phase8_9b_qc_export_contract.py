from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
EXPORT = ROOT / "core" / "reconstruction" / "export.py"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def _method(page: str, name: str, next_name: str) -> str:
    return page[page.index(f"def {name}") : page.index(f"def {next_name}")]


def test_b_reuses_the_existing_a1_inspection_report_export_source() -> None:
    page = PAGE.read_text(encoding="utf-8")
    export = EXPORT.read_text(encoding="utf-8")
    assert "InspectionReportExportSource" in page
    assert "InspectionReportExportSource" in export
    assert "write_inspection_report_json(source.report, destination)" in export
    assert 'role="inspection_qc_report"' in export
    assert "ReconstructionArtifactAuthority.INSPECTION_DECISION" in export
    assert "write_inspection_report_json" not in page


def test_b_requires_explicit_typed_authority_and_reuses_e4_presentation() -> None:
    page = PAGE.read_text(encoding="utf-8")
    method = _method(
        page,
        "set_reconstruction_inspection_qc_export_source",
        "clear_reconstruction_inspection_qc_evidence",
    )
    assert "InspectionReportExportSource" in method
    assert "set_reconstruction_inspection_qc_evidence" in method
    assert "source.report" in method
    assert "self._inspection_selection" not in method  # Session validation stays delegated to E4.
    assert "inspection_qc_export_source = source" in method
    for forbidden in (
        "inspect_semantic_geometry",
        "write_inspection_report_json",
        "run_reconstruction_pipeline",
        "start_offline(",
        "start_live(",
    ):
        assert forbidden not in method


def test_b_display_only_report_cannot_silently_replace_export_authority() -> None:
    page = PAGE.read_text(encoding="utf-8")
    method = _method(
        page,
        "set_reconstruction_inspection_qc_evidence",
        "set_reconstruction_inspection_qc_export_source",
    )
    assert "export_source.report is not inspection_report" in method
    assert "self._inspection_qc_export_source = None" in method
    clear = _method(
        page,
        "clear_reconstruction_inspection_qc_evidence",
        "_refresh_reconstruction_inspection_qc_presentation",
    )
    assert "self._inspection_qc_export_source = None" in clear
    assert "self._refresh_reconstruction_export_controls()" in clear


def test_b_qc_is_one_source_in_the_existing_product_bundle_not_a_second_export_system() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("def _qualified_reconstruction_export_sources")
    end = page.index("def _load_persistent_scene_interaction", start)
    block = page[start:end]
    assert "self._inspection_qc_export_source" in block
    assert "active_session" in block
    assert "InspectionReportExportSource" in block
    assert block.count("export_reconstruction_bundle(") == 1
    assert block.count("verify_reconstruction_export_bundle(") == 1
    assert 'QPushButton("Export Bundle")' not in block  # Button stays in the existing session group.
    assert "inspection/QC evidence included" in block
    assert "Certified metrology remains false" in block


def test_b_preserves_locked_sidebar_and_dependency_isolation() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('QPushButton("Export Bundle")') == 1
    assert 'self._settings_group("QC EXPORT")' not in page
    assert 'self._settings_group("INSPECTION / QC")' in page
    assert 'self._settings_group("SESSION & PERSISTENCE")' in page
    heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    imports = _imports(PAGE)
    assert not any(any(value.startswith(prefix) for prefix in heavy) for value in imports)
