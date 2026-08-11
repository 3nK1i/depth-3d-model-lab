from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
DISCLOSURE = ROOT / "app" / "widgets" / "reconstruction_settings_disclosure.py"
G4 = ROOT / "app" / "reconstruction_failure_recovery.py"
G6_HELPERS = ROOT / "tests" / "reconstruction_g6_helpers.py"
G6_SMOKE = ROOT / "scripts" / "run_phase8_8g6_full_ux_review_smoke.py"
G6_PREVIEW = ROOT / "scripts" / "run_phase8_8g6_full_ux_review_preview.py"
G6_DOC = ROOT / "docs" / "phase_history" / "PHASE8_8G6_VALIDATION.md"

EXPECTED_GROUPS = (
    "WORKFLOW",
    "SESSION & PERSISTENCE",
    "OFFLINE INSPECTION",
    "SEMANTIC / OBJECTS",
    "INSPECTION TARGET",
    "RECONSTRUCTION MEASUREMENTS",
    "MEASUREMENT VALIDITY",
    "INSPECTION / QC",
    "REFERENCE COMPARISON",
    "APPEARANCE REPRESENTATIONS",
    "GAUSSIAN SPLATTING VISUALIZATION",
    "NERF VISUALIZATION",
    "RESOURCE & FALLBACK",
    "AUTHORITY & SAFETY",
)


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def test_g6_reviews_the_complete_reconstruction_settings_information_architecture() -> None:
    page = PAGE.read_text(encoding="utf-8")
    positions = []
    for title in EXPECTED_GROUPS:
        anchor = f'self._settings_group("{title}")'
        assert page.count(anchor) == 1
        positions.append(page.index(anchor))
    assert positions == sorted(positions)


def test_g6_preserves_one_clear_copy_of_each_primary_runtime_and_session_action() -> None:
    page = PAGE.read_text(encoding="utf-8")
    for label in (
        "Start",
        "Pause",
        "Stop",
        "Load Scene",
        "Save Scene",
        "Reload",
        "Load Checkpoint",
        "Prepare Resume",
        "Retry From Source",
    ):
        assert page.count(f'QPushButton("{label}")') == 1


def test_g6_preserves_resume_and_fresh_retry_execution_boundaries() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "controller.prepare_loaded_engine_resume()" in page
    retry = page[page.index("def _retry_failed_runtime_interaction"):]
    retry = retry[: retry.index("def ", 8)]
    assert "self._start_runtime()" in retry
    assert "prepare_loaded_engine_resume" not in retry
    g4 = G4.read_text(encoding="utf-8")
    assert "Checkpoint evidence is preserved" in g4
    assert "Retry starts a fresh reconstruction session" in g4


def test_g6_preserves_progressive_disclosure_and_vertical_only_scroll() -> None:
    page = PAGE.read_text(encoding="utf-8")
    widget = DISCLOSURE.read_text(encoding="utf-8")
    assert "ScrollBarAlwaysOff" in page
    assert "ScrollBarAsNeeded" in page
    assert "class ReconstructionSettingsDisclosureGroup" in widget
    assert 'self._session_group = session' in page
    for name in (
        "_reconstruction_measurement_group",
        "_measurement_validity_group",
        "_inspection_qc_group",
        "_reference_comparison_group",
        "_gaussian_visualization_group",
        "_nerf_visualization_group",
        "_appearance_resource_group",
    ):
        assert f"self.{name}.set_expanded(False)" in page


def test_g6_keeps_authority_and_safety_language_explicit() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'self._settings_group("AUTHORITY & SAFETY")' in page
    assert 'self._add_settings_row(workflow_layout, "Geometry", "Canonical authority")' in page
    assert 'self._metrology_value = self._settings_value("Not certified")' in page
    assert 'self._add_settings_row(authority_layout, "Heavy backends", "Isolated workers")' in page


def test_g6_review_scripts_are_repo_root_safe_and_do_not_import_heavy_backends() -> None:
    for path in (G6_SMOKE, G6_PREVIEW):
        text = path.read_text(encoding="utf-8")
        assert "ROOT = Path(__file__).resolve().parents[1]" in text
        assert "sys.path.insert(0, str(ROOT))" in text
        imports = _imports(path)
        heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
        assert not any(any(value.startswith(prefix) for prefix in heavy) for value in imports)


def test_g6_fixture_uses_existing_scene_checkpoint_and_mock_pipeline_contracts_only() -> None:
    helper = G6_HELPERS.read_text(encoding="utf-8")
    assert "ReconstructionPersistenceController" in helper
    assert "create_reconstruction_engine_checkpoint" in helper
    assert "write_reconstruction_engine_checkpoint" in helper
    assert "DeterministicMockStageExecutor" in helper
    for forbidden in (
        "subprocess",
        "torch",
        "pycolmap",
        "nerfstudio",
        "gsplat",
        "start_selected_offline",
    ):
        assert forbidden not in helper


def test_g6_is_review_only_and_preserves_generic_scope() -> None:
    doc = G6_DOC.read_text(encoding="utf-8").lower()
    assert "review-only" in doc
    assert "no production behavior change" in doc
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (G6_HELPERS, G6_SMOKE, G6_PREVIEW)
    )
    for forbidden in (
        'class_name == "connector"',
        'class_name == "pin"',
        'class_name == "gear"',
        'use_case == "pcb"',
        'industry == "automotive"',
    ):
        assert forbidden not in combined
