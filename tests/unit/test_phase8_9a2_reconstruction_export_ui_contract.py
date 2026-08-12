from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def test_a2_wires_only_the_existing_a1_reconstruction_export_contract() -> None:
    page = PAGE.read_text(encoding="utf-8")
    for symbol in (
        "OfflineSparseExportSource",
        "CanonicalSurfaceMeshExportSource",
        "InspectionReportExportSource",  # Added by the Phase 8.9B extension.
        "ReconstructionExportRequest",
        "ReconstructionExportSource",
        "export_reconstruction_bundle",
        "verify_reconstruction_export_bundle",
    ):
        assert symbol in page
    imports = _imports(PAGE)
    for heavy in ("pycolmap", "nerfstudio", "gsplat"):
        assert not any(value.startswith(heavy) for value in imports)


def test_a2_keeps_export_separate_from_persistent_scene_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('session = self._settings_group("SESSION & PERSISTENCE")')
    end = page.index("layout.addWidget(session)", start)
    session = page[start:end]
    assert session.count('QPushButton("Load Scene")') == 1
    assert session.count('QPushButton("Save Scene")') == 1
    assert session.count('QPushButton("Reload")') == 1
    assert session.count('QPushButton("Export Bundle")') == 1
    assert session.count('QPushButton("Open Export")') == 1
    assert '"Product export"' in session
    assert "separate from Save Scene" in session
    assert session.index('QPushButton("Export Bundle")') > session.index('QPushButton("Reload")')
    assert session.index('QPushButton("Export Bundle")') < session.index('QPushButton("Load Checkpoint")')


def test_a2_does_not_change_the_locked_phase88_sidebar_group_architecture() -> None:
    page = PAGE.read_text(encoding="utf-8")
    expected = (
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
    positions = []
    for title in expected:
        anchor = f'self._settings_group("{title}")'
        assert page.count(anchor) == 1
        positions.append(page.index(anchor))
    assert positions == sorted(positions)
    assert 'self._settings_group("EXPORT")' not in page


def test_a2_is_fail_closed_and_never_deletes_or_reconstructs_for_export() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index("def _qualified_reconstruction_export_sources")
    end = page.index("def _load_persistent_scene_interaction", start)
    export_block = page[start:end]
    assert "identity_status is ReconstructionSessionIdentityStatus.CONFLICT" in export_block
    assert "len(session_ids) != 1" in export_block
    assert "controller.is_active" in export_block
    assert "verify_reconstruction_export_bundle" in export_block
    for forbidden in (
        "shutil.rmtree",
        "os.remove",
        "unlink(",
        "start_offline(",
        "start_live(",
        "run_reconstruction_pipeline",
    ):
        assert forbidden not in export_block


def test_a2_loaded_scene_cannot_manufacture_a_surface_mesh_export() -> None:
    page = PAGE.read_text(encoding="utf-8")
    method = page[
        page.index("def set_reconstruction_canonical_surface_export_source") :
        page.index("def _qualified_reconstruction_export_sources")
    ]
    assert "CanonicalSurfaceMeshExportSource" in method
    assert "Persistent manifests do" in method
    assert "not contain full mesh arrays" in method
    assert "payload_snapshot" not in method


def test_a2_export_buttons_remain_sidebar_width_safe() -> None:
    page = PAGE.read_text(encoding="utf-8")
    start = page.index('self._reconstruction_export_button = QPushButton("Export Bundle")')
    end = page.index('self._session_resume_state_value', start)
    block = page[start:end]
    assert "setMinimumWidth(0)" in block
    assert "QSizePolicy.Policy.Ignored" in block
    assert "ScrollBarAlwaysOff" in page


def test_a2_remains_dependency_light() -> None:
    heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    imports = _imports(PAGE)
    assert not any(any(value.startswith(prefix) for prefix in heavy) for value in imports)
