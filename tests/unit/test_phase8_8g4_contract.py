from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "app" / "reconstruction_failure_recovery.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
G3 = ROOT / "app" / "reconstruction_checkpoint_resume.py"
LIVE = ROOT / "core" / "reconstruction" / "live_runtime.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_g4_observes_existing_live_recovery_instead_of_adding_worker_restart_logic() -> None:
    module = RECOVERY.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    live = LIVE.read_text(encoding="utf-8")
    assert "AUTO_RECOVERING" in module
    assert "existing live runtime owns bounded worker restart/recovery" in module
    assert "max_worker_restarts" in live
    assert "_recover_backend" in live
    assert "restart_backoff" not in page
    assert "_recover_backend" not in page


def test_g4_retry_is_explicit_fresh_source_attempt_not_checkpoint_resume() -> None:
    page = PAGE.read_text(encoding="utf-8")
    module = RECOVERY.read_text(encoding="utf-8")
    assert 'QPushButton("Retry From Source")' in page
    start = page.index("def _retry_failed_runtime_interaction")
    end = page.index("def set_source_selection", start)
    retry = page[start:end]
    assert "self._start_runtime()" in retry
    assert "recovery_owns_retry" in page
    assert "and not recovery_owns_retry" in page
    assert "prepare_loaded_engine_resume" not in retry
    assert "resume_live" not in retry
    assert "fresh reconstruction session" in module


def test_g4_fails_closed_when_persisted_evidence_or_identity_conflict_exists() -> None:
    module = RECOVERY.read_text(encoding="utf-8")
    assert "ReconstructionSessionIdentityStatus.CONFLICT" in module
    assert "has_scene or has_checkpoint" in module
    assert "can_retry_from_source=False" in module
    assert "evidence preserved" in module.lower()


def test_g4_keeps_g3_checkpoint_execution_boundary_unchanged() -> None:
    module = RECOVERY.read_text(encoding="utf-8")
    g3 = G3.read_text(encoding="utf-8")
    assert "ReconstructionCheckpointResumeState.PREPARED" in module
    assert "resumed execution remains explicit outside G4" in module
    assert "does not launch the resumed engine" in g3
    assert "resume_stage_results" not in module


def test_g4_preserves_authority_and_never_certifies_metrology() -> None:
    module = RECOVERY.read_text(encoding="utf-8")
    assert "canonical_geometry_authoritative" in module
    assert "certified_metrology: bool = False" in module
    assert "G4 failure/recovery UX cannot certify metrology" in module
    assert "appearance artifacts" in module


def test_g4_is_generic_and_dependency_light() -> None:
    text = RECOVERY.read_text(encoding="utf-8").lower()
    for forbidden in (
        'class_name == "connector"',
        'class_name == "pin"',
        'class_name == "gear"',
        'use_case == "pcb"',
        'industry == "automotive"',
        "connector_body",
        "missing_pin",
        "bent_pin",
    ):
        assert forbidden not in text
    heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    assert not any(any(value.startswith(prefix) for prefix in heavy) for value in _imports(RECOVERY))


def test_g4_does_not_modify_or_create_persistence_checkpoint_formats() -> None:
    module = RECOVERY.read_text(encoding="utf-8")
    assert "json.dump" not in module
    assert "write_reconstruction_engine_checkpoint" not in module
    assert "write_persistent_reconstruction_scene" not in module
    assert "ENGINE_CHECKPOINT_SCHEMA_VERSION" not in module
    assert "PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION" not in module


def test_g4_ui_is_one_compact_recovery_row_and_one_action() -> None:
    page = PAGE.read_text(encoding="utf-8")
    session_start = page.index('session = self._settings_group("SESSION & PERSISTENCE")')
    session_end = page.index("layout.addWidget(session)", session_start)
    group = page[session_start:session_end]
    assert group.count('"Recovery"') == 1
    assert group.count('QPushButton("Retry From Source")') == 1
    assert 'QPushButton("Recover Worker")' not in group
    assert 'QPushButton("Execute Resume")' not in group
