from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESUME = ROOT / "app" / "reconstruction_checkpoint_resume.py"
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
PERSISTENCE = ROOT / "app" / "reconstruction_persistence_controller.py"
CORE = ROOT / "core" / "reconstruction" / "engine_checkpoint.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_g3_reuses_existing_checkpoint_integrity_and_compatibility_contracts() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    persistence = PERSISTENCE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    assert "def load_engine_checkpoint(" in controller
    assert "def prepare_loaded_engine_resume(" in controller
    assert "check_resume_compatibility(" in controller
    assert "read_reconstruction_engine_checkpoint" in persistence
    assert "check_reconstruction_engine_resume_compatibility" in persistence
    assert "ENGINE_CHECKPOINT_CHECKSUM" in core
    assert "SHA-256" in core


def test_g3_requires_current_plan_provider_payload_context_and_never_guesses() -> None:
    module = RESUME.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "ReconstructionResumeContext" in module
    assert "plan: ReconstructionPipelinePlan" in module
    assert "providers: tuple[ReconstructionProviderDescriptor" in module
    assert "available_payload_refs: frozenset[str]" in module
    assert "will not guess compatibility" in module
    assert "No current reconstruction resume context is registered" in controller


def test_g3_ui_is_explicit_load_then_prepare_not_automatic_resume() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'QPushButton("Load Checkpoint")' in page
    assert 'QPushButton("Prepare Resume")' in page
    assert "prepare_loaded_engine_resume()" in page
    assert "G3 does not launch the resumed engine" in page
    assert "start_selected_offline" not in page[page.index("def _prepare_engine_resume_interaction"):page.index("def _on_resume_compatibility_changed")]


def test_g3_prepare_handoff_uses_existing_reconstruction_resume_preparation() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    persistence = PERSISTENCE.read_text(encoding="utf-8")
    assert "ReconstructionResumePreparation" in controller
    assert "self._persistence_controller.prepare_resume(" in controller
    assert "resume_stage_results" in persistence
    assert "reused_stage_count" in persistence


def test_g3_is_idle_guarded_and_session_conflict_fails_closed() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    module = RESUME.read_text(encoding="utf-8")
    assert '_require_idle_for_persistence("load a reconstruction checkpoint")' in controller
    assert '_require_idle_for_persistence("prepare reconstruction resume")' in controller
    assert "ReconstructionSessionIdentityStatus.CONFLICT" in module
    assert "ReconstructionSessionIdentityStatus.CONFLICT" in controller
    assert "session identity conflict" in module.lower()


def test_g3_checkpoint_boundary_remains_intact_as_g4_adds_failure_recovery_ux() -> None:
    module = RESUME.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "recover failed runtimes" in module
    assert "certified_metrology: bool = False" in module
    assert "G3 checkpoint/resume UX cannot certify metrology" in module
    assert "canonical_geometry_authoritative" in module
    group = page[page.index('QPushButton("Load Checkpoint")'):page.index('layout.addWidget(session)', page.index('QPushButton("Load Checkpoint")'))]
    assert 'QPushButton("Prepare Resume")' in group
    assert 'QPushButton("Retry From Source")' in group  # G4 ownership


def test_g3_remains_generic_and_dependency_light() -> None:
    text = (RESUME.read_text(encoding="utf-8") + CONTROLLER.read_text(encoding="utf-8")).lower()
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
    imports = _imports(RESUME)
    heavy = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    assert not any(any(value.startswith(prefix) for prefix in heavy) for value in imports)


def test_g3_does_not_create_another_checkpoint_format_or_touch_core_contracts() -> None:
    module = RESUME.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "write_reconstruction_engine_checkpoint" not in module
    assert "create_reconstruction_engine_checkpoint" not in module
    assert "json.dump" not in module
    assert "ENGINE_CHECKPOINT_SCHEMA_VERSION" not in controller
