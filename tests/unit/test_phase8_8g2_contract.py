from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTERACTION = ROOT / "app" / "reconstruction_session_interaction.py"
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
PERSISTENCE = ROOT / "app" / "reconstruction_persistence_controller.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return values


def test_g2_reuses_existing_persistence_bridge_and_g1_session_model() -> None:
    interaction = INTERACTION.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "ReconstructionSessionModel" in interaction
    assert "def load_persistent_scene(" in controller
    assert "def save_persistent_scene(" in controller
    assert "def reload_persistent_scene(" in controller
    assert "ReconstructionPersistenceController" in PERSISTENCE.read_text(encoding="utf-8")
    assert "_session_load_button" in page
    assert "_session_save_button" in page
    assert "_session_reload_button" in page


def test_g2_does_not_introduce_second_persistence_or_checkpoint_resume() -> None:
    interaction = INTERACTION.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    for text in (interaction, page):
        assert "prepare_engine_resume(" not in text
        assert "check_engine_resume_compatibility(" not in text
        assert "create_engine_checkpoint(" not in text
    assert "write_persistent_reconstruction_scene(" not in interaction
    assert "write_reconstruction_engine_checkpoint(" not in interaction
    assert "No checkpoint resume is executed" in page


def test_g2_reload_requires_current_manifest_same_session_and_integrity_checks() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    start = controller.index("def reload_persistent_scene(")
    end = controller.index("def clear_persistent_scene(", start)
    block = controller[start:end]
    assert "scene.session_id" in block
    assert "expected_session_id=scene.session_id" in block
    assert "verify_file_references=True" in block
    assert "scene_manifest" in block


def test_g2_scene_actions_are_idle_only_and_ui_availability_is_state_driven() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    interaction = INTERACTION.read_text(encoding="utf-8")
    assert '_require_idle_for_persistence("reload the persistent reconstruction scene")' in controller
    assert "controller_active" in interaction
    assert "RUNTIME_ACTIVE" in interaction
    assert "can_load=False" in interaction
    assert "can_save=False" in interaction
    assert "can_reload=False" in interaction


def test_g2_save_is_explicit_about_manifest_only_not_artifact_bundle() -> None:
    page = PAGE.read_text(encoding="utf-8")
    interaction = INTERACTION.read_text(encoding="utf-8")
    assert "Referenced artifact payloads remain at their recorded paths" in page
    assert "referenced artifact payloads remain at their recorded paths" in interaction
    assert "export bundle" in interaction


def test_g2_keeps_multi_use_case_contract_generic() -> None:
    text = (INTERACTION.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
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


def test_g2_adds_no_heavy_backend_dependency() -> None:
    imports = _imports(INTERACTION)
    forbidden = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
