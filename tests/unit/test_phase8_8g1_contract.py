from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_session_model.py"
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
CORE_SESSION = ROOT / "core" / "reconstruction" / "session.py"
PERSISTENCE = ROOT / "app" / "reconstruction_persistence_controller.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_g1_composes_existing_session_evidence_without_replacing_core_session() -> None:
    module = MODULE.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "build_reconstruction_session_model" in module
    assert "ReconstructionSessionModel" in controller
    assert "core.reconstruction.session" not in _imports(MODULE)
    assert "class ReconstructionSession:" in CORE_SESSION.read_text(encoding="utf-8")
    assert "ReconstructionPersistenceController" in PERSISTENCE.read_text(encoding="utf-8")


def test_g1_does_not_implement_g2_g3_g4_actions_or_new_persistence_format() -> None:
    module = MODULE.read_text(encoding="utf-8")
    forbidden_calls = (
        "QFileDialog",
        "save_persistent_scene(",
        "load_persistent_scene(",
        "prepare_engine_resume(",
        "resume_live(",
        "restart_worker(",
        "write_persistent_reconstruction_scene(",
        "write_reconstruction_engine_checkpoint(",
    )
    for token in forbidden_calls:
        assert token not in module
    assert "no second persistence format" in module


def test_g1_model_is_generic_artifact_provider_representation_driven() -> None:
    text = MODULE.read_text(encoding="utf-8").lower()
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
    for required in (
        "artifacts",
        "providers",
        "appearance_representation_kinds",
        "scene_id",
        "checkpoint_id",
        "session_id",
    ):
        assert required in text


def test_g1_preserves_authority_and_does_not_certify_metrology() -> None:
    module = MODULE.read_text(encoding="utf-8")
    assert "canonical_geometry_authoritative" in module
    assert "certified_metrology" in module
    assert 'object.__setattr__(self, "certified_metrology", False)' in module
    assert "G1 cannot promote reconstruction session state to certified metrology" in module


def test_g1_adds_passive_context_to_existing_session_group_without_actions() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('session = self._settings_group("SESSION & PERSISTENCE")') == 1
    start = page.index('session = self._settings_group("SESSION & PERSISTENCE")')
    end = page.index('self._offline_inspection_group = self._settings_group("OFFLINE INSPECTION")', start)
    group = page[start:end]
    assert '"Context"' in group
    assert '"Evidence"' in group
    assert "QPushButton(" not in group
    assert "QFileDialog" not in group


def test_g1_controller_exposes_computed_model_and_change_signal_only() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "session_model_changed = pyqtSignal(object)" in controller
    assert "def session_model(self) -> ReconstructionSessionModel:" in controller
    assert "return build_reconstruction_session_model(" in controller
    assert "self._session_model =" not in controller


def test_g1_does_not_pull_heavy_reconstruction_backends_into_session_layer() -> None:
    imports = _imports(MODULE)
    forbidden = ("torch", "pycolmap", "nerfstudio", "tinycudann", "gsplat", "cv2")
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
