from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "app/reconstruction_controller.py"
PAGE = ROOT / "app/widgets/reconstruction_page.py"
PERSISTENCE = ROOT / "app/reconstruction_persistence_controller.py"
SESSION = ROOT / "app/reconstruction_session_model.py"
RESUME = ROOT / "app/reconstruction_checkpoint_resume.py"
RECOVERY = ROOT / "app/reconstruction_failure_recovery.py"


def test_h1_reuses_existing_application_coordinator_and_persistence_bridge() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    assert "class ReconstructionController(QObject)" in controller
    assert "def adopt_pipeline_result(" in controller
    assert "self._persistence_controller.adopt_pipeline_result(" in controller
    assert "def load_persistent_scene(" in controller
    assert "def create_engine_checkpoint(" in controller


def test_h1_source_workflow_and_runtime_paths_remain_existing_controller_ownership() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    for token in (
        "def select_live_source(",
        "def select_image_sequence(",
        "def select_recorded_video(",
        "def start_selected_live(",
        "def start_selected_offline(",
    ):
        assert token in controller


def test_h1_scene_signal_drives_semantic_and_appearance_presentation_without_recompute() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "controller.persistent_scene_changed.connect(self.set_persistent_scene)" in page
    group = page[page.index("def set_persistent_scene("):page.index("def set_checkpoint(")]
    assert "semantic_presentation_from_persistent_scene(scene)" in group
    assert "_refresh_appearance_representation_controls()" in group
    assert "start_selected_offline" not in group
    assert "start_selected_live" not in group


def test_h1_persistence_keeps_canonical_authority_and_uncertified_metrology() -> None:
    persistence = PERSISTENCE.read_text(encoding="utf-8")
    assert "persistent_scene_from_pipeline_result(" in persistence
    assert "canonical_geometry_authoritative" in persistence
    assert "certified_metrology" in persistence
    assert "cannot silently claim certified metrology" in persistence


def test_h1_session_identity_is_composed_not_reimplemented() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    session = SESSION.read_text(encoding="utf-8")
    assert "build_reconstruction_session_model(" in controller
    assert "ReconstructionSessionIdentityStatus" in session
    assert "CONFLICT" in session


def test_h1_checkpoint_resume_remains_prepare_only_at_ui_boundary() -> None:
    page = PAGE.read_text(encoding="utf-8")
    resume = RESUME.read_text(encoding="utf-8")
    assert 'QPushButton("Prepare Resume")' in page
    assert "prepare_loaded_engine_resume()" in page
    assert "G3 does not launch the resumed engine" in page
    assert "does not launch the resumed engine automatically" in resume


def test_h1_recovery_remains_distinct_from_checkpoint_resume() -> None:
    page = PAGE.read_text(encoding="utf-8")
    recovery = RECOVERY.read_text(encoding="utf-8")
    assert 'QPushButton("Retry From Source")' in page
    assert "self._start_runtime()" in page
    assert "checkpoint resume" in recovery.lower()
    assert "fresh" in recovery.lower()


def test_h1_adds_no_domain_specific_or_heavy_backend_application_branch() -> None:
    paths = (
        ROOT / "tests/reconstruction_h1_helpers.py",
        ROOT / "tests/integration/test_phase8_8h1_whole_workflow_integration.py",
        ROOT / "tests/ui/test_phase8_8h1_whole_reconstruction_workflow.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in paths)
    for forbidden in ("connector_body", "pin-top", "gear", "pcb"):
        assert forbidden not in text
    for forbidden in ("pycolmap", "nerfstudio", "gsplat", "torch.cuda"):
        assert forbidden not in text
