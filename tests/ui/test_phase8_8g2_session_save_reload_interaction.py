from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from app.main_window import MainWindow
from app.reconstruction_session_interaction import ReconstructionSceneInteractionState
from app.reconstruction_session_model import build_reconstruction_session_model
from core.reconstruction.errors import ReconstructionValidationError

SESSION = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def _selection():
    return SimpleNamespace(
        workflow=SimpleNamespace(value="image_sequence"),
        ready=True,
        display_name="Generic reconstruction source",
        detail="Selected image sequence",
        path=Path("/tmp/g2-ui/source"),
    )


def _scene():
    return SimpleNamespace(
        scene_id="prs-g2-ui",
        session_id=SESSION,
        state=SimpleNamespace(value="ready"),
        artifacts=(
            SimpleNamespace(
                artifact_id="artifact-1",
                stage=SimpleNamespace(value="surface_mesh"),
                kind=SimpleNamespace(value="surface_mesh"),
                authority=SimpleNamespace(value="canonical_geometry"),
                provider_id="provider-1",
            ),
        ),
        provider_provenance=(
            SimpleNamespace(
                provider_id="provider-1",
                provider_version="1",
                execution=SimpleNamespace(value="in_process"),
                stages=(SimpleNamespace(value="surface_mesh"),),
            ),
        ),
        canonical_geometry_authoritative=True,
        certified_metrology=False,
        created_at_utc="2026-08-11T12:00:00+00:00",
    )


def _ready_model(manifest: str = "/tmp/g2-ui/persistent_reconstruction_scene.json"):
    return build_reconstruction_session_model(
        source_selection=_selection(),
        scene=_scene(),
        scene_manifest=manifest,
        scene_availability=SimpleNamespace(value="ready"),
    )


def test_g2_initial_session_actions_are_load_only(window) -> None:
    page = window._reconstruction_page
    assert page._session_load_button.isEnabled()
    assert not page._session_save_button.isEnabled()
    assert not page._session_reload_button.isEnabled()
    assert page.scene_interaction_presentation is not None
    assert page.scene_interaction_presentation.state is ReconstructionSceneInteractionState.LOAD_ONLY
    assert hasattr(page, "_session_resume_button")  # G3 ownership; G2 scene actions remain separate


def test_g2_loaded_scene_enables_save_reload_and_keeps_compact_session_ui(window, qtbot) -> None:
    page = window._reconstruction_page
    page.set_session_model(_ready_model())
    qtbot.wait(10)

    assert page._session_value.text() == SESSION[:8]
    assert page._session_load_button.isEnabled()
    assert page._session_save_button.isEnabled()
    assert page._session_reload_button.isEnabled()
    assert page._session_action_state_value.text() == "Scene actions ready • guarded persistence"
    assert "No checkpoint resume" in page._session_action_state_value.toolTip()


def test_g2_load_button_uses_existing_controller_integrity_checked_load(window, qtbot, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    selected = "/tmp/g2-ui-load/persistent_reconstruction_scene.json"
    calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *_args, **_kwargs: (selected, "Persistent reconstruction scene"),
    )
    monkeypatch.setattr(
        controller,
        "load_persistent_scene",
        lambda path, *, verify_file_references=True, **_kwargs: calls.append(
            (Path(path), verify_file_references)
        ),
    )

    page._session_load_button.click()
    qtbot.wait(10)
    assert calls == [(Path(selected).resolve(), True)]


def test_g2_save_button_uses_existing_scene_save_without_copying_payloads(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    page.set_session_model(_ready_model())
    output = (tmp_path / "saved-scene").resolve()
    output.mkdir()
    calls: list[Path] = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(output))
    monkeypatch.setattr(controller, "save_persistent_scene", lambda path: calls.append(Path(path)))

    page._session_save_button.click()
    qtbot.wait(10)
    assert calls == [output]
    assert "Referenced artifact payloads remain" in page._session_save_button.toolTip()


def test_g2_save_refuses_silent_overwrite_of_another_scene(
    window, qtbot, monkeypatch, tmp_path: Path
) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    page.set_session_model(_ready_model())
    output = (tmp_path / "existing-scene").resolve()
    output.mkdir()
    (output / "persistent_reconstruction_scene.json").write_text("{}\n", encoding="utf-8")
    calls: list[Path] = []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_args, **_kwargs: str(output))
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Cancel,
    )
    monkeypatch.setattr(controller, "save_persistent_scene", lambda path: calls.append(Path(path)))

    page._session_save_button.click()
    qtbot.wait(10)
    assert calls == []


def test_g2_reload_button_delegates_to_same_session_reload(window, qtbot, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    page.set_session_model(_ready_model())
    calls: list[bool] = []
    monkeypatch.setattr(controller, "reload_persistent_scene", lambda: calls.append(True))

    page._session_reload_button.click()
    qtbot.wait(10)
    assert calls == [True]


def test_g2_scene_actions_disable_while_reconstruction_is_active(window, monkeypatch) -> None:
    page = window._reconstruction_page
    controller = window._reconstruction_controller
    page.set_session_model(_ready_model())
    monkeypatch.setattr(type(controller), "is_active", property(lambda _self: True))
    page._refresh_scene_interaction_controls()

    assert not page._session_load_button.isEnabled()
    assert not page._session_save_button.isEnabled()
    assert not page._session_reload_button.isEnabled()
    assert page.scene_interaction_presentation is not None
    assert page.scene_interaction_presentation.state is ReconstructionSceneInteractionState.RUNTIME_ACTIVE


def test_g2_controller_reload_requires_manifest_and_preserves_session_identity(window, monkeypatch, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    bridge = controller.persistence_controller
    manifest = (tmp_path / "scene" / "persistent_reconstruction_scene.json").resolve()
    scene = SimpleNamespace(session_id=SESSION)
    bridge._scene = scene
    bridge._scene_manifest = manifest
    calls: list[tuple[Path, str, bool]] = []

    def fake_load(path, *, expected_session_id=None, verify_file_references=True):
        calls.append((Path(path), expected_session_id, verify_file_references))
        return scene

    monkeypatch.setattr(bridge, "load_scene", fake_load)
    assert controller.reload_persistent_scene() is scene
    assert calls == [(manifest, SESSION, True)]


def test_g2_controller_reload_fails_closed_without_active_manifest(window) -> None:
    controller = window._reconstruction_controller
    with pytest.raises(ReconstructionValidationError, match="manifest is available to reload"):
        controller.reload_persistent_scene()
