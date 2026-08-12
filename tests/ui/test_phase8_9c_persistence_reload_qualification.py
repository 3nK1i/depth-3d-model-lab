from __future__ import annotations

from pathlib import Path

import pytest

from app.main_window import MainWindow
from core.reconstruction import ReconstructionValidationError
from tests.reconstruction_f5_helpers import run_f5_fixture


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    win = MainWindow()
    qtbot.addWidget(win)
    yield win
    win.close()


def test_c_application_save_reload_preserves_scene_identity_and_references(window, tmp_path: Path) -> None:
    controller = window._reconstruction_controller
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase8-9c-ui-reload"
    )
    source_dir = (tmp_path / "source-scene").resolve()
    saved_dir = (tmp_path / "saved-scene").resolve()

    adopted = controller.adopt_pipeline_result(
        result,
        source_dir,
        payload_snapshots_by_kind=snapshots,
    )
    source_id = adopted.scene_id
    source_fingerprint = adopted.scene_fingerprint_sha256

    saved_manifest = controller.save_persistent_scene(saved_dir)
    assert saved_manifest == (saved_dir / "persistent_reconstruction_scene.json").resolve()
    reloaded = controller.reload_persistent_scene()

    assert reloaded.scene_id == source_id
    assert reloaded.scene_fingerprint_sha256 == source_fingerprint
    assert controller.persistent_scene_manifest == saved_manifest
    assert reloaded.canonical_geometry_authoritative
    assert reloaded.certified_metrology is False


def test_c_application_reload_fails_closed_on_referenced_artifact_tamper_and_keeps_bound_scene(
    window, tmp_path: Path
) -> None:
    controller = window._reconstruction_controller
    _fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase8-9c-ui-tamper"
    )
    scene = controller.adopt_pipeline_result(
        result,
        (tmp_path / "scene").resolve(),
        payload_snapshots_by_kind=snapshots,
    )
    assert scene.file_references
    before = controller.persistent_scene
    target = Path(scene.file_references[0].path)
    original = target.read_bytes()
    target.write_bytes(original + b"tamper")
    try:
        with pytest.raises(ReconstructionValidationError, match="file size changed"):
            controller.reload_persistent_scene()
        assert controller.persistent_scene is before
        assert controller.persistent_scene_manifest is not None
    finally:
        target.write_bytes(original)
