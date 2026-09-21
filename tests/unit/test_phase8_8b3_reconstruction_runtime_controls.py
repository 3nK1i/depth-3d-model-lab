from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import (
    ReconstructionController,
    ReconstructionWorkflow,
    default_live_stella_launch_settings,
    default_pycolmap_python_path,
    default_reconstruction_output_root,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.reconstruction import CameraIntrinsicsPolicy, ReconstructionQualityProfile
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_live_helpers import make_calibrated_live_camera


@pytest.fixture
def application_controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    controller = ApplicationController(source_bar, original_view)
    yield controller
    controller.shutdown()


def _images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for index in range(5):
        (root / f"frame-{index:02d}.jpg").write_bytes(b"fixture")
    return root


def test_b3_selected_offline_builds_real_application_configuration_without_importing_pycolmap(
    application_controller, monkeypatch, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    images = _images(tmp_path)
    sfm_python = tmp_path / "eia-pycolmap-probe-python"
    sfm_python.write_text("#!/bin/sh\n", encoding="utf-8")
    sfm_python.chmod(0o755)
    output_root = tmp_path / "outputs"
    controller.select_image_sequence(images)

    captured: dict[str, object] = {}

    def fake_start(configuration, *, session_id=None):
        captured["configuration"] = configuration
        captured["session_id"] = session_id
        return session_id

    monkeypatch.setattr(controller, "start_offline", fake_start)
    sid = controller.start_selected_offline(
        sfm_python=sfm_python,
        output_root=output_root,
        max_keyframes=42,
    )

    UUID(sid)
    configuration = captured["configuration"]
    assert configuration.inputs == (images.resolve(),)
    assert configuration.output_dir == (output_root / sid).resolve()
    assert configuration.sfm_python == sfm_python.resolve()
    assert configuration.max_keyframes == 42
    assert configuration.profile is ReconstructionQualityProfile.EFFICIENT
    assert captured["session_id"] == sid
    controller.shutdown()


def test_b3_selected_offline_requires_ready_offline_workflow(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    sfm_python = tmp_path / "python"
    sfm_python.write_bytes(b"x")

    with pytest.raises(ReconstructionValidationError, match="Select Image sequence"):
        controller.start_selected_offline(sfm_python=sfm_python, output_root=tmp_path / "out")

    controller.shutdown()


def test_b3_selected_live_reuses_existing_live_bridge_with_explicit_camera_and_settings(
    application_controller, monkeypatch, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    application_controller._active_source = source
    controller.select_live_source()

    settings = LiveStellaLaunchSettings(
        install_prefix=tmp_path / "stella",
        worker_executable=tmp_path / "worker",
        vocabulary_path=tmp_path / "vocab",
    )
    camera = make_calibrated_live_camera()
    captured: dict[str, object] = {}

    def fake_start_live(*, camera, settings, profile, session_id=None):
        captured.update(camera=camera, settings=settings, profile=profile, session_id=session_id)
        return "00000000-0000-4000-8000-0000000088b3"

    monkeypatch.setattr(controller, "start_live", fake_start_live)
    sid = controller.start_selected_live(camera=camera, settings=settings)

    assert sid == "00000000-0000-4000-8000-0000000088b3"
    assert captured["camera"] is camera
    assert captured["settings"] is settings
    assert captured["profile"] is ReconstructionQualityProfile.EFFICIENT
    controller.shutdown()


def test_b3_build_estimated_live_camera_is_explicitly_non_calibrated(
    application_controller,
) -> None:
    controller = ReconstructionController(application_controller)
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    application_controller._active_source = source
    controller.select_live_source()

    camera = controller.build_estimated_live_camera()

    assert camera.image_size.width == 1280
    assert camera.image_size.height == 720
    assert camera.intrinsics.source.value == "estimated"
    assert camera.intrinsics.calibration_status.value == "not_calibrated"
    assert camera.intrinsics.calibration_record_id == ""
    assert camera.distortion_model.value == "none"
    controller.shutdown()


def test_b3_selected_live_estimated_mode_opts_into_existing_allow_estimated_contract(
    application_controller, monkeypatch, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    source = CameraSource(0, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    application_controller._active_source = source
    controller.select_live_source()
    camera = controller.build_estimated_live_camera()
    settings = LiveStellaLaunchSettings(
        install_prefix=tmp_path / "stella",
        worker_executable=tmp_path / "worker",
        vocabulary_path=tmp_path / "vocab",
    )
    captured: dict[str, object] = {}

    def fake_start_live(**kwargs):
        captured.update(kwargs)
        return "00000000-0000-4000-8000-00000000e8b3"

    monkeypatch.setattr(controller, "start_live", fake_start_live)
    sid = controller.start_selected_live(
        camera=camera,
        settings=settings,
        camera_intrinsics_policy=CameraIntrinsicsPolicy.ALLOW_ESTIMATED,
    )

    assert sid == "00000000-0000-4000-8000-00000000e8b3"
    assert captured["camera"] is camera
    assert captured["camera_intrinsics_policy"] is CameraIntrinsicsPolicy.ALLOW_ESTIMATED
    controller.shutdown()


def test_b3_selected_live_rejects_not_ready_capture(application_controller, tmp_path: Path) -> None:
    controller = ReconstructionController(application_controller)
    selection = controller.select_live_source()
    assert selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
    assert selection.ready is False

    settings = LiveStellaLaunchSettings(
        install_prefix=tmp_path / "stella",
        worker_executable=tmp_path / "worker",
        vocabulary_path=tmp_path / "vocab",
    )
    with pytest.raises(ReconstructionValidationError, match="Connect a Camera"):
        controller.start_selected_live(
            camera=make_calibrated_live_camera(),
            settings=settings,
        )
    controller.shutdown()


def test_b3_runtime_defaults_support_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    pycolmap = tmp_path / "pycolmap-python"
    pycolmap.write_bytes(b"python")
    prefix = tmp_path / "stella-prefix"
    prefix.mkdir()
    worker = tmp_path / "worker"
    worker.write_bytes(b"worker")
    vocab = tmp_path / "orb_vocab.fbow"
    vocab.write_bytes(b"vocab")
    output_root = tmp_path / "runtime-output"

    monkeypatch.setenv("EIA_RECONSTRUCTION_PYCOLMAP_PYTHON", str(pycolmap))
    monkeypatch.setenv("EIA_RECONSTRUCTION_STELLA_PREFIX", str(prefix))
    monkeypatch.setenv("EIA_RECONSTRUCTION_STELLA_WORKER", str(worker))
    monkeypatch.setenv("EIA_RECONSTRUCTION_STELLA_VOCAB", str(vocab))
    monkeypatch.setenv("EIA_RECONSTRUCTION_OUTPUT_ROOT", str(output_root))

    assert default_pycolmap_python_path() == pycolmap.resolve()
    assert default_reconstruction_output_root() == output_root.resolve()
    settings = default_live_stella_launch_settings()
    assert settings.install_prefix == prefix.resolve()
    assert settings.worker_executable == worker.resolve()
    assert settings.vocabulary_path == vocab.resolve()


def test_b3_missing_external_runtime_fails_before_backend_start(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "missing-python"
    monkeypatch.setenv("EIA_RECONSTRUCTION_PYCOLMAP_PYTHON", str(missing))
    with pytest.raises(ReconstructionValidationError, match="isolated PyCOLMAP interpreter"):
        default_pycolmap_python_path()
