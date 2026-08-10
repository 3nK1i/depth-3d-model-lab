from __future__ import annotations

from pathlib import Path

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    ReconstructionController,
    ReconstructionWorkflow,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.capture.image_source import ImageSource
from core.capture.video_source import VideoSource
from core.reconstruction.errors import ReconstructionValidationError


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


def test_b2_default_selection_is_dormant(application_controller) -> None:
    controller = ReconstructionController(application_controller)

    selection = controller.source_selection
    assert selection.workflow is ReconstructionWorkflow.NONE
    assert selection.ready is False
    assert selection.path is None
    assert controller.is_active is False

    controller.shutdown()


def test_b2_select_image_sequence_validates_supported_top_level_images(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    images = tmp_path / "south-building-images"
    images.mkdir()
    for index, suffix in enumerate((".JPG", ".png", ".jpeg", ".txt")):
        (images / f"frame-{index:02d}{suffix}").write_bytes(b"fixture")

    emitted = []
    controller.source_selection_changed.connect(emitted.append)
    selection = controller.select_image_sequence(images)

    assert selection.workflow is ReconstructionWorkflow.IMAGE_SEQUENCE
    assert selection.ready is True
    assert selection.path == images.resolve()
    assert selection.item_count == 3
    assert selection.display_name == images.name
    assert emitted == [selection]
    assert controller.is_active is False

    controller.shutdown()


def test_b2_invalid_image_sequence_does_not_replace_previous_selection(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    good = tmp_path / "good"
    good.mkdir()
    for index in range(3):
        (good / f"{index}.jpg").write_bytes(b"x")
    previous = controller.select_image_sequence(good)

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "only.jpg").write_bytes(b"x")
    with pytest.raises(ReconstructionValidationError, match="at least 3"):
        controller.select_image_sequence(bad)

    assert controller.source_selection == previous
    controller.shutdown()


def test_b2_recorded_video_selection_is_path_only_and_starts_no_worker(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    video = tmp_path / "walkthrough.mp4"
    video.write_bytes(b"video-identity")

    selection = controller.select_recorded_video(video)

    assert selection.workflow is ReconstructionWorkflow.RECORDED_VIDEO
    assert selection.ready is True
    assert selection.path == video.resolve()
    assert selection.item_count == 1
    assert controller.offline_is_running is False
    assert controller.live_slam_controller.runtime is None

    controller.shutdown()


def test_b2_rejects_unsupported_recorded_video_extension(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    video = tmp_path / "walkthrough.xyz"
    video.write_bytes(b"not-a-supported-container")

    with pytest.raises(ReconstructionValidationError, match="Unsupported recorded-video"):
        controller.select_recorded_video(video)

    assert controller.source_selection.workflow is ReconstructionWorkflow.NONE
    controller.shutdown()


def test_b2_live_selection_reuses_and_tracks_application_capture_source(
    application_controller,
) -> None:
    controller = ReconstructionController(application_controller)
    source = CameraSource(2, width=1280, height=720, fps=30)
    source._status = SourceStatus.CONNECTED
    application_controller._active_source = source

    selection = controller.select_live_source()
    assert selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
    assert selection.ready is True
    assert selection.display_name == "Camera 2"
    assert selection.live_source_kind == "camera"
    assert application_controller.active_source is source
    assert controller.live_slam_controller.runtime is None

    source._status = SourceStatus.RECONNECTING
    application_controller.source_status_changed.emit(SourceStatus.RECONNECTING)
    updated = controller.source_selection
    assert updated.workflow is ReconstructionWorkflow.LIVE_SOURCE
    assert updated.ready is False
    assert "Reconnecting" in updated.detail
    assert application_controller.active_source is source

    controller.shutdown()


def test_b2_static_image_is_explicitly_not_live_compatible(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    image = tmp_path / "single.jpg"
    image.write_bytes(b"image")
    source = ImageSource(str(image))
    source._status = SourceStatus.CONNECTED
    application_controller._active_source = source

    selection = controller.select_live_source()

    assert selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
    assert selection.ready is False
    assert selection.live_source_kind == "unsupported_image"
    assert "Image sequence" in selection.detail
    controller.shutdown()
