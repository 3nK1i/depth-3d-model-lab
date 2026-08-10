from __future__ import annotations

from pathlib import Path
import sys

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    ReconstructionController,
    build_offline_reconstruction_command,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction import ReconstructionQualityProfile, SfmMatchingStrategy
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


def _image_configuration(tmp_path: Path) -> OfflineReconstructionConfiguration:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    return OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=tmp_path / "out",
        sfm_python=Path(sys.executable),
        profile=ReconstructionQualityProfile.EFFICIENT,
        matching=SfmMatchingStrategy.SEQUENTIAL,
        max_keyframes=24,
        maximum_image_size=1280,
        sequential_overlap=6,
        image_sequence_fps=8.0,
        recursive=True,
        use_gpu=False,
        timeout_minutes=15.0,
    )


def test_offline_configuration_rejects_invalid_source_shapes(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    extra = tmp_path / "extra"
    extra.mkdir()

    with pytest.raises(ReconstructionValidationError, match="exactly one"):
        OfflineReconstructionConfiguration(
            source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
            inputs=(images, extra),
            output_dir=tmp_path / "out",
            sfm_python=Path(sys.executable),
        )

    with pytest.raises(ReconstructionValidationError, match="at least 3"):
        OfflineReconstructionConfiguration(
            source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
            inputs=(images,),
            output_dir=tmp_path / "out2",
            sfm_python=Path(sys.executable),
            max_keyframes=2,
        )


def test_build_offline_image_directory_command_is_external_pycolmap_safe(tmp_path: Path) -> None:
    configuration = _image_configuration(tmp_path)
    worker = tmp_path / "worker.py"
    worker.write_text("print('worker')\n", encoding="utf-8")

    program, arguments, output_dir = build_offline_reconstruction_command(
        configuration,
        session_id="00000000-0000-4000-8000-0000000088a3",
        worker_script=worker,
    )

    assert program == sys.executable
    assert arguments[0] == str(worker.resolve())
    assert "--images-dir" in arguments
    assert str(configuration.inputs[0]) in arguments
    assert "--sfm-python" in arguments
    assert str(configuration.sfm_python) in arguments
    assert "--cpu" in arguments
    assert "--recursive" in arguments
    assert output_dir == configuration.output_dir


def test_build_offline_recorded_video_command_uses_video_input(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"offline-video-identity")
    worker = tmp_path / "worker.py"
    worker.write_text("print('worker')\n", encoding="utf-8")
    configuration = OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.RECORDED_VIDEO,
        inputs=(video,),
        output_dir=tmp_path / "out",
        sfm_python=Path(sys.executable),
        video_target_fps=1.5,
    )

    _program, arguments, _output = build_offline_reconstruction_command(
        configuration,
        session_id="00000000-0000-4000-8000-0000000088a4",
        worker_script=worker,
    )

    assert "--video" in arguments
    assert str(video.resolve()) in arguments
    assert "--images-dir" not in arguments
    assert "--image" not in arguments


def test_controller_offline_defaults_are_dormant(application_controller) -> None:
    controller = ReconstructionController(application_controller)

    assert controller.offline_state is OfflineReconstructionRunState.IDLE
    assert controller.offline_session_id is None
    assert controller.offline_output_dir is None
    assert controller.offline_is_running is False
    assert controller.is_active is False

    controller.shutdown()


def test_offline_line_protocol_is_forwarded_without_backend_imports(application_controller) -> None:
    controller = ReconstructionController(application_controller)
    events: list[object] = []
    logs: list[str] = []
    controller.offline_event_received.connect(events.append)
    controller.offline_log_received.connect(logs.append)

    controller.offline_controller._consume_line(
        '{"protocol":"eia.reconstruction.offline.v1","event":"progress","stage":"camera_pose","message":"ok"}'
    )
    controller.offline_controller._consume_line("native worker log")

    assert events and events[0]["event"] == "progress"
    assert logs == ["native worker log"]
    controller.shutdown()


def test_offline_start_is_blocked_while_live_runtime_is_active(
    application_controller, monkeypatch, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    monkeypatch.setattr(
        type(controller.live_slam_controller),
        "is_active",
        property(lambda _self: True),
    )

    with pytest.raises(ReconstructionValidationError, match="Live reconstruction is active"):
        controller.start_offline(_image_configuration(tmp_path))

    controller.shutdown()


def test_a3_controller_and_worker_keep_heavy_backends_external() -> None:
    root = Path(__file__).resolve().parents[2]
    texts = [
        (root / "app/reconstruction_controller.py").read_text(encoding="utf-8").lower(),
        (root / "scripts/run_reconstruction_offline_job.py").read_text(encoding="utf-8").lower(),
    ]
    for text in texts:
        for forbidden in (
            "import pycolmap",
            "from pycolmap",
            "import gsplat",
            "from gsplat",
            "import nerfstudio",
            "from nerfstudio",
            "import depth_anything_3",
            "from depth_anything_3",
            "import stella_vslam",
            "from stella_vslam",
        ):
            assert forbidden not in text

    worker = texts[1]
    assert "from tests" not in worker
    assert "import tests" not in worker
