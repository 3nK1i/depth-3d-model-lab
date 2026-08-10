from __future__ import annotations

from pathlib import Path
import sys

import pytest

from app.application_controller import ApplicationController
from app.live_slam_controller import LiveStellaLaunchSettings
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionSourceKind,
    ReconstructionController,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction.errors import ReconstructionValidationError
from tests.reconstruction_live_helpers import make_calibrated_live_camera


ROOT = Path(__file__).resolve().parents[2]


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


def _offline_configuration(tmp_path: Path, name: str = "offline") -> OfflineReconstructionConfiguration:
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    return OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=tmp_path / name,
        sfm_python=Path(sys.executable),
        max_keyframes=3,
        use_gpu=False,
    )


def _live_settings(tmp_path: Path) -> LiveStellaLaunchSettings:
    return LiveStellaLaunchSettings(
        install_prefix=tmp_path / "stella",
        worker_executable=tmp_path / "worker",
        vocabulary_path=tmp_path / "vocab",
    )


def test_start_guards_fail_before_live_or_offline_delegate(
    application_controller, monkeypatch, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    live_calls: list[bool] = []
    offline_calls: list[bool] = []
    monkeypatch.setattr(
        controller.live_slam_controller,
        "start_stella_for_active_source",
        lambda **_kwargs: live_calls.append(True) or "unexpected",
    )
    monkeypatch.setattr(
        controller.offline_controller,
        "start",
        lambda *_args, **_kwargs: offline_calls.append(True) or "unexpected",
    )
    controller.configure_start_guards(
        live=lambda: "Benchmark is running.",
        offline=lambda: "Inspection is running.",
    )

    with pytest.raises(ReconstructionValidationError, match="Benchmark is running"):
        controller.start_live(
            camera=make_calibrated_live_camera(),
            settings=_live_settings(tmp_path),
        )
    with pytest.raises(ReconstructionValidationError, match="Inspection is running"):
        controller.start_offline(_offline_configuration(tmp_path))

    assert live_calls == []
    assert offline_calls == []
    controller.shutdown()


def test_start_guard_exception_fails_closed(application_controller, tmp_path: Path) -> None:
    controller = ReconstructionController(application_controller)

    def broken_guard() -> None:
        raise RuntimeError("guard probe failed")

    controller.configure_start_guards(offline=broken_guard)
    with pytest.raises(ReconstructionValidationError, match="preflight failed: guard probe failed"):
        controller.start_offline(_offline_configuration(tmp_path))
    controller.shutdown()


def test_persistence_mutation_is_rejected_while_reconstruction_active(
    application_controller, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    monkeypatch.setattr(
        type(controller),
        "is_active",
        property(lambda _self: True),
    )

    with pytest.raises(ReconstructionValidationError, match="while a reconstruction session is active"):
        controller.clear_persistent_scene()
    with pytest.raises(ReconstructionValidationError, match="while a reconstruction session is active"):
        controller.clear_engine_checkpoint()

    controller.shutdown()


def test_offline_start_rejects_non_uuid_session_synchronously(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    with pytest.raises(ReconstructionValidationError, match="session_id must be a valid UUID"):
        controller.start_offline(
            _offline_configuration(tmp_path),
            session_id="phase8-8a5-not-a-uuid",
        )
    assert controller.offline_is_running is False
    controller.shutdown()


def test_offline_start_rechecks_output_directory_to_prevent_stale_result_race(
    application_controller, tmp_path: Path
) -> None:
    configuration = _offline_configuration(tmp_path)
    configuration.output_dir.mkdir(parents=True)
    (configuration.output_dir / "offline_result.json").write_text("{}", encoding="utf-8")
    controller = ReconstructionController(application_controller)

    with pytest.raises(ReconstructionValidationError, match="became non-empty"):
        controller.start_offline(configuration)

    assert controller.offline_is_running is False
    controller.shutdown()


def test_shutdown_is_idempotent_and_blocks_new_controller_work(
    application_controller, tmp_path: Path
) -> None:
    controller = ReconstructionController(application_controller)
    controller.shutdown()
    controller.shutdown()

    with pytest.raises(ReconstructionValidationError, match="shut down"):
        controller.start_offline(_offline_configuration(tmp_path))
    with pytest.raises(ReconstructionValidationError, match="shut down"):
        controller.configure_start_guards(live=lambda: None)
    with pytest.raises(ReconstructionValidationError, match="shut down"):
        controller.clear_persistent_scene()
    assert controller.cancel_offline() is False


def test_a5_application_layer_keeps_heavy_backends_external_and_ui_disabled() -> None:
    texts = {
        "reconstruction": (ROOT / "app" / "reconstruction_controller.py").read_text(encoding="utf-8"),
        "offline": (ROOT / "app" / "offline_reconstruction_controller.py").read_text(encoding="utf-8"),
        "main": (ROOT / "app" / "main_window.py").read_text(encoding="utf-8"),
        "tabs": (ROOT / "app" / "widgets" / "mode_tabs.py").read_text(encoding="utf-8"),
    }
    lowered = "\n".join(texts.values()).lower()
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
        assert forbidden not in lowered
    assert "AppMode.RECONSTRUCTION" not in texts["tabs"].split("_ENABLED_MODES =", 1)[1].split("}", 1)[0]
