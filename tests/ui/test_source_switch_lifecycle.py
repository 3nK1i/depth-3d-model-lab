"""Regression tests for source-type switching and post-unload header telemetry."""

from __future__ import annotations

import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from app.main_window import MainWindow
from app.widgets.source_toolbar_snapshot import ConnectionState, SourceType
from core.capture.video_source import VideoSource
from core.models.catalog import MOGE2_MODEL_ID
from core.runtime.messages import Ack, CommandType
from core.runtime.model_lifecycle import (
    ModelLifecycleState,
    model_config_signature,
)
from core.runtime.state_machine import ApplicationState


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)

    win = MainWindow()
    qtbot.addWidget(win)

    controller = win._inference_controller
    monkeypatch.setattr(controller._process_manager, "start_worker", lambda: None)
    monkeypatch.setattr(controller._process_manager, "load_model", lambda *_a, **_k: None)
    monkeypatch.setattr(controller._process_manager, "resume", lambda: None)
    monkeypatch.setattr(controller._process_manager, "pause", lambda: None)
    monkeypatch.setattr(controller._process_manager, "stop", lambda *_a, **_k: None)
    monkeypatch.setattr(controller._process_manager, "shutdown", lambda *_a, **_k: None)
    monkeypatch.setattr(controller._process_manager, "unload_model", lambda: None)
    monkeypatch.setattr(
        type(controller._process_manager),
        "is_running",
        property(lambda self: True),
    )
    return win


def _install_resident_moge(window: MainWindow) -> None:
    controller = window._inference_controller
    settings = window._settings_panel
    model_combo = settings._model_combo
    index = model_combo.findData(MOGE2_MODEL_ID)
    assert index >= 0
    model_combo.setCurrentIndex(index)

    controller._model_loaded = True
    controller._active_model_id = MOGE2_MODEL_ID
    controller._active_model_config_signature = model_config_signature(
        dict(settings.model_runtime_config)
    )
    controller._active_model_metadata = {
        "display_name": "MoGe-2 ViT-S Normal",
        "precision": "FP16",
        "device": "cuda:0",
        "outputs": {"normals": True},
    }
    controller._model_lifecycle.transition_to(ModelLifecycleState.LOADING)
    controller._model_lifecycle.transition_to(ModelLifecycleState.READY)
    settings.set_model_loaded(controller._active_model_metadata)


def _advance_to_result_ready(window: MainWindow) -> None:
    machine = window._inference_controller.state_machine
    for target in (
        ApplicationState.CONNECTING,
        ApplicationState.SOURCE_READY,
        ApplicationState.LOADING_MODEL,
        ApplicationState.MODEL_READY,
        ApplicationState.RUNNING,
        ApplicationState.RESULT_READY,
    ):
        machine.transition_to(target)


def test_switching_source_type_clears_stale_source_and_outputs(window, qtbot) -> None:
    source_bar = window._source_bar
    controller = window._inference_controller

    source_bar._on_type_clicked(SourceType.IMAGE)
    _install_resident_moge(window)
    _advance_to_result_ready(window)

    source_bar.set_connection_enum(ConnectionState.CONNECTED)
    source_pixmap = QPixmap(24, 18)
    source_pixmap.fill(Qt.GlobalColor.green)
    depth_pixmap = QPixmap(24, 18)
    depth_pixmap.fill(Qt.GlobalColor.red)
    window._original_view.set_frame(source_pixmap)
    window._original_view.set_ready()
    window._depth_view.set_depth_frame(depth_pixmap)
    controller._point_cloud_canvas.update_points(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[255, 255, 255]], dtype=np.uint8),
    )

    source_bar._on_type_clicked(SourceType.VIDEO)
    qtbot.wait(1)

    assert source_bar.active_type is SourceType.VIDEO
    assert source_bar._connection_badge.text() == "DISCONNECTED"
    assert window._original_view.activity_text == "IDLE"
    assert not window._original_view._display.has_pixmap
    assert not window._depth_view._display.has_pixmap
    assert not controller._point_cloud_canvas.has_points
    assert controller.state_machine.state is ApplicationState.DISCONNECTED
    assert controller._model_loaded is True
    assert controller._active_model_id == MOGE2_MODEL_ID
    assert window._runtime_toolbar._connect_btn.isEnabled()
    assert not window._runtime_toolbar._start_btn.isEnabled()


def test_empty_video_cannot_claim_source_ready(window, qtbot) -> None:
    source_bar = window._source_bar
    source_bar._on_type_clicked(SourceType.VIDEO)

    qtbot.mouseClick(window._runtime_toolbar._connect_btn, Qt.MouseButton.LeftButton)

    assert window._inference_controller.state_machine.state is ApplicationState.DISCONNECTED
    assert source_bar._connection_badge.text() == "DISCONNECTED"
    assert not window._runtime_toolbar._start_btn.isEnabled()


def test_reconnected_source_reuses_resident_model_without_reload(
    window, qtbot, monkeypatch
) -> None:
    source_bar = window._source_bar
    controller = window._inference_controller
    load_calls: list[str] = []
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, _config=None: load_calls.append(model_id),
    )

    source_bar._on_type_clicked(SourceType.IMAGE)
    _install_resident_moge(window)
    _advance_to_result_ready(window)
    source_bar._on_type_clicked(SourceType.VIDEO)

    window._controller._active_source = VideoSource("unused.mp4")
    qtbot.mouseClick(window._runtime_toolbar._connect_btn, Qt.MouseButton.LeftButton)
    assert controller.state_machine.state is ApplicationState.SOURCE_READY

    qtbot.mouseClick(window._runtime_toolbar._start_btn, Qt.MouseButton.LeftButton)

    assert controller.state_machine.state is ApplicationState.RUNNING
    assert load_calls == []


def test_source_controls_are_locked_only_during_unsafe_states(window) -> None:
    machine = window._inference_controller.state_machine
    source_buttons = tuple(window._source_bar._type_buttons.values())

    machine.transition_to(ApplicationState.CONNECTING)
    machine.transition_to(ApplicationState.SOURCE_READY)
    assert all(button.isEnabled() for button in source_buttons)

    machine.transition_to(ApplicationState.LOADING_MODEL)
    assert all(not button.isEnabled() for button in source_buttons)

    machine.transition_to(ApplicationState.MODEL_READY)
    assert all(button.isEnabled() for button in source_buttons)

    machine.transition_to(ApplicationState.RUNNING)
    assert all(not button.isEnabled() for button in source_buttons)


def test_unload_ack_refreshes_header_device_vram(window) -> None:
    controller = window._inference_controller
    _install_resident_moge(window)
    controller._model_lifecycle.transition_to(ModelLifecycleState.UNLOADING)

    controller._on_ack_received(
        Ack(
            CommandType.UNLOAD_MODEL,
            True,
            model_id=MOGE2_MODEL_ID,
            metadata={
                "after_device_used_gib": 0.68,
                "total_gib": 6.0,
            },
            unload_time_ms=40.0,
        )
    )

    assert window._header._vram_value.text() == "0.7 / 6 GB"
    assert window._header._gpu_value.text() == window._settings_panel.hardware_info.name
