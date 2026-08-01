"""Main-window shell and runtime lifecycle regression tests."""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.main_window import MainWindow
from app.widgets.mode_tabs import AppMode
from app.widgets.performance_snapshot import PerformanceSnapshot
from app.widgets.viewport.viewport_panel import ViewportStatus
from core.capture.base_source import SourceStatus
from core.capture.camera_source import CameraSource
from core.capture.frame import SourceFrame
from core.capture.image_source import ImageSource
from core.capture.rtsp_source import RtspSource
from core.capture.video_source import VideoSource
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID
from core.runtime.messages import Ack, CommandType, InferenceResult
from core.runtime.state_machine import ApplicationState
from core.runtime.telemetry import GeometryTiming


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)

    win = MainWindow()
    qtbot.addWidget(win)

    controller = win._inference_controller
    monkeypatch.setattr(win._controller, "connect_active_source", lambda: None)
    monkeypatch.setattr(win._controller, "disconnect_active_source", lambda: None)
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


def _moge_ack() -> Ack:
    return Ack(
        CommandType.LOAD_MODEL,
        True,
        model_id=MOGE2_MODEL_ID,
        metadata={
            "id": MOGE2_MODEL_ID,
            "display_name": "MoGe-2 ViT-S Normal",
            "is_metric": True,
            "precision": "FP16",
            "device": "cuda:0",
        },
        load_time_ms=100.0,
        model_load_time_ms=70.0,
        warmup_time_ms=30.0,
    )


def test_window_launches(window):
    assert window.windowTitle() == "EIA TECH — Depth & 3D Model Lab"


def test_minimum_resolution_supported(window):
    window.resize(1366, 768)
    assert window.size().width() >= 1366
    assert window.size().height() >= 768


def test_only_live_3d_mode_enabled(window):
    for mode, btn in window._mode_tabs._buttons.items():
        assert btn.isEnabled() is (mode is AppMode.LIVE_3D)


def test_initial_button_states_are_disconnected(window):
    toolbar = window._runtime_toolbar
    assert toolbar._connect_btn.isEnabled()
    assert not toolbar._disconnect_btn.isEnabled()
    assert not toolbar._start_btn.isEnabled()
    assert not toolbar._pause_resume_btn.isEnabled()
    assert not toolbar._stop_btn.isEnabled()


def test_connect_start_stop_lifecycle(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.SOURCE_READY

    qtbot.mouseClick(toolbar._start_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.LOADING_MODEL

    controller._on_ack_received(_moge_ack())
    assert controller.state_machine.state is ApplicationState.RUNNING
    assert toolbar._pause_resume_btn.isEnabled()
    assert toolbar._stop_btn.isEnabled()

    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert toolbar._disconnect_btn.isEnabled()
    assert toolbar._start_btn.isEnabled()


def test_pause_resume_has_one_valid_state_transition(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())

    qtbot.mouseClick(toolbar._pause_resume_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.PAUSED
    assert window._viewport_panel.status is ViewportStatus.PAUSED
    qtbot.mouseClick(toolbar._pause_resume_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.RUNNING
    assert window._viewport_panel.status is ViewportStatus.LIVE


def test_restart_from_model_ready_reuses_loaded_model(window, qtbot, monkeypatch):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    load_calls: list[str] = []
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, _config=None: load_calls.append(model_id),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    assert load_calls == [MOGE2_MODEL_ID]

    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    assert controller.state_machine.state is ApplicationState.RUNNING
    assert load_calls == [MOGE2_MODEL_ID]


def test_load_failure_keeps_source_retryable(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    controller._on_ack_received(
        Ack(CommandType.LOAD_MODEL, False, "MoGe package is missing", model_id=MOGE2_MODEL_ID)
    )

    assert controller.state_machine.state is ApplicationState.SOURCE_READY
    assert toolbar._start_btn.isEnabled()
    assert window._settings_panel._model_combo.isEnabled()


def test_mock_performance_snapshot_reports_real_point_count(window):
    controller = window._inference_controller
    controller._run_started_at = time.monotonic()
    result = InferenceResult(
        session_id=1,
        frame_id=0,
        model_id=MOCK_MODEL_ID,
        depth=np.ones((24, 32), dtype=np.float32),
        latency_ms=12.5,
    )

    controller._update_performance_snapshot(result, point_count=12_345)

    assert window._performance_bar._points._value.text() == "12,345"
    assert window._performance_bar._precision._value.text() == "FP32"



def test_step3_snapshot_reports_truthful_latency_vram_and_diagnostics(window):
    controller = window._inference_controller
    controller._run_started_at = time.monotonic()
    controller._active_model_metadata.update(
        {
            "display_name": "MoGe-2 ViT-S Normal",
            "device": "cuda:0",
            "precision": "FP16",
            "pretrained": "Ruicheng/moge-2-vits-normal",
            "num_tokens": 1200,
            "resolution_level": 5,
            "model_load_time_ms": 1200.0,
            "warmup_time_ms": 240.0,
            "load_time_ms": 1440.0,
        }
    )
    controller._counters.source_frames_seen = 12
    controller._counters.pending_frames_replaced = 4
    controller._counters.frames_submitted = 8
    controller._counters.results_received = 7
    controller._counters.results_rendered = 6
    controller._counters.pending_results_replaced = 1

    result = InferenceResult(
        session_id=1,
        frame_id=2,
        model_id=MOGE2_MODEL_ID,
        depth=np.ones((24, 32), dtype=np.float32),
        latency_ms=64.2,
        preprocess_ms=1.5,
        postprocess_ms=15.0,
        worker_total_ms=82.0,
        queue_wait_ms=2.0,
        model_name="MoGe-2 ViT-S Normal",
        precision="FP16",
        vram_used_gib=0.8,
        vram_allocated_gib=0.6,
        vram_reserved_gib=0.8,
        vram_peak_reserved_gib=0.9,
        vram_device_used_gib=1.1,
        vram_total_gib=6.0,
        point_stride=4,
    )

    controller._update_performance_snapshot(
        result,
        geometry=GeometryTiming(
            point_count=73_728,
            depth_view_ms=3.0,
            geometry_cpu_ms=8.0,
            canvas_update_ms=4.0,
            total_ms=15.0,
        ),
        pipeline_latency_ms=105.0,
        source_to_render_ms=118.0,
    )

    bar = window._performance_bar
    assert bar._latency._value.text() == "105.0 ms"
    assert "Model forward pass: 64.2 ms" in bar._latency.toolTip()
    assert bar._vram._value.text() == "0.8 / 6 GB"
    assert "Allocated tensors: 0.60 GiB" in bar._vram.toolTip()
    assert "Dropped/replaced total: 5" in bar._status_cell.toolTip()
    assert "Warm-up: 240.0 ms" in bar._model._info_btn.toolTip()
    assert bar.snapshot.latency_ms == 105.0
    assert window._header._vram_value.text() == "1.1 / 6 GB"

    controller._refresh_uptime()
    assert bar.snapshot.source_frames_seen == 12


def test_live_stream_keeps_only_the_latest_pending_frame(window, qtbot, monkeypatch):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    submitted: list[tuple[int, int, int, int | None]] = []

    monkeypatch.setattr(
        controller._process_manager,
        "submit_frame",
        lambda session_id, frame_id, image, *, point_budget=None: (
            submitted.append((session_id, frame_id, int(image[0, 0, 0]), point_budget)) or True
        ),
    )
    monkeypatch.setattr(controller, "_try_render_latest_result", lambda: None)

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())

    controller.on_source_frame(SourceFrame(np.full((8, 8, 3), 1, dtype=np.uint8), frame_id=1))
    controller.on_source_frame(SourceFrame(np.full((8, 8, 3), 2, dtype=np.uint8), frame_id=2))
    controller.on_source_frame(SourceFrame(np.full((8, 8, 3), 3, dtype=np.uint8), frame_id=3))

    assert [value for _session, _frame, value, _budget in submitted] == [1]
    assert submitted[0][3] is not None
    assert controller._pending_source_frame is not None
    assert int(controller._pending_source_frame.image[0, 0, 0]) == 3

    controller._last_submit_at = time.monotonic() - 1.0
    controller._on_result_ready(
        InferenceResult(
            session_id=controller._session.active_session_id,
            frame_id=submitted[0][1],
            model_id=MOGE2_MODEL_ID,
            depth=np.ones((8, 8), dtype=np.float32),
            latency_ms=1.0,
        )
    )

    assert [value for _session, _frame, value, _budget in submitted] == [1, 3]


def test_continuous_preview_paints_only_the_latest_frame(window, monkeypatch):
    from PyQt6.QtGui import QPixmap

    presented_values: list[int] = []

    def _fake_pixmap(image):
        presented_values.append(int(image[0, 0, 0]))
        return QPixmap(1, 1)

    monkeypatch.setattr("app.application_controller.ndarray_to_qpixmap", _fake_pixmap)

    window._controller._on_frame_ready(
        SourceFrame(np.full((8, 8, 3), 10, dtype=np.uint8), frame_id=10)
    )
    window._controller._on_frame_ready(
        SourceFrame(np.full((8, 8, 3), 20, dtype=np.uint8), frame_id=20)
    )

    assert presented_values == []
    window._controller._present_latest_preview()
    assert presented_values == [20]


def test_image_preview_is_ready_connected_and_has_no_camera_fps(window, monkeypatch):
    from PyQt6.QtGui import QPixmap

    source = ImageSource("unused.png")
    window._controller._active_source = source
    monkeypatch.setattr(
        "app.application_controller.ndarray_to_qpixmap",
        lambda _image: QPixmap(1, 1),
    )

    window._controller._on_status_changed(SourceStatus.CONNECTED)
    window._controller._on_frame_ready(
        SourceFrame(np.zeros((8, 8, 3), dtype=np.uint8), frame_id=0)
    )

    assert window._source_bar._connection_badge.text() == "CONNECTED"
    assert window._original_view.activity_text == "READY"
    assert window._controller.capture_fps is None


def test_static_image_finishes_in_result_ready_state(window, qtbot, monkeypatch):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    window._controller._active_source = ImageSource("unused.png")

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    assert controller.state_machine.state is ApplicationState.RUNNING

    monkeypatch.setattr(controller, "_update_geometry", lambda *_a, **_k: 174_801)
    controller._pending_render = (
        InferenceResult(
            session_id=controller._session.active_session_id,
            frame_id=0,
            model_id=MOGE2_MODEL_ID,
            depth=np.ones((24, 32), dtype=np.float32),
            latency_ms=64.2,
            vram_used_gib=0.6,
            vram_total_gib=6.0,
        ),
        np.zeros((24, 32, 3), dtype=np.uint8),
    )

    controller._try_render_latest_result()

    assert controller.state_machine.state is ApplicationState.RESULT_READY
    assert window._viewport_panel.status is ViewportStatus.RESULT_READY
    assert window._performance_bar._status_cell._badge.text() == "RESULT READY"
    assert window._performance_bar._cam_fps._value.text() == "—"
    assert window._performance_bar._points._value.text() == "174,801"
    assert toolbar._start_btn.isEnabled()
    assert not toolbar._pause_resume_btn.isEnabled()
    assert not toolbar._stop_btn.isEnabled()


@pytest.mark.parametrize("source_kind", ["video", "camera", "rtsp"])
def test_stream_sources_remain_running_after_each_result(
    window, qtbot, monkeypatch, source_kind
):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    sources = {
        "video": VideoSource("unused.mp4"),
        "camera": CameraSource(0),
        "rtsp": RtspSource("rtsp://127.0.0.1/example"),
    }
    window._controller._active_source = sources[source_kind]

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())

    monkeypatch.setattr(controller, "_update_geometry", lambda *_a, **_k: 73_728)
    controller._pending_render = (
        InferenceResult(
            session_id=controller._session.active_session_id,
            frame_id=0,
            model_id=MOGE2_MODEL_ID,
            depth=np.ones((24, 32), dtype=np.float32),
            latency_ms=40.0,
        ),
        np.zeros((24, 32, 3), dtype=np.uint8),
    )

    controller._try_render_latest_result()

    assert controller.state_machine.state is ApplicationState.RUNNING
    assert window._viewport_panel.status is ViewportStatus.LIVE
    assert toolbar._pause_resume_btn.isEnabled()
    assert toolbar._stop_btn.isEnabled()


def test_stop_preserves_last_metrics_and_visible_result(window, qtbot):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())

    window._performance_bar.update_snapshot(
        PerformanceSnapshot(
            status="RUNNING",
            latency_ms=64.2,
            point_count=174_801,
            model_name="MoGe-2 ViT-S Normal",
            precision="FP16",
        )
    )
    controller._point_cloud_canvas._has_points = True

    qtbot.mouseClick(toolbar._stop_btn, _left_button())

    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert window._viewport_panel.status is ViewportStatus.RESULT_READY
    assert window._performance_bar._points._value.text() == "174,801"
    assert window._performance_bar._latency._value.text() == "64.2 ms"
    assert window._performance_bar._status_cell._badge.text() == "MODEL READY"


def _left_button():
    from PyQt6.QtCore import Qt

    return Qt.MouseButton.LeftButton
