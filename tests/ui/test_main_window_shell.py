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
from core.runtime.failures import FailureInfo, FailureKind, FailureStage
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


def _select_model(window: MainWindow, model_id: str) -> None:
    combo = window._settings_panel._model_combo
    index = combo.findData(model_id)
    assert index >= 0
    combo.setCurrentIndex(index)
    assert window._settings_panel.selected_model_id == model_id


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
            "outputs": {
                "depth": True,
                "points": True,
                "normals": True,
                "intrinsics": True,
                "valid_mask": True,
            },
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


def test_live_3d_snapshot_and_benchmark_modes_are_enabled(window):
    enabled = {AppMode.LIVE_3D, AppMode.SNAPSHOT, AppMode.BENCHMARK}
    for mode, btn in window._mode_tabs._buttons.items():
        assert btn.isEnabled() is (mode in enabled)


def test_benchmark_mode_switches_to_interactive_page(window, qtbot):
    button = window._mode_tabs._buttons[AppMode.BENCHMARK]
    qtbot.mouseClick(button, _left_button())

    assert window._mode_tabs.active_mode is AppMode.BENCHMARK
    assert window._mode_stack.currentWidget() is window._benchmark_page
    assert window._benchmark_page._run_btn.isEnabled()

    qtbot.mouseClick(window._mode_tabs._buttons[AppMode.LIVE_3D], _left_button())
    assert window._mode_stack.currentWidget() is window._page


def test_benchmark_run_is_blocked_while_live_source_is_connected(window, qtbot):
    qtbot.mouseClick(window._runtime_toolbar._connect_btn, _left_button())
    assert window._inference_controller.state_machine.state is ApplicationState.SOURCE_READY

    qtbot.mouseClick(window._mode_tabs._buttons[AppMode.BENCHMARK], _left_button())

    assert not window._benchmark_page._run_btn.isEnabled()
    assert "Disconnect" in window._benchmark_page._safety_message.text()


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


def test_load_failure_keeps_source_retryable(window, qtbot, monkeypatch):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    # The shared window fixture replaces connect_active_source() with a no-op.
    # Represent a genuinely configured source explicitly so recovery can
    # preserve it and return to SOURCE_READY, as the real application does.
    monkeypatch.setattr(window._controller, "_active_source", object())

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    controller._on_ack_received(
        Ack(
            CommandType.LOAD_MODEL,
            False,
            "MoGe package is missing",
            model_id=MOGE2_MODEL_ID,
        )
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


def test_loaded_model_can_switch_to_mock_without_disconnecting_source(
    window, qtbot, monkeypatch
):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    unload_calls: list[bool] = []
    load_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        controller._process_manager,
        "unload_model",
        lambda: unload_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, config=None: load_calls.append((model_id, dict(config or {}))),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    qtbot.mouseClick(toolbar._stop_btn, _left_button())

    mock_index = window._settings_panel._model_combo.findData(MOCK_MODEL_ID)
    window._settings_panel._model_combo.setCurrentIndex(mock_index)

    assert controller.state_machine.state is ApplicationState.SWITCHING_MODEL
    assert unload_calls == [True]
    assert window._runtime_toolbar._disconnect_btn.isEnabled() is False

    controller._on_ack_received(
        Ack(
            CommandType.UNLOAD_MODEL,
            True,
            model_id=MOGE2_MODEL_ID,
            metadata={"released": True},
            unload_time_ms=5.0,
        )
    )
    assert load_calls[-1][0] == MOCK_MODEL_ID

    controller._on_ack_received(
        Ack(
            CommandType.LOAD_MODEL,
            True,
            model_id=MOCK_MODEL_ID,
            metadata={
                "id": MOCK_MODEL_ID,
                "display_name": "Development Mock",
                "is_metric": False,
                "precision": "FP32",
                "device": "cpu",
                "outputs": {"depth": True, "normals": False},
            },
        )
    )

    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert controller._active_model_id == MOCK_MODEL_ID
    assert window._settings_panel._model_combo.isEnabled()
    assert window._runtime_toolbar._disconnect_btn.isEnabled()


def test_worker_model_stage_updates_visible_badge(window):
    from core.runtime.messages import ModelStage, ModelStageEvent

    controller = window._inference_controller
    controller._on_model_stage_changed(
        ModelStageEvent(
            ModelStage.WARMING_UP,
            model_id=MOGE2_MODEL_ID,
            message="Priming kernels",
        )
    )

    assert window._settings_panel._model_badge.text() == "WARMING UP"
    assert "Priming kernels" in window._settings_panel._model_badge.toolTip()


def test_point_budget_change_does_not_reload_resident_model(
    window, qtbot, monkeypatch
):
    toolbar = window._runtime_toolbar
    controller = window._inference_controller

    # UI tests share an isolated QSettings store for the pytest session.
    # Earlier persistence tests may leave Development Mock selected, so make
    # this regression test self-contained and explicitly select MoGe-2 before
    # simulating a successful MoGe load acknowledgement.
    model_combo = window._settings_panel._model_combo
    moge_index = model_combo.findData(MOGE2_MODEL_ID)
    assert moge_index >= 0
    model_combo.setCurrentIndex(moge_index)
    assert window._settings_panel.selected_model_id == MOGE2_MODEL_ID
    unload_calls: list[bool] = []
    load_calls: list[str] = []
    monkeypatch.setattr(
        controller._process_manager,
        "unload_model",
        lambda: unload_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, _config=None: load_calls.append(model_id),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    assert controller.state_machine.state is ApplicationState.MODEL_READY

    budget = window._settings_panel._point_budget
    target = budget.findText("500K")
    if target < 0:
        target = (budget.currentIndex() + 1) % budget.count()
    budget.setCurrentIndex(target)
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    assert controller.state_machine.state is ApplicationState.RUNNING
    assert unload_calls == []
    assert load_calls == [MOGE2_MODEL_ID]

def test_structured_network_load_failure_preserves_source_and_allows_retry(
    window, qtbot
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())

    failure = FailureInfo(
        FailureKind.NETWORK_UNAVAILABLE,
        FailureStage.LOAD,
        "The model is not cached and the network is unavailable.",
        "ConnectionError: offline",
        "Reconnect or use a cached model.",
    )
    controller._on_ack_received(
        Ack(
            CommandType.LOAD_MODEL,
            False,
            "offline",
            model_id=MOGE2_MODEL_ID,
            failure=failure,
        )
    )

    assert controller.state_machine.state is ApplicationState.SOURCE_READY
    assert controller.last_failure == failure
    assert toolbar._start_btn.isEnabled()
    assert window._settings_panel._model_combo.isEnabled()
    assert window._settings_panel._model_badge.text() == "OFFLINE"
    assert "Reconnect" in window._settings_panel._model_badge.toolTip()


def test_inference_oom_restarts_worker_and_returns_to_retryable_source(
    window, qtbot, monkeypatch
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    restart_calls: list[bool] = []
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    session_id = controller._session.active_session_id

    failure = FailureInfo(
        FailureKind.CUDA_OUT_OF_MEMORY,
        FailureStage.INFERENCE,
        "The selected model ran out of GPU memory.",
        "OutOfMemoryError",
        "Choose Efficient or Development Mock.",
        requires_worker_restart=True,
    )
    controller._on_result_ready(
        InferenceResult(
            session_id=session_id,
            frame_id=0,
            model_id=MOGE2_MODEL_ID,
            depth=None,
            latency_ms=0.0,
            error="CUDA out of memory",
            failure=failure,
        )
    )

    assert restart_calls == [True]
    assert controller.state_machine.state is ApplicationState.SOURCE_READY
    assert not controller._model_loaded
    assert toolbar._start_btn.isEnabled()
    assert window._settings_panel._model_badge.text() == "OOM RECOVERED"


def test_worker_crash_restarts_and_reloads_resident_model_without_autostart(
    window, qtbot, monkeypatch
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    restart_calls: list[bool] = []
    load_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, config=None: load_calls.append((model_id, dict(config or {}))),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    load_calls.clear()

    controller._on_worker_crashed()

    assert restart_calls == [True]
    assert controller.state_machine.state is ApplicationState.LOADING_MODEL
    assert load_calls and load_calls[-1][0] == MOGE2_MODEL_ID

    controller._on_ack_received(_moge_ack())
    assert controller.state_machine.state is ApplicationState.MODEL_READY
    assert toolbar._start_btn.isEnabled()
    assert controller._app_controller.active_source is not None


def test_load_timeout_restarts_worker_without_automatic_retry(
    window, qtbot, monkeypatch
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    restart_calls: list[bool] = []
    load_calls: list[str] = []
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, _config=None: load_calls.append(model_id),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    controller._on_worker_unresponsive(CommandType.LOAD_MODEL)

    assert restart_calls == [True]
    assert load_calls == []
    assert controller.state_machine.state is ApplicationState.SOURCE_READY
    assert toolbar._start_btn.isEnabled()
    assert window._settings_panel._model_badge.text() == "WORKER TIMEOUT"


def test_missing_inference_result_restarts_worker_and_preserves_source(
    window, qtbot, monkeypatch
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    restart_calls: list[bool] = []
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    assert controller.state_machine.state is ApplicationState.RUNNING

    controller._inference_in_flight = True
    controller._inflight_frame_id = 42
    controller._on_inference_result_timeout()

    assert restart_calls == [True]
    assert controller.state_machine.state is ApplicationState.SOURCE_READY
    assert controller.last_failure is not None
    assert controller.last_failure.kind is FailureKind.WORKER_TIMEOUT
    assert "frame 42" in controller.last_failure.technical_details
    assert controller._app_controller.active_source is not None
    assert toolbar._start_btn.isEnabled()


def test_worker_crash_during_unload_keeps_pending_model_selection(
    window, qtbot, monkeypatch
):
    _select_model(window, MOGE2_MODEL_ID)
    toolbar = window._runtime_toolbar
    controller = window._inference_controller
    controller._app_controller._active_source = object()
    restart_calls: list[bool] = []
    load_calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        controller._process_manager,
        "restart_worker",
        lambda: restart_calls.append(True),
    )
    monkeypatch.setattr(
        controller._process_manager,
        "load_model",
        lambda model_id, config=None: load_calls.append((model_id, dict(config or {}))),
    )

    qtbot.mouseClick(toolbar._connect_btn, _left_button())
    qtbot.mouseClick(toolbar._start_btn, _left_button())
    controller._on_ack_received(_moge_ack())
    qtbot.mouseClick(toolbar._stop_btn, _left_button())
    load_calls.clear()

    _select_model(window, MOCK_MODEL_ID)
    assert controller.state_machine.state is ApplicationState.SWITCHING_MODEL
    assert controller._pending_model_request is not None

    controller._on_worker_crashed()

    assert restart_calls == [True]
    assert load_calls and load_calls[-1][0] == MOCK_MODEL_ID
    assert controller.state_machine.state is ApplicationState.LOADING_MODEL


def test_model_metadata_can_extend_inference_timeout_without_model_branching(window) -> None:
    controller = window._inference_controller
    controller._active_model_metadata = {
        "inference_timeout_ms": 120_000,
        "stop_timeout_ms": 125_000,
    }
    controller._apply_active_inference_timeout()
    assert controller._inference_watchdog_timer.interval() == 120_000
    assert controller._active_command_timeout_ms == 125_000

    controller._active_model_metadata = {"inference_timeout_ms": "invalid"}
    controller._apply_active_inference_timeout()
    assert controller._inference_watchdog_timer.interval() == 30_000

    controller._active_model_metadata = {"inference_timeout_ms": 999_999}
    controller._apply_active_inference_timeout()
    assert controller._inference_watchdog_timer.interval() == 600_000
