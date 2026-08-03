"""Inference lifecycle, latest-frame scheduling, and geometry rendering.

Phase 5 Step 2 activates the first real adapter, MoGe-2, while preserving the
bounded live-stream pipeline introduced in Step 1.2. Heavy model imports and all
CUDA allocations remain isolated in the spawned worker process.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

from app.application_controller import ApplicationController
from app.widgets.badge import BadgeKind
from app.widgets.depth_view import DepthMapView
from app.widgets.header_widget import HeaderWidget
from app.widgets.performance_bar import PerformanceBar
from app.widgets.performance_snapshot import PerformanceSnapshot
from app.widgets.runtime_toolbar import RuntimeToolbar
from app.widgets.settings_panel import SettingsPanel
from app.widgets.viewport.viewport_panel import ViewportPanel, ViewportStatus
from app.widgets.viewport.vispy_canvas import PointCloudCanvas
from core.capture.frame import SourceFrame
from core.geometry.colors import apply_colormap
from core.geometry.filtering import edge_filter_mask, remove_invalid, valid_mask
from core.geometry.projection import depth_to_points
from core.geometry.sampling import adaptive_stride, enforce_point_budget, stride_for_point_budget
from core.geometry.temporal import TemporalGeometrySmoother
from core.inference.process_manager import InferenceProcessManager
from core.models.catalog import (
    MOGE2_DISPLAY_LABEL,
    MOGE2_MODEL_ID,
    MOCK_DISPLAY_LABEL,
    MOCK_MODEL_ID,
)
from core.runtime.messages import Ack, CommandType, InferenceResult
from core.runtime.session import SessionTracker
from core.runtime.state_machine import ApplicationState, StateMachine
from core.runtime.telemetry import GeometryTiming, RateMeter, RuntimeCounters

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_MS = 5000
_MOCK_LATENCY_MS = 30.0
_RECENT_FRAMES_CACHE_SIZE = 3
_LIVE_PIPELINE_FPS = 12.0
_LIVE_PIPELINE_INTERVAL_S = 1.0 / _LIVE_PIPELINE_FPS
_PIPELINE_TICK_MS = 10

_STATE_TO_FOOTER_BADGE: dict[ApplicationState, tuple[BadgeKind, str]] = {
    ApplicationState.DISCONNECTED: (BadgeKind.NOT_INSTALLED, "IDLE"),
    ApplicationState.CONNECTING: (BadgeKind.LOADING, "CONNECTING"),
    ApplicationState.SOURCE_READY: (BadgeKind.COMPATIBLE, "SOURCE READY"),
    ApplicationState.LOADING_MODEL: (BadgeKind.LOADING, "LOADING MODEL"),
    ApplicationState.MODEL_READY: (BadgeKind.COMPATIBLE, "MODEL READY"),
    ApplicationState.RESULT_READY: (BadgeKind.RECOMMENDED, "RESULT READY"),
    ApplicationState.RUNNING: (BadgeKind.RECOMMENDED, "RUNNING"),
    ApplicationState.PAUSED: (BadgeKind.EXPERIMENTAL, "PAUSED"),
    ApplicationState.STOPPING: (BadgeKind.EXPERIMENTAL, "STOPPING"),
    ApplicationState.SWITCHING_MODEL: (BadgeKind.LOADING, "SWITCHING MODEL"),
    ApplicationState.RECONNECTING: (BadgeKind.EXPERIMENTAL, "RECONNECTING"),
    ApplicationState.ERROR: (BadgeKind.NOT_RECOMMENDED, "ERROR"),
    ApplicationState.SHUTTING_DOWN: (BadgeKind.NOT_INSTALLED, "SHUTTING DOWN"),
}


def _ndarray_to_qpixmap_rgb(rgb: np.ndarray) -> QPixmap:
    """Convert an RGB H×W×3 uint8 array to an independent QPixmap."""
    height, width, _ = rgb.shape
    contiguous = np.ascontiguousarray(rgb)
    qimage = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class InferenceController(QObject):
    """Drive model lifecycle, live backpressure, and geometry rendering."""

    viewport_escape_requested = pyqtSignal()

    def __init__(
        self,
        application_controller: ApplicationController,
        runtime_toolbar: RuntimeToolbar,
        header: HeaderWidget,
        performance_bar: PerformanceBar,
        depth_view: DepthMapView,
        viewport_panel: ViewportPanel,
        settings_panel: SettingsPanel,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._app_controller = application_controller
        self._runtime_toolbar = runtime_toolbar
        self._header = header
        self._performance_bar = performance_bar
        self._depth_view = depth_view
        self._viewport_panel = viewport_panel
        self._settings_panel = settings_panel

        self._state_machine = StateMachine(self)
        self._process_manager = InferenceProcessManager(
            stop_timeout_ms=_STOP_TIMEOUT_MS,
            parent=self,
        )
        self._session = SessionTracker()
        self._model_loaded = False
        self._requested_model_id: str | None = None
        self._active_model_id: str | None = None
        self._active_model_metadata: dict[str, object] = {}
        self._run_started_at: float | None = None

        self._inference_rate = RateMeter(window_seconds=2.0)
        self._render_rate = RateMeter(window_seconds=2.0)
        self._last_inference_fps: float | None = None
        self._last_render_fps: float | None = None
        self._counters = RuntimeCounters()

        # Latest-frame-only policy: one worker request in flight, one newest
        # pending capture frame, and one newest pending render result.
        self._pending_source_frame: SourceFrame | None = None
        self._inference_in_flight = False
        self._inflight_frame_id: int | None = None
        self._last_submit_at: float | None = None
        self._pending_render: tuple[InferenceResult, np.ndarray | None] | None = None
        self._last_render_at: float | None = None
        self._recent_frames: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._submitted_at_by_frame: "OrderedDict[int, float]" = OrderedDict()
        self._source_timestamp_by_frame: "OrderedDict[int, float]" = OrderedDict()

        # The most recently rendered result is retained so display controls can
        # refresh a static image immediately without another model inference.
        self._cached_raw_result: InferenceResult | None = None
        self._cached_render_result: InferenceResult | None = None
        self._cached_rgb: np.ndarray | None = None
        self._temporal_smoother = TemporalGeometrySmoother()

        self._pipeline_timer = QTimer(self)
        self._pipeline_timer.setInterval(_PIPELINE_TICK_MS)
        self._pipeline_timer.timeout.connect(self._on_pipeline_tick)

        self._settings_refresh_timer = QTimer(self)
        self._settings_refresh_timer.setSingleShot(True)
        self._settings_refresh_timer.setInterval(40)
        self._settings_refresh_timer.timeout.connect(self._rerender_cached_result)

        self._uptime_timer = QTimer(self)
        self._uptime_timer.setInterval(500)
        self._uptime_timer.timeout.connect(self._refresh_uptime)

        # Created once; only its data buffers are updated per result.
        self._point_cloud_canvas = PointCloudCanvas(self)
        self._viewport_panel.attach_canvas(self._point_cloud_canvas.native)
        self._point_cloud_canvas.escape_pressed.connect(
            self.viewport_escape_requested.emit
        )
        self._initialize_viewport_controls()

        self._state_machine.state_changed.connect(self._on_state_changed)
        self._process_manager.result_ready.connect(self._on_result_ready)
        self._process_manager.ack_received.connect(self._on_ack_received)
        self._process_manager.worker_unresponsive.connect(self._on_worker_unresponsive)
        self._process_manager.worker_crashed.connect(self._on_worker_crashed)

        runtime_toolbar.connect_clicked.connect(self._on_connect_clicked)
        runtime_toolbar.disconnect_clicked.connect(self._on_disconnect_clicked)
        runtime_toolbar.start_clicked.connect(self._on_start_clicked)
        runtime_toolbar.pause_resume_clicked.connect(self._on_pause_resume_clicked)
        runtime_toolbar.stop_clicked.connect(self._on_stop_clicked)
        viewport_panel.reset_view_clicked.connect(self._on_reset_view_clicked)
        viewport_panel.fit_requested.connect(self._on_fit_view_clicked)
        viewport_panel.tool_changed.connect(self._point_cloud_canvas.set_interaction_mode)
        viewport_panel.point_size_changed.connect(self._point_cloud_canvas.set_point_size)
        viewport_panel.smoothing_changed.connect(self._point_cloud_canvas.set_smoothing)
        viewport_panel.density_changed.connect(self._schedule_cached_geometry_refresh)
        viewport_panel.render_mode_changed.connect(self._schedule_cached_geometry_refresh)

        depth_view.colormap_changed.connect(settings_panel.set_colormap_name)
        depth_view.depth_range_changed.connect(settings_panel.set_depth_range)
        settings_panel.depth_visual_changed.connect(self._sync_depth_view_controls)
        settings_panel.depth_visual_changed.connect(self._schedule_cached_geometry_refresh)
        settings_panel.geometry_settings_changed.connect(self._schedule_cached_geometry_refresh)
        settings_panel.temporal_smoothing_changed.connect(self._on_temporal_smoothing_changed)
        settings_panel.background_changed.connect(self._on_background_changed)
        settings_panel.grid_visibility_changed.connect(self._point_cloud_canvas.set_grid_visible)
        settings_panel.axes_visibility_changed.connect(viewport_panel.set_axis_visible)
        settings_panel.point_budget_changed.connect(self._schedule_cached_geometry_refresh)

        self._apply_button_states()

    def _initialize_viewport_controls(self) -> None:
        """Apply the visible control state to the freshly created canvas."""
        self._point_cloud_canvas.set_interaction_mode(self._viewport_panel.active_tool)
        self._point_cloud_canvas.set_point_size(self._viewport_panel.point_size)
        self._point_cloud_canvas.set_smoothing(self._viewport_panel.smoothing)
        self._on_background_changed(self._settings_panel.background_name)
        self._point_cloud_canvas.set_grid_visible(self._settings_panel.grid_visible)
        self._viewport_panel.set_axis_visible(self._settings_panel.axes_visible)
        self._viewport_panel.set_surface_normals_available(False)
        self._sync_depth_view_controls()

    def _sync_depth_view_controls(self) -> None:
        self._depth_view.sync_controls(
            self._settings_panel.colormap_name,
            self._settings_panel.min_depth_m,
            self._settings_panel.max_depth_m,
        )

    def _on_background_changed(self, background: str) -> None:
        self._point_cloud_canvas.set_background(background)
        self._viewport_panel.set_light_background(background == "white")

    def _schedule_cached_geometry_refresh(self, *_args: object) -> None:
        if self._cached_render_result is None and self._cached_raw_result is None:
            return
        self._settings_refresh_timer.start()

    def _on_temporal_smoothing_changed(self, _strength: float) -> None:
        # Do not repeatedly blend the already displayed frame. The new
        # strength takes effect on the next continuous-source result.
        self._temporal_smoother.reset()

    # -- Runtime toolbar handlers ----------------------------------------

    def _on_connect_clicked(self) -> None:
        self._state_machine.transition_to(ApplicationState.CONNECTING)
        self._app_controller.connect_active_source()
        # Capture widgets expose the asynchronous source status independently.
        self._state_machine.transition_to(ApplicationState.SOURCE_READY)
        self._apply_button_states()

    def _on_disconnect_clicked(self) -> None:
        if self._state_machine.state in (ApplicationState.RUNNING, ApplicationState.PAUSED):
            self._stop_inference()

        if self._model_loaded:
            self._process_manager.unload_model()
        self._process_manager.shutdown()

        self._model_loaded = False
        self._requested_model_id = None
        self._active_model_id = None
        self._active_model_metadata.clear()
        self._viewport_panel.set_surface_normals_available(False)
        self._settings_panel.set_model_unloaded()

        self._app_controller.disconnect_active_source()
        self._depth_view.clear_frame()
        self._point_cloud_canvas.clear_points()
        self._viewport_panel.invalidate_camera()
        self._viewport_panel.set_status(ViewportStatus.IDLE)
        self._performance_bar.reset_metrics()
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()
        self._state_machine.transition_to(ApplicationState.DISCONNECTED)
        self._apply_button_states()

    def _on_start_clicked(self) -> None:
        # Stop keeps the worker model resident for a fast restart.
        if (
            self._state_machine.state in (
                ApplicationState.MODEL_READY,
                ApplicationState.RESULT_READY,
            )
            and self._model_loaded
            and self._process_manager.is_running
        ):
            self._begin_running_session()
            return

        load_state = (
            ApplicationState.SWITCHING_MODEL
            if self._state_machine.state in (
                ApplicationState.MODEL_READY,
                ApplicationState.RESULT_READY,
            )
            else ApplicationState.LOADING_MODEL
        )
        self._state_machine.transition_to(load_state)
        self._settings_panel.set_model_loading()
        self._header.set_status(BadgeKind.LOADING, "LOADING MODEL")
        self._apply_button_states()

        model_id = self._settings_panel.selected_model_id
        self._requested_model_id = model_id
        if model_id == MOGE2_MODEL_ID:
            config = dict(self._settings_panel.model_runtime_config)
        elif model_id == MOCK_MODEL_ID:
            config = {
                "device": "cpu",
                "precision": "fp32",
                "latency_ms": _MOCK_LATENCY_MS,
            }
        else:
            self._handle_model_load_failure(f"Unknown model selection: {model_id!r}")
            return

        if not self._process_manager.is_running:
            self._process_manager.start_worker()
        self._process_manager.load_model(model_id, config)
        # LOAD_MODEL ack advances MODEL_READY -> RUNNING.

    def _begin_running_session(self) -> None:
        """Start a fresh session using the model already resident in the worker."""
        if self._state_machine.state not in (
            ApplicationState.MODEL_READY,
            ApplicationState.RESULT_READY,
        ):
            return

        self._session.start_new_session()
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()
        self._viewport_panel.invalidate_camera()
        self._process_manager.resume()
        self._state_machine.transition_to(ApplicationState.RUNNING)
        self._run_started_at = time.monotonic()
        self._performance_bar.update_snapshot(self._initial_session_snapshot())
        self._viewport_panel.set_status(ViewportStatus.LIVE)
        self._set_active_model_header()
        self._pipeline_timer.start()
        self._uptime_timer.start()
        self._apply_button_states()

        # Static images emitted before inference starts need an explicit resend.
        self._app_controller.resend_current_frame()

    def _on_pause_resume_clicked(self) -> None:
        if self._state_machine.state == ApplicationState.RUNNING:
            self._process_manager.pause()
            self._state_machine.transition_to(ApplicationState.PAUSED)
            self._runtime_toolbar.set_paused_label(is_paused=True)
            self._viewport_panel.set_status(ViewportStatus.PAUSED)
            self._pipeline_timer.stop()
            if self._pending_source_frame is not None:
                self._counters.pending_frames_replaced += 1
            if self._pending_render is not None:
                dropped_result, _dropped_rgb = self._pending_render
                self._counters.pending_results_replaced += 1
                self._submitted_at_by_frame.pop(dropped_result.frame_id, None)
                self._source_timestamp_by_frame.pop(dropped_result.frame_id, None)
            self._pending_source_frame = None
            self._pending_render = None
            self._refresh_uptime()
        elif self._state_machine.state == ApplicationState.PAUSED:
            self._process_manager.resume()
            self._state_machine.transition_to(ApplicationState.RUNNING)
            self._runtime_toolbar.set_paused_label(is_paused=False)
            self._viewport_panel.set_status(ViewportStatus.LIVE)
            self._set_active_model_header()
            self._pipeline_timer.start()
            self._app_controller.resend_current_frame()
        self._apply_button_states()

    def _on_stop_clicked(self) -> None:
        self._stop_inference()
        self._apply_button_states()

    def _stop_inference(self) -> None:
        self._state_machine.transition_to(ApplicationState.STOPPING)
        self._apply_button_states()
        self._refresh_uptime()
        stopped_session_id = self._session.active_session_id
        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.stop(session_id=stopped_session_id)
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._viewport_panel.set_status(
            ViewportStatus.RESULT_READY
            if self._point_cloud_canvas.has_points
            else ViewportStatus.IDLE
        )
        # Preserve the last result and its metrics for inspection. Only an
        # explicit source disconnect clears the panels and footer.
        self._state_machine.transition_to(ApplicationState.MODEL_READY)
        self._set_active_model_header()

    def _on_reset_view_clicked(self) -> None:
        self._point_cloud_canvas.reset_view()
        self._viewport_panel.mark_camera_initialized()

    def _on_fit_view_clicked(self) -> None:
        if not self._point_cloud_canvas.has_points:
            return
        self._point_cloud_canvas.fit_scene()
        self._viewport_panel.mark_camera_initialized()

    # -- Latest-frame inference scheduling -------------------------------

    def on_source_frame(self, frame: SourceFrame) -> None:
        if self._state_machine.state != ApplicationState.RUNNING:
            return
        self._counters.source_frames_seen += 1
        if self._pending_source_frame is not None:
            self._counters.pending_frames_replaced += 1
        self._pending_source_frame = frame
        self._try_submit_latest_frame()

    def _on_pipeline_tick(self) -> None:
        if self._state_machine.state != ApplicationState.RUNNING:
            return
        self._try_submit_latest_frame()
        self._try_render_latest_result()

    def _try_submit_latest_frame(self) -> None:
        if self._state_machine.state != ApplicationState.RUNNING:
            return
        if self._inference_in_flight or self._pending_source_frame is None:
            return
        if not self._process_manager.is_running:
            return

        now = time.monotonic()
        if (
            self._app_controller.is_stream_source
            and self._last_submit_at is not None
            and now - self._last_submit_at < _LIVE_PIPELINE_INTERVAL_S
        ):
            return

        source_frame = self._pending_source_frame
        self._pending_source_frame = None
        frame_id = self._session.next_frame_id()
        session_id = self._session.active_session_id
        if session_id == 0:
            return

        self._recent_frames[frame_id] = source_frame.image
        while len(self._recent_frames) > _RECENT_FRAMES_CACHE_SIZE:
            self._recent_frames.popitem(last=False)

        point_budget = (
            self._settings_panel.live_point_budget
            if self._app_controller.is_stream_source
            else self._settings_panel.point_budget
        )
        try:
            accepted = self._process_manager.submit_frame(
                session_id,
                frame_id,
                source_frame.image,
                point_budget=point_budget,
                submitted_at=now,
                source_timestamp=source_frame.timestamp,
            )
        except TypeError as exc:
            # Compatibility with focused legacy test doubles that implement
            # the pre-telemetry submit_frame signature.
            if "unexpected keyword" not in str(exc):
                raise
            accepted = self._process_manager.submit_frame(
                session_id,
                frame_id,
                source_frame.image,
                point_budget=point_budget,
            )
        # Only explicit False is rejection; legacy test doubles may return None.
        if accepted is False:
            self._counters.submit_rejected += 1
            self._recent_frames.pop(frame_id, None)
            self._pending_source_frame = source_frame
            return

        self._counters.frames_submitted += 1
        self._submitted_at_by_frame[frame_id] = now
        self._source_timestamp_by_frame[frame_id] = source_frame.timestamp
        self._inference_in_flight = True
        self._inflight_frame_id = frame_id
        self._last_submit_at = now

    # -- Worker signal handlers ------------------------------------------

    def _on_ack_received(self, ack: Ack) -> None:
        if ack.command_type != CommandType.LOAD_MODEL:
            return

        if not ack.success:
            self._handle_model_load_failure(ack.message or "Model loading failed.")
            return

        self._model_loaded = True
        self._active_model_id = ack.model_id or self._requested_model_id
        self._active_model_metadata = dict(ack.metadata or {})
        if ack.load_time_ms is not None:
            self._active_model_metadata["load_time_ms"] = ack.load_time_ms
        if ack.model_load_time_ms is not None:
            self._active_model_metadata["model_load_time_ms"] = ack.model_load_time_ms
        if ack.warmup_time_ms is not None:
            self._active_model_metadata["warmup_time_ms"] = ack.warmup_time_ms
        self._settings_panel.set_model_loaded(self._active_model_metadata)
        outputs = self._active_model_metadata.get("outputs", {})
        normals_available = bool(
            isinstance(outputs, dict) and outputs.get("normals", False)
        )
        self._viewport_panel.set_surface_normals_available(normals_available)
        self._set_active_model_header()

        if self._state_machine.state in (
            ApplicationState.LOADING_MODEL,
            ApplicationState.SWITCHING_MODEL,
        ):
            self._state_machine.transition_to(ApplicationState.MODEL_READY)
            self._begin_running_session()

    def _handle_model_load_failure(self, message: str) -> None:
        logger.error("Model load failed: %s", message)
        self._model_loaded = False
        self._active_model_id = None
        self._active_model_metadata.clear()
        self._viewport_panel.set_surface_normals_available(False)
        self._settings_panel.set_model_error(message, label="LOAD FAILED")
        self._header.set_status(BadgeKind.NOT_RECOMMENDED, "LOAD FAILED")

        # Keep the source connected and return to a retryable state. This lets
        # the user select Development Mock when the optional MoGe dependency or
        # checkpoint is unavailable, without restarting the whole application.
        if self._state_machine.state in (
            ApplicationState.LOADING_MODEL,
            ApplicationState.SWITCHING_MODEL,
        ):
            self._state_machine.transition_to(ApplicationState.SOURCE_READY)
        else:
            self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
        self._performance_bar.set_status(BadgeKind.NOT_RECOMMENDED, "ERROR")
        self._apply_button_states()

    def _on_result_ready(self, result: InferenceResult) -> None:
        if not self._session.is_current(result.session_id):
            self._counters.stale_results_ignored += 1
            return

        self._counters.results_received += 1
        rgb = self._recent_frames.pop(result.frame_id, None)
        if result.frame_id == self._inflight_frame_id:
            self._inference_in_flight = False
            self._inflight_frame_id = None

        if self._state_machine.state != ApplicationState.RUNNING:
            self._submitted_at_by_frame.pop(result.frame_id, None)
            self._source_timestamp_by_frame.pop(result.frame_id, None)
            return

        if result.error is not None:
            self._handle_inference_failure(result.error)
            return
        if result.depth is None:
            self._handle_inference_failure("The model returned no depth map.")
            return

        self._update_inference_fps()
        if self._pending_render is not None:
            replaced_result, _replaced_rgb = self._pending_render
            self._counters.pending_results_replaced += 1
            self._submitted_at_by_frame.pop(replaced_result.frame_id, None)
            self._source_timestamp_by_frame.pop(replaced_result.frame_id, None)
        self._pending_render = (result, rgb)

        # Overlap newest inference with the bounded UI render slot without
        # allowing more than one request to queue in the worker.
        self._try_submit_latest_frame()
        self._try_render_latest_result()

    def _handle_inference_failure(self, message: str) -> None:
        logger.error("Inference failed: %s", message)
        self._counters.inference_errors += 1
        self._refresh_uptime()
        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.pause()
        self._viewport_panel.set_status(ViewportStatus.ERROR)
        self._settings_panel.set_model_error(
            message,
            label="INFERENCE ERROR",
            unlock_controls=False,
        )
        self._header.set_status(BadgeKind.NOT_RECOMMENDED, "INFERENCE ERROR")
        self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
        self._performance_bar.set_status(BadgeKind.NOT_RECOMMENDED, "ERROR")
        self._apply_button_states()

    def _try_render_latest_result(self) -> None:
        if self._state_machine.state != ApplicationState.RUNNING:
            return
        if self._pending_render is None:
            return

        now = time.monotonic()
        if (
            self._app_controller.is_stream_source
            and self._last_render_at is not None
            and now - self._last_render_at < _LIVE_PIPELINE_INTERVAL_S
        ):
            return

        result, rgb = self._pending_render
        self._pending_render = None
        self._cached_raw_result = result
        self._cached_rgb = rgb
        try:
            geometry = self._update_geometry(result, rgb, apply_temporal=True)
            if isinstance(geometry, int):  # compatibility with focused test doubles
                geometry = GeometryTiming(point_count=geometry)
        except Exception as exc:  # noqa: BLE001 - renderer errors must be visible
            self._handle_inference_failure(f"Geometry rendering failed: {exc}")
            return

        rendered_at = time.monotonic()
        self._last_render_at = rendered_at
        self._counters.results_rendered += 1
        self._update_render_fps()

        submitted_at = self._submitted_at_by_frame.pop(result.frame_id, None)
        source_timestamp = self._source_timestamp_by_frame.pop(result.frame_id, None)
        pipeline_latency_ms = (
            (rendered_at - submitted_at) * 1000.0
            if submitted_at is not None
            else None
        )
        source_to_render_ms = (
            (rendered_at - source_timestamp) * 1000.0
            if source_timestamp is not None
            else None
        )
        self._update_performance_snapshot(
            result,
            geometry=geometry,
            pipeline_latency_ms=pipeline_latency_ms,
            source_to_render_ms=source_to_render_ms,
        )

        if not self._app_controller.is_stream_source:
            self._complete_static_result()

    def _complete_static_result(self) -> None:
        """Finish one-shot image inference while keeping the result visible."""
        if self._state_machine.state != ApplicationState.RUNNING:
            return

        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.pause()
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._state_machine.transition_to(ApplicationState.RESULT_READY)
        self._viewport_panel.set_status(ViewportStatus.RESULT_READY)
        self._set_active_model_header()
        self._apply_button_states()

    # -- Geometry ---------------------------------------------------------

    def _update_geometry(
        self,
        result: InferenceResult,
        rgb: np.ndarray | None,
        *,
        apply_temporal: bool = True,
    ) -> GeometryTiming:
        """Render geometry and return measured UI-side stage timings."""
        total_started = time.perf_counter()
        effective_result = self._prepare_render_result(
            result,
            apply_temporal=apply_temporal,
        )
        self._cached_render_result = effective_result

        depth = effective_result.depth
        if depth is None or depth.ndim != 2:
            raise ValueError("Expected a two-dimensional depth map.")

        settings = self._settings_panel
        near = settings.min_depth_m
        far = max(settings.max_depth_m, near + 0.1)
        depth_started = time.perf_counter()

        finite_depth_mask = np.isfinite(depth) & (depth > 0.0)
        model_valid_mask = _aligned_mask(effective_result.valid_mask, depth.shape)
        full_valid_mask = finite_depth_mask.copy()
        if settings.background_removal_enabled:
            if model_valid_mask is not None:
                full_valid_mask &= model_valid_mask
            full_valid_mask &= (depth >= near) & (depth <= far)

        depth_rgb = apply_colormap(
            depth,
            colormap_name=settings.colormap_name,
            near=near,
            far=far,
            valid_mask=full_valid_mask,
        )
        self._depth_view.set_depth_frame(_ndarray_to_qpixmap_rgb(depth_rgb))
        depth_view_ms = (time.perf_counter() - depth_started) * 1000.0
        geometry_started = time.perf_counter()

        point_budget = (
            settings.live_point_budget
            if self._app_controller.is_stream_source
            else settings.point_budget
        )
        density_ratio = float(np.clip(self._viewport_panel.density_ratio, 0.01, 1.0))
        # Density is enforced after validity/edge filtering so 50% means
        # approximately half of the points that would otherwise be rendered.
        # The input stride remains available for temporary interaction-time
        # decimation and live point-budget protection.
        density_stride = 1

        if effective_result.points is not None:
            points_grid, sampled_depth, sampled_colors, sampled_mask = self._direct_geometry_inputs(
                effective_result,
                rgb,
                depth_rgb,
                full_valid_mask,
                density_stride,
            )
        else:
            points_grid, sampled_depth, sampled_colors, sampled_mask = self._depth_geometry_inputs(
                effective_result,
                rgb,
                depth_rgb,
                full_valid_mask,
                density_stride,
                point_budget,
            )

        mask = valid_mask(points_grid, sampled_depth)
        if sampled_mask is not None:
            mask &= sampled_mask
        mask &= edge_filter_mask(sampled_depth, settings.edge_filter_strength)

        flat_points, flat_colors = remove_invalid(
            np.where(mask[..., None], points_grid, np.nan),
            sampled_colors,
            np.where(mask, sampled_depth, 0.0),
        )
        density_budget = max(1, int(round(flat_points.shape[0] * density_ratio)))
        flat_points, flat_colors = enforce_point_budget(
            flat_points,
            flat_colors,
            min(point_budget, density_budget),
        )

        geometry_cpu_ms = (time.perf_counter() - geometry_started) * 1000.0
        canvas_started = time.perf_counter()
        self._point_cloud_canvas.set_point_size(self._viewport_panel.point_size)
        self._point_cloud_canvas.set_grid_visible(settings.grid_visible)
        self._viewport_panel.set_axis_visible(settings.axes_visible)
        if flat_points.shape[0] > 0:
            self._point_cloud_canvas.update_points(flat_points, flat_colors)
        else:
            self._point_cloud_canvas.clear_points()

        if flat_points.shape[0] > 0 and not self._viewport_panel.camera_initialized:
            self._point_cloud_canvas.fit_scene()
            self._viewport_panel.mark_camera_initialized()
        canvas_update_ms = (time.perf_counter() - canvas_started) * 1000.0

        self._set_active_model_header()
        return GeometryTiming(
            point_count=int(flat_points.shape[0]),
            depth_view_ms=depth_view_ms,
            geometry_cpu_ms=geometry_cpu_ms,
            canvas_update_ms=canvas_update_ms,
            total_ms=(time.perf_counter() - total_started) * 1000.0,
        )

    def _prepare_render_result(
        self,
        result: InferenceResult,
        *,
        apply_temporal: bool,
    ) -> InferenceResult:
        if (
            not apply_temporal
            or not self._app_controller.is_stream_source
            or result.depth is None
        ):
            return result

        signature = (
            result.model_id,
            result.depth.shape,
            max(1, int(result.point_stride)),
            None if result.points is None else result.points.shape,
            None if result.normals is None else result.normals.shape,
        )
        smoothed = self._temporal_smoother.apply(
            depth=result.depth,
            points=result.points,
            normals=result.normals,
            strength=self._settings_panel.temporal_smoothing,
            signature=signature,
        )
        return replace(
            result,
            depth=smoothed.depth,
            points=smoothed.points,
            normals=smoothed.normals,
        )

    def _rerender_cached_result(self) -> None:
        """Refresh the visible depth/geometry using cached model outputs."""
        result = self._cached_render_result or self._cached_raw_result
        if result is None or result.depth is None:
            return

        try:
            geometry = self._update_geometry(
                result,
                self._cached_rgb,
                apply_temporal=False,
            )
        except Exception:  # noqa: BLE001 - retain the last valid visual
            logger.exception("Cached geometry refresh failed")
            return

        point_budget = (
            self._settings_panel.live_point_budget
            if self._app_controller.is_stream_source
            else self._settings_panel.point_budget
        )
        snapshot = replace(
            self._performance_bar.snapshot,
            point_count=geometry.point_count,
            geometry_ms=geometry.total_ms,
            geometry_cpu_ms=geometry.geometry_cpu_ms,
            depth_view_ms=geometry.depth_view_ms,
            canvas_update_ms=geometry.canvas_update_ms,
            point_stride=max(1, int(result.point_stride)),
            point_budget=point_budget,
        )
        self._performance_bar.update_snapshot(snapshot)

    def _clear_cached_geometry(self) -> None:
        self._settings_refresh_timer.stop()
        self._cached_raw_result = None
        self._cached_render_result = None
        self._cached_rgb = None
        self._temporal_smoother.reset()

    def _direct_geometry_inputs(
        self,
        result: InferenceResult,
        rgb: np.ndarray | None,
        depth_rgb: np.ndarray,
        full_valid_mask: np.ndarray | None,
        density_stride: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Align MoGe's worker-sampled point grid with depth/RGB/mask data."""
        assert result.depth is not None
        assert result.points is not None

        base_stride = max(1, int(result.point_stride))
        base_depth = result.depth[::base_stride, ::base_stride]
        points_grid = np.asarray(result.points, dtype=np.float32)
        if points_grid.shape[:2] != base_depth.shape or points_grid.shape[-1] != 3:
            raise ValueError(
                "Direct point-map shape does not match depth and point_stride: "
                f"points={points_grid.shape}, depth={result.depth.shape}, "
                f"stride={base_stride}."
            )

        extra_stride = adaptive_stride(
            density_stride,
            self._point_cloud_canvas.is_interacting,
        )
        points_grid = points_grid[::extra_stride, ::extra_stride]
        sampled_depth = base_depth[::extra_stride, ::extra_stride]

        render_mode = self._viewport_panel.render_mode
        if render_mode == "Surface Normals" and result.normals is not None:
            normal_grid = np.asarray(result.normals, dtype=np.float32)
            if normal_grid.shape == result.points.shape:
                sampled_colors = _normal_colors(normal_grid[::extra_stride, ::extra_stride])
            else:
                sampled_colors = depth_rgb[
                    :: base_stride * extra_stride,
                    :: base_stride * extra_stride,
                ]
        elif render_mode == "Depth Colors":
            sampled_colors = depth_rgb[
                :: base_stride * extra_stride,
                :: base_stride * extra_stride,
            ]
        else:
            aligned_rgb = _aligned_rgb(rgb, result.depth.shape)
            source = aligned_rgb if aligned_rgb is not None else depth_rgb
            sampled_colors = source[
                :: base_stride * extra_stride,
                :: base_stride * extra_stride,
            ]

        sampled_mask = None
        if full_valid_mask is not None:
            sampled_mask = full_valid_mask[
                :: base_stride * extra_stride,
                :: base_stride * extra_stride,
            ]
        return points_grid, sampled_depth, sampled_colors, sampled_mask

    def _depth_geometry_inputs(
        self,
        result: InferenceResult,
        rgb: np.ndarray | None,
        depth_rgb: np.ndarray,
        full_valid_mask: np.ndarray | None,
        density_stride: int,
        point_budget: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Keep the existing virtual-FOV path for mock/depth-only adapters."""
        assert result.depth is not None
        depth = result.depth
        if self._app_controller.is_stream_source:
            budget_stride = stride_for_point_budget(depth.shape[0], depth.shape[1], point_budget)
        else:
            budget_stride = 1
        stride = adaptive_stride(
            max(density_stride, budget_stride),
            self._point_cloud_canvas.is_interacting,
        )

        sampled_depth = depth[::stride, ::stride]
        points_grid = depth_to_points(sampled_depth)

        render_mode = self._viewport_panel.render_mode
        if render_mode == "Depth Colors" or render_mode == "Surface Normals":
            color_source = depth_rgb
        else:
            aligned_rgb = _aligned_rgb(rgb, depth.shape)
            color_source = aligned_rgb if aligned_rgb is not None else depth_rgb
        sampled_colors = color_source[::stride, ::stride]
        sampled_mask = full_valid_mask[::stride, ::stride] if full_valid_mask is not None else None
        return points_grid, sampled_depth, sampled_colors, sampled_mask

    # -- Performance metrics ---------------------------------------------

    def _update_performance_snapshot(
        self,
        result: InferenceResult,
        *,
        geometry: GeometryTiming | None = None,
        point_count: int | None = None,
        pipeline_latency_ms: float | None = None,
        source_to_render_ms: float | None = None,
    ) -> None:
        if geometry is None:
            geometry = GeometryTiming(point_count=point_count or 0)
        model_label = result.model_name or self._active_model_display_name(result.model_id)
        precision = result.precision or (
            "FP32" if result.model_id == MOCK_MODEL_ID else None
        )
        uptime_seconds = (
            int(time.monotonic() - self._run_started_at)
            if self._run_started_at is not None
            else None
        )
        resolution = None
        if result.depth is not None:
            resolution = (result.depth.shape[1], result.depth.shape[0])

        metadata = self._active_model_metadata
        reserved_gib = (
            result.vram_reserved_gib
            if result.vram_reserved_gib is not None
            else result.vram_used_gib
        )
        visible_latency = (
            pipeline_latency_ms
            if pipeline_latency_ms is not None
            else result.worker_total_ms
            if result.worker_total_ms is not None
            else result.latency_ms
        )
        point_budget = (
            self._settings_panel.live_point_budget
            if self._app_controller.is_stream_source
            else self._settings_panel.point_budget
        )

        self._performance_bar.update_snapshot(
            PerformanceSnapshot(
                status="RUNNING",
                camera_fps=self._app_controller.capture_fps,
                inference_fps=self._last_inference_fps,
                latency_ms=visible_latency,
                model_latency_ms=result.latency_ms,
                preprocess_ms=result.preprocess_ms,
                postprocess_ms=result.postprocess_ms,
                worker_total_ms=result.worker_total_ms,
                queue_wait_ms=result.queue_wait_ms,
                geometry_ms=geometry.total_ms,
                geometry_cpu_ms=geometry.geometry_cpu_ms,
                depth_view_ms=geometry.depth_view_ms,
                canvas_update_ms=geometry.canvas_update_ms,
                source_to_render_ms=source_to_render_ms,
                render_fps=self._last_render_fps,
                point_count=geometry.point_count,
                model_name=model_label,
                precision=precision,
                resolution=resolution,
                vram_used_gb=reserved_gib,
                vram_allocated_gb=result.vram_allocated_gib,
                vram_reserved_gb=reserved_gib,
                vram_peak_reserved_gb=result.vram_peak_reserved_gib,
                vram_device_used_gb=result.vram_device_used_gib,
                vram_total_gb=result.vram_total_gib,
                uptime_seconds=uptime_seconds,
                model_load_time_ms=_float_metadata(metadata, "model_load_time_ms"),
                warmup_time_ms=_float_metadata(metadata, "warmup_time_ms"),
                total_load_time_ms=_float_metadata(metadata, "load_time_ms"),
                model_device=_string_metadata(metadata, "device"),
                checkpoint=_string_metadata(metadata, "pretrained"),
                num_tokens=_int_metadata(metadata, "num_tokens"),
                resolution_level=_int_metadata(metadata, "resolution_level"),
                source_frames_seen=self._counters.source_frames_seen,
                frames_submitted=self._counters.frames_submitted,
                results_received=self._counters.results_received,
                results_rendered=self._counters.results_rendered,
                dropped_before_inference=self._counters.dropped_before_inference,
                dropped_before_render=self._counters.dropped_before_render,
                submit_rejected=self._counters.submit_rejected,
                inference_errors=self._counters.inference_errors,
                point_stride=max(1, int(result.point_stride)),
                point_budget=point_budget,
            )
        )

        header_used_gib = (
            result.vram_device_used_gib
            if result.vram_device_used_gib is not None
            else reserved_gib
        )
        if header_used_gib is not None and result.vram_total_gib is not None:
            # The compact header is device-wide VRAM usage. The footer is
            # deliberately PyTorch-reserved memory and explains its allocator
            # breakdown in the tooltip.
            self._header.update_gpu_status(
                self._settings_panel.hardware_info.name,
                header_used_gib,
                result.vram_total_gib,
            )

    def _initial_session_snapshot(self) -> PerformanceSnapshot:
        metadata = self._active_model_metadata
        precision = _string_metadata(metadata, "precision")
        if not precision and self._active_model_id == MOCK_MODEL_ID:
            precision = "FP32"
        return PerformanceSnapshot(
            status="RUNNING",
            model_name=self._active_model_display_name(),
            precision=precision,
            uptime_seconds=0,
            model_load_time_ms=_float_metadata(metadata, "model_load_time_ms"),
            warmup_time_ms=_float_metadata(metadata, "warmup_time_ms"),
            total_load_time_ms=_float_metadata(metadata, "load_time_ms"),
            model_device=_string_metadata(metadata, "device"),
            checkpoint=_string_metadata(metadata, "pretrained"),
            num_tokens=_int_metadata(metadata, "num_tokens"),
            resolution_level=_int_metadata(metadata, "resolution_level"),
        )

    def _refresh_uptime(self) -> None:
        if self._run_started_at is None:
            return
        snapshot = self._performance_bar.snapshot.with_runtime_counters(
            uptime_seconds=int(time.monotonic() - self._run_started_at),
            source_frames_seen=self._counters.source_frames_seen,
            frames_submitted=self._counters.frames_submitted,
            results_received=self._counters.results_received,
            results_rendered=self._counters.results_rendered,
            dropped_before_inference=self._counters.dropped_before_inference,
            dropped_before_render=self._counters.dropped_before_render,
            submit_rejected=self._counters.submit_rejected,
            inference_errors=self._counters.inference_errors,
        )
        self._performance_bar.update_snapshot(snapshot)

    def _active_model_display_name(self, model_id: str | None = None) -> str:
        metadata_name = str(self._active_model_metadata.get("display_name", ""))
        if metadata_name:
            return metadata_name
        active_id = model_id or self._active_model_id
        if active_id == MOCK_MODEL_ID:
            return MOCK_DISPLAY_LABEL
        if active_id == MOGE2_MODEL_ID:
            return self._settings_panel.resolved_profile.spec.model_label
        return active_id or ""

    def _set_active_model_header(self) -> None:
        if self._active_model_id == MOCK_MODEL_ID:
            self._header.set_status(BadgeKind.EXPERIMENTAL, "MOCK")
        elif self._active_model_id == MOGE2_MODEL_ID:
            self._header.set_status(BadgeKind.RECOMMENDED, "MOGE-2")

    def _update_inference_fps(self) -> None:
        self._last_inference_fps = self._inference_rate.mark(time.monotonic())

    def _update_render_fps(self) -> None:
        self._last_render_fps = self._render_rate.mark(time.monotonic())

    def _reset_pipeline_runtime(self) -> None:
        self._pipeline_timer.stop()
        self._uptime_timer.stop()
        self._pending_source_frame = None
        self._pending_render = None
        self._inference_in_flight = False
        self._inflight_frame_id = None
        self._last_submit_at = None
        self._last_render_at = None
        self._recent_frames.clear()
        self._submitted_at_by_frame.clear()
        self._source_timestamp_by_frame.clear()

        self._run_started_at = None
        self._inference_rate.reset()
        self._render_rate.reset()
        self._last_inference_fps = None
        self._last_render_fps = None
        self._counters.reset()
        self._temporal_smoother.reset()

    # -- Worker recovery --------------------------------------------------

    def _on_worker_unresponsive(self, command_type: CommandType) -> None:
        logger.warning("Inference worker did not acknowledge %s in time", command_type)

    def _on_worker_crashed(self) -> None:
        logger.error("Inference worker process crashed")
        self._model_loaded = False
        self._active_model_id = None
        self._active_model_metadata.clear()
        self._viewport_panel.set_surface_normals_available(False)
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()
        self._settings_panel.set_model_error(
            "The inference worker process crashed.",
            label="WORKER ERROR",
        )
        self._header.set_status(BadgeKind.NOT_RECOMMENDED, "WORKER ERROR")
        self._viewport_panel.set_status(ViewportStatus.ERROR)
        self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
        self._apply_button_states()

    def restart_worker(self) -> None:
        """Public entry point for a future Restart Worker UI action."""
        self._process_manager.restart_worker()
        self._model_loaded = False
        self._active_model_id = None
        self._active_model_metadata.clear()
        self._viewport_panel.set_surface_normals_available(False)
        self._settings_panel.set_model_unloaded()
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()

    # -- Button state application ----------------------------------------

    def _on_state_changed(self, _old_state, new_state) -> None:
        self._apply_button_states()
        kind, text = _STATE_TO_FOOTER_BADGE.get(
            new_state,
            (BadgeKind.NOT_INSTALLED, "IDLE"),
        )
        self._performance_bar.set_status(kind, text)

    def _apply_button_states(self) -> None:
        buttons = self._state_machine.button_states
        self._runtime_toolbar.apply_button_states(
            connect_enabled=buttons.connect_enabled,
            disconnect_enabled=buttons.disconnect_enabled,
            start_enabled=buttons.start_enabled,
            pause_resume_enabled=buttons.pause_resume_enabled,
            stop_enabled=buttons.stop_enabled,
        )

    def shutdown(self) -> None:
        """Called from MainWindow.closeEvent for a clean app exit."""
        if self._state_machine.state in (ApplicationState.RUNNING, ApplicationState.PAUSED):
            self._stop_inference()
        self._pipeline_timer.stop()
        self._settings_refresh_timer.stop()
        self._process_manager.shutdown()
        self._settings_panel.set_model_unloaded()
        self._point_cloud_canvas.close()

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine



def _float_metadata(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_metadata(metadata: dict[str, object], key: str) -> int | None:
    value = metadata.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _string_metadata(metadata: dict[str, object], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _aligned_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    array = np.asarray(mask, dtype=bool)
    return array if array.shape == shape else None


def _aligned_rgb(rgb: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if rgb is None or rgb.shape[:2] != shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _normal_colors(normals: np.ndarray) -> np.ndarray:
    """Map camera-space normals from [-1, 1] into RGB [0, 255]."""
    finite = np.isfinite(normals).all(axis=-1)
    normalized = np.clip((normals.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    colors = (normalized * 255.0).astype(np.uint8)
    colors[~finite] = 0
    return colors
