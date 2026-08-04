"""Inference lifecycle, latest-frame scheduling, and geometry rendering.

Phase 6 keeps the bounded Phase 5 runtime stable while making model selection,
configuration, capabilities, and presentation registry-driven. Heavy model
imports and all CUDA allocations remain isolated in the spawned worker process.
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
from core.geometry.projection import depth_to_points, scale_intrinsics
from core.geometry.sampling import adaptive_stride, enforce_point_budget, stride_for_point_budget
from core.geometry.temporal import TemporalGeometrySmoother
from core.inference.process_manager import InferenceProcessManager
from core.models.registry import get_model_descriptor
from core.runtime.failures import (
    FailureInfo,
    FailureKind,
    FailureStage,
    classify_failure,
    crashed_worker_failure,
    format_failure_tooltip,
    timeout_failure,
)
from core.runtime.messages import (
    Ack,
    CommandType,
    InferenceResult,
    ModelStage,
    ModelStageEvent,
)
from core.runtime.model_lifecycle import (
    ModelLifecycle,
    ModelLifecycleState,
    model_config_signature,
)
from core.runtime.session import SessionTracker
from core.runtime.state_machine import ApplicationState, StateMachine
from core.runtime.telemetry import GeometryTiming, RateMeter, RuntimeCounters

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_MS = 5000
_RECENT_FRAMES_CACHE_SIZE = 3
_LIVE_PIPELINE_FPS = 12.0
_LIVE_PIPELINE_INTERVAL_S = 1.0 / _LIVE_PIPELINE_FPS
_PIPELINE_TICK_MS = 10
_DEFAULT_INFERENCE_RESULT_TIMEOUT_MS = 30_000
_MAX_AUTO_WORKER_RECOVERIES = 1

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
    ApplicationState.RECOVERING_WORKER: (BadgeKind.LOADING, "WORKER RESTARTING"),
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
        self._model_lifecycle = ModelLifecycle()
        self._model_loaded = False
        self._requested_model_id: str | None = None
        self._requested_model_config: dict[str, object] = {}
        self._active_model_id: str | None = None
        self._active_model_config_signature: tuple | None = None
        self._active_model_metadata: dict[str, object] = {}
        self._pending_model_request: tuple[str, dict[str, object]] | None = None
        self._auto_start_after_load = False
        self._run_started_at: float | None = None
        self._last_failure: FailureInfo | None = None
        self._worker_recovery_attempts = 0
        self._recovery_in_progress = False
        self._shutdown_started = False

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

        self._inference_watchdog_timer = QTimer(self)
        self._inference_watchdog_timer.setSingleShot(True)
        self._active_inference_timeout_ms = _DEFAULT_INFERENCE_RESULT_TIMEOUT_MS
        self._active_command_timeout_ms = _STOP_TIMEOUT_MS
        self._inference_watchdog_timer.setInterval(self._active_inference_timeout_ms)
        self._inference_watchdog_timer.timeout.connect(self._on_inference_result_timeout)

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
        self._process_manager.model_stage_changed.connect(self._on_model_stage_changed)
        self._process_manager.worker_unresponsive.connect(self._on_worker_unresponsive)
        self._process_manager.worker_crashed.connect(self._on_worker_crashed)
        application_controller.source_type_changed.connect(self._on_source_type_changed)

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
        settings_panel.model_selection_changed.connect(self._on_model_selection_changed)
        settings_panel.compute_profile_changed.connect(self._on_compute_profile_changed)

        self._apply_button_states()
        self._sync_model_controls_for_state()
        self._sync_source_controls_for_state()

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

    # -- Model selection and lifecycle -----------------------------------

    def _selected_model_request(self) -> tuple[str, dict[str, object]]:
        model_id = self._settings_panel.selected_model_id
        # Registry dispatch in SettingsPanel resolves every adapter config.
        get_model_descriptor(model_id)  # fail early on corrupted persisted ids
        return model_id, dict(self._settings_panel.model_runtime_config)

    def _selected_model_matches_active(self) -> bool:
        if not self._model_loaded or self._active_model_id is None:
            return False
        try:
            model_id, config = self._selected_model_request()
        except ValueError:
            return False
        return (
            model_id == self._active_model_id
            and model_config_signature(config) == self._active_model_config_signature
        )

    def _on_model_selection_changed(self, _model_id: str) -> None:
        self._request_safe_model_switch_from_settings()

    def _on_compute_profile_changed(self, _resolved_profile: object) -> None:
        self._request_safe_model_switch_from_settings()

    def _request_safe_model_switch_from_settings(self) -> None:
        if not self._model_loaded:
            return
        if self._state_machine.state not in (
            ApplicationState.SOURCE_READY,
            ApplicationState.MODEL_READY,
            ApplicationState.RESULT_READY,
        ):
            return
        if self._selected_model_matches_active():
            return
        try:
            model_id, config = self._selected_model_request()
        except ValueError as exc:
            self._handle_model_load_failure(str(exc))
            return
        self._initiate_model_switch(model_id, config, auto_start=False)

    def _begin_model_load(
        self,
        model_id: str,
        config: dict[str, object],
        *,
        auto_start: bool,
        switching: bool,
    ) -> None:
        if self._shutdown_started:
            return
        self._requested_model_id = model_id
        self._requested_model_config = dict(config)
        self._last_failure = None
        self._auto_start_after_load = bool(auto_start)
        self._pending_model_request = None
        self._model_lifecycle.transition_to(ModelLifecycleState.LOADING, strict=False)

        target_state = (
            ApplicationState.SWITCHING_MODEL
            if switching
            else ApplicationState.LOADING_MODEL
        )
        if self._state_machine.state != target_state:
            self._state_machine.transition_to(target_state)
        self._settings_panel.set_model_stage(
            "LOADING",
            "Loading model weights in the isolated inference worker.",
        )
        self._header.set_status(BadgeKind.LOADING, "LOADING MODEL")
        self._apply_button_states()

        if not self._process_manager.is_running:
            self._process_manager.start_worker()
        self._process_manager.load_model(model_id, config)

    def _initiate_model_switch(
        self,
        model_id: str,
        config: dict[str, object],
        *,
        auto_start: bool,
    ) -> None:
        if self._shutdown_started:
            return
        if self._state_machine.state not in (
            ApplicationState.SOURCE_READY,
            ApplicationState.MODEL_READY,
            ApplicationState.RESULT_READY,
        ):
            return

        self._pending_model_request = (model_id, dict(config))
        self._auto_start_after_load = bool(auto_start)
        self._model_lifecycle.transition_to(ModelLifecycleState.UNLOADING, strict=False)
        self._state_machine.transition_to(ApplicationState.SWITCHING_MODEL)
        self._settings_panel.set_model_stage(
            "UNLOADING",
            "Releasing the current model before loading the new selection.",
        )
        self._header.set_status(BadgeKind.LOADING, "UNLOADING MODEL")
        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.unload_model()
        self._apply_button_states()

    def _load_pending_model_after_unload(self) -> None:
        pending = self._pending_model_request
        if pending is None:
            self._settings_panel.set_model_unloaded()
            self._sync_model_controls_for_state()
            return
        model_id, config = pending
        self._pending_model_request = None
        self._begin_model_load(
            model_id,
            config,
            auto_start=self._auto_start_after_load,
            switching=True,
        )

    def _on_model_stage_changed(self, event: ModelStageEvent) -> None:
        if self._shutdown_started:
            return
        if event.stage is ModelStage.LOADING_WEIGHTS:
            self._model_lifecycle.transition_to(ModelLifecycleState.LOADING, strict=False)
            self._settings_panel.set_model_stage("LOADING", event.message)
            self._header.set_status(BadgeKind.LOADING, "LOADING MODEL")
        elif event.stage is ModelStage.WARMING_UP:
            self._model_lifecycle.transition_to(ModelLifecycleState.WARMING_UP, strict=False)
            self._settings_panel.set_model_stage("WARMING UP", event.message)
            self._header.set_status(BadgeKind.LOADING, "WARMING UP")
        elif event.stage is ModelStage.UNLOADING:
            self._model_lifecycle.transition_to(ModelLifecycleState.UNLOADING, strict=False)
            self._settings_panel.set_model_stage("UNLOADING", event.message)
            self._header.set_status(BadgeKind.LOADING, "UNLOADING MODEL")

    def _on_source_type_changed(self, _source_type: object) -> None:
        """Clear source-specific output while preserving a resident model.

        Source controls are locked during inference and model transitions, so
        this normally runs from SOURCE_READY, MODEL_READY, RESULT_READY, or
        DISCONNECTED. The model stays resident for a fast start after the next
        source is configured.
        """
        if self._state_machine.state in (ApplicationState.RUNNING, ApplicationState.PAUSED):
            self._stop_inference()

        self._session.invalidate()
        self._depth_view.clear_frame()
        self._point_cloud_canvas.clear_points()
        self._viewport_panel.invalidate_camera()
        self._viewport_panel.set_status(ViewportStatus.IDLE)
        self._performance_bar.reset_metrics()
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()

        if self._state_machine.state is not ApplicationState.DISCONNECTED:
            self._state_machine.transition_to(ApplicationState.DISCONNECTED, strict=False)
        self._set_active_model_header()
        self._apply_button_states()

    def _update_header_vram_after_unload(self, metadata: dict[str, object]) -> None:
        """Refresh device-wide VRAM from worker cleanup telemetry."""
        used_gib = _float_metadata(metadata, "after_device_used_gib")
        total_gib = _float_metadata(metadata, "total_gib")
        if total_gib is None:
            total_gib = self._settings_panel.hardware_info.total_memory_gib
        if used_gib is None or total_gib is None:
            return
        self._header.update_gpu_status(
            self._settings_panel.hardware_info.name,
            max(0.0, used_gib),
            max(0.0, total_gib),
        )

    # -- Runtime toolbar handlers ----------------------------------------

    def _on_connect_clicked(self) -> None:
        self._state_machine.transition_to(ApplicationState.CONNECTING)
        connected = self._app_controller.connect_active_source()
        # Only explicit False means the source is not configured. Legacy test
        # doubles and third-party controllers may still return None.
        if connected is False:
            self._state_machine.transition_to(ApplicationState.DISCONNECTED)
            self._apply_button_states()
            return
        # Capture widgets expose the asynchronous source status independently.
        self._state_machine.transition_to(ApplicationState.SOURCE_READY)
        self._apply_button_states()

    def _on_disconnect_clicked(self) -> None:
        if self._state_machine.state in (ApplicationState.RUNNING, ApplicationState.PAUSED):
            self._stop_inference()

        # Shutdown performs a best-effort worker-side unload synchronously.
        # Do not queue a separate asynchronous UNLOAD immediately before it.
        self._process_manager.shutdown()

        self._model_lifecycle.reset()
        self._model_loaded = False
        self._requested_model_id = None
        self._requested_model_config.clear()
        self._pending_model_request = None
        self._last_failure = None
        self._worker_recovery_attempts = 0
        self._recovery_in_progress = False
        self._active_model_id = None
        self._active_model_config_signature = None
        self._active_model_metadata.clear()
        self._reset_inference_timeout()
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
        if self._shutdown_started:
            return
        # A direct user retry starts a fresh recovery budget. Automatic worker
        # recovery never loops indefinitely.
        self._worker_recovery_attempts = 0
        try:
            model_id, config = self._selected_model_request()
        except ValueError as exc:
            self._handle_model_load_failure(str(exc))
            return

        # Stop and static RESULT_READY keep the worker model resident. Reuse it
        # only when both the adapter id and its resolved runtime config match.
        if (
            self._state_machine.state in (
                ApplicationState.SOURCE_READY,
                ApplicationState.MODEL_READY,
                ApplicationState.RESULT_READY,
            )
            and self._selected_model_matches_active()
            and self._process_manager.is_running
        ):
            self._begin_running_session()
            return

        if self._model_loaded and self._state_machine.state in (
            ApplicationState.SOURCE_READY,
            ApplicationState.MODEL_READY,
            ApplicationState.RESULT_READY,
        ):
            self._initiate_model_switch(model_id, config, auto_start=True)
            return

        self._begin_model_load(
            model_id,
            config,
            auto_start=True,
            switching=False,
        )
        # LOAD_MODEL ack advances to MODEL_READY and optionally RUNNING.

    def _begin_running_session(self) -> None:
        """Start a fresh session using the model already resident in the worker."""
        if self._model_lifecycle.state is not ModelLifecycleState.READY:
            return
        if self._state_machine.state not in (
            ApplicationState.SOURCE_READY,
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
        self._inference_watchdog_timer.start()

    def _on_inference_result_timeout(self) -> None:
        """Recover when a submitted frame never returns from the worker."""
        if self._shutdown_started:
            return
        if (
            self._state_machine.state is not ApplicationState.RUNNING
            or not self._inference_in_flight
        ):
            return
        frame_id = self._inflight_frame_id
        failure = timeout_failure(
            f"inference result for frame {frame_id}"
            if frame_id is not None
            else "inference result"
        )
        self._handle_inference_failure(failure)

    # -- Worker signal handlers ------------------------------------------

    def _on_ack_received(self, ack: Ack) -> None:
        if self._shutdown_started:
            return
        if ack.command_type is CommandType.UNLOAD_MODEL:
            if not ack.success:
                failure = ack.failure or classify_failure(
                    RuntimeError(ack.message or "Model unloading failed."),
                    stage=FailureStage.UNLOAD,
                    model_id=ack.model_id,
                )
                self._handle_model_unload_failure(failure)
                return

            self._model_lifecycle.transition_to(
                ModelLifecycleState.UNLOADED,
                strict=False,
            )
            self._update_header_vram_after_unload(dict(ack.metadata or {}))
            self._model_loaded = False
            self._active_model_id = None
            self._active_model_config_signature = None
            self._active_model_metadata.clear()
            self._reset_inference_timeout()
            self._viewport_panel.set_surface_normals_available(False)
            self._clear_cached_geometry()
            self._depth_view.clear_frame()
            self._point_cloud_canvas.clear_points()
            self._viewport_panel.invalidate_camera()
            self._viewport_panel.set_status(ViewportStatus.IDLE)
            self._performance_bar.reset_metrics()
            self._load_pending_model_after_unload()
            return

        if ack.command_type is not CommandType.LOAD_MODEL:
            return

        if not ack.success:
            failure = ack.failure or classify_failure(
                RuntimeError(ack.message or "Model loading failed."),
                stage=FailureStage.LOAD,
                model_id=ack.model_id or self._requested_model_id or "",
            )
            self._handle_model_load_failure(failure)
            return

        self._model_lifecycle.transition_to(ModelLifecycleState.READY, strict=False)
        self._model_loaded = True
        self._active_model_id = ack.model_id or self._requested_model_id
        self._active_model_config_signature = model_config_signature(
            self._requested_model_config
        )
        self._active_model_metadata = dict(ack.metadata or {})
        self._apply_active_inference_timeout()
        if ack.load_time_ms is not None:
            self._active_model_metadata["load_time_ms"] = ack.load_time_ms
        if ack.model_load_time_ms is not None:
            self._active_model_metadata["model_load_time_ms"] = ack.model_load_time_ms
        if ack.warmup_time_ms is not None:
            self._active_model_metadata["warmup_time_ms"] = ack.warmup_time_ms
        self._last_failure = None
        self._worker_recovery_attempts = 0
        self._recovery_in_progress = False
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
            ApplicationState.RECOVERING_WORKER,
        ):
            self._state_machine.transition_to(ApplicationState.MODEL_READY)
            should_start = self._auto_start_after_load
            self._auto_start_after_load = False
            self._sync_model_controls_for_state()
            if should_start:
                self._begin_running_session()


    def _apply_active_inference_timeout(self) -> None:
        """Apply a registry/adapter-provided timeout without model branching."""

        raw = self._active_model_metadata.get(
            "inference_timeout_ms", _DEFAULT_INFERENCE_RESULT_TIMEOUT_MS
        )
        try:
            timeout_ms = int(raw)
        except (TypeError, ValueError):
            timeout_ms = _DEFAULT_INFERENCE_RESULT_TIMEOUT_MS
        self._active_inference_timeout_ms = min(600_000, max(5_000, timeout_ms))
        self._inference_watchdog_timer.setInterval(self._active_inference_timeout_ms)
        raw_command = self._active_model_metadata.get("stop_timeout_ms", _STOP_TIMEOUT_MS)
        try:
            command_timeout_ms = int(raw_command)
        except (TypeError, ValueError):
            command_timeout_ms = _STOP_TIMEOUT_MS
        self._active_command_timeout_ms = min(600_000, max(1_000, command_timeout_ms))
        self._process_manager.set_command_timeout_ms(self._active_command_timeout_ms)

    def _reset_inference_timeout(self) -> None:
        self._active_inference_timeout_ms = _DEFAULT_INFERENCE_RESULT_TIMEOUT_MS
        self._active_command_timeout_ms = _STOP_TIMEOUT_MS
        self._inference_watchdog_timer.setInterval(self._active_inference_timeout_ms)
        self._process_manager.set_command_timeout_ms(self._active_command_timeout_ms)

    def _handle_model_unload_failure(self, failure: FailureInfo) -> None:
        logger.error("Model unload failed: %s", failure.technical_details or failure.message)
        self._mark_model_unavailable()
        self._pending_model_request = None
        self._auto_start_after_load = False
        if failure.requires_worker_restart:
            self._recover_worker(failure, reload_request=None)
            return
        self._present_recoverable_failure(failure, model_ready=False)

    def _handle_model_load_failure(self, failure: FailureInfo | str) -> None:
        if isinstance(failure, str):
            failure = classify_failure(
                RuntimeError(failure),
                stage=FailureStage.LOAD,
                model_id=self._requested_model_id or "",
            )
        logger.error("Model load failed: %s", failure.technical_details or failure.message)
        self._mark_model_unavailable()
        self._pending_model_request = None
        self._auto_start_after_load = False
        if failure.requires_worker_restart:
            self._recover_worker(failure, reload_request=None)
            return
        self._present_recoverable_failure(failure, model_ready=False)

    def _mark_model_unavailable(self) -> None:
        self._model_lifecycle.mark_failed()
        self._model_loaded = False
        self._active_model_id = None
        self._active_model_config_signature = None
        self._active_model_metadata.clear()
        self._reset_inference_timeout()
        self._viewport_panel.set_surface_normals_available(False)

    def _present_recoverable_failure(
        self,
        failure: FailureInfo,
        *,
        model_ready: bool,
    ) -> None:
        """Expose a failure without discarding the connected capture source."""
        self._last_failure = failure
        tooltip = format_failure_tooltip(failure)
        self._settings_panel.set_model_error(
            tooltip,
            label=failure.label,
            unlock_controls=True,
        )
        badge_kind = (
            BadgeKind.EXPERIMENTAL
            if failure.retryable
            else BadgeKind.NOT_RECOMMENDED
        )
        self._header.set_status(badge_kind, failure.label)
        self._performance_bar.set_status(badge_kind, failure.label)
        self._viewport_panel.set_status(
            ViewportStatus.IDLE if failure.preserve_source else ViewportStatus.ERROR
        )

        target = self._recoverable_application_state(model_ready=model_ready)
        if self._state_machine.state != target:
            self._state_machine.transition_to(target, strict=False)
        self._apply_button_states()
        self._sync_model_controls_for_state()

    def _recoverable_application_state(self, *, model_ready: bool) -> ApplicationState:
        if self._app_controller.active_source is None:
            return ApplicationState.DISCONNECTED
        return ApplicationState.MODEL_READY if model_ready else ApplicationState.SOURCE_READY

    def _on_result_ready(self, result: InferenceResult) -> None:
        if self._shutdown_started:
            return
        if not self._session.is_current(result.session_id):
            self._counters.stale_results_ignored += 1
            return

        self._counters.results_received += 1
        rgb = self._recent_frames.pop(result.frame_id, None)
        if result.frame_id == self._inflight_frame_id:
            self._inference_watchdog_timer.stop()
            self._inference_in_flight = False
            self._inflight_frame_id = None

        if self._state_machine.state != ApplicationState.RUNNING:
            self._submitted_at_by_frame.pop(result.frame_id, None)
            self._source_timestamp_by_frame.pop(result.frame_id, None)
            return

        if result.error is not None:
            failure = result.failure or classify_failure(
                RuntimeError(result.error),
                stage=FailureStage.INFERENCE,
                model_id=result.model_id or self._active_model_id or "",
            )
            self._handle_inference_failure(failure)
            return
        if result.depth is None:
            self._handle_inference_failure(
                classify_failure(
                    RuntimeError("The model returned no depth map."),
                    stage=FailureStage.INFERENCE,
                    model_id=result.model_id or self._active_model_id or "",
                )
            )
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

    def _handle_inference_failure(self, failure: FailureInfo | str) -> None:
        if isinstance(failure, str):
            failure = classify_failure(
                RuntimeError(failure),
                stage=FailureStage.INFERENCE,
                model_id=self._active_model_id or "",
            )
        logger.error("Inference failed: %s", failure.technical_details or failure.message)
        self._counters.inference_errors += 1
        self._refresh_uptime()
        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.pause()

        if failure.requires_worker_restart:
            self._mark_model_unavailable()
            self._recover_worker(failure, reload_request=None)
            return

        # The worker pauses after publishing an inference failure.  Keep a
        # healthy resident model available for a deliberate retry while the
        # capture source remains connected.
        self._present_recoverable_failure(
            failure,
            model_ready=self._model_loaded,
        )

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
        """Align any adapter-provided point grid with depth/RGB/mask data."""
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
        elif result.colors is not None and result.colors.shape == result.points.shape:
            sampled_colors = np.asarray(result.colors, dtype=np.uint8)[
                ::extra_stride,
                ::extra_stride,
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
        """Use estimated-intrinsics projection for depth-only adapters."""
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
        sampled_intrinsics = None
        if result.intrinsics is not None:
            sampled_intrinsics = scale_intrinsics(
                result.intrinsics,
                1.0 / float(stride),
            )
        points_grid = depth_to_points(sampled_depth, intrinsics=sampled_intrinsics)

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
        precision = (
            result.precision
            or _string_metadata(self._active_model_metadata, "precision")
            or _registry_default_precision_label(result.model_id)
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
                hardware_tier=_string_metadata(metadata, "hardware_tier"),
                warmup_runs=_int_metadata(metadata, "warmup_runs"),
                tf32_enabled=_bool_metadata(metadata, "allow_tf32"),
                native_sdpa_enabled=_bool_metadata(metadata, "native_sdpa"),
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
        precision = (
            _string_metadata(metadata, "precision")
            or _registry_default_precision_label(self._active_model_id)
        )
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
            hardware_tier=_string_metadata(metadata, "hardware_tier"),
            warmup_runs=_int_metadata(metadata, "warmup_runs"),
            tf32_enabled=_bool_metadata(metadata, "allow_tf32"),
            native_sdpa_enabled=_bool_metadata(metadata, "native_sdpa"),
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
        if not active_id:
            return ""
        try:
            return get_model_descriptor(active_id).display_name
        except ValueError:
            return active_id

    def _set_active_model_header(self) -> None:
        active_id = self._active_model_id
        if not active_id:
            return
        try:
            descriptor = get_model_descriptor(active_id)
        except ValueError:
            return
        try:
            kind = BadgeKind(descriptor.presentation.header_badge_kind)
        except ValueError:
            kind = BadgeKind.COMPATIBLE
        label = descriptor.presentation.header_label or descriptor.display_name.upper()
        self._header.set_status(kind, label)

    def _update_inference_fps(self) -> None:
        self._last_inference_fps = self._inference_rate.mark(time.monotonic())

    def _update_render_fps(self) -> None:
        self._last_render_fps = self._render_rate.mark(time.monotonic())

    def _reset_pipeline_runtime(self) -> None:
        self._pipeline_timer.stop()
        self._uptime_timer.stop()
        self._pending_source_frame = None
        self._pending_render = None
        self._inference_watchdog_timer.stop()
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
        if self._shutdown_started:
            return
        failure = timeout_failure(command_type.value)
        logger.warning("%s", failure.message)

        reload_request: tuple[str, dict[str, object]] | None = None
        if command_type is CommandType.UNLOAD_MODEL:
            reload_request = self._pending_model_request
        elif command_type is CommandType.STOP and self._model_loaded:
            reload_request = self._resident_model_request()
        # A timed-out load is not retried automatically: it may have stalled
        # because of network or hardware pressure.  The user can retry after
        # the fresh worker is ready.
        self._recover_worker(failure, reload_request=reload_request)

    def _on_worker_crashed(self) -> None:
        if self._shutdown_started:
            return
        if self._state_machine.state is ApplicationState.SHUTTING_DOWN:
            return
        failure = crashed_worker_failure(self._process_manager.last_exitcode)
        logger.error("%s", failure.message)
        reload_request: tuple[str, dict[str, object]] | None = None
        if self._app_controller.active_source is not None:
            # If the worker dies while unloading for a model switch, preserve
            # the user's new selection instead of silently restoring the old
            # resident model. A crash during the new load itself is not retried
            # automatically, which prevents repeated download/OOM loops.
            if self._pending_model_request is not None:
                model_id, config = self._pending_model_request
                reload_request = model_id, dict(config)
            elif self._model_loaded:
                reload_request = self._resident_model_request()
        self._recover_worker(failure, reload_request=reload_request)

    def _resident_model_request(self) -> tuple[str, dict[str, object]] | None:
        model_id = self._active_model_id or self._requested_model_id
        if not model_id:
            return None
        return model_id, dict(self._requested_model_config)

    def _recover_worker(
        self,
        failure: FailureInfo,
        *,
        reload_request: tuple[str, dict[str, object]] | None,
    ) -> None:
        """Restart a failed worker once while preserving the capture source."""
        if self._shutdown_started:
            return
        self._last_failure = failure
        if self._recovery_in_progress:
            return
        if self._worker_recovery_attempts >= _MAX_AUTO_WORKER_RECOVERIES:
            self._present_recoverable_failure(failure, model_ready=False)
            return

        self._worker_recovery_attempts += 1
        self._recovery_in_progress = True
        self._session.invalidate()
        self._clear_cached_geometry()
        self._reset_pipeline_runtime()
        self._depth_view.clear_frame()
        self._point_cloud_canvas.clear_points()
        self._viewport_panel.invalidate_camera()
        self._viewport_panel.set_surface_normals_available(False)
        self._viewport_panel.set_status(ViewportStatus.IDLE)

        if self._state_machine.state is not ApplicationState.RECOVERING_WORKER:
            transitioned = self._state_machine.transition_to(
                ApplicationState.RECOVERING_WORKER,
                strict=False,
            )
            if not transitioned and self._app_controller.active_source is None:
                self._recovery_in_progress = False
                self._process_manager.restart_worker()
                self._mark_model_unavailable()
                return

        self._settings_panel.set_model_stage(
            "RESTARTING",
            format_failure_tooltip(failure),
        )
        self._header.set_status(BadgeKind.LOADING, "WORKER RESTARTING")
        self._performance_bar.set_status(BadgeKind.LOADING, "WORKER RESTARTING")
        self._apply_button_states()

        self._process_manager.restart_worker()
        self._model_lifecycle.reset()
        self._model_loaded = False
        self._active_model_id = None
        self._active_model_config_signature = None
        self._active_model_metadata.clear()
        self._pending_model_request = None
        self._auto_start_after_load = False

        if reload_request is not None and self._app_controller.active_source is not None:
            model_id, config = reload_request
            # A recovered model is loaded but never auto-started.  This avoids
            # a crash/OOM loop and gives the user control over the next frame.
            self._recovery_in_progress = False
            self._begin_model_load(
                model_id,
                config,
                auto_start=False,
                switching=False,
            )
            return

        self._recovery_in_progress = False
        self._present_recoverable_failure(failure, model_ready=False)

    def restart_worker(self) -> None:
        """Explicitly restart the worker while preserving the current source."""
        if self._shutdown_started:
            return
        failure = crashed_worker_failure(self._process_manager.last_exitcode)
        self._worker_recovery_attempts = 0
        self._recover_worker(failure, reload_request=None)

    # -- Button state application ----------------------------------------

    def _on_state_changed(self, _old_state, new_state) -> None:
        self._apply_button_states()
        self._sync_model_controls_for_state()
        self._sync_source_controls_for_state()
        kind, text = _STATE_TO_FOOTER_BADGE.get(
            new_state,
            (BadgeKind.NOT_INSTALLED, "IDLE"),
        )
        self._performance_bar.set_status(kind, text)

    def _sync_model_controls_for_state(self) -> None:
        locked = self._state_machine.state in {
            ApplicationState.CONNECTING,
            ApplicationState.LOADING_MODEL,
            ApplicationState.RUNNING,
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.SWITCHING_MODEL,
            ApplicationState.RECOVERING_WORKER,
            ApplicationState.RECONNECTING,
            ApplicationState.SHUTTING_DOWN,
        }
        self._settings_panel.set_model_controls_locked(locked)

    def _sync_source_controls_for_state(self) -> None:
        locked = self._state_machine.state in {
            ApplicationState.CONNECTING,
            ApplicationState.LOADING_MODEL,
            ApplicationState.RUNNING,
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.SWITCHING_MODEL,
            ApplicationState.RECOVERING_WORKER,
            ApplicationState.RECONNECTING,
            ApplicationState.SHUTTING_DOWN,
        }
        self._app_controller.set_source_controls_enabled(not locked)

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
        """Stop all asynchronous runtime work and close the worker exactly once.

        Shutdown can begin while a model is loading, inference is running, a
        switch is in progress, or worker recovery is active.  We therefore do
        not enqueue an additional Stop command; the worker's SHUTDOWN command
        owns final model cleanup and the process manager supplies terminate/kill
        fallbacks. Late acknowledgements/results are ignored by the guards on
        the asynchronous handlers.
        """
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._state_machine.transition_to(
            ApplicationState.SHUTTING_DOWN,
            strict=False,
        )
        self._session.invalidate()
        self._pending_model_request = None
        self._auto_start_after_load = False
        self._pipeline_timer.stop()
        self._inference_watchdog_timer.stop()
        self._settings_refresh_timer.stop()
        self._uptime_timer.stop()
        self._reset_pipeline_runtime()
        self._process_manager.shutdown()
        self._model_lifecycle.reset()
        self._model_loaded = False
        self._requested_model_id = None
        self._requested_model_config.clear()
        self._active_model_id = None
        self._active_model_config_signature = None
        self._active_model_metadata.clear()
        self._last_failure = None
        self._recovery_in_progress = False
        self._settings_panel.set_model_unloaded()
        self._point_cloud_canvas.close()

    @property
    def model_lifecycle_state(self) -> ModelLifecycleState:
        return self._model_lifecycle.state

    @property
    def last_failure(self) -> FailureInfo | None:
        return self._last_failure

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine



def _float_metadata(metadata: dict[str, object], key: str) -> float | None:
    value = metadata.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _bool_metadata(metadata: dict[str, object], key: str) -> bool | None:
    value = metadata.get(key)
    return value if isinstance(value, bool) else None


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


def _registry_default_precision_label(model_id: str | None) -> str | None:
    """Return a registry-owned fallback precision label for partial results.

    Normal worker results carry their resolved precision explicitly. This
    fallback preserves Phase 5 behavior for synthetic tests and early-session
    snapshots where runtime metadata has not yet been populated.
    """
    if not model_id:
        return None
    try:
        descriptor = get_model_descriptor(model_id)
    except ValueError:
        return None

    precision = str(descriptor.default_config.get("precision", "")).strip()
    return precision.upper() or None


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
