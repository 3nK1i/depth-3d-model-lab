"""Safe inference lifecycle plus real depth/point-cloud rendering.

Phase 5 Step 1.2 adds a live-stream performance guard around the existing
Phase 3/4 runtime:

* the UI keeps only the latest source frame while one request is in flight;
* continuous sources are submitted at a bounded 12 FPS;
* obsolete inference results are replaced before geometry rendering;
* live point clouds are pre-decimated before dense 3D projection;
* source preview, inference and 3D rendering remain independent.

The development mock remains the only registered adapter until the real
MoGe-2 integration in the next step.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QObject, QTimer
from PyQt6.QtGui import QImage, QPixmap

from app.application_controller import ApplicationController
from app.widgets.badge import BadgeKind
from app.widgets.depth_view import DepthMapView
from app.widgets.header_widget import HeaderWidget
from app.widgets.performance_bar import PerformanceBar
from app.widgets.performance_snapshot import PerformanceSnapshot
from app.widgets.runtime_toolbar import RuntimeToolbar
from app.widgets.settings_panel import SettingsPanel
from app.widgets.viewport.viewport_panel import ViewportPanel
from app.widgets.viewport.vispy_canvas import PointCloudCanvas
from core.capture.frame import SourceFrame
from core.geometry.colors import apply_colormap
from core.geometry.filtering import edge_filter_mask, remove_invalid, valid_mask
from core.geometry.projection import depth_to_points
from core.geometry.sampling import (
    adaptive_stride,
    enforce_point_budget,
    stride_for_point_budget,
)
from core.inference.process_manager import InferenceProcessManager
from core.models.adapters.mock_adapter import MOCK_DISPLAY_LABEL, MOCK_MODEL_ID
from core.runtime.messages import CommandType, InferenceResult
from core.runtime.session import SessionTracker
from core.runtime.state_machine import ApplicationState, StateMachine

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_MS = 5000
_MOCK_LATENCY_MS = 30.0
_RECENT_FRAMES_CACHE_SIZE = 3
_LIVE_PIPELINE_FPS = 12.0
_LIVE_PIPELINE_INTERVAL_S = 1.0 / _LIVE_PIPELINE_FPS
_PIPELINE_TICK_MS = 10
_BACKGROUND_COLOR_MAP = {
    "Dark": "#0d0f12",
    "Black": "#000000",
    "White": "#f2f2f2",
    "Transparent": "#0d0f12",  # a true alpha background is a later polish item
}

_STATE_TO_FOOTER_BADGE: dict[ApplicationState, tuple[BadgeKind, str]] = {
    ApplicationState.DISCONNECTED: (BadgeKind.NOT_INSTALLED, "IDLE"),
    ApplicationState.CONNECTING: (BadgeKind.LOADING, "CONNECTING"),
    ApplicationState.SOURCE_READY: (BadgeKind.COMPATIBLE, "SOURCE READY"),
    ApplicationState.LOADING_MODEL: (BadgeKind.LOADING, "LOADING MODEL"),
    ApplicationState.MODEL_READY: (BadgeKind.COMPATIBLE, "MODEL READY"),
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
    """Drive model lifecycle, stream backpressure and geometry rendering."""

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
        self._run_started_at: float | None = None

        # Measured model-completion rate.
        self._fps_window_start: float | None = None
        self._fps_window_count = 0
        self._last_inference_fps: float | None = None

        # Measured successful geometry/VisPy update rate.
        self._render_fps_window_start: float | None = None
        self._render_fps_window_count = 0
        self._last_render_fps: float | None = None

        # Latest-frame-only stream policy. At most one request is queued or
        # executing in the worker; every newer captured frame replaces this
        # pending slot until that request completes.
        self._pending_source_frame: SourceFrame | None = None
        self._inference_in_flight = False
        self._inflight_frame_id: int | None = None
        self._last_submit_at: float | None = None

        # Results are similarly latest-only before the CPU-heavy depth-to-3D
        # pipeline. A newer result replaces an older result that has not yet
        # reached the bounded render cadence.
        self._pending_render: tuple[InferenceResult, np.ndarray | None] | None = None
        self._last_render_at: float | None = None

        # Pair submitted frame ids with their RGB images. With one request in
        # flight this stays tiny, but the bounded map also protects unusual
        # restart races and stale worker messages.
        self._recent_frames: "OrderedDict[int, np.ndarray]" = OrderedDict()

        self._pipeline_timer = QTimer(self)
        self._pipeline_timer.setInterval(_PIPELINE_TICK_MS)
        self._pipeline_timer.timeout.connect(self._on_pipeline_tick)

        # The VisPy canvas is created exactly once and only its buffers change.
        self._point_cloud_canvas = PointCloudCanvas(self)
        self._viewport_panel.attach_canvas(self._point_cloud_canvas.native)

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

        self._apply_button_states()

    # -- Runtime toolbar handlers -------------------------------------------

    def _on_connect_clicked(self) -> None:
        self._state_machine.transition_to(ApplicationState.CONNECTING)
        self._app_controller.connect_active_source()
        # Source connection status remains independently visible in its badge.
        self._state_machine.transition_to(ApplicationState.SOURCE_READY)
        self._apply_button_states()

    def _on_disconnect_clicked(self) -> None:
        if self._state_machine.state in (ApplicationState.RUNNING, ApplicationState.PAUSED):
            self._stop_inference()
        if self._model_loaded:
            self._process_manager.unload_model()
            self._model_loaded = False
        self._process_manager.shutdown()
        self._app_controller.disconnect_active_source()
        self._depth_view.clear_frame()
        self._point_cloud_canvas.clear_points()
        self._viewport_panel.invalidate_camera()
        self._viewport_panel.set_live(False)
        self._performance_bar.reset_metrics()
        self._reset_pipeline_runtime()
        self._state_machine.transition_to(ApplicationState.DISCONNECTED)
        self._apply_button_states()

    def _on_start_clicked(self) -> None:
        # After Stop the model stays loaded for a fast restart.
        if (
            self._state_machine.state == ApplicationState.MODEL_READY
            and self._model_loaded
            and self._process_manager.is_running
        ):
            self._begin_running_session()
            return

        self._state_machine.transition_to(ApplicationState.LOADING_MODEL)
        self._apply_button_states()

        if not self._process_manager.is_running:
            self._process_manager.start_worker()
        self._process_manager.load_model(MOCK_MODEL_ID, {"latency_ms": _MOCK_LATENCY_MS})
        # The LOAD_MODEL ack advances MODEL_READY -> RUNNING.

    def _begin_running_session(self) -> None:
        """Start a fresh session using the model already loaded in the worker."""
        if self._state_machine.state != ApplicationState.MODEL_READY:
            return

        self._session.start_new_session()
        self._reset_pipeline_runtime()
        self._viewport_panel.invalidate_camera()
        self._process_manager.resume()
        self._state_machine.transition_to(ApplicationState.RUNNING)
        self._run_started_at = time.monotonic()
        self._viewport_panel.set_live(True)
        self._pipeline_timer.start()
        self._apply_button_states()

        # Static images emitted before inference started need one explicit resend.
        self._app_controller.resend_current_frame()

    def _on_pause_resume_clicked(self) -> None:
        if self._state_machine.state == ApplicationState.RUNNING:
            self._process_manager.pause()
            self._state_machine.transition_to(ApplicationState.PAUSED)
            self._runtime_toolbar.set_paused_label(is_paused=True)
            self._viewport_panel.set_live(False)
            self._pipeline_timer.stop()
            self._pending_source_frame = None
            self._pending_render = None
        elif self._state_machine.state == ApplicationState.PAUSED:
            self._process_manager.resume()
            self._state_machine.transition_to(ApplicationState.RUNNING)
            self._runtime_toolbar.set_paused_label(is_paused=False)
            self._viewport_panel.set_live(True)
            self._pipeline_timer.start()
            self._app_controller.resend_current_frame()
        self._apply_button_states()

    def _on_stop_clicked(self) -> None:
        self._stop_inference()
        self._apply_button_states()

    def _stop_inference(self) -> None:
        self._state_machine.transition_to(ApplicationState.STOPPING)
        self._apply_button_states()
        stopped_session_id = self._session.active_session_id
        self._session.invalidate()
        self._reset_pipeline_runtime()
        self._process_manager.stop(session_id=stopped_session_id)
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._viewport_panel.set_live(False)
        self._performance_bar.reset_metrics()
        # Model and last geometry remain loaded/visible for fast restart.
        self._state_machine.transition_to(ApplicationState.MODEL_READY)

    def _on_reset_view_clicked(self) -> None:
        self._point_cloud_canvas.reset_view()
        self._viewport_panel.mark_camera_initialized()

    def _on_fit_view_clicked(self) -> None:
        """Frame the active cloud without resetting orbit orientation."""
        if not self._point_cloud_canvas.has_points:
            return
        self._point_cloud_canvas.fit_scene()
        self._viewport_panel.mark_camera_initialized()

    # -- Latest-frame inference scheduling ----------------------------------

    def on_source_frame(self, frame: SourceFrame) -> None:
        """Receive capture frames without ever building an inference backlog."""
        if self._state_machine.state != ApplicationState.RUNNING:
            return
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

        accepted = self._process_manager.submit_frame(
            session_id,
            frame_id,
            source_frame.image,
        )
        # Treat only an explicit False as rejection so legacy test doubles that
        # return None remain compatible.
        if accepted is False:
            self._recent_frames.pop(frame_id, None)
            self._pending_source_frame = source_frame
            return

        self._inference_in_flight = True
        self._inflight_frame_id = frame_id
        self._last_submit_at = now

    # -- Worker signal handlers ---------------------------------------------

    def _on_ack_received(self, ack) -> None:  # noqa: ANN001 - avoids an import cycle in signature
        if ack.command_type == CommandType.LOAD_MODEL and ack.success:
            self._model_loaded = True
            if self._state_machine.state == ApplicationState.LOADING_MODEL:
                self._state_machine.transition_to(ApplicationState.MODEL_READY)
                self._begin_running_session()
        elif ack.command_type == CommandType.LOAD_MODEL and not ack.success:
            logger.error("Model load failed: %s", ack.message)
            self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
            self._apply_button_states()

    def _on_result_ready(self, result: InferenceResult) -> None:
        if not self._session.is_current(result.session_id):
            return  # stale result from a stopped/replaced session

        rgb = self._recent_frames.pop(result.frame_id, None)
        if result.frame_id == self._inflight_frame_id:
            self._inference_in_flight = False
            self._inflight_frame_id = None

        if self._state_machine.state != ApplicationState.RUNNING:
            return

        if result.error is not None:
            logger.error("Inference error: %s", result.error)
            self._try_submit_latest_frame()
            return
        if result.depth is None:
            self._try_submit_latest_frame()
            return

        self._update_inference_fps()
        self._pending_render = (result, rgb)

        # The worker may begin the newest pending frame while the UI waits for
        # the next bounded 3D render slot. This overlaps compute and rendering
        # without allowing more than one model request in flight.
        self._try_submit_latest_frame()
        self._try_render_latest_result()

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
        point_count = self._update_geometry(result.depth, rgb, result.model_id)
        self._last_render_at = time.monotonic()
        self._update_render_fps()
        self._update_performance_snapshot(result, point_count=point_count)

    # -- Geometry ------------------------------------------------------------

    def _update_geometry(
        self,
        depth: np.ndarray,
        rgb: np.ndarray | None,
        model_id: str,
    ) -> int:
        """Colorize depth, build a filtered cloud and update VisPy in place.

        For continuous sources the regular-grid stride is chosen *before*
        projection. On the user's 1280×1440 RTSP stream this avoids creating
        1.84 million 3D points merely to discard most of them afterward.
        Static images retain the selected full-quality point budget.
        """
        settings = self._settings_panel

        depth_rgb = apply_colormap(
            depth,
            colormap_name=settings.colormap_name,
            near=settings.min_depth_m,
            far=settings.max_depth_m,
        )
        self._depth_view.set_depth_frame(_ndarray_to_qpixmap_rgb(depth_rgb))

        if self._viewport_panel.render_mode == "Depth Colors":
            color_source = depth_rgb
        elif rgb is not None and rgb.shape[:2] == depth.shape[:2]:
            # Capture frames are BGR; VisPy expects RGB.
            color_source = rgb[:, :, ::-1]
        else:
            # A future adapter may emit depth at a different resolution. Until
            # explicit intrinsics-aware color resampling lands, depth colors are
            # the safe aligned fallback rather than risking a shape mismatch.
            color_source = depth_rgb

        density_ratio = max(self._viewport_panel.density_ratio, 0.02)
        density_stride = max(1, round(1.0 / density_ratio))

        if self._app_controller.is_stream_source:
            point_budget = settings.live_point_budget
            budget_stride = stride_for_point_budget(
                depth.shape[0],
                depth.shape[1],
                point_budget,
            )
        else:
            point_budget = settings.point_budget
            budget_stride = 1

        stride = adaptive_stride(
            max(density_stride, budget_stride),
            self._point_cloud_canvas.is_interacting,
        )
        sampled_depth = depth[::stride, ::stride]
        sampled_colors = color_source[::stride, ::stride]

        points_grid = depth_to_points(sampled_depth)
        mask = valid_mask(points_grid, sampled_depth)
        mask &= edge_filter_mask(sampled_depth, settings.edge_filter_strength)

        flat_points, flat_colors = remove_invalid(
            np.where(mask[..., None], points_grid, np.nan),
            sampled_colors,
            np.where(mask, sampled_depth, 0.0),
        )
        flat_points, flat_colors = enforce_point_budget(
            flat_points,
            flat_colors,
            point_budget,
        )

        self._point_cloud_canvas.set_point_size(self._viewport_panel.point_size)
        self._point_cloud_canvas.set_grid_visible(settings.grid_visible)
        self._viewport_panel.set_axis_visible(settings.axes_visible)
        self._point_cloud_canvas.update_points(flat_points, flat_colors)

        if not self._viewport_panel.camera_initialized:
            self._point_cloud_canvas.fit_scene()
            self._viewport_panel.mark_camera_initialized()

        if model_id == MOCK_MODEL_ID:
            self._header.set_status(BadgeKind.EXPERIMENTAL, "MOCK")

        return int(flat_points.shape[0])

    # -- Performance metrics -------------------------------------------------

    def _update_performance_snapshot(
        self,
        result: InferenceResult,
        *,
        point_count: int,
    ) -> None:
        model_label = MOCK_DISPLAY_LABEL if result.model_id == MOCK_MODEL_ID else result.model_id
        uptime_seconds = (
            int(time.monotonic() - self._run_started_at)
            if self._run_started_at is not None
            else None
        )
        self._performance_bar.update_snapshot(
            PerformanceSnapshot(
                status="RUNNING",
                camera_fps=self._app_controller.capture_fps,
                inference_fps=self._last_inference_fps,
                latency_ms=result.latency_ms,
                render_fps=self._last_render_fps,
                point_count=point_count,
                model_name=model_label,
                precision="FP32" if result.model_id == MOCK_MODEL_ID else None,
                resolution=(result.depth.shape[1], result.depth.shape[0]),
                uptime_seconds=uptime_seconds,
                # Real worker-side GPU memory instrumentation lands with the
                # real CUDA adapter; no preview/fabricated value is shown here.
            )
        )

    def _update_inference_fps(self) -> None:
        """Update a rolling model-completion FPS measurement."""
        now = time.monotonic()
        if self._fps_window_start is None:
            self._fps_window_start = now
            self._fps_window_count = 1
            return
        self._fps_window_count += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._last_inference_fps = self._fps_window_count / elapsed
            self._fps_window_start = now
            self._fps_window_count = 0

    def _update_render_fps(self) -> None:
        """Update the measured successful point-cloud buffer-update rate."""
        now = time.monotonic()
        if self._render_fps_window_start is None:
            self._render_fps_window_start = now
            self._render_fps_window_count = 1
            return
        self._render_fps_window_count += 1
        elapsed = now - self._render_fps_window_start
        if elapsed >= 1.0:
            self._last_render_fps = self._render_fps_window_count / elapsed
            self._render_fps_window_start = now
            self._render_fps_window_count = 0

    def _reset_pipeline_runtime(self) -> None:
        self._pipeline_timer.stop()
        self._pending_source_frame = None
        self._pending_render = None
        self._inference_in_flight = False
        self._inflight_frame_id = None
        self._last_submit_at = None
        self._last_render_at = None
        self._recent_frames.clear()

        self._run_started_at = None
        self._fps_window_start = None
        self._fps_window_count = 0
        self._last_inference_fps = None
        self._render_fps_window_start = None
        self._render_fps_window_count = 0
        self._last_render_fps = None

    # -- Worker recovery -----------------------------------------------------

    def _on_worker_unresponsive(self, command_type: CommandType) -> None:
        logger.warning("Inference worker did not acknowledge %s in time", command_type)

    def _on_worker_crashed(self) -> None:
        logger.error("Inference worker process crashed")
        self._model_loaded = False
        self._reset_pipeline_runtime()
        self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
        self._apply_button_states()

    def restart_worker(self) -> None:
        """Public entry point for a future Restart Worker UI action."""
        self._process_manager.restart_worker()
        self._model_loaded = False
        self._reset_pipeline_runtime()

    # -- Button state application -------------------------------------------

    def _on_state_changed(self, _old_state, new_state) -> None:
        self._apply_button_states()
        kind, text = _STATE_TO_FOOTER_BADGE.get(
            new_state,
            (BadgeKind.NOT_INSTALLED, "IDLE"),
        )
        self._performance_bar.set_status(kind, text)
        if new_state == ApplicationState.RUNNING:
            self._header.set_status(BadgeKind.READY, "READY")

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
        self._process_manager.shutdown()
        self._point_cloud_canvas.close()

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine
