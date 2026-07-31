"""Wires the safe runtime system (Phase 3) and real geometry (Phase 4)
into the UI.

Owns the :class:`StateMachine` and :class:`InferenceProcessManager`
(Phase 3), and now also the real geometry pipeline: colorized depth in
the Depth Map panel and a live point cloud in the VisPy 3D viewport,
built from :mod:`core.geometry`'s colormap/projection/filtering/
sampling functions and driven by the sidebar's actual Colormap/Min-Max
Depth/Point Size/Density controls (spec §11, §16).

Only the mock model is registered so far — its depth output feeds the
same colorization/projection pipeline any real model's output will use
starting Phase 5, so this pipeline does not need to change when a real
model adapter is added, only the model_id driving it.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QObject
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
from core.geometry.sampling import adaptive_stride, decimate_grid, enforce_point_budget
from core.inference.process_manager import InferenceProcessManager
from core.models.adapters.mock_adapter import MOCK_DISPLAY_LABEL, MOCK_MODEL_ID
from core.runtime.messages import CommandType, InferenceResult
from core.runtime.session import SessionTracker
from core.runtime.state_machine import ApplicationState, StateMachine

logger = logging.getLogger(__name__)

_STOP_TIMEOUT_MS = 5000
_MOCK_LATENCY_MS = 30.0
_RECENT_FRAMES_CACHE_SIZE = 6
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
    """RGB (not BGR) HxWx3 uint8 array to QPixmap — depth colormap output
    is already RGB, unlike core.capture's BGR camera frames."""
    height, width, _ = rgb.shape
    contiguous = np.ascontiguousarray(rgb)
    qimage = QImage(contiguous.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


class InferenceController(QObject):
    """Drives the safe start/pause/stop lifecycle and the real geometry
    pipeline (colorized depth + live point cloud) against the mock model."""

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
        self._process_manager = InferenceProcessManager(stop_timeout_ms=_STOP_TIMEOUT_MS, parent=self)
        self._session = SessionTracker()
        self._model_loaded = False
        self._run_started_at: float | None = None
        self._fps_window_start: float | None = None
        self._fps_window_count = 0
        self._last_inference_fps: float | None = None

        # Small ring buffer pairing a submitted frame_id with the RGB
        # image that produced it, so a returned depth result (which only
        # carries depth over the process boundary, not the RGB it came
        # from) can still be shown/colored/projected alongside its
        # matching color frame.
        self._recent_frames: "OrderedDict[int, np.ndarray]" = OrderedDict()

        # The real VisPy canvas (spec §10: created once, never rebuilt).
        # Attached immediately, even while idle, so the viewport always
        # shows a live canvas rather than swapping it in only once
        # inference starts.
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
        # Phase 2's source connection is fire-and-forget from here; the
        # source badge reflects the real connect/error state independently.
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
        self._recent_frames.clear()
        self._state_machine.transition_to(ApplicationState.DISCONNECTED)
        self._apply_button_states()

    def _on_start_clicked(self) -> None:
        # After Stop, the model intentionally remains loaded. Restart the
        # session directly instead of trying the invalid MODEL_READY ->
        # LOADING_MODEL transition and reloading the weights.
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
        # The ack (see _on_ack_received) advances MODEL_READY -> RUNNING.

    def _begin_running_session(self) -> None:
        """Start a fresh session using the model already loaded in the worker."""
        if self._state_machine.state != ApplicationState.MODEL_READY:
            return

        self._session.start_new_session()
        self._recent_frames.clear()
        self._viewport_panel.invalidate_camera()
        self._process_manager.resume()
        self._state_machine.transition_to(ApplicationState.RUNNING)
        self._run_started_at = time.monotonic()
        self._fps_window_start = None
        self._fps_window_count = 0
        self._last_inference_fps = None
        self._viewport_panel.set_live(True)
        self._apply_button_states()

        # Static images emit only once when opened, before inference starts.
        self._app_controller.resend_current_frame()

    def _on_pause_resume_clicked(self) -> None:
        if self._state_machine.state == ApplicationState.RUNNING:
            self._process_manager.pause()
            self._state_machine.transition_to(ApplicationState.PAUSED)
            self._runtime_toolbar.set_paused_label(is_paused=True)
            self._viewport_panel.set_live(False)
        elif self._state_machine.state == ApplicationState.PAUSED:
            self._process_manager.resume()
            self._state_machine.transition_to(ApplicationState.RUNNING)
            self._runtime_toolbar.set_paused_label(is_paused=False)
            self._viewport_panel.set_live(True)
        self._apply_button_states()

    def _on_stop_clicked(self) -> None:
        self._stop_inference()
        self._apply_button_states()

    def _stop_inference(self) -> None:
        self._state_machine.transition_to(ApplicationState.STOPPING)
        self._apply_button_states()
        stopped_session_id = self._session.active_session_id
        self._session.invalidate()
        self._recent_frames.clear()
        self._process_manager.stop(session_id=stopped_session_id)
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._viewport_panel.set_live(False)
        self._performance_bar.reset_metrics()
        self._run_started_at = None
        self._fps_window_start = None
        self._fps_window_count = 0
        self._last_inference_fps = None
        # Model AND last geometry stay visible/loaded (spec §13/§15: "the
        # last depth and 3D result may remain visible" and "keep the
        # model loaded for fast restart") — only Disconnect clears them.
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

    # -- Frame forwarding (only while RUNNING) ------------------------------

    def on_source_frame(self, frame: SourceFrame) -> None:
        """Called by main_window for every captured frame. A no-op unless
        inference is actually running — the source preview must stay
        independent of inference either way (spec §5)."""
        if self._state_machine.state != ApplicationState.RUNNING:
            return
        frame_id = self._session.next_frame_id()
        self._recent_frames[frame_id] = frame.image
        while len(self._recent_frames) > _RECENT_FRAMES_CACHE_SIZE:
            self._recent_frames.popitem(last=False)
        self._process_manager.submit_frame(self._session.active_session_id, frame_id, frame.image)

    # -- Worker signal handlers ----------------------------------------------

    def _on_ack_received(self, ack) -> None:  # noqa: ANN001 - Ack, avoid import cycle in signature
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
            return  # stale result from a stopped/replaced session — drop it
        if result.error is not None:
            logger.error("Inference error: %s", result.error)
            return
        if result.depth is None:
            return

        rgb = self._recent_frames.pop(result.frame_id, None)
        self._update_geometry(result.depth, rgb, result.model_id)
        self._update_inference_fps()
        self._update_performance_snapshot(result)

    def _update_geometry(self, depth: np.ndarray, rgb: np.ndarray | None, model_id: str) -> None:
        """The real Phase 4 pipeline: colorize depth for the Depth Map
        panel, back-project to a point cloud, filter/sample it per the
        sidebar's live settings, and push it into the VisPy canvas
        without ever recreating the canvas or its visuals."""
        settings = self._settings_panel

        depth_rgb = apply_colormap(
            depth,
            colormap_name=settings.colormap_name,
            near=settings.min_depth_m,
            far=settings.max_depth_m,
        )
        self._depth_view.set_depth_frame(_ndarray_to_qpixmap_rgb(depth_rgb))

        points_grid = depth_to_points(depth)

        if self._viewport_panel.render_mode == "Depth Colors":
            color_source = depth_rgb
        elif rgb is not None:
            # "RGB Points" and the not-yet-implemented Surface
            # Normals/Mesh modes all fall back to RGB coloring — there
            # is no normals/mesh data from a depth-only mock model, and
            # showing RGB is preferable to showing nothing.
            color_source = rgb[:, :, ::-1]  # BGR (capture) -> RGB
        else:
            color_source = depth_rgb

        mask = valid_mask(points_grid, depth)
        edge_mask = edge_filter_mask(depth, settings.edge_filter_strength)
        combined_mask = mask & edge_mask

        density_ratio = max(self._viewport_panel.density_ratio, 0.02)
        base_stride = max(1, round(1.0 / density_ratio))
        stride = adaptive_stride(base_stride, self._point_cloud_canvas.is_interacting)

        strided_points, strided_colors = decimate_grid(points_grid, color_source, stride)
        strided_mask = combined_mask[::stride, ::stride]
        strided_depth = depth[::stride, ::stride]

        flat_points, flat_colors = remove_invalid(
            np.where(strided_mask[..., None], strided_points, np.nan),
            strided_colors,
            np.where(strided_mask, strided_depth, 0.0),
        )
        flat_points, flat_colors = enforce_point_budget(
            flat_points, flat_colors, settings.point_budget
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

    def _update_performance_snapshot(self, result: InferenceResult) -> None:
        model_label = MOCK_DISPLAY_LABEL if result.model_id == MOCK_MODEL_ID else result.model_id
        uptime_seconds = (
            int(time.monotonic() - self._run_started_at) if self._run_started_at is not None else None
        )
        self._performance_bar.update_snapshot(
            PerformanceSnapshot(
                status="RUNNING",
                inference_fps=self._last_inference_fps,
                latency_ms=result.latency_ms,
                model_name=model_label,
                resolution=(result.depth.shape[1], result.depth.shape[0])
                if result.depth is not None
                else None,
                uptime_seconds=uptime_seconds,
                # camera_fps/render_fps/points/vram: real capture-FPS
                # metering and GPU monitoring land in Phase 5 — left as
                # None (renders as "—") rather than fabricated.
            )
        )

    def _update_inference_fps(self) -> None:
        """A simple rolling-window FPS count — real, measured values only."""
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

    def _on_worker_unresponsive(self, command_type: CommandType) -> None:
        logger.warning("Inference worker did not acknowledge %s in time", command_type)
        # A real "Continue Waiting / Restart Worker" dialog is a UI
        # affordance for a later polish pass; the safe recovery mechanism
        # itself (restart_worker) is already fully implemented and tested.

    def _on_worker_crashed(self) -> None:
        logger.error("Inference worker process crashed")
        self._model_loaded = False
        self._state_machine.transition_to(ApplicationState.ERROR, strict=False)
        self._apply_button_states()

    def restart_worker(self) -> None:
        """Public entry point for a future "Restart Worker" UI action."""
        self._process_manager.restart_worker()
        self._model_loaded = False

    # -- Button state application --------------------------------------------

    def _on_state_changed(self, _old_state, new_state) -> None:
        self._apply_button_states()
        kind, text = _STATE_TO_FOOTER_BADGE.get(new_state, (BadgeKind.NOT_INSTALLED, "IDLE"))
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
        self._process_manager.shutdown()
        self._point_cloud_canvas.close()

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine