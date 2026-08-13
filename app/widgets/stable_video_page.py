from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.icons import render_icon
from app.stable_video_controller import StableVideoRunState
from app.theme import PALETTE
from app.widgets.depth_view import DepthMapView
from app.widgets.original_view import OriginalSourceView
from app.widgets.viewport.viewport_panel import ViewportPanel, ViewportStatus
from app.widgets.viewport.vispy_canvas import PointCloudCanvas
from core.capture.qt_convert import ndarray_to_qpixmap
from core.geometry.colors import apply_colormap
from core.geometry.filtering import valid_mask
from core.geometry.projection import depth_to_points, scale_intrinsics
from core.geometry.sampling import enforce_point_budget
from core.runtime.messages import InferenceResult
from core.snapshot.export import align_point_colors, align_point_mask


_STABILIZATION_VALUES = {
    "Off": 0.0,
    "Low": 0.25,
    "Medium": 0.50,
    "High": 0.75,
}


class StableVideoPage(QWidget):
    start_requested = pyqtSignal()
    pause_resume_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    stabilization_changed = pyqtSignal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._compact_controls = False
        self._paused = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("panel")
        row = QHBoxLayout(controls)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(8)
        self._model = QLabel("Model: —")
        self._temporal = QLabel("Temporal path: —")
        self._temporal.setObjectName("muted")
        self._stabilization = QComboBox()
        self._stabilization.addItems(list(_STABILIZATION_VALUES))
        self._stabilization.setCurrentText("Medium")
        self._fps = QSpinBox()
        self._fps.setRange(1, 60)
        self._fps.setValue(20)
        self._fps.setSuffix(" FPS export")
        self._export = QCheckBox("Export RGB + Depth")
        self._export.setChecked(True)
        self._numeric = QCheckBox("Numerical depth")
        self._start = QPushButton("Start Stable Video")
        self._pause = QPushButton("Pause")
        self._stop = QPushButton("Stop & Finalize")
        self._open = QPushButton("Open Export")
        self._pause.setEnabled(False)
        self._stop.setEnabled(False)
        self._open.setEnabled(False)
        self._stabilization_label = QLabel("Stabilization")
        for widget in (
            self._model,
            self._temporal,
            self._stabilization,
            self._fps,
            self._export,
            self._numeric,
            self._start,
            self._pause,
            self._stop,
            self._open,
        ):
            widget.setMinimumWidth(0)
        self._temporal.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        for widget in (self._start, self._pause, self._stop, self._open):
            # Runtime actions must never collapse to zero width. Compact mode
            # gives them fixed icon-sized widths; normal mode keeps their
            # preferred text width.
            widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._start.setIcon(render_icon("play", PALETTE.toolbar_text_muted, 14))
        self._pause.setIcon(render_icon("pause", PALETTE.toolbar_text_muted, 14))
        self._stop.setIcon(render_icon("stop", PALETTE.toolbar_text_muted, 14))
        self._open.setIcon(render_icon("snapshot", PALETTE.toolbar_text_muted, 14))
        self._start.setToolTip("Start Stable Video")
        self._pause.setToolTip("Pause or resume Stable Video")
        self._stop.setToolTip("Stop and finalize Stable Video export")
        self._open.setToolTip("Open Stable Video export")
        row.addWidget(self._model)
        row.addWidget(self._temporal)
        row.addStretch(1)
        row.addWidget(self._stabilization_label)
        row.addWidget(self._stabilization)
        row.addWidget(self._fps)
        row.addWidget(self._export)
        row.addWidget(self._numeric)
        row.addWidget(self._start)
        row.addWidget(self._pause)
        row.addWidget(self._stop)
        row.addWidget(self._open)
        layout.addWidget(controls)

        self._source = OriginalSourceView()
        self._depth = DepthMapView()
        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(self._source)
        top.addWidget(self._depth)
        top.setSizes([1, 1])

        self._viewport = ViewportPanel()
        self._canvas = PointCloudCanvas(self)
        self._viewport.attach_canvas(self._canvas.native)
        self._viewport.fit_requested.connect(self._fit_view)
        self._viewport.reset_view_clicked.connect(self._reset_view)
        self._viewport.tool_changed.connect(self._canvas.set_interaction_mode)
        self._viewport.point_size_changed.connect(self._canvas.set_point_size)

        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(top)
        body.addWidget(self._viewport)
        body.setSizes([270, 520])
        layout.addWidget(body, stretch=1)

        stats = QFrame()
        stats.setObjectName("panel")
        grid = QGridLayout(stats)
        grid.setContentsMargins(12, 8, 12, 8)
        self._stats: dict[str, QLabel] = {}
        labels = [
            ("source_fps", "SOURCE FPS"),
            ("processing_fps", "PROCESSING FPS"),
            ("average_latency_ms", "AVG LATENCY"),
            ("peak_latency_ms", "PEAK LATENCY"),
            ("dropped_frames", "DROPPED"),
            ("reset_count", "RESETS"),
            ("model_vram_gib", "MODEL VRAM"),
            ("elapsed_seconds", "ELAPSED"),
        ]
        for column, (key, title) in enumerate(labels):
            title_label = QLabel(title)
            title_label.setObjectName("muted")
            value = QLabel("—")
            self._stats[key] = value
            grid.addWidget(title_label, 0, column)
            grid.addWidget(value, 1, column)
        layout.addWidget(stats)

        self._status = QLabel(
            "Connect a Video, Camera, or RTSP source in Live 3D, then start Stable Video."
        )
        self._status.setObjectName("muted")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(90)
        layout.addWidget(self._status)
        layout.addWidget(self._log)

        self._start.clicked.connect(self.start_requested.emit)
        self._pause.clicked.connect(self.pause_resume_requested.emit)
        self._stop.clicked.connect(self.stop_requested.emit)
        self._open.clicked.connect(self._open_export)
        self._stabilization.currentTextChanged.connect(
            lambda text: self.stabilization_changed.emit(_STABILIZATION_VALUES[text])
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._set_compact_controls(self.width() < 1180)

    def _set_compact_controls(self, compact: bool) -> None:
        compact = bool(compact)
        if compact == self._compact_controls:
            return
        self._compact_controls = compact
        self._temporal.setVisible(not compact)
        self._stabilization_label.setText("Stab." if compact else "Stabilization")
        self._fps.setSuffix(" FPS" if compact else " FPS export")
        self._export.setText("RGB + Depth" if compact else "Export RGB + Depth")
        self._numeric.setText("Numeric" if compact else "Numerical depth")
        self._sync_runtime_action_presentation()

    def _sync_runtime_action_presentation(self) -> None:
        """Keep Stable Video runtime actions visible at every supported width."""

        compact = self._compact_controls
        buttons = (self._start, self._pause, self._stop, self._open)
        if compact:
            for button in buttons:
                button.setFixedWidth(34)
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            self._start.setText("")
            self._pause.setText("")
            self._stop.setText("")
            self._open.setText("")
        else:
            for button in buttons:
                button.setMinimumWidth(0)
                button.setMaximumWidth(16_777_215)
                button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            self._start.setText("Start Stable Video")
            self._pause.setText("Resume" if self._paused else "Pause")
            self._stop.setText("Stop & Finalize")
            self._open.setText("Open Export")

        # Pause becomes Resume semantically while retaining a compact icon-only
        # presentation in constrained windows.
        self._pause.setIcon(
            render_icon(
                "play" if self._paused else "pause",
                PALETTE.toolbar_text_muted,
                14,
            )
        )
        self._pause.setToolTip(
            "Resume Stable Video" if self._paused else "Pause Stable Video"
        )

    @property
    def compact_controls(self) -> bool:
        return self._compact_controls

    @property
    def smoothing_strength(self) -> float:
        return _STABILIZATION_VALUES[self._stabilization.currentText()]

    @property
    def export_enabled(self) -> bool:
        return self._export.isChecked()

    @property
    def export_numeric_depth(self) -> bool:
        return self._numeric.isChecked()

    @property
    def export_fps(self) -> float:
        return float(self._fps.value())

    def set_model(self, label: str, *, native_temporal: bool) -> None:
        self._model.setText(f"Model: {label}")
        self._temporal.setText(
            "Temporal path: Native model cache + optional smoothing"
            if native_temporal
            else "Temporal path: Generic geometry smoothing"
        )
        self._model.setToolTip(self._temporal.text())

    def set_source_eligible(self, eligible: bool, reason: str = "") -> None:
        if self._start.isEnabled() == eligible and not reason:
            return
        if not self._is_running_state():
            self._start.setEnabled(eligible)
        if reason:
            self._status.setText(reason)

    def apply_state(self, state: StableVideoRunState) -> None:
        running = state in {StableVideoRunState.STARTING, StableVideoRunState.RUNNING}
        paused = state is StableVideoRunState.PAUSED
        self._paused = paused
        stopping = state is StableVideoRunState.STOPPING
        self._start.setEnabled(state in {StableVideoRunState.IDLE, StableVideoRunState.COMPLETED, StableVideoRunState.FAILED})
        self._pause.setEnabled(running or paused)
        self._stop.setEnabled(running or paused or stopping)
        self._sync_runtime_action_presentation()
        self._stabilization.setEnabled(not running and not paused)
        self._fps.setEnabled(not running and not paused)
        self._export.setEnabled(not running and not paused)
        self._numeric.setEnabled(not running and not paused)
        messages = {
            StableVideoRunState.STARTING: "Loading the selected model and starting the temporal sequence…",
            StableVideoRunState.RUNNING: "Stable Video is processing the newest source frames.",
            StableVideoRunState.PAUSED: "Stable Video is paused; temporal state is preserved.",
            StableVideoRunState.STOPPING: "Stopping inference and finalizing export…",
            StableVideoRunState.COMPLETED: "Stable Video completed and export finalized.",
            StableVideoRunState.FAILED: "Stable Video failed. Review the log.",
            StableVideoRunState.IDLE: "Stable Video is idle.",
        }
        self._status.setText(messages[state])

    def show_result(self, result: InferenceResult, rgb: np.ndarray, _snapshot: object) -> None:
        self._source.set_frame(ndarray_to_qpixmap(rgb))
        depth = np.asarray(result.depth, dtype=np.float32)
        finite = np.isfinite(depth) & (depth > 0.0)
        if finite.any():
            lo, hi = np.percentile(depth[finite], [2.0, 98.0])
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        depth_rgb = apply_colormap(depth, near=float(lo), far=float(hi), valid_mask=finite)
        self._depth.set_depth_frame(ndarray_to_qpixmap(depth_rgb[:, :, ::-1]))

        point_stride = max(1, int(result.point_stride))
        if result.points is not None:
            points = np.asarray(result.points, dtype=np.float32)
        else:
            sampled_depth = depth[::point_stride, ::point_stride]
            intrinsics = result.intrinsics
            if intrinsics is not None and point_stride > 1:
                intrinsics = scale_intrinsics(intrinsics, 1.0 / point_stride)
            points = depth_to_points(sampled_depth, intrinsics)
        colors = align_point_colors(
            rgb_bgr=rgb,
            points=points,
            colors=result.colors,
            point_stride=point_stride,
            depth_shape=depth.shape,
        )
        mask = valid_mask(points, depth[::point_stride, ::point_stride])
        mask &= align_point_mask(
            valid_mask=result.valid_mask,
            points=points,
            point_stride=point_stride,
        ).reshape(mask.shape)
        flat_points = points.reshape(-1, 3)[mask.reshape(-1)]
        flat_colors = colors.reshape(-1, 3)[mask.reshape(-1)]
        flat_points, flat_colors = enforce_point_budget(flat_points, flat_colors, 250_000)
        if flat_points.size:
            self._canvas.update_points(flat_points, flat_colors)
            if not self._viewport.camera_initialized:
                self._canvas.fit_scene()
                self._viewport.mark_camera_initialized()
            self._viewport.set_status(ViewportStatus.LIVE)
            self._viewport.set_idle_message_visible(False)

    def update_statistics(self, data: dict[str, object]) -> None:
        formatters = {
            "source_fps": lambda v: "—" if v is None else f"{float(v):.1f}",
            "processing_fps": lambda v: "—" if v is None else f"{float(v):.1f}",
            "average_latency_ms": lambda v: "—" if v is None else f"{float(v):.1f} ms",
            "peak_latency_ms": lambda v: "—" if v is None else f"{float(v):.1f} ms",
            "dropped_frames": lambda v: f"{int(v or 0):,}",
            "reset_count": lambda v: str(int(v or 0)),
            "model_vram_gib": lambda v: "—" if v is None else f"{float(v):.1f} GB",
            "elapsed_seconds": lambda v: self._format_elapsed(float(v or 0.0)),
        }
        for key, label in self._stats.items():
            label.setText(formatters[key](data.get(key)))

    def set_export_dir(self, path: str) -> None:
        self._output_dir = Path(path)
        self._open.setEnabled(True)
        self.append_log(f"Export finalized: {path}")

    def append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def _fit_view(self) -> None:
        if self._canvas.has_points:
            self._canvas.fit_scene()
            self._viewport.mark_camera_initialized()

    def _reset_view(self) -> None:
        self._canvas.reset_view()
        self._viewport.mark_camera_initialized()

    def _open_export(self) -> None:
        if self._output_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._output_dir)))

    def _is_running_state(self) -> bool:
        return self._pause.isEnabled() or self._stop.isEnabled()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
