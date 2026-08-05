from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.snapshot_controller import SnapshotRunState
from app.widgets.depth_view import DepthMapView
from app.widgets.original_view import OriginalSourceView
from app.widgets.viewport.viewport_panel import ViewportPanel, ViewportStatus
from app.widgets.viewport.vispy_canvas import PointCloudCanvas
from core.capture.qt_convert import ndarray_to_qpixmap
from core.geometry.colors import apply_colormap
from core.geometry.projection import depth_to_points
from core.snapshot.export import align_point_colors, align_point_mask


class SnapshotPage(QWidget):
    capture_requested = pyqtSignal()
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        controls = QFrame()
        controls.setObjectName("panel")
        row = QHBoxLayout(controls)
        row.setContentsMargins(12, 8, 12, 8)
        self._model_label = QLabel("Model: —")
        self._frame_label = QLabel("No frame captured")
        self._frame_label.setObjectName("muted")
        self._capture = QPushButton("Capture Latest Frame")
        self._run = QPushButton("Run Snapshot")
        self._cancel = QPushButton("Cancel")
        self._open = QPushButton("Open Export Folder")
        self._run.setEnabled(False)
        self._cancel.setEnabled(False)
        self._open.setEnabled(False)
        row.addWidget(self._model_label)
        row.addSpacing(12)
        row.addWidget(self._frame_label)
        row.addStretch(1)
        for button in (self._capture, self._run, self._cancel, self._open):
            row.addWidget(button)
        layout.addWidget(controls)

        self._source = OriginalSourceView()
        self._depth = DepthMapView()
        top = QSplitter()
        top.addWidget(self._source)
        top.addWidget(self._depth)
        top.setSizes([1, 1])

        self._viewport = ViewportPanel()
        self._canvas = PointCloudCanvas(self)
        self._viewport.attach_canvas(self._canvas.native)
        self._viewport.fit_requested.connect(self._on_fit_view_requested)
        self._viewport.reset_view_clicked.connect(self._on_reset_view_requested)
        self._viewport.tool_changed.connect(self._canvas.set_interaction_mode)
        self._viewport.point_size_changed.connect(self._canvas.set_point_size)

        body = QSplitter(Qt.Orientation.Vertical)
        body.addWidget(top)
        body.addWidget(self._viewport)
        body.setSizes([280, 520])
        layout.addWidget(body, stretch=1)

        self._status = QLabel("Capture a frame from the currently connected Live 3D source.")
        self._status.setObjectName("muted")
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        layout.addWidget(self._status)
        layout.addWidget(self._log)

        self._capture.clicked.connect(self.capture_requested.emit)
        self._run.clicked.connect(self.run_requested.emit)
        self._cancel.clicked.connect(self.cancel_requested.emit)
        self._open.clicked.connect(self._choose_folder)

    def set_model(self, label: str) -> None:
        self._model_label.setText(f"Model: {label}")

    def set_captured_frame(self, image: np.ndarray) -> None:
        self._source.set_frame(ndarray_to_qpixmap(image))
        h, w = image.shape[:2]
        self._frame_label.setText(f"Captured {w} × {h}")
        self._run.setEnabled(True)
        self._status.setText("Frozen frame ready for high-quality inference.")

    def apply_state(self, state: SnapshotRunState) -> None:
        running = state in {SnapshotRunState.RUNNING, SnapshotRunState.CANCELLING}
        self._capture.setEnabled(not running)
        self._run.setEnabled(not running and self._frame_label.text().startswith("Captured"))
        self._cancel.setEnabled(state is SnapshotRunState.RUNNING)
        self._status.setText({
            SnapshotRunState.RUNNING: "Snapshot inference is running in an isolated process…",
            SnapshotRunState.CANCELLING: "Cancelling snapshot inference…",
            SnapshotRunState.COMPLETED: "Snapshot completed and exported.",
            SnapshotRunState.FAILED: "Snapshot failed. Review the log.",
            SnapshotRunState.CANCELLED: "Snapshot cancelled.",
        }.get(state, self._status.text()))

    def append_log(self, text: str) -> None:
        self._log.appendPlainText(text)

    def show_result(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        data = np.load(self._output_dir / "result.npz", allow_pickle=False)
        rgb = data["rgb"]
        depth = data["depth"]
        self._source.set_frame(ndarray_to_qpixmap(rgb))
        if depth.size:
            finite = np.isfinite(depth)
            lo, hi = np.percentile(depth[finite], [2, 98]) if finite.any() else (0.0, 1.0)
            if hi <= lo:
                hi = lo + 1.0
            preview_rgb = apply_colormap(depth, near=float(lo), far=float(hi))
            self._depth.set_depth_frame(ndarray_to_qpixmap(preview_rgb[:, :, ::-1]))
        points = data["points"]
        colors = data["colors"]
        intrinsics = data["intrinsics"]
        valid_mask = data["valid_mask"] if "valid_mask" in data.files else np.array([])
        point_stride = (
            int(np.asarray(data["point_stride"]).reshape(-1)[0])
            if "point_stride" in data.files and data["point_stride"].size
            else 1
        )
        if not points.size and depth.size:
            points = depth_to_points(depth, intrinsics if intrinsics.size else None)
        if points.size:
            point_array = np.asarray(points, dtype=np.float32)
            flat_points = point_array.reshape(-1, 3)
            flat_colors = align_point_colors(
                rgb_bgr=rgb,
                points=point_array,
                colors=colors if colors.size else None,
                point_stride=point_stride,
                depth_shape=depth.shape[:2] if depth.size else None,
            )
            valid = np.isfinite(flat_points).all(axis=1)
            valid &= align_point_mask(
                valid_mask=valid_mask if valid_mask.size else None,
                points=point_array,
                point_stride=point_stride,
            )
            self._canvas.update_points(flat_points[valid], flat_colors[valid])
            self._canvas.fit_scene()
            self._viewport.mark_camera_initialized()
            self._viewport.set_status(ViewportStatus.RESULT_READY)
            self._viewport.set_idle_message_visible(False)
        self._open.setEnabled(True)

    def _on_fit_view_requested(self) -> None:
        if not self._canvas.has_points:
            return
        self._canvas.fit_scene()
        self._viewport.mark_camera_initialized()

    def _on_reset_view_requested(self) -> None:
        self._canvas.reset_view()
        self._viewport.mark_camera_initialized()

    def _choose_folder(self) -> None:
        if self._output_dir is None:
            return
        QFileDialog.getExistingDirectory(self, "Snapshot export", str(self._output_dir))
