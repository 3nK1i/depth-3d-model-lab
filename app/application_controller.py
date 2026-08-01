"""Wires real source capture into the UI (Phase 2).

Owns the single active :class:`~core.capture.base_source.BaseSource`
instance, converts its frames to pixmaps for :class:`OriginalSourceView`,
and reflects its connection status onto :class:`SourceBar`'s badge.

This is deliberately independent of inference: nothing here knows or
cares whether inference is running, per spec §5 ("the original source
preview must be independent of AI inference"). Phase 3 introduces the
inference worker and state machine; this controller only ever manages
the source side.
"""

from __future__ import annotations

import logging
import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QWidget

from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from app.widgets.source_toolbar_snapshot import ConnectionState, SourceType
from core.capture.base_source import BaseSource, SourceStatus
from core.capture.camera_source import CameraEnumerator, CameraSource
from core.capture.frame import SourceFrame
from core.capture.image_source import SUPPORTED_EXTENSIONS as IMAGE_EXTENSIONS
from core.capture.image_source import ImageSource
from core.capture.qt_convert import ndarray_to_qpixmap
from core.capture.rtsp_source import RtspSource, sanitize_rtsp_url
from core.capture.video_source import VideoSource

logger = logging.getLogger(__name__)

_STATUS_TO_CONNECTION_STATE: dict[SourceStatus, ConnectionState] = {
    SourceStatus.IDLE: ConnectionState.DISCONNECTED,
    SourceStatus.CONNECTING: ConnectionState.CONNECTING,
    SourceStatus.CONNECTED: ConnectionState.CONNECTED,
    SourceStatus.RECONNECTING: ConnectionState.RECONNECTING,
    SourceStatus.ERROR: ConnectionState.ERROR,
    SourceStatus.STOPPED: ConnectionState.DISCONNECTED,
}

_IMAGE_FILE_FILTER = "Images (" + " ".join(f"*{ext}" for ext in IMAGE_EXTENSIONS) + ")"
_VIDEO_FILE_FILTER = "Videos (*.mp4 *.avi *.mkv *.mov *.webm)"
_PREVIEW_TARGET_FPS = 30.0
_PREVIEW_INTERVAL_MS = round(1000.0 / _PREVIEW_TARGET_FPS)
_FPS_WINDOW_SECONDS = 1.0


class ApplicationController(QObject):
    """Owns the active capture source and drives the source-facing UI."""

    frame_captured = pyqtSignal(object)  # SourceFrame — for anything else
    # (e.g. Phase 3's InferenceController) that wants live frames without
    # this controller needing to know inference exists at all.

    def __init__(
        self,
        source_bar: SourceBar,
        original_view: OriginalSourceView,
        parent_widget: QWidget | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_bar = source_bar
        self._original_view = original_view
        self._parent_widget = parent_widget
        self._active_source: BaseSource | None = None

        # High-rate cameras can deliver 60-100+ decoded frames per second.
        # Converting every one to a QPixmap in the GUI thread would starve
        # depth/geometry work. Keep only the newest preview frame and paint it
        # at a smooth, bounded 30 FPS. The raw SourceFrame still reaches the
        # inference controller immediately through ``frame_captured``.
        self._latest_preview_frame: SourceFrame | None = None
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(_PREVIEW_INTERVAL_MS)
        self._preview_timer.timeout.connect(self._present_latest_preview)
        self._preview_timer.start()

        self._capture_fps_window_start = time.monotonic()
        self._capture_fps_window_count = 0
        self._last_capture_fps: float | None = None

        source_bar.browse_image_clicked.connect(self._on_browse_image)
        source_bar.browse_video_clicked.connect(self._on_browse_video)
        source_bar.rtsp_test_clicked.connect(self._on_rtsp_test)

        self._test_source: BaseSource | None = None
        self._enumerator = CameraEnumerator()
        self._enumerator.devices_found.connect(self._source_bar.set_camera_devices)
        self._enumerator.start()

    # -- Connect/Disconnect entry points, called from the runtime toolbar --

    def connect_active_source(self) -> None:
        """Connect whichever source type is currently selected.

        Image/Video are normally started directly from Browse (spec §5A:
        "run automatically after opening"); calling Connect while one of
        them is already loaded simply re-sends the last frame.
        """
        source_type = self._source_bar.active_type
        if source_type is SourceType.CAMERA:
            self._start_camera()
        elif source_type is SourceType.RTSP:
            self._start_rtsp()
        else:
            self.resend_current_frame()
        # VIDEO: nothing to do — Browse already started playback.

    def resend_current_frame(self) -> None:
        """Re-emit the active source's current frame without re-reading
        anything from disk/device.

        Needed specifically for a static ImageSource: it only ever emits
        ONE frame, immediately on connect/browse — well before Start
        Inference has been clicked. Without an explicit resend when
        inference actually begins RUNNING, that single frame opportunity
        is already gone and the image source can never feed inference
        anything (Camera/Video/RTSP don't have this gap since they keep
        emitting frames continuously).
        """
        if isinstance(self._active_source, ImageSource):
            self._active_source.resend()

    def disconnect_active_source(self) -> None:
        if self._active_source is not None:
            self._active_source.stop()
            self._active_source = None
        if self._test_source is not None:
            self._test_source.stop()
            self._test_source = None
        self._source_bar.set_connection_enum(ConnectionState.DISCONNECTED)
        self._original_view.set_live(False)
        self._original_view.clear_frame()
        self._latest_preview_frame = None
        self._reset_capture_fps()

    # -- Browse handlers -----------------------------------------------------

    def _on_browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Open Image", "", _IMAGE_FILE_FILTER
        )
        if not path:
            return
        self._replace_active_source(ImageSource(path))

    def _on_browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._parent_widget, "Open Video", "", _VIDEO_FILE_FILTER
        )
        if not path:
            return
        self._source_bar.set_video_path_display(path)
        self._replace_active_source(VideoSource(path))

    def _start_rtsp(self) -> None:
        """Persistent RTSP connect — stays connected until Disconnect."""
        url = self._source_bar.rtsp_url()
        if not url:
            return
        logger.info("Connecting to RTSP stream: %s", sanitize_rtsp_url(url))
        self._replace_active_source(RtspSource(url))

    def _on_rtsp_test(self) -> None:
        """Bounded connectivity probe: opens the stream just long enough
        to report CONNECTED or ERROR on the badge, then stops itself. It
        never becomes the active source — a separate, explicit Connect
        is what keeps a stream running (spec §7: "Test Connection" is
        distinct from actually connecting)."""
        url = self._source_bar.rtsp_url()
        if not url:
            return
        if self._test_source is not None:
            self._test_source.stop()

        logger.info("Testing RTSP connection: %s", sanitize_rtsp_url(url))
        test_source = RtspSource(url, max_reconnect_attempts=0, connection_timeout_ms=4000)
        self._test_source = test_source

        def _on_test_status(status: SourceStatus) -> None:
            if status in (SourceStatus.CONNECTED, SourceStatus.ERROR):
                self._source_bar.set_connection_enum(_STATUS_TO_CONNECTION_STATE[status])
                test_source.stop()

        test_source.status_changed.connect(_on_test_status)
        test_source.error_occurred.connect(self._on_error)
        test_source.start()

    def _start_camera(self) -> None:
        device_index, width, height, fps = self._source_bar.camera_config()
        self._replace_active_source(
            CameraSource(device_index, width=width, height=height, fps=fps)
        )

    # -- Source lifecycle plumbing ------------------------------------------

    def _replace_active_source(self, source: BaseSource) -> None:
        if self._active_source is not None:
            self._active_source.stop()

        self._latest_preview_frame = None
        self._reset_capture_fps()
        source.frame_ready.connect(self._on_frame_ready)
        source.status_changed.connect(self._on_status_changed)
        source.error_occurred.connect(self._on_error)
        self._active_source = source
        source.start()

    def _on_frame_ready(self, frame: SourceFrame) -> None:
        if self.is_stream_source:
            self._record_capture_frame()
        self._latest_preview_frame = frame

        # Static images should appear immediately. Continuous sources are
        # presented by the timer so a 100 FPS stream cannot flood Qt painting.
        if isinstance(self._active_source, ImageSource):
            self._present_latest_preview()

        # No QPixmap conversion happens before this signal. InferenceController
        # stores only its newest pending frame, so this path remains cheap.
        self.frame_captured.emit(frame)

    def _present_latest_preview(self) -> None:
        frame = self._latest_preview_frame
        if frame is None:
            return
        self._latest_preview_frame = None
        pixmap = ndarray_to_qpixmap(frame.image)
        self._original_view.set_frame(pixmap)
        if isinstance(self._active_source, ImageSource):
            self._original_view.set_ready()
        else:
            self._original_view.set_live(True)

    def _record_capture_frame(self) -> None:
        now = time.monotonic()
        self._capture_fps_window_count += 1
        elapsed = now - self._capture_fps_window_start
        if elapsed < _FPS_WINDOW_SECONDS:
            return
        self._last_capture_fps = self._capture_fps_window_count / elapsed
        self._capture_fps_window_start = now
        self._capture_fps_window_count = 0

    def _reset_capture_fps(self) -> None:
        self._capture_fps_window_start = time.monotonic()
        self._capture_fps_window_count = 0
        self._last_capture_fps = None

    def _on_status_changed(self, status: SourceStatus) -> None:
        self._source_bar.set_connection_enum(_STATUS_TO_CONNECTION_STATE[status])
        # Explicit disconnect clears the preview directly. A real source
        # error is the only asynchronous state that should mark it idle.
        if status is SourceStatus.ERROR:
            self._original_view.set_live(False)

    def _on_error(self, message: str) -> None:
        logger.error("Source error: %s", message)

    @property
    def active_source(self) -> BaseSource | None:
        return self._active_source

    @property
    def is_stream_source(self) -> bool:
        """Whether the current source continuously produces frames.

        Video files use the same backpressure policy as cameras and RTSP.
        Before a source object exists, the selected source type is a safe
        fallback for tests and for the short pre-connect state.
        """
        if self._active_source is not None:
            return isinstance(self._active_source, (VideoSource, CameraSource, RtspSource))
        return self._source_bar.active_type is not SourceType.IMAGE

    @property
    def capture_fps(self) -> float | None:
        """Measured FPS for continuous sources; unavailable for images."""
        if not self.is_stream_source:
            return None
        return self._last_capture_fps
