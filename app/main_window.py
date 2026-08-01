"""Main application window for EIA TECH — Depth & 3D Model Lab.

Assembles the full visual shell (spec §4). Runtime button enable/disable
is driven by the real application state machine via
:class:`~app.inference_controller.InferenceController` (Phase 3) — the
Phase 1 demo lifecycle that used to live here has been fully replaced.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.application_controller import ApplicationController
from app.inference_controller import InferenceController
from app.settings import AppSettings
from app.theme import (
    BODY_SPACING,
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from app.widgets.depth_view import DepthMapView
from app.widgets.header_widget import HeaderWidget
from app.widgets.mode_tabs import ModeTabs
from app.widgets.original_view import OriginalSourceView
from app.widgets.performance_bar import PerformanceBar
from app.widgets.settings_panel import SettingsPanel
from app.widgets.source_runtime_bar import SourceRuntimeBar
from app.widgets.viewport.viewport_panel import ViewportPanel

MIN_WINDOW_WIDTH = 1366
MIN_WINDOW_HEIGHT = 768


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EIA TECH — Depth & 3D Model Lab")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self._settings = AppSettings()
        self._viewport_focus_mode = False
        self._saved_main_splitter_sizes: list[int] | None = None
        self._saved_left_splitter_sizes: list[int] | None = None

        self._build_ui()
        self._viewport_panel.fullscreen_requested.connect(self._toggle_viewport_focus)
        self._sync_header_hardware()

        self._controller = ApplicationController(
            self._source_bar, self._original_view, parent_widget=self
        )
        self._inference_controller = InferenceController(
            self._controller,
            self._runtime_toolbar,
            self._header,
            self._performance_bar,
            self._depth_view,
            self._viewport_panel,
            self._settings_panel,
        )
        self._inference_controller.viewport_escape_requested.connect(
            self._exit_viewport_focus
        )
        self._controller.frame_captured.connect(self._inference_controller.on_source_frame)
        self._restore_layout()

    # -- Construction --------------------------------------------------

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header and mode tabs are full-bleed chrome; they intentionally sit
        # outside the page gutters.
        self._header = HeaderWidget()
        outer.addWidget(self._header)

        self._mode_tabs = ModeTabs()
        outer.addWidget(self._mode_tabs)

        self._page = self._build_page()
        outer.addWidget(self._page, stretch=1)

    def _build_page(self) -> QWidget:
        """The Live 3D page: source bar, main body and footer as siblings.

        All outer gutters and inter-card gaps come from this one layout
        (spec: one shared spacing grid), never from padding inside the
        individual cards or the VisPy canvas.
        """
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(
            PAGE_MARGIN_H, PAGE_MARGIN_TOP, PAGE_MARGIN_H, PAGE_MARGIN_BOTTOM
        )
        layout.setSpacing(PAGE_SPACING)

        self._control_row = self._build_control_row()
        layout.addWidget(self._control_row, stretch=0)
        layout.addWidget(self._build_main_splitter(), stretch=1)

        self._performance_bar = PerformanceBar()
        layout.addWidget(self._performance_bar, stretch=0)
        return page

    def _build_control_row(self) -> QWidget:
        """Three independent grouped cards: source type, source config,
        and runtime actions — see :class:`SourceRuntimeBar`. No outer
        border wraps them; the page background shows through the gaps.
        """
        bar = SourceRuntimeBar()
        self._source_bar = bar.source_bar
        self._runtime_toolbar = bar.runtime_toolbar
        return bar

    def _build_main_splitter(self) -> QSplitter:
        self._original_view = OriginalSourceView()
        self._depth_view = DepthMapView()

        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._top_splitter.setHandleWidth(SPLITTER_HANDLE)
        self._top_splitter.setChildrenCollapsible(False)
        self._top_splitter.addWidget(self._original_view)
        self._top_splitter.addWidget(self._depth_view)
        self._top_splitter.setSizes([1, 1])

        self._viewport_panel = ViewportPanel()

        self._left_splitter = QSplitter(Qt.Orientation.Vertical)
        self._left_splitter.setHandleWidth(SPLITTER_HANDLE)
        self._left_splitter.setChildrenCollapsible(False)
        self._left_splitter.addWidget(self._top_splitter)
        self._left_splitter.addWidget(self._viewport_panel)
        self._left_splitter.setStretchFactor(0, 0)
        self._left_splitter.setStretchFactor(1, 1)
        # The 3D viewport is the visually dominant element (spec §4).
        self._left_splitter.setSizes([250, 640])

        self._settings_panel = SettingsPanel(app_settings=self._settings)

        self._main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._main_splitter.setHandleWidth(BODY_SPACING)
        self._main_splitter.setChildrenCollapsible(False)
        self._main_splitter.addWidget(self._left_splitter)
        self._main_splitter.addWidget(self._settings_panel)
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 0)
        self._main_splitter.setCollapsible(1, False)
        self._main_splitter.setSizes([980, 348])

        return self._main_splitter

    def _sync_header_hardware(self) -> None:
        """Display the detected startup hardware snapshot in the header."""
        hardware = self._settings_panel.hardware_info

        total_gib = hardware.total_memory_gib
        free_gib = hardware.free_memory_gib

        if total_gib is not None:
            used_gib = (
                max(total_gib - free_gib, 0.0)
                if free_gib is not None
                else 0.0
            )

            self._header.update_gpu_status(
                gpu_name=hardware.name,
                vram_used_gb=used_gib,
                vram_total_gb=total_gib,
            )
        else:
            self._header.update_gpu_status(
                gpu_name=hardware.name,
                vram_used_gb=0.0,
                vram_total_gb=0.0,
            )

        cuda_label = (
            hardware.torch_cuda_version
            if hardware.cuda_ready
            else None
        )
        self._header.update_cuda_status(cuda_label)

    # -- Viewport focus mode ----------------------------------------------

    def _toggle_viewport_focus(self) -> None:
        self._set_viewport_focus_mode(not self._viewport_focus_mode)

    def _exit_viewport_focus(self) -> None:
        if self._viewport_focus_mode:
            self._set_viewport_focus_mode(False)

    def _set_viewport_focus_mode(self, enabled: bool) -> None:
        """Maximize only the 3D viewport while preserving the window state.

        This is an in-window focus mode rather than an operating-system
        fullscreen toggle. It avoids recreating the VisPy/OpenGL surface and
        restores the user's splitter proportions exactly.
        """
        enabled = bool(enabled)
        if enabled == self._viewport_focus_mode:
            self._viewport_panel.set_fullscreen_active(enabled)
            return

        if enabled:
            self._saved_main_splitter_sizes = self._main_splitter.sizes()
            self._saved_left_splitter_sizes = self._left_splitter.sizes()

            self._header.hide()
            self._mode_tabs.hide()
            self._control_row.hide()
            self._top_splitter.hide()
            self._settings_panel.hide()
            self._performance_bar.hide()

            self._main_splitter.setSizes([1, 0])
            self._left_splitter.setSizes([0, 1])
        else:
            self._header.show()
            self._mode_tabs.show()
            self._control_row.show()
            self._top_splitter.show()
            self._settings_panel.show()
            self._performance_bar.show()

            if self._saved_main_splitter_sizes:
                self._main_splitter.setSizes(self._saved_main_splitter_sizes)
            if self._saved_left_splitter_sizes:
                self._left_splitter.setSizes(self._saved_left_splitter_sizes)

        self._viewport_focus_mode = enabled
        self._viewport_panel.set_fullscreen_active(enabled)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._viewport_focus_mode and event.key() == Qt.Key.Key_Escape:
            self._set_viewport_focus_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    # -- Persistence ------------------------------------------------------

    def _restore_layout(self) -> None:
        geometry = self._settings.load_window_geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        window_state = self._settings.load_window_state()
        if window_state is not None:
            self.restoreState(window_state)

        main_state = self._settings.load_splitter_state(AppSettings.KEY_SPLITTER_MAIN)
        if main_state is not None:
            self._main_splitter.restoreState(main_state)

        left_state = self._settings.load_splitter_state(AppSettings.KEY_SPLITTER_LEFT)
        if left_state is not None:
            self._left_splitter.restoreState(left_state)

        top_state = self._settings.load_splitter_state(AppSettings.KEY_SPLITTER_TOP)
        if top_state is not None:
            self._top_splitter.restoreState(top_state)

    def _save_layout(self) -> None:
        self._settings.save_window_geometry(self.saveGeometry())
        self._settings.save_window_state(self.saveState())
        self._settings.save_splitter_state(AppSettings.KEY_SPLITTER_MAIN, self._main_splitter.saveState())
        self._settings.save_splitter_state(AppSettings.KEY_SPLITTER_LEFT, self._left_splitter.saveState())
        self._settings.save_splitter_state(AppSettings.KEY_SPLITTER_TOP, self._top_splitter.saveState())
        self._settings.sync()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override signature
        if self._viewport_focus_mode:
            self._set_viewport_focus_mode(False)
        self._inference_controller.shutdown()
        self._controller.disconnect_active_source()
        self._save_layout()
        super().closeEvent(event)