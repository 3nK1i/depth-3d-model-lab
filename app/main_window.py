"""Main application window for EIA TECH — Depth & 3D Model Lab.

Phase 1 scope: assemble the full visual shell described in spec §4 with
responsive QSplitter-based layout, and demonstrate runtime-button
enable/disable with a small local demo (``_DemoRuntimeDemo``). That demo
is explicitly temporary — Phase 3 replaces it with the real application
state machine (spec §14) driving :meth:`RuntimeToolbar.apply_button_states`
from actual source/inference lifecycle events instead of button clicks.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.settings import AppSettings
from app.theme import (
    BODY_SPACING,
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from app.widgets.badge import BadgeKind
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
        self.setWindowTitle("EIA TECH - Depth & 3D Model Lab")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self._settings = AppSettings()
        self._is_connected = False
        self._is_running = False

        self._build_ui()
        self._wire_demo_lifecycle()
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

        outer.addWidget(self._build_page(), stretch=1)

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

        layout.addWidget(self._build_control_row(), stretch=0)
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

        self._settings_panel = SettingsPanel()

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

    # -- Phase-1 demo lifecycle -----------------------------------------
    # Deliberately minimal and self-contained. Replaced wholesale by the
    # real state machine + application controller in Phase 3.

    def _wire_demo_lifecycle(self) -> None:
        self._runtime_toolbar.connect_clicked.connect(self._demo_on_connect)
        self._runtime_toolbar.disconnect_clicked.connect(self._demo_on_disconnect)
        self._runtime_toolbar.start_clicked.connect(self._demo_on_start)
        self._runtime_toolbar.pause_resume_clicked.connect(self._demo_on_pause_resume)
        self._runtime_toolbar.stop_clicked.connect(self._demo_on_stop)

    def _demo_on_connect(self) -> None:
        self._is_connected = True
        self._source_bar.set_connection_state(True)
        self._original_view.set_live(True)
        self._header.set_status(BadgeKind.READY, "READY")
        self._performance_bar.set_status(BadgeKind.COMPATIBLE, "SOURCE READY")
        self._runtime_toolbar.apply_button_states(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=True,
            pause_resume_enabled=False,
            stop_enabled=False,
        )

    def _demo_on_disconnect(self) -> None:
        self._is_connected = False
        self._is_running = False
        self._source_bar.set_connection_state(False)
        self._original_view.set_live(False)
        self._original_view.clear_frame()
        self._depth_view.clear_frame()
        self._viewport_panel.set_live(False)
        self._performance_bar.set_status(BadgeKind.NOT_INSTALLED, "IDLE")
        self._performance_bar.reset_metrics()
        self._runtime_toolbar.apply_button_states(
            connect_enabled=True,
            disconnect_enabled=False,
            start_enabled=False,
            pause_resume_enabled=False,
            stop_enabled=False,
        )

    def _demo_on_start(self) -> None:
        self._is_running = True
        self._viewport_panel.set_live(True)
        self._performance_bar.set_status(BadgeKind.RECOMMENDED, "RUNNING")
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._runtime_toolbar.apply_button_states(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=False,
            pause_resume_enabled=True,
            stop_enabled=True,
        )

    def _demo_on_pause_resume(self) -> None:
        is_paused = not self._runtime_toolbar.is_paused
        self._runtime_toolbar.set_paused_label(is_paused)
        self._viewport_panel.set_live(not is_paused)
        self._performance_bar.set_status(
            BadgeKind.EXPERIMENTAL if is_paused else BadgeKind.RECOMMENDED,
            "PAUSED" if is_paused else "RUNNING",
        )

    def _demo_on_stop(self) -> None:
        self._is_running = False
        self._viewport_panel.set_live(False)
        self._runtime_toolbar.set_paused_label(is_paused=False)
        self._performance_bar.set_status(BadgeKind.COMPATIBLE, "SOURCE READY")
        self._performance_bar.reset_metrics()
        # Source preview stays live: stopping inference must never touch
        # the source connection (spec §5, §13).
        self._runtime_toolbar.apply_button_states(
            connect_enabled=False,
            disconnect_enabled=True,
            start_enabled=True,
            pause_resume_enabled=False,
            stop_enabled=False,
        )

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
        self._save_layout()
        super().closeEvent(event)