"""Main application window for EIA TECH — Depth & 3D Model Lab.

Assembles the full visual shell (spec §4). Runtime button enable/disable
is driven by the real application state machine via
:class:`~app.inference_controller.InferenceController` (Phase 3) — the
Phase 1 demo lifecycle that used to live here has been fully replaced.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.benchmark_controller import (
    BenchmarkController,
    BenchmarkRunConfiguration,
    BenchmarkRunState,
)
from app.inference_controller import InferenceController
from app.inspection_export_controller import InspectionExportController, InspectionExportState
from app.measurement_inspection_controller import (
    InspectionControllerState,
    MeasurementInspectionController,
)
from app.settings import AppSettings
from app.snapshot_controller import SnapshotController, SnapshotRunState
from app.stable_video_controller import StableVideoController, StableVideoRunState
from app.theme import (
    BODY_SPACING,
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from app.widgets.benchmark_page import BenchmarkPage
from app.widgets.depth_view import DepthMapView
from app.widgets.header_widget import HeaderWidget
from app.widgets.inspection_page import InspectionPage
from app.widgets.mode_tabs import AppMode, ModeTabs
from app.widgets.original_view import OriginalSourceView
from app.widgets.reconstruction_page import ReconstructionPage
from app.widgets.performance_bar import PerformanceBar
from app.widgets.settings_panel import SettingsPanel
from app.widgets.snapshot_page import SnapshotPage
from app.widgets.stable_video_page import StableVideoPage
from app.widgets.source_runtime_bar import SourceRuntimeBar
from app.widgets.viewport.viewport_panel import ViewportPanel
from core.inspection import load_inspection_bundle
from core.models.registry import get_model_descriptor
from core.runtime.state_machine import ApplicationState

MIN_WINDOW_WIDTH = 1366
MIN_WINDOW_HEIGHT = 768


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EIA TECH — Depth & 3D Model Lab")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)

        self._settings = AppSettings()
        self._focus_target: str | None = None
        self._viewport_focus_mode = False  # Backward-compatible test/public state.
        self._saved_main_splitter_sizes: list[int] | None = None
        self._saved_left_splitter_sizes: list[int] | None = None
        self._saved_top_splitter_sizes: list[int] | None = None

        self._build_ui()
        self._original_view.fullscreen_requested.connect(
            lambda: self._toggle_panel_focus("source")
        )
        self._depth_view.fullscreen_requested.connect(
            lambda: self._toggle_panel_focus("depth")
        )
        self._viewport_panel.fullscreen_requested.connect(self._toggle_viewport_focus)
        self._reconstruction_page.viewport_fullscreen_requested.connect(
            self._toggle_reconstruction_focus
        )
        self._reconstruction_page.viewport_escape_requested.connect(
            self._exit_focus_mode
        )
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
            self._exit_focus_mode
        )
        self._controller.frame_captured.connect(self._inference_controller.on_source_frame)

        # Phase 8.8A1 established the application-facing reconstruction
        # coordinator around the existing Phase 8.4 live-SLAM bridge. Phase
        # 8.8B5 enables navigation to its UI, while backend startup remains
        # explicit and controller-guarded. Keep the live-SLAM attribute as a
        # compatibility alias for the already-qualified Phase 8.4 boundary.
        self._reconstruction_controller = ReconstructionController(self._controller, self)
        self._live_slam_controller = self._reconstruction_controller.live_slam_controller
        # The page remains passive on construction: navigation alone never starts
        # a backend. Runtime actions continue to flow through the hardened
        # ReconstructionController boundary.
        self._reconstruction_page.bind_controller(self._reconstruction_controller)
        self._reconstruction_page.camera_calibration_profile_changed.connect(
            lambda _profile: self._refresh_inspection_context()
        )

        self._benchmark_controller = BenchmarkController(self)
        self._benchmark_page.run_requested.connect(self._on_benchmark_run_requested)
        self._benchmark_page.cancel_requested.connect(self._benchmark_controller.cancel)
        self._benchmark_controller.state_changed.connect(self._on_benchmark_state_changed)
        self._benchmark_controller.event_received.connect(self._benchmark_page.handle_event)
        self._benchmark_controller.log_received.connect(self._benchmark_page.append_log)
        self._benchmark_controller.results_ready.connect(self._benchmark_page.set_results)
        self._snapshot_controller = SnapshotController(Path(__file__).resolve().parents[1], self)
        self._stable_video_controller = StableVideoController(Path(__file__).resolve().parents[1], self)
        self._inspection_controller = MeasurementInspectionController(Path(__file__).resolve().parents[1], self)
        self._inspection_export_controller = InspectionExportController(Path(__file__).resolve().parents[1], self)
        # Phase 8.8A5 established Reconstruction-side start guards. Phase 8.8H2
        # keeps that ownership model and closes the reciprocal application boundary:
        # other long-running GPU/process jobs must also refuse to start while
        # Reconstruction owns resources. Live reconstruction may still share the
        # existing capture/inference path; offline reconstruction additionally
        # requires idle inference and releases any resident depth model before PyCOLMAP.
        self._reconstruction_controller.configure_start_guards(
            live=self._reconstruction_live_start_guard,
            offline=self._reconstruction_offline_start_guard,
        )
        self._reconstruction_controller.live_state_changed.connect(
            self._on_reconstruction_activity_changed
        )
        self._reconstruction_controller.offline_state_changed.connect(
            self._on_reconstruction_activity_changed
        )
        self._reconstruction_controller.offline_run_finished.connect(
            self._on_reconstruction_activity_changed
        )
        self._inspection_frame: np.ndarray | None = None
        self._snapshot_frame: np.ndarray | None = None
        self._stable_previous_smoothing: float | None = None
        self._snapshot_page.capture_requested.connect(self._capture_snapshot_frame)
        self._snapshot_page.run_requested.connect(self._run_snapshot)
        self._snapshot_page.cancel_requested.connect(self._snapshot_controller.cancel)
        self._snapshot_controller.state_changed.connect(self._on_snapshot_state_changed)
        self._snapshot_controller.log_received.connect(self._snapshot_page.append_log)
        self._snapshot_controller.result_ready.connect(self._snapshot_page.show_result)
        self._inspection_page.capture_requested.connect(self._capture_inspection_frame)
        self._inspection_page.calibration_requested.connect(
            self._reconstruction_page.open_camera_calibration_workspace
        )
        self._inspection_page.analysis_requested.connect(self._start_inspection_analysis)
        self._inspection_page.cancel_requested.connect(self._inspection_controller.cancel)
        self._inspection_page.recompute_requested.connect(self._recompute_inspection)
        self._inspection_page.export_requested.connect(self._export_inspection)
        self._inspection_controller.state_changed.connect(self._on_inspection_state_changed)
        self._inspection_controller.progress_changed.connect(self._inspection_page.set_progress)
        self._inspection_controller.result_ready.connect(self._on_inspection_result)
        self._inspection_controller.log_received.connect(self._inspection_page.append_job_log)
        self._inspection_controller.failure_received.connect(self._inspection_page.show_failure)
        self._inspection_export_controller.progress_changed.connect(self._inspection_page.set_export_progress)
        self._inspection_export_controller.result_ready.connect(self._on_inspection_export_result)
        self._inspection_export_controller.failure_received.connect(self._inspection_page.show_export_failure)
        self._inspection_export_controller.state_changed.connect(self._on_inspection_export_state_changed)
        self._stable_video_page.start_requested.connect(self._start_stable_video)
        self._stable_video_page.pause_resume_requested.connect(self._pause_resume_stable_video)
        self._stable_video_page.stop_requested.connect(self._stop_stable_video)
        self._stable_video_page.stabilization_changed.connect(
            self._settings_panel.set_temporal_smoothing
        )
        self._stable_video_controller.state_changed.connect(
            self._stable_video_page.apply_state
        )
        self._stable_video_controller.log_received.connect(
            self._stable_video_page.append_log
        )
        self._stable_video_controller.result_ready.connect(
            self._stable_video_page.show_result
        )
        self._stable_video_controller.statistics_changed.connect(
            self._stable_video_page.update_statistics
        )
        self._stable_video_controller.export_ready.connect(
            self._stable_video_page.set_export_dir
        )
        self._inference_controller.frame_rendered.connect(
            self._stable_video_controller.consume_result
        )
        self._inference_controller.state_machine.state_changed.connect(
            self._on_stable_runtime_state_changed
        )
        self._controller.source_type_changed.connect(
            lambda _source: self._on_stable_sequence_reset("source changed")
        )
        self._mode_tabs.mode_changed.connect(self._on_mode_changed)
        self._inference_controller.state_machine.state_changed.connect(
            self._refresh_benchmark_safety
        )
        self._refresh_benchmark_safety()
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
        self._inspection_page = InspectionPage()
        self._benchmark_page = BenchmarkPage()
        self._snapshot_page = SnapshotPage()
        self._stable_video_page = StableVideoPage()
        # Phase 8.8B1 constructed the Reconstruction workspace shell; Phase
        # 8.8B5 now exposes it through normal application navigation.
        self._reconstruction_page = ReconstructionPage()
        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._page)
        self._mode_stack.addWidget(self._inspection_page)
        self._mode_stack.addWidget(self._stable_video_page)
        self._mode_stack.addWidget(self._snapshot_page)
        self._mode_stack.addWidget(self._reconstruction_page)
        self._mode_stack.addWidget(self._benchmark_page)
        outer.addWidget(self._mode_stack, stretch=1)

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

    # -- Reconstruction resource interlocks ------------------------------

    def _reconstruction_shared_job_conflict(self) -> str | None:
        if self._benchmark_controller.is_running:
            return "Benchmark is running; finish or cancel it before starting Reconstruction."
        if self._snapshot_controller.is_running:
            return "Snapshot processing is running; finish or cancel it before starting Reconstruction."
        if self._stable_video_controller.is_active:
            return "Stable Video is active; stop it before starting Reconstruction."
        if self._inspection_controller.is_running:
            return "Measure & Inspect analysis is running; finish or cancel it before starting Reconstruction."
        if self._inspection_export_controller.is_running:
            return "Inspection export is running; finish it before starting Reconstruction."
        return None

    def _reconstruction_conflict_for_application_job(self, job_label: str) -> str:
        """Return a fail-closed reason when Reconstruction owns long-running resources.

        A5 already prevents Reconstruction from starting over Benchmark, Snapshot,
        Stable Video, inspection, or export. H2 makes that ownership reciprocal at
        the application entry points rather than relying only on navigation locking.
        Persistent scenes, static appearance previews, and prepared resume metadata
        do not count as active resource owners.
        """

        if not self._reconstruction_controller.is_active:
            return ""
        label = str(job_label).strip() or "this application job"
        return f"Reconstruction is active; stop it before starting {label}."

    def _reconstruction_live_start_guard(self) -> str | None:
        """Live SLAM may share capture/inference but not another long-running mode job."""

        return self._reconstruction_shared_job_conflict()

    def _reconstruction_offline_start_guard(self) -> str | None:
        """Reserve GPU/process capacity before an isolated offline reconstruction job."""

        conflict = self._reconstruction_shared_job_conflict()
        if conflict:
            return conflict
        state = self._inference_controller.current_state
        if state in {
            ApplicationState.CONNECTING,
            ApplicationState.LOADING_MODEL,
            ApplicationState.RUNNING,
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.SWITCHING_MODEL,
            ApplicationState.RECOVERING_WORKER,
            ApplicationState.RECONNECTING,
            ApplicationState.SHUTTING_DOWN,
        }:
            return (
                "Live 3D inference is busy; stop inference and wait for an idle source/model state "
                "before starting offline Reconstruction."
            )
        if not self._inference_controller.release_model_for_external_job():
            return "Live 3D model resources could not be released for offline Reconstruction."
        return None

    def _on_reconstruction_activity_changed(self, *_args: object) -> None:
        """Lock primary navigation while any long-running owner is active.

        Reconstruction is now a normal application mode, so starting it from the
        page must preserve the same leave-no-job-behind navigation discipline as
        Benchmark, Snapshot, Stable Video, and inspection jobs.
        """

        locked = (
            self._reconstruction_controller.is_active
            or self._benchmark_controller.is_running
            or self._snapshot_controller.is_running
            or self._stable_video_controller.is_active
            or self._inspection_controller.is_running
            or self._inspection_export_controller.is_running
        )
        self._mode_tabs.set_navigation_locked(locked)
        # Keep the other mode-specific readiness surfaces synchronized with the
        # same ownership decision. This is advisory UI state; the start handlers
        # below remain the authoritative admission boundary.
        self._refresh_benchmark_safety()
        self._refresh_stable_video_context()

    # -- Primary application modes ---------------------------------------

    def _on_mode_changed(self, mode: AppMode) -> None:
        if mode is AppMode.LIVE_3D:
            self._mode_stack.setCurrentWidget(self._page)
            return
        if mode is AppMode.INSPECTION:
            self._exit_focus_mode()
            self._refresh_inspection_context()
            self._mode_stack.setCurrentWidget(self._inspection_page)
            return
        if mode is AppMode.STABLE_VIDEO:
            self._exit_focus_mode()
            self._refresh_stable_video_context()
            self._mode_stack.setCurrentWidget(self._stable_video_page)
            return
        if mode is AppMode.SNAPSHOT:
            self._exit_focus_mode()
            self._snapshot_page.set_model(self._settings_panel.selected_model_label)
            self._mode_stack.setCurrentWidget(self._snapshot_page)
            return
        if mode is AppMode.RECONSTRUCTION:
            self._exit_focus_mode()
            self._mode_stack.setCurrentWidget(self._reconstruction_page)
            return
        if mode is AppMode.BENCHMARK:
            self._exit_focus_mode()
            self._refresh_benchmark_safety()
            self._mode_stack.setCurrentWidget(self._benchmark_page)

    def _refresh_inspection_context(self) -> None:
        self._inspection_page.set_context(
            source_available=self._controller.latest_source_frame is not None,
            depth_model_id=self._settings_panel.selected_model_id,
            depth_model_label=self._settings_panel.selected_model_label,
        )
        self._inspection_page.set_camera_calibration_context(
            self._reconstruction_page.selected_camera_calibration_profile
        )

    def _capture_inspection_frame(self) -> None:
        frame = self._controller.latest_source_frame
        if frame is None:
            self._inspection_page.show_failure(
                "No source frame is available. Connect or load a source in Live 3D first."
            )
            return
        self._inspection_frame = np.array(frame.image, dtype=np.uint8, copy=True)
        self._inspection_page.set_context(
            source_available=True,
            depth_model_id=self._settings_panel.selected_model_id,
            depth_model_label=self._settings_panel.selected_model_label,
        )
        self._inspection_page.set_captured_frame(self._inspection_frame)

    def _inspection_jobs_conflict(self) -> str:
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Measure & Inspect"
        )
        if reconstruction_conflict:
            return reconstruction_conflict
        if self._benchmark_controller.is_running:
            return "Stop the active Benchmark before measurement inspection."
        if self._snapshot_controller.is_running:
            return "Stop the active Snapshot job before measurement inspection."
        if self._stable_video_controller.is_active:
            return "Stop Stable Video before measurement inspection."
        return ""

    def _start_inspection_analysis(self, configuration: dict[str, object]) -> None:
        conflict = self._inspection_jobs_conflict()
        if conflict:
            self._inspection_page.show_failure(conflict)
            return
        image = configuration.get("image_bgr")
        if not isinstance(image, np.ndarray):
            self._inspection_page.show_failure("Capture a frame before analysis.")
            return
        if not self._inference_controller.release_model_for_external_job():
            self._inspection_page.show_failure(
                "Stop or finish the active Live 3D inference operation before analysis."
            )
            return
        try:
            self._inspection_controller.start_analysis(
                image_bgr=image,
                depth_model_id=str(configuration["depth_model_id"]),
                depth_config=dict(self._settings_panel.model_runtime_config),
                perception_model_id=str(configuration["perception_model_id"]),
                depth_profile=str(configuration["depth_profile"]),
                perception_profile=str(configuration["perception_profile"]),
                max_objects=int(configuration["max_objects"]),
                box_method=str(configuration.get("box_method", "auto")),
                calibration_profile=self._reconstruction_page.selected_camera_calibration_profile,
            )
            self._mode_tabs.set_navigation_locked(True)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            self._inspection_page.show_failure(str(exc))
            self._mode_tabs.set_navigation_locked(False)

    def _recompute_inspection(self, configuration: dict[str, object]) -> None:
        conflict = self._inspection_jobs_conflict()
        if conflict:
            self._inspection_page.show_failure(conflict)
            return
        try:
            self._inspection_controller.recompute(
                base_state_path=Path(configuration["base_state_path"]),
                observation_id=str(configuration["observation_id"]),
                manual_mask=configuration.get("manual_mask"),
                manual_revision=int(configuration["manual_revision"]),
                box_method=str(configuration.get("box_method", "auto")),
                reset_manual=bool(configuration.get("reset_manual", False)),
            )
            self._mode_tabs.set_navigation_locked(True)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._inspection_page.show_failure(str(exc))
            self._mode_tabs.set_navigation_locked(False)

    def _on_inspection_state_changed(self, state: InspectionControllerState) -> None:
        self._inspection_page.set_controller_state(state)
        if state in {
            InspectionControllerState.COMPLETED,
            InspectionControllerState.FAILED,
            InspectionControllerState.CANCELLED,
        }:
            self._mode_tabs.set_navigation_locked(False)

    def _on_inspection_result(self, bundle_dir: str) -> None:
        try:
            summary, arrays = load_inspection_bundle(bundle_dir)
            self._inspection_page.load_bundle(summary, arrays)
        except (OSError, ValueError, TypeError) as exc:
            self._inspection_page.show_failure(f"Inspection bundle could not be loaded: {exc}")
        finally:
            self._mode_tabs.set_navigation_locked(False)

    def _export_inspection(self, scope: str) -> None:
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "inspection export"
        )
        if reconstruction_conflict:
            self._inspection_page.show_export_failure(reconstruction_conflict)
            return
        if self._inspection_export_controller.is_running:
            self._inspection_page.show_export_failure("An inspection export is already running.")
            return
        try:
            context = self._inspection_page.export_context(scope)
        except RuntimeError as exc:
            self._inspection_page.show_export_failure(str(exc))
            return
        default_name = datetime.now().strftime("inspection_export_%Y%m%d_%H%M%S")
        parent = QFileDialog.getExistingDirectory(
            self,
            "Choose Inspection Export Parent Folder",
            str(Path.home() / "Documents"),
        )
        if not parent:
            return
        target = Path(parent) / default_name
        try:
            self._inspection_export_controller.start(
                bundle_dir=context["summary"].bundle_dir,
                output_dir=target,
                observation_ids=tuple(context["observation_ids"]),
                locked_ids=tuple(context["locked_ids"]),
            )
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self._inspection_page.show_export_failure(str(exc))

    def _on_inspection_export_state_changed(self, state: InspectionExportState) -> None:
        running = state is InspectionExportState.RUNNING
        self._inspection_page.set_export_running(running)
        if running:
            self._mode_tabs.set_navigation_locked(True)
        elif not self._inspection_controller.is_running:
            self._mode_tabs.set_navigation_locked(False)

    def _on_inspection_export_result(self, path: str, object_count: int, file_count: int) -> None:
        self._inspection_page.show_export_success(path, object_count, file_count)
        self._mode_tabs.set_navigation_locked(False)
        QMessageBox.information(
            self,
            "Inspection Export Complete",
            f"Created {file_count} files for {object_count} object(s).\n\n{path}",
        )
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _refresh_stable_video_context(self) -> None:
        descriptor = get_model_descriptor(self._settings_panel.selected_model_id)
        self._stable_video_page.set_model(
            self._settings_panel.selected_model_label,
            native_temporal=descriptor.capabilities.stateful_sequence,
        )
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Stable Video"
        )
        eligible = (
            not reconstruction_conflict
            and self._controller.active_source is not None
            and self._controller.is_stream_source
            and self._inference_controller.current_state in {
                ApplicationState.SOURCE_READY,
                ApplicationState.MODEL_READY,
                ApplicationState.RESULT_READY,
            }
        )
        if reconstruction_conflict:
            reason = reconstruction_conflict
        else:
            reason = "" if eligible else (
                "Connect a Video, USB Camera, or RTSP source in Live 3D first. "
                "Static images belong in Snapshot Mode."
            )
        self._stable_video_page.set_source_eligible(eligible, reason)

    def _start_stable_video(self) -> None:
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Stable Video"
        )
        if reconstruction_conflict:
            self._stable_video_page.append_log(reconstruction_conflict)
            return
        if self._benchmark_controller.is_running or self._snapshot_controller.is_running:
            self._stable_video_page.append_log(
                "Stop the active Benchmark or Snapshot job before Stable Video."
            )
            return
        if self._controller.active_source is None or not self._controller.is_stream_source:
            self._refresh_stable_video_context()
            return
        state = self._inference_controller.current_state
        if state not in {
            ApplicationState.SOURCE_READY,
            ApplicationState.MODEL_READY,
            ApplicationState.RESULT_READY,
        }:
            self._stable_video_page.append_log(
                "Stable Video requires a connected, idle continuous source."
            )
            return
        descriptor = get_model_descriptor(self._settings_panel.selected_model_id)
        strength = self._stable_video_page.smoothing_strength
        self._stable_previous_smoothing = self._settings_panel.temporal_smoothing
        self._settings_panel.set_temporal_smoothing(strength)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = Path.cwd() / "outputs" / "stable_video" / f"stable_video_{stamp}"
        temporal_mode = (
            "native_temporal_cache_plus_smoothing"
            if descriptor.capabilities.stateful_sequence
            else "generic_geometry_smoothing"
        )
        try:
            self._stable_video_controller.begin(
                output_dir=output,
                model_id=descriptor.model_id,
                model_name=descriptor.display_name,
                temporal_mode=temporal_mode,
                smoothing_strength=strength,
                export_enabled=self._stable_video_page.export_enabled,
                export_numeric_depth=self._stable_video_page.export_numeric_depth,
                export_fps=self._stable_video_page.export_fps,
            )
            self._mode_tabs.set_navigation_locked(True)
            self._inference_controller.start_inference()
        except (OSError, RuntimeError, ValueError) as exc:
            self._stable_video_page.append_log(f"Stable Video could not start: {exc}")
            self._stable_video_controller.abort()
            self._mode_tabs.set_navigation_locked(False)

    def _pause_resume_stable_video(self) -> None:
        state = self._inference_controller.current_state
        if state not in {ApplicationState.RUNNING, ApplicationState.PAUSED}:
            return
        self._inference_controller.pause_or_resume_inference()

    def _stop_stable_video(self) -> None:
        self._inference_controller.stop_inference()
        self._stable_video_controller.finish(completed=True)
        if self._stable_previous_smoothing is not None:
            self._settings_panel.set_temporal_smoothing(self._stable_previous_smoothing)
            self._stable_previous_smoothing = None
        self._mode_tabs.set_navigation_locked(False)
        self._refresh_stable_video_context()

    def _on_stable_runtime_state_changed(self, _old: object, new: ApplicationState) -> None:
        if not self._stable_video_controller.is_active:
            self._refresh_stable_video_context()
            return
        if new is ApplicationState.RUNNING:
            self._stable_video_controller.mark_running()
        elif new is ApplicationState.PAUSED:
            self._stable_video_controller.mark_paused()
        elif new is ApplicationState.RECOVERING_WORKER:
            self._on_stable_sequence_reset(new.value)
        elif new is ApplicationState.ERROR:
            self._stable_video_controller.fail("Stable Video stopped because the inference runtime entered ERROR.")
            if self._stable_previous_smoothing is not None:
                self._settings_panel.set_temporal_smoothing(self._stable_previous_smoothing)
                self._stable_previous_smoothing = None
            self._mode_tabs.set_navigation_locked(False)

    def _on_stable_sequence_reset(self, reason: str) -> None:
        if self._stable_video_controller.is_active:
            self._stable_video_controller.mark_reset(reason)

    def _capture_snapshot_frame(self) -> None:
        frame = self._controller.latest_source_frame
        if frame is None:
            self._snapshot_page.append_log(
                "No source frame is available. Connect or load a source in Live 3D first."
            )
            return
        self._snapshot_frame = np.array(frame.image, dtype=np.uint8, copy=True)
        self._snapshot_page.set_model(self._settings_panel.selected_model_label)
        self._snapshot_page.set_captured_frame(self._snapshot_frame)

    def _run_snapshot(self) -> None:
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Snapshot"
        )
        if reconstruction_conflict:
            self._snapshot_page.append_log(reconstruction_conflict)
            return
        if self._snapshot_frame is None:
            self._snapshot_page.append_log("Capture a frame before running Snapshot Mode.")
            return
        if self._benchmark_controller.is_running:
            self._snapshot_page.append_log("Stop the active benchmark before running a snapshot.")
            return
        state = self._inference_controller.state_machine.state
        if state not in {ApplicationState.DISCONNECTED, ApplicationState.SOURCE_READY}:
            self._snapshot_page.append_log(
                "Stop Live 3D inference before running isolated snapshot inference."
            )
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = Path.cwd() / "outputs" / "snapshots" / f"snapshot_{stamp}"
        try:
            self._snapshot_controller.start(
                image=self._snapshot_frame,
                model_id=self._settings_panel.selected_model_id,
                config=dict(self._settings_panel.model_runtime_config),
                output_dir=output,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            self._snapshot_page.append_log(f"Snapshot could not start: {exc}")
            self._snapshot_page.apply_state(SnapshotRunState.FAILED)

    def _on_snapshot_state_changed(self, state: SnapshotRunState) -> None:
        self._snapshot_page.apply_state(state)
        running = state in {SnapshotRunState.RUNNING, SnapshotRunState.CANCELLING}
        self._mode_tabs.set_navigation_locked(running)

    def _on_benchmark_run_requested(
        self, configuration: BenchmarkRunConfiguration
    ) -> None:
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Benchmark"
        )
        if reconstruction_conflict:
            self._benchmark_page.append_log(reconstruction_conflict)
            self._refresh_benchmark_safety()
            return
        if self._inference_controller.state_machine.state is not ApplicationState.DISCONNECTED:
            self._refresh_benchmark_safety()
            return
        try:
            self._benchmark_controller.start(configuration)
        except (OSError, RuntimeError, ValueError) as exc:
            self._benchmark_page.append_log(f"Benchmark could not start: {exc}")
            self._benchmark_page.apply_state(BenchmarkRunState.FAILED)

    def _on_benchmark_state_changed(self, state: BenchmarkRunState) -> None:
        self._benchmark_page.apply_state(state)
        running = state in {
            BenchmarkRunState.STARTING,
            BenchmarkRunState.RUNNING,
            BenchmarkRunState.CANCELLING,
        }
        self._mode_tabs.set_navigation_locked(running)
        if not running:
            self._refresh_benchmark_safety()

    def _refresh_benchmark_safety(self, *_args: object) -> None:
        state = self._inference_controller.state_machine.state
        reconstruction_conflict = self._reconstruction_conflict_for_application_job(
            "Benchmark"
        )
        safe = (
            not reconstruction_conflict
            and state is ApplicationState.DISCONNECTED
            and not self._benchmark_controller.is_running
        )
        reason = ""
        if reconstruction_conflict:
            reason = reconstruction_conflict
        elif state is not ApplicationState.DISCONNECTED:
            reason = (
                "Benchmarking requires an idle GPU runtime. Return to Live 3D, "
                "Stop inference, and Disconnect the active source first."
            )
        elif self._benchmark_controller.is_running:
            reason = "A benchmark is already running."
        self._benchmark_page.set_runtime_safe(safe, reason)

    # -- Panel focus mode -------------------------------------------------

    def _toggle_panel_focus(self, target: str) -> None:
        self._set_focus_target(None if self._focus_target == target else target)

    def _toggle_viewport_focus(self) -> None:
        self._toggle_panel_focus("viewport")

    def _toggle_reconstruction_focus(self) -> None:
        self._toggle_panel_focus("reconstruction")

    def _exit_viewport_focus(self) -> None:
        """Backward-compatible alias retained for existing integrations."""
        self._exit_focus_mode()

    def _exit_focus_mode(self) -> None:
        if self._focus_target is not None:
            self._set_focus_target(None)

    def _set_viewport_focus_mode(self, enabled: bool) -> None:
        """Backward-compatible entry point used by the existing tests."""
        self._set_focus_target("viewport" if enabled else None)

    def _set_focus_target(self, target: str | None) -> None:
        """Focus a visual surface without recreating any render surface."""
        if target not in {None, "source", "depth", "viewport", "reconstruction"}:
            raise ValueError(f"Unknown focus target: {target!r}")
        if target == "reconstruction" and self._mode_stack.currentWidget() is not self._reconstruction_page:
            return
        if target == self._focus_target:
            self._sync_focus_buttons()
            return

        live_target = target in {"source", "depth", "viewport"}
        if self._focus_target is None and live_target:
            self._saved_main_splitter_sizes = self._main_splitter.sizes()
            self._saved_left_splitter_sizes = self._left_splitter.sizes()
            self._saved_top_splitter_sizes = self._top_splitter.sizes()

        # Restore both focus-capable pages first. This supports a deterministic
        # Escape/close path without recreating the Live-3D or Reconstruction GL
        # surfaces, and keeps the existing Live-3D focus contract unchanged.
        self._header.show()
        self._mode_tabs.show()
        self._reconstruction_page.set_viewport_focus_mode(False)

        self._control_row.show()
        self._top_splitter.show()
        self._original_view.show()
        self._depth_view.show()
        self._viewport_panel.show()
        self._settings_panel.show()
        self._performance_bar.show()

        if target is None:
            if self._saved_main_splitter_sizes:
                self._main_splitter.setSizes(self._saved_main_splitter_sizes)
            if self._saved_left_splitter_sizes:
                self._left_splitter.setSizes(self._saved_left_splitter_sizes)
            if self._saved_top_splitter_sizes:
                self._top_splitter.setSizes(self._saved_top_splitter_sizes)
        elif target == "reconstruction":
            self._header.hide()
            self._mode_tabs.hide()
            self._reconstruction_page.set_viewport_focus_mode(True)
        else:
            self._header.hide()
            self._mode_tabs.hide()
            self._control_row.hide()
            self._settings_panel.hide()
            self._performance_bar.hide()
            self._main_splitter.setSizes([1, 0])

            if target == "viewport":
                self._top_splitter.hide()
                self._viewport_panel.show()
                self._left_splitter.setSizes([0, 1])
            else:
                self._viewport_panel.hide()
                self._top_splitter.show()
                self._left_splitter.setSizes([1, 0])
                if target == "source":
                    self._original_view.show()
                    self._depth_view.hide()
                    self._top_splitter.setSizes([1, 0])
                else:
                    self._original_view.hide()
                    self._depth_view.show()
                    self._top_splitter.setSizes([0, 1])

        self._focus_target = target
        self._viewport_focus_mode = target == "viewport"
        self._sync_focus_buttons()

    def _sync_focus_buttons(self) -> None:
        self._original_view.set_fullscreen_active(self._focus_target == "source")
        self._depth_view.set_fullscreen_active(self._focus_target == "depth")
        self._viewport_panel.set_fullscreen_active(self._focus_target == "viewport")
        self._reconstruction_page.set_viewport_fullscreen_active(
            self._focus_target == "reconstruction"
        )

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._focus_target is not None and event.key() == Qt.Key.Key_Escape:
            self._set_focus_target(None)
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
        if self._focus_target is not None:
            self._set_focus_target(None)
        self._benchmark_controller.shutdown()
        self._snapshot_controller.shutdown()
        self._stable_video_controller.shutdown()
        self._inspection_controller.shutdown()
        self._inspection_export_controller.shutdown()
        self._inspection_page.close()
        # Stop reconstruction producers before closing the dedicated VisPy canvas
        # so no late worker observation can target a released GL surface.
        self._reconstruction_controller.shutdown()
        self._reconstruction_page.shutdown_viewport()
        self._inference_controller.shutdown()
        self._controller.shutdown()
        self._save_layout()
        super().closeEvent(event)
