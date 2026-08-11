"""Reconstruction workspace through Phase 8.8F1.

C1-C5 established the live viewer. D1-D5 added bounded offline sparse-SfM,
camera/health/coverage inspection, and semantic-object presentation. E1 binds
the active scene/object inspection target. E2 presents already-computed
measurement values. E3 presents source-authored measurement validity and
uncertainty. E4 presents authoritative inspection/QC reports. E5 adds the
reference-comparison workflow around an explicit reference identity and existing
Phase 8.7C comparison evidence without rerunning registration or tolerances. F1
adds metadata-only appearance-representation controls over persisted Gaussian/NeRF
artifacts; visualization remains deferred to F2/F3 and switching to F4.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.offline_reconstruction_controller import OfflineReconstructionRunState
from app.offline_reconstruction_visualization import (
    OfflineSparseVisualization,
    load_offline_sparse_visualization,
)
from app.offline_reconstruction_inspection import (
    build_offline_camera_inspection,
    format_offline_structure_summary,
)
from app.offline_reconstruction_quality import (
    OfflineSfmHealthPresentation,
    build_offline_sfm_health_presentation,
    unavailable_offline_sfm_health,
)
from app.offline_reconstruction_spatial import (
    OfflineSparseSpatialCoverage,
    build_offline_sparse_spatial_coverage,
)
from app.offline_reconstruction_semantics import (
    OfflineSemanticPresentation,
    semantic_presentation_from_payload,
    semantic_presentation_from_persistent_scene,
)
from app.reconstruction_diagnostics import LiveTrackingDiagnosticsAccumulator
from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    build_reconstruction_inspection_selection,
)
from app.reconstruction_measurement_presentation import (
    ReconstructionMeasurementPresentation,
    build_reconstruction_measurement_presentation,
    empty_reconstruction_measurement_text,
)
from app.reconstruction_measurement_validity_presentation import (
    ReconstructionMeasurementValidityPresentation,
    build_reconstruction_measurement_validity_presentation,
    empty_reconstruction_measurement_validity_text,
)
from app.reconstruction_inspection_qc_presentation import (
    ReconstructionInspectionQcPresentation,
    build_reconstruction_inspection_qc_presentation,
    empty_reconstruction_inspection_qc_text,
)
from app.reconstruction_reference_comparison import (
    ReconstructionReferenceComparisonPresentation,
    ReconstructionReferenceDescriptor,
    build_reconstruction_reference_comparison_presentation,
    empty_reference_comparison_text,
    load_reference_descriptor_json,
    reference_descriptor_from_payload,
)
from app.reconstruction_appearance_presentation import (
    AppearanceRepresentationEntry,
    ReconstructionAppearancePresentation,
    build_reconstruction_appearance_presentation,
    empty_appearance_representation_text,
)
from app.reconstruction_controller import (
    ReconstructionSourceSelection,
    ReconstructionWorkflow,
)
from app.reconstruction_persistence_controller import ReconstructionSceneAvailability
from app.reconstruction_status import (
    ReconstructionStatusPresentation,
    format_elapsed_seconds,
    format_live_telemetry,
    present_live_state,
    present_offline_event,
    present_offline_state,
)
from app.theme import (
    PAGE_MARGIN_BOTTOM,
    PAGE_MARGIN_H,
    PAGE_MARGIN_TOP,
    PAGE_SPACING,
    SPLITTER_HANDLE,
)
from app.widgets.icon_button import IconButton
from app.widgets.reconstruction_diagnostics_hud import ReconstructionDiagnosticsHud
from app.widgets.reconstruction_orientation_gizmo import ReconstructionOrientationGizmo
from app.widgets.viewport.vispy_canvas import ReconstructionCanvas
from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveCaptureRuntimeState,
    ReconstructionCamera,
    load_live_camera_calibration,
)


class ReconstructionPage(QWidget):
    """Professional Reconstruction workspace shell with passive state binding."""

    inspection_target_changed = pyqtSignal(object)
    reconstruction_measurement_changed = pyqtSignal(object)
    reconstruction_measurement_validity_changed = pyqtSignal(object)
    reconstruction_inspection_qc_changed = pyqtSignal(object)
    reconstruction_reference_comparison_changed = pyqtSignal(object)
    appearance_representation_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("reconstructionPage")
        self._controller = None
        self._source_buttons: dict[ReconstructionWorkflow, QPushButton] = {}
        self._live_state = LiveCaptureRuntimeState.IDLE
        self._offline_state = OfflineReconstructionRunState.IDLE
        self._live_camera: ReconstructionCamera | None = None
        self._live_calibration_path: Path | None = None
        self._live_camera_mode: str | None = None
        self._last_live_source_identity: tuple[str, str, str] | None = None
        self._last_workflow = ReconstructionWorkflow.NONE
        self._runtime_started_at: float | None = None
        self._runtime_elapsed_frozen = 0.0
        self._active_runtime_kind: str | None = None
        self._live_pose_count = 0
        self._live_keyframe_count = 0
        self._live_landmark_count: int | None = None
        self._last_runtime_error = ""
        self._scene_availability = ReconstructionSceneAvailability.NONE
        self._persistent_scene = None
        self._inspection_selection = build_reconstruction_inspection_selection()
        self._reconstruction_measurement_presentations: dict[
            tuple[str, str], ReconstructionMeasurementPresentation
        ] = {}
        self._reconstruction_measurement_presentation: ReconstructionMeasurementPresentation | None = None
        self._reconstruction_measurement_validity_presentations: dict[
            tuple[str, str], ReconstructionMeasurementValidityPresentation
        ] = {}
        self._reconstruction_measurement_validity_presentation: ReconstructionMeasurementValidityPresentation | None = None
        self._reconstruction_inspection_qc_reports: dict[str, tuple[object, str]] = {}
        self._reconstruction_inspection_qc_presentation: ReconstructionInspectionQcPresentation | None = None
        self._reconstruction_reference_descriptors: dict[str, ReconstructionReferenceDescriptor] = {}
        self._reconstruction_reference_comparison_presentation: ReconstructionReferenceComparisonPresentation | None = None
        self._appearance_presentation: ReconstructionAppearancePresentation | None = None
        self._selected_appearance_artifact_id = ""
        self._viewport_shutdown = False
        self._pending_visual_pose: np.ndarray | None = None
        self._pending_sparse_landmarks: tuple[np.ndarray, int, int] | None = None
        self._pending_keyframes: tuple[np.ndarray, tuple[str, ...], int, int] | None = None
        self._offline_visualization_path: Path | None = None
        self._offline_visualization_error = ""
        self._offline_visualization: OfflineSparseVisualization | None = None
        self._offline_camera_selection_index: int | None = None
        self._offline_sfm_health: OfflineSfmHealthPresentation | None = None
        self._offline_spatial_coverage: OfflineSparseSpatialCoverage | None = None
        self._offline_semantics: OfflineSemanticPresentation | None = None
        self._semantic_object_selection_index: int | None = None
        self._visualizer_fitted_once = False
        self._viewport_tool_buttons: dict[str, IconButton] = {}
        self._live_diagnostics = LiveTrackingDiagnosticsAccumulator()
        self._build_ui()
        self._visual_pose_timer = QTimer(self)
        self._visual_pose_timer.setSingleShot(True)
        self._visual_pose_timer.setInterval(33)
        self._visual_pose_timer.timeout.connect(self._flush_pending_visual_pose)
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.setInterval(500)
        self._telemetry_timer.timeout.connect(self._refresh_runtime_telemetry)
        self.set_live_state(LiveCaptureRuntimeState.IDLE)
        self.set_offline_state(OfflineReconstructionRunState.IDLE)
        self.set_scene_availability(ReconstructionSceneAvailability.NONE)
        self.set_checkpoint(None, "")
        self._apply_runtime_presentation(
            ReconstructionStatusPresentation(
                "Waiting for source",
                "Choose a Reconstruction workflow to begin.",
            )
        )

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            PAGE_MARGIN_H,
            PAGE_MARGIN_TOP,
            PAGE_MARGIN_H,
            PAGE_MARGIN_BOTTOM,
        )
        outer.setSpacing(PAGE_SPACING)

        outer.addWidget(self._build_header_card())

        body = QSplitter(Qt.Orientation.Horizontal)
        body.setObjectName("reconstructionBodySplitter")
        body.setHandleWidth(SPLITTER_HANDLE)
        body.setChildrenCollapsible(False)

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(PAGE_SPACING)

        top = QHBoxLayout()
        top.setSpacing(PAGE_SPACING)
        top.addWidget(self._build_source_card(), stretch=1)
        top.addWidget(self._build_runtime_card(), stretch=1)
        main_layout.addLayout(top)
        main_layout.addWidget(self._build_viewport_card(), stretch=1)

        body.addWidget(main)
        body.addWidget(self._build_settings_panel())
        body.setStretchFactor(0, 1)
        body.setStretchFactor(1, 0)
        body.setSizes([1010, 320])
        outer.addWidget(body, stretch=1)

    def _build_header_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionHeaderCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 11, 16, 11)
        layout.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Reconstruction")
        title.setObjectName("reconstructionTitle")
        subtitle = QLabel(
            "Live and offline 3D reconstruction workspace. Controller, isolation, persistence, and recovery foundations are ready."
        )
        subtitle.setObjectName("reconstructionSubtitle")
        subtitle.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text, stretch=1)

        self._readiness_badge = QLabel("CONTROLLER READY")
        self._readiness_badge.setObjectName("reconstructionReadinessBadge")
        self._readiness_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._readiness_badge)
        return card

    def _build_source_card(self) -> QFrame:
        card = self._card("SOURCE & WORKFLOW")
        layout = card.layout()

        self._source_value = self._value_label("No reconstruction source selected")
        self._workflow_value = self._value_label("Choose a workflow")
        self._source_readiness_value = self._status_value()
        self._add_status_row(layout, "Source", self._source_value)
        self._add_status_row(layout, "Workflow", self._workflow_value)
        self._add_status_row(layout, "Readiness", self._source_readiness_value)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        button_specs = (
            (ReconstructionWorkflow.LIVE_SOURCE, "Live source", self._select_live_source),
            (ReconstructionWorkflow.IMAGE_SEQUENCE, "Image sequence", self._choose_image_sequence),
            (ReconstructionWorkflow.RECORDED_VIDEO, "Recorded video", self._choose_recorded_video),
        )
        for workflow, text, callback in button_specs:
            button = QPushButton(text)
            button.setObjectName("reconstructionWorkflowButton")
            button.setCheckable(True)
            button.setEnabled(False)
            button.setToolTip("Select this Reconstruction input workflow.")
            button.clicked.connect(callback)
            self._source_buttons[workflow] = button
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return card

    def _build_runtime_card(self) -> QFrame:
        card = self._card("RUNTIME STATUS")
        layout = card.layout()

        self._live_value = self._status_value()
        self._offline_value = self._status_value()
        self._scene_value = self._status_value()
        self._stage_value = self._status_value()
        self._elapsed_value = self._value_label("—")
        self._add_status_row(layout, "Live SLAM", self._live_value)
        self._add_status_row(layout, "Offline", self._offline_value)
        self._add_status_row(layout, "Scene", self._scene_value)
        self._add_status_row(layout, "Stage", self._stage_value)

        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("reconstructionProgressBar")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("Waiting")
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setToolTip(
            "Stage-local progress when the backend exposes a meaningful finite fraction. "
            "Live tracking is intentionally shown as ongoing activity rather than a fake percentage."
        )
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_row.addWidget(self._progress_bar, stretch=1)
        self._elapsed_value.setMinimumWidth(42)
        self._elapsed_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_row.addWidget(self._elapsed_value)
        layout.addLayout(progress_row)

        self._runtime_message = QLabel("Choose a Reconstruction workflow to begin.")
        self._runtime_message.setObjectName("reconstructionRuntimeMessage")
        self._runtime_message.setWordWrap(True)
        self._runtime_message.setProperty("statusKind", "neutral")
        layout.addWidget(self._runtime_message)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._start_button = QPushButton("Start")
        self._start_button.setObjectName("reconstructionRuntimeAction")
        self._start_button.setProperty("actionRole", "primary")
        self._start_button.setToolTip("Start the selected Reconstruction workflow.")
        self._start_button.clicked.connect(self._start_runtime)
        controls.addWidget(self._start_button)

        self._pause_button = QPushButton("Pause")
        self._pause_button.setObjectName("reconstructionRuntimeAction")
        self._pause_button.setToolTip("Pause or resume a running live reconstruction session.")
        self._pause_button.clicked.connect(self._pause_or_resume_runtime)
        controls.addWidget(self._pause_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setObjectName("reconstructionRuntimeAction")
        self._stop_button.setToolTip("Stop live reconstruction or cancel an offline run.")
        self._stop_button.clicked.connect(self._stop_runtime)
        controls.addWidget(self._stop_button)
        layout.addLayout(controls)
        return card

    def _build_viewport_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionViewportCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("RECONSTRUCTION VIEWPORT")
        title.setObjectName("reconstructionSectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._viewport_state = QLabel("Awaiting scene")
        self._viewport_state.setObjectName("reconstructionViewportState")
        header.addWidget(self._viewport_state)
        layout.addLayout(header)

        host = QFrame()
        host.setObjectName("reconstructionViewportCanvasHost")
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._reconstruction_canvas = ReconstructionCanvas(self)
        grid.addWidget(self._reconstruction_canvas.native, 0, 0)

        self._viewport_placeholder = QWidget()
        self._viewport_placeholder.setObjectName("reconstructionViewportPlaceholder")
        self._viewport_placeholder.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        center = QVBoxLayout(self._viewport_placeholder)
        center.setContentsMargins(24, 24, 24, 24)
        center.setSpacing(6)
        center.addStretch(1)
        glyph = QLabel("◇")
        glyph.setObjectName("reconstructionViewportGlyph")
        glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(glyph)
        headline = QLabel("Reconstruction viewport")
        headline.setObjectName("reconstructionViewportHeadline")
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(headline)
        detail = QLabel(
            "World origin, current camera pose, trajectory, sparse SLAM landmarks, and historical keyframe structure are active. Dense/canonical geometry and neural appearance are introduced progressively in later Phase 8.8 steps."
        )
        detail.setObjectName("reconstructionViewportHint")
        detail.setWordWrap(True)
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center.addWidget(detail)
        center.addStretch(1)
        grid.addWidget(self._viewport_placeholder, 0, 0)

        tools = QFrame()
        tools.setObjectName("reconstructionViewportTools")
        tool_row = QHBoxLayout(tools)
        tool_row.setContentsMargins(5, 4, 5, 4)
        tool_row.setSpacing(2)

        group = QButtonGroup(tools)
        group.setExclusive(True)
        for icon, mode, tip in (
            ("orbit", "orbit", "Orbit reconstruction view"),
            ("pan", "pan", "Pan reconstruction view"),
            ("zoom", "zoom", "Zoom reconstruction view"),
        ):
            button = IconButton(icon, size=28, icon_size=15, tooltip=tip, checkable=True)
            button.clicked.connect(lambda _checked, value=mode: self._set_viewport_tool(value))
            group.addButton(button)
            self._viewport_tool_buttons[mode] = button
            tool_row.addWidget(button)
        self._viewport_tool_buttons["orbit"].setChecked(True)

        fit_button = IconButton("fit", size=28, icon_size=15, tooltip="Fit known camera path to view")
        fit_button.clicked.connect(self._reconstruction_canvas.fit_scene)
        tool_row.addWidget(fit_button)
        focus_button = IconButton("camera", size=28, icon_size=15, tooltip="Focus current camera")
        focus_button.clicked.connect(self._reconstruction_canvas.focus_current_camera)
        tool_row.addWidget(focus_button)
        reset_button = IconButton("reset", size=28, icon_size=15, tooltip="Reset reconstruction view")
        reset_button.clicked.connect(self._reconstruction_canvas.reset_view)
        tool_row.addWidget(reset_button)
        grid_button = IconButton("grid", size=28, icon_size=15, tooltip="Show or hide world grid", checkable=True)
        grid_button.setChecked(True)
        grid_button.toggled.connect(self._reconstruction_canvas.set_grid_visible)
        self._viewport_tool_buttons["grid"] = grid_button
        tool_row.addWidget(grid_button)
        trajectory_button = IconButton(
            "trajectory",
            size=28,
            icon_size=15,
            tooltip="Show or hide live camera trajectory",
            checkable=True,
        )
        trajectory_button.setChecked(True)
        trajectory_button.toggled.connect(self._reconstruction_canvas.set_trajectory_visible)
        self._viewport_tool_buttons["trajectory"] = trajectory_button
        tool_row.addWidget(trajectory_button)
        landmarks_button = IconButton(
            "landmarks",
            size=28,
            icon_size=15,
            tooltip="Show or hide live sparse SLAM landmarks",
            checkable=True,
        )
        landmarks_button.setChecked(True)
        landmarks_button.toggled.connect(self._reconstruction_canvas.set_sparse_landmarks_visible)
        self._viewport_tool_buttons["landmarks"] = landmarks_button
        tool_row.addWidget(landmarks_button)
        keyframes_button = IconButton(
            "keyframes",
            size=28,
            icon_size=15,
            tooltip="Show or hide historical SLAM keyframe cameras",
            checkable=True,
        )
        keyframes_button.setChecked(True)
        keyframes_button.toggled.connect(self._reconstruction_canvas.set_keyframes_visible)
        self._viewport_tool_buttons["keyframes"] = keyframes_button
        tool_row.addWidget(keyframes_button)
        envelope_button = IconButton(
            "cube",
            size=28,
            icon_size=15,
            tooltip="Show or hide offline sparse presentation envelope",
            checkable=True,
        )
        envelope_button.setChecked(True)
        envelope_button.toggled.connect(self._reconstruction_canvas.set_offline_sparse_envelope_visible)
        self._viewport_tool_buttons["offline_envelope"] = envelope_button
        tool_row.addWidget(envelope_button)
        semantic_button = IconButton(
            "semantics",
            size=28,
            icon_size=15,
            tooltip="Show or hide semantic object context",
            checkable=True,
        )
        semantic_button.setChecked(True)
        semantic_button.setEnabled(False)
        semantic_button.toggled.connect(self._reconstruction_canvas.set_semantic_objects_visible)
        self._viewport_tool_buttons["semantic_objects"] = semantic_button
        tool_row.addWidget(semantic_button)
        diagnostics_button = IconButton(
            "diagnostics",
            size=28,
            icon_size=15,
            tooltip="Show or hide live tracking diagnostics",
            checkable=True,
        )
        diagnostics_button.setChecked(True)
        self._viewport_tool_buttons["diagnostics"] = diagnostics_button
        tool_row.addWidget(diagnostics_button)

        grid.addWidget(
            tools,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self._diagnostics_hud = ReconstructionDiagnosticsHud(host)
        diagnostics_button.toggled.connect(self._sync_diagnostics_visibility)
        self._sync_diagnostics_visibility()
        grid.addWidget(
            self._diagnostics_hud,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )

        safety = QLabel("VIEWER Z-UP • ARBITRARY GAUGE")
        safety.setObjectName("reconstructionViewportSafety")
        safety.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        grid.addWidget(
            safety,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )

        self._orientation_gizmo = ReconstructionOrientationGizmo(
            self._reconstruction_canvas.orientation_gizmo_axes,
            host,
        )
        grid.addWidget(
            self._orientation_gizmo,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        layout.addWidget(host, stretch=1)
        return card

    def _set_viewport_tool(self, mode: str) -> None:
        self._reconstruction_canvas.set_interaction_mode(mode)

    def _sync_diagnostics_visibility(self, _checked: bool | None = None) -> None:
        if not hasattr(self, "_diagnostics_hud"):
            return
        button = self._viewport_tool_buttons.get("diagnostics")
        requested = bool(button is not None and button.isChecked())
        live_context = (
            self._last_workflow is ReconstructionWorkflow.LIVE_SOURCE
            or self._live_state in {
                LiveCaptureRuntimeState.STARTING,
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
                LiveCaptureRuntimeState.RECOVERING,
                LiveCaptureRuntimeState.STOPPING,
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            }
        )
        self._diagnostics_hud.setVisible(requested and live_context)

    def _apply_offline_sfm_health(self, presentation: OfflineSfmHealthPresentation) -> None:
        self._offline_sfm_health = presentation
        self._offline_quality_status_value.setText(presentation.status_label)
        self._offline_quality_status_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._offline_quality_status_value)
        self._offline_quality_status_value.setToolTip(presentation.gate_tooltip)
        self._offline_quality_registration_value.setText(presentation.registration_text)
        self._offline_quality_reprojection_value.setText(presentation.reprojection_text)
        self._offline_quality_support_value.setText(presentation.support_text)
        self._offline_quality_gate_value.setText(presentation.gate_text)
        self._offline_quality_gate_value.setToolTip(presentation.gate_tooltip)
        self._offline_quality_registration_value.setToolTip(
            "Registered sparse-SfM cameras versus input keyframes. This is coverage, not metric accuracy."
        )
        self._offline_quality_reprojection_value.setToolTip(
            "Mean sparse feature reprojection residual in image pixels, as reported by the qualified SfM backend."
        )
        self._offline_quality_support_value.setToolTip(
            "Mean landmark track support and sparse landmark density. Values are operational SfM health telemetry."
        )
        self._offline_inspection_group.setVisible(True)

    def _reset_offline_sfm_health(self) -> None:
        self._offline_sfm_health = None
        if not hasattr(self, "_offline_quality_status_value"):
            return
        self._offline_quality_status_value.setText("—")
        self._offline_quality_status_value.setProperty("statusKind", "")
        self._refresh_style(self._offline_quality_status_value)
        self._offline_quality_status_value.setToolTip("")
        self._offline_quality_registration_value.setText("—")
        self._offline_quality_reprojection_value.setText("—")
        self._offline_quality_support_value.setText("—")
        self._offline_quality_gate_value.setText("Operational SfM health only • not metrology")
        self._offline_quality_gate_value.setToolTip(
            "D3 never promotes sparse-SfM health to metric accuracy or certified metrology."
        )

    def _apply_offline_spatial_coverage(self, presentation: OfflineSparseSpatialCoverage) -> None:
        self._offline_spatial_coverage = presentation
        self._offline_spatial_envelope_value.setText(presentation.envelope_text)
        self._offline_spatial_envelope_value.setToolTip(presentation.envelope_tooltip)
        self._offline_spatial_camera_span_value.setText(presentation.camera_span_text)
        self._offline_spatial_camera_span_value.setToolTip(
            "Axis-aligned span of registered camera centers in arbitrary reconstruction gauge units."
        )
        self._offline_spatial_standoff_value.setText(presentation.standoff_text)
        self._offline_spatial_standoff_value.setToolTip(
            "Camera-center distance to the robust sparse-SfM sample center. Values are gauge units, not metres."
        )
        self._offline_spatial_basis_value.setText(presentation.basis_text)
        self._offline_spatial_basis_value.setToolTip(
            "D4 uses the bounded D1 visualization sample only. The envelope is a presentation aid, not canonical geometry."
        )
        if presentation.has_envelope:
            self._reconstruction_canvas.set_offline_sparse_envelope(
                presentation.robust_min_reconstruction,
                presentation.robust_max_reconstruction,
            )
        else:
            self._reconstruction_canvas.clear_offline_sparse_envelope()
        self._offline_inspection_group.setVisible(True)

    def _reset_offline_spatial_coverage(self) -> None:
        self._offline_spatial_coverage = None
        if hasattr(self, "_offline_spatial_envelope_value"):
            self._offline_spatial_envelope_value.setText("—")
            self._offline_spatial_envelope_value.setToolTip("")
            self._offline_spatial_camera_span_value.setText("—")
            self._offline_spatial_standoff_value.setText("—")
            self._offline_spatial_basis_value.setText(
                "Sparse presentation envelope only • not object dimensions"
            )
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_offline_sparse_envelope()

    def _activate_offline_inspection(self, visual: OfflineSparseVisualization) -> None:
        self._offline_visualization = visual
        self._offline_camera_selection_index = None
        self._offline_camera_combo.blockSignals(True)
        self._offline_camera_combo.clear()
        self._offline_camera_combo.addItem("Select camera…", -1)
        for index, camera_id in enumerate(visual.camera_ids):
            self._offline_camera_combo.addItem(f"{index + 1:02d} • {camera_id}", index)
        self._offline_camera_combo.setCurrentIndex(0)
        self._offline_camera_combo.blockSignals(False)
        self._offline_camera_combo.setEnabled(bool(visual.camera_ids))
        self._offline_structure_value.setText(format_offline_structure_summary(visual))
        self._offline_structure_value.setToolTip(
            "Bounded D1 visualization sample versus authoritative full sparse-SfM counts."
        )
        self._offline_camera_position_value.setText("—")
        self._offline_camera_delta_value.setText("—")
        self._offline_inspection_group.setVisible(True)
        self._refresh_offline_camera_navigation()
        self._refresh_inspection_selection()

    def _reset_offline_inspection(self, *, hide: bool = True) -> None:
        self._offline_visualization = None
        self._offline_camera_selection_index = None
        if hasattr(self, "_offline_camera_combo"):
            self._offline_camera_combo.blockSignals(True)
            self._offline_camera_combo.clear()
            self._offline_camera_combo.addItem("Select camera…", -1)
            self._offline_camera_combo.setCurrentIndex(0)
            self._offline_camera_combo.setEnabled(False)
            self._offline_camera_combo.blockSignals(False)
            self._offline_camera_position_value.setText("—")
            self._offline_camera_delta_value.setText("—")
            self._offline_structure_value.setText("No offline sparse SfM")
            self._offline_inspection_group.setVisible(
                (not hide)
                or self._offline_sfm_health is not None
                or self._offline_spatial_coverage is not None
            )
            self._refresh_offline_camera_navigation()
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_offline_camera_selection()
        self._refresh_inspection_selection()

    def _on_offline_camera_selected(self, combo_index: int) -> None:
        visual = self._offline_visualization
        if visual is None:
            return
        item_data = self._offline_camera_combo.itemData(combo_index)
        camera_index = -1 if item_data is None else int(item_data)
        if camera_index < 0:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._offline_camera_position_value.setText("—")
            self._offline_camera_delta_value.setText("—")
            self._refresh_offline_camera_navigation()
            return
        try:
            inspection = build_offline_camera_inspection(visual, camera_index)
            self._reconstruction_canvas.select_offline_camera(camera_index)
        except (IndexError, TypeError, ValueError) as exc:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._offline_camera_position_value.setText("Unavailable")
            self._offline_camera_delta_value.setText("Unavailable")
            self._offline_camera_position_value.setToolTip(str(exc))
            self._refresh_offline_camera_navigation()
            return
        self._offline_camera_selection_index = camera_index
        self._offline_camera_position_value.setText(inspection.position_text)
        self._offline_camera_position_value.setToolTip(
            "Camera translation in the arbitrary reconstruction gauge; values are not metres."
        )
        self._offline_camera_delta_value.setText(inspection.delta_previous_text)
        self._offline_camera_delta_value.setToolTip(
            "Euclidean displacement from the previous registered camera in arbitrary SfM gauge units."
        )
        self._viewport_state.setToolTip(
            f"Selected registered camera {inspection.ordinal_text}: {inspection.camera_id}"
        )
        self._refresh_offline_camera_navigation()

    def _refresh_offline_camera_navigation(self) -> None:
        visual = self._offline_visualization
        selected = self._offline_camera_selection_index
        count = 0 if visual is None else len(visual.camera_ids)
        has_selection = selected is not None and 0 <= selected < count
        self._offline_focus_camera_button.setEnabled(has_selection)
        self._offline_clear_camera_button.setEnabled(has_selection)
        self._offline_prev_camera_button.setEnabled(has_selection and selected > 0)
        self._offline_next_camera_button.setEnabled(has_selection and selected + 1 < count)

    def _select_previous_offline_camera(self) -> None:
        if self._offline_camera_selection_index is None:
            return
        target = self._offline_camera_selection_index - 1
        if target >= 0:
            self._offline_camera_combo.setCurrentIndex(target + 1)

    def _select_next_offline_camera(self) -> None:
        visual = self._offline_visualization
        if visual is None or self._offline_camera_selection_index is None:
            return
        target = self._offline_camera_selection_index + 1
        if target < len(visual.camera_ids):
            self._offline_camera_combo.setCurrentIndex(target + 1)

    def _focus_selected_offline_camera(self) -> None:
        if self._offline_camera_selection_index is None:
            return
        self._reconstruction_canvas.focus_selected_offline_camera()

    def _clear_offline_camera_selection(self) -> None:
        if self._offline_camera_combo.count():
            self._offline_camera_combo.setCurrentIndex(0)
        else:
            self._offline_camera_selection_index = None
            self._reconstruction_canvas.clear_offline_camera_selection()
            self._refresh_offline_camera_navigation()

    def set_semantic_geometry(self, payload: object, *, source: str = "semantic_geometry") -> None:
        """Present an already-built semantic result without recomputing semantics."""

        presentation = semantic_presentation_from_payload(payload, source=source)
        self._apply_semantic_presentation(presentation)

    def _apply_semantic_presentation(self, presentation: OfflineSemanticPresentation) -> None:
        if not isinstance(presentation, OfflineSemanticPresentation):
            raise TypeError("presentation must be OfflineSemanticPresentation.")
        self._offline_semantics = presentation
        self._semantic_object_selection_index = None
        self._semantic_object_combo.blockSignals(True)
        self._semantic_object_combo.clear()
        self._semantic_object_combo.addItem("Select object…", -1)
        for index, item in enumerate(presentation.objects):
            self._semantic_object_combo.addItem(
                f"{index + 1:02d} • {item.display_name}", index
            )
        self._semantic_object_combo.setCurrentIndex(0)
        self._semantic_object_combo.setEnabled(bool(presentation.objects))
        self._semantic_object_combo.blockSignals(False)
        self._semantic_structure_value.setText(presentation.structure_text)
        self._semantic_structure_value.setToolTip(
            "Bounded semantic-object presentation from existing Phase 8.7 semantic context."
        )
        self._semantic_role_value.setText("—")
        self._semantic_support_value.setText("—")
        self._semantic_quality_value.setText("—")
        if presentation.metric_geometry_status == "arbitrary":
            self._semantic_scale_value.setText("Arbitrary reconstruction gauge")
        elif presentation.metric_geometry_status == "provisional_metric":
            self._semantic_scale_value.setText("Provisional metric geometry • not certified")
        elif presentation.metric_geometry_status == "validated_metric":
            self._semantic_scale_value.setText("Validated metric geometry • metrology not certified")
        else:
            self._semantic_scale_value.setText("Scale semantics unavailable")
        self._semantic_authority_value.setText(presentation.authority_text)
        self._semantic_authority_value.setToolTip(
            "Semantic context labels geometry; inspection/QC and measurement retain their own authority."
        )
        if presentation.objects:
            self._viewport_placeholder.setVisible(False)
            lower = np.asarray(
                [item.bounds_min_reconstruction for item in presentation.objects], dtype=np.float64
            )
            upper = np.asarray(
                [item.bounds_max_reconstruction for item in presentation.objects], dtype=np.float64
            )
            centroids = np.asarray(
                [item.centroid_reconstruction for item in presentation.objects], dtype=np.float64
            )
            self._reconstruction_canvas.set_semantic_objects(
                bounds_min_reconstruction=lower,
                bounds_max_reconstruction=upper,
                centroids_reconstruction=centroids,
                object_ids=tuple(item.object_id for item in presentation.objects),
                total_count=presentation.total_object_count,
            )
        else:
            self._reconstruction_canvas.clear_semantic_objects()
        semantic_button = self._viewport_tool_buttons.get("semantic_objects")
        if semantic_button is not None:
            semantic_button.setEnabled(bool(presentation.objects))
            self._reconstruction_canvas.set_semantic_objects_visible(semantic_button.isChecked())
        self._semantic_objects_group.setVisible(True)
        self._refresh_semantic_navigation()
        self._refresh_inspection_selection()
        self._refresh_viewport_state()

    def _reset_semantic_presentation(self, *, hide: bool = True) -> None:
        self._offline_semantics = None
        self._semantic_object_selection_index = None
        if hasattr(self, "_semantic_object_combo"):
            self._semantic_object_combo.blockSignals(True)
            self._semantic_object_combo.clear()
            self._semantic_object_combo.addItem("Select object…", -1)
            self._semantic_object_combo.setCurrentIndex(0)
            self._semantic_object_combo.setEnabled(False)
            self._semantic_object_combo.blockSignals(False)
            self._semantic_structure_value.setText("No semantic context")
            self._semantic_role_value.setText("—")
            self._semantic_support_value.setText("—")
            self._semantic_quality_value.setText("—")
            self._semantic_scale_value.setText("—")
            self._semantic_authority_value.setText(
                "Semantic context only • not inspection/QC decision"
            )
            self._semantic_objects_group.setVisible(not hide)
            self._refresh_semantic_navigation()
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.clear_semantic_objects()
        button = getattr(self, "_viewport_tool_buttons", {}).get("semantic_objects")
        if button is not None:
            button.setEnabled(False)
        self._refresh_inspection_selection()

    def _on_semantic_object_selected(self, combo_index: int) -> None:
        presentation = self._offline_semantics
        if presentation is None:
            return
        data = self._semantic_object_combo.itemData(combo_index)
        index = -1 if data is None else int(data)
        if index < 0:
            self._semantic_object_selection_index = None
            self._reconstruction_canvas.clear_semantic_object_selection()
            self._semantic_role_value.setText("—")
            self._semantic_support_value.setText("—")
            self._semantic_quality_value.setText("—")
            if presentation.metric_geometry_status == "arbitrary":
                self._semantic_scale_value.setText("Arbitrary reconstruction gauge")
            elif presentation.metric_geometry_status == "provisional_metric":
                self._semantic_scale_value.setText("Provisional metric geometry • not certified")
            elif presentation.metric_geometry_status == "validated_metric":
                self._semantic_scale_value.setText("Validated metric geometry • metrology not certified")
            else:
                self._semantic_scale_value.setText("Scale semantics unavailable")
            self._semantic_authority_value.setText(presentation.authority_text)
            self._refresh_semantic_navigation()
            self._refresh_inspection_selection()
            return
        if index >= len(presentation.objects):
            return
        item = presentation.objects[index]
        self._semantic_object_selection_index = index
        self._reconstruction_canvas.select_semantic_object(index)
        self._semantic_role_value.setText(item.role_text)
        self._semantic_support_value.setText(item.support_text)
        self._semantic_quality_value.setText(item.quality_text)
        self._semantic_scale_value.setText(item.scale_text)
        self._semantic_authority_value.setText(item.authority_text)
        self._semantic_quality_value.setToolTip(
            "Semantic/model confidence and mesh-vertex quality are context signals, not measurement accuracy."
        )
        self._viewport_state.setToolTip(
            f"Selected semantic object {item.object_id}: {item.class_name} ({item.semantic_role})."
        )
        self._refresh_semantic_navigation()
        self._refresh_inspection_selection()

    def _refresh_semantic_navigation(self) -> None:
        presentation = self._offline_semantics
        selected = self._semantic_object_selection_index
        count = 0 if presentation is None else len(presentation.objects)
        valid = selected is not None and 0 <= selected < count
        if hasattr(self, "_semantic_focus_button"):
            self._semantic_focus_button.setEnabled(valid)
            self._semantic_clear_button.setEnabled(valid)
            self._semantic_prev_button.setEnabled(valid and selected > 0)
            self._semantic_next_button.setEnabled(valid and selected + 1 < count)

    def _select_previous_semantic_object(self) -> None:
        if self._semantic_object_selection_index is None:
            return
        target = self._semantic_object_selection_index - 1
        if target >= 0:
            self._semantic_object_combo.setCurrentIndex(target + 1)

    def _select_next_semantic_object(self) -> None:
        presentation = self._offline_semantics
        if presentation is None or self._semantic_object_selection_index is None:
            return
        target = self._semantic_object_selection_index + 1
        if target < len(presentation.objects):
            self._semantic_object_combo.setCurrentIndex(target + 1)

    def _focus_selected_semantic_object(self) -> None:
        if self._semantic_object_selection_index is None:
            return
        self._reconstruction_canvas.focus_selected_semantic_object()

    def _clear_semantic_object_selection(self) -> None:
        if self._semantic_object_combo.count():
            self._semantic_object_combo.setCurrentIndex(0)
        else:
            self._semantic_object_selection_index = None
            self._reconstruction_canvas.clear_semantic_object_selection()
            self._refresh_semantic_navigation()

    @property
    def inspection_selection(self) -> ReconstructionInspectionSelection:
        return self._inspection_selection

    def _refresh_inspection_selection(self) -> None:
        """Bind the active scene + existing D5 selection into the E1 target."""

        scene = self._persistent_scene
        scene_id = str(getattr(scene, "scene_id", "")).strip() if scene is not None else ""
        session_id = str(getattr(scene, "session_id", "")).strip() if scene is not None else ""
        scene_source = "persistent_scene" if scene is not None else ""
        canonical = bool(
            getattr(scene, "canonical_geometry_authoritative", False)
        ) if scene is not None else False

        if not session_id and self._offline_semantics is not None:
            session_id = self._offline_semantics.session_id
            scene_source = self._offline_semantics.source
        if not session_id and self._offline_visualization is not None:
            session_id = self._offline_visualization.session_id
            scene_source = "offline_sparse_reconstruction"

        selection = build_reconstruction_inspection_selection(
            scene_id=scene_id,
            session_id=session_id,
            scene_source=scene_source,
            scene_availability=getattr(self._scene_availability, "value", str(self._scene_availability)),
            canonical_geometry_authoritative=canonical,
            semantic_presentation=self._offline_semantics,
            selected_object_index=self._semantic_object_selection_index,
        )
        changed = selection != self._inspection_selection
        self._inspection_selection = selection

        if hasattr(self, "_inspection_scope_value"):
            self._inspection_scope_value.setText(selection.scope_text)
            self._inspection_scene_value.setText(selection.scene_text)
            self._inspection_scene_value.setToolTip(selection.scene_tooltip)
            self._inspection_object_value.setText(selection.object_text)
            self._inspection_object_value.setToolTip(selection.object_tooltip)
            self._inspection_geometry_value.setText(selection.geometry_text)
            self._inspection_state_value.setText(selection.state_text)
            self._inspection_authority_value.setText(selection.authority_text)

        self._refresh_reconstruction_measurement_presentation()
        if changed:
            self.inspection_target_changed.emit(selection)

    @property
    def reconstruction_measurement_presentation(self) -> ReconstructionMeasurementPresentation | None:
        return self._reconstruction_measurement_presentation

    def set_reconstruction_measurement_evidence(
        self,
        measurement_object: object,
        *,
        source: str = "phase7_5_measurement",
    ) -> None:
        """Attach already-computed measurement evidence to the active E1 object.

        The caller owns correspondence between the selected persistent semantic
        object and the Phase 7.5 measurement observation. E2 never guesses that
        association from class names, roles, or geometry.
        """

        presentation = build_reconstruction_measurement_presentation(
            self._inspection_selection,
            measurement_object,
            source=source,
        )
        key = (presentation.session_id, presentation.target_object_id)
        self._reconstruction_measurement_presentations[key] = presentation
        # Replacing E2 evidence invalidates any E3 qualification previously
        # attached to this exact target/provenance. The caller must reattach
        # source-authored validity evidence explicitly.
        self._reconstruction_measurement_validity_presentations.pop(key, None)
        self._refresh_reconstruction_measurement_presentation()

    def clear_reconstruction_measurement_evidence(self) -> None:
        self._reconstruction_measurement_presentations.clear()
        self._reconstruction_measurement_validity_presentations.clear()
        self._refresh_reconstruction_measurement_presentation()

    def _refresh_reconstruction_measurement_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        presentation = (
            self._reconstruction_measurement_presentations.get(key)
            if key is not None
            else None
        )
        changed = presentation != self._reconstruction_measurement_presentation
        self._reconstruction_measurement_presentation = presentation

        if not hasattr(self, "_measurement_target_value"):
            return
        if presentation is None:
            target, source, values, state, authority = empty_reconstruction_measurement_text(selection)
            self._measurement_target_value.setText(target)
            self._measurement_source_value.setText(source)
            self._measurement_source_value.setToolTip("")
            self._measurement_values_value.setText(values)
            self._measurement_state_value.setText(state)
            self._measurement_authority_value.setText(authority)
        else:
            self._measurement_target_value.setText(presentation.target_label)
            self._measurement_source_value.setText(presentation.source)
            self._measurement_source_value.setToolTip(presentation.provenance_tooltip)
            self._measurement_values_value.setText(presentation.values_text)
            self._measurement_state_value.setText(presentation.state_text)
            self._measurement_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_measurement_changed.emit(presentation)
        self._refresh_reconstruction_measurement_validity_presentation()

    @property
    def reconstruction_measurement_validity_presentation(self) -> ReconstructionMeasurementValidityPresentation | None:
        return self._reconstruction_measurement_validity_presentation

    def set_reconstruction_measurement_validity_evidence(
        self,
        measurement_object: object,
        *,
        source: str = "phase7_5_measurement",
        context: object | None = None,
    ) -> None:
        """Attach E3 qualification to the exact E2-bound measurement evidence.

        E3 requires an existing E2 presentation for the same target and exact
        source observation. It never guesses correspondence, computes new
        uncertainty/reliability, runs QC, or changes reconstruction scale.
        """

        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        e2 = self._reconstruction_measurement_presentations.get(key) if key is not None else None
        if e2 is None:
            raise ValueError("E3 validity evidence requires existing E2 measurement evidence for the active target.")
        presentation = build_reconstruction_measurement_validity_presentation(
            selection,
            measurement_object,
            source=source,
            context=context,
        )
        if presentation.source != e2.source:
            raise ValueError("E3 validity source must match the exact E2 measurement source.")
        if presentation.source_observation_id != e2.source_observation_id:
            raise ValueError("E3 validity observation must match the exact E2 measurement observation.")
        self._reconstruction_measurement_validity_presentations[key] = presentation
        self._refresh_reconstruction_measurement_validity_presentation()

    def clear_reconstruction_measurement_validity_evidence(self) -> None:
        self._reconstruction_measurement_validity_presentations.clear()
        self._refresh_reconstruction_measurement_validity_presentation()

    def _refresh_reconstruction_measurement_validity_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = selection.session_id or selection.scene_id
        key = (session_key, selection.object_id) if selection.has_object else None
        e2 = self._reconstruction_measurement_presentations.get(key) if key is not None else None
        presentation = (
            self._reconstruction_measurement_validity_presentations.get(key)
            if key is not None and e2 is not None
            else None
        )
        changed = presentation != self._reconstruction_measurement_validity_presentation
        self._reconstruction_measurement_validity_presentation = presentation

        if not hasattr(self, "_measurement_validity_qualified_value"):
            return
        if presentation is None:
            qualified, validity, reliability, uncertainty, scale, calibration, support, authority = (
                empty_reconstruction_measurement_validity_text(
                    selection,
                    has_e2_evidence=e2 is not None,
                )
            )
            self._measurement_validity_qualified_value.setText(qualified)
            self._measurement_validity_qualified_value.setToolTip("")
            self._measurement_validity_status_value.setText(validity)
            self._measurement_validity_reliability_value.setText(reliability)
            self._measurement_validity_uncertainty_value.setText(uncertainty)
            self._measurement_validity_scale_value.setText(scale)
            self._measurement_validity_calibration_value.setText(calibration)
            self._measurement_validity_support_value.setText(support)
            self._measurement_validity_reasons_value.setText("—")
            self._measurement_validity_authority_value.setText(authority)
        else:
            self._measurement_validity_qualified_value.setText(presentation.qualified_values_text)
            self._measurement_validity_qualified_value.setToolTip(presentation.provenance_tooltip)
            self._measurement_validity_status_value.setText(presentation.validity_text)
            self._measurement_validity_reliability_value.setText(presentation.reliability_text)
            self._measurement_validity_uncertainty_value.setText(presentation.uncertainty_text)
            self._measurement_validity_scale_value.setText(presentation.scale_text)
            self._measurement_validity_calibration_value.setText(presentation.calibration_text)
            self._measurement_validity_support_value.setText(presentation.support_text)
            self._measurement_validity_reasons_value.setText(presentation.reasons_text)
            self._measurement_validity_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_measurement_validity_changed.emit(presentation)
        self._refresh_reconstruction_inspection_qc_presentation()

    @property
    def reconstruction_inspection_qc_presentation(self) -> ReconstructionInspectionQcPresentation | None:
        return self._reconstruction_inspection_qc_presentation

    def set_reconstruction_inspection_qc_evidence(
        self,
        inspection_report: object,
        *,
        source: str = "phase8_7c.inspection_report",
    ) -> None:
        """Attach an existing authoritative inspection/QC report to the active scene.

        E4 consumes the serialized Phase 8.7C contract only. It does not run
        registration, apply tolerance rules, infer PASS/FAIL from semantic or
        measurement confidence, or load/select references.
        """

        presentation = build_reconstruction_inspection_qc_presentation(
            self._inspection_selection, inspection_report, source=source
        )
        self._reconstruction_inspection_qc_reports[presentation.session_id] = (
            inspection_report, str(source).strip()
        )
        self._refresh_reconstruction_inspection_qc_presentation()

    def clear_reconstruction_inspection_qc_evidence(self) -> None:
        self._reconstruction_inspection_qc_reports.clear()
        self._refresh_reconstruction_inspection_qc_presentation()

    def _refresh_reconstruction_inspection_qc_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = (selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        attached = self._reconstruction_inspection_qc_reports.get(session_key) if session_key else None
        presentation = None
        if attached is not None:
            report, source = attached
            presentation = build_reconstruction_inspection_qc_presentation(
                selection, report, source=source
            )
        changed = presentation != self._reconstruction_inspection_qc_presentation
        self._reconstruction_inspection_qc_presentation = presentation

        if not hasattr(self, "_inspection_qc_target_value"):
            return
        if presentation is None:
            target, decision, findings, coverage, registration, comparison, authority = (
                empty_reconstruction_inspection_qc_text(selection)
            )
            self._inspection_qc_target_value.setText(target)
            self._inspection_qc_target_value.setToolTip("")
            self._inspection_qc_decision_value.setText(decision)
            self._inspection_qc_findings_value.setText(findings)
            self._inspection_qc_coverage_value.setText(coverage)
            self._inspection_qc_decisions_value.setText("—")
            self._inspection_qc_registration_value.setText(registration)
            self._inspection_qc_comparison_value.setText(comparison)
            self._inspection_qc_authority_value.setText(authority)
        else:
            self._inspection_qc_target_value.setText(presentation.target_label)
            self._inspection_qc_target_value.setToolTip(presentation.provenance_tooltip)
            self._inspection_qc_decision_value.setText(presentation.decision_text)
            self._inspection_qc_findings_value.setText(presentation.findings_text)
            self._inspection_qc_coverage_value.setText(presentation.coverage_text)
            self._inspection_qc_decisions_value.setText(presentation.decisions_text)
            self._inspection_qc_registration_value.setText(presentation.registration_text)
            self._inspection_qc_comparison_value.setText(presentation.comparison_text)
            self._inspection_qc_authority_value.setText(presentation.authority_text)

        if changed:
            self.reconstruction_inspection_qc_changed.emit(presentation)
        self._refresh_reconstruction_reference_comparison_presentation()

    @property
    def reconstruction_reference_comparison_presentation(self) -> ReconstructionReferenceComparisonPresentation | None:
        return self._reconstruction_reference_comparison_presentation

    def set_reconstruction_reference_comparison_reference(
        self,
        reference: object,
        *,
        source: str = "phase8_7c.reference_assembly",
    ) -> None:
        """Attach an explicit E5 reference identity to the active E1 scene.

        The reference is only validated/presented here. E5 does not build a
        reference assembly, parse CAD geometry, run registration, or apply
        tolerances. Existing Phase 8.7C report evidence is paired only when its
        session and reference_id match exactly.
        """

        selection = self._inspection_selection
        if not selection.has_scene:
            raise ValueError("E5 reference comparison requires an active reconstruction scene.")
        session_key = str(selection.session_id or selection.scene_id).strip()
        descriptor = (
            reference
            if isinstance(reference, ReconstructionReferenceDescriptor)
            else reference_descriptor_from_payload(reference, provenance=source)
        )
        self._reconstruction_reference_descriptors[session_key] = descriptor
        self._refresh_reconstruction_reference_comparison_presentation()

    def load_reconstruction_reference_comparison_json(self, path: str | Path) -> None:
        """Attach metadata from an existing Phase 8.7C ReferenceAssembly JSON artifact."""

        selection = self._inspection_selection
        if not selection.has_scene:
            raise ValueError("E5 reference comparison requires an active reconstruction scene.")
        descriptor = load_reference_descriptor_json(path)
        session_key = str(selection.session_id or selection.scene_id).strip()
        self._reconstruction_reference_descriptors[session_key] = descriptor
        self._refresh_reconstruction_reference_comparison_presentation()

    def clear_reconstruction_reference_comparison(self) -> None:
        """Clear all E5 reference bindings without deleting E4 source reports."""

        self._reconstruction_reference_descriptors.clear()
        self._refresh_reconstruction_reference_comparison_presentation()

    def _choose_reconstruction_reference_json(self) -> None:
        selection = self._inspection_selection
        if not selection.has_scene:
            QMessageBox.information(
                self,
                "Reference Comparison",
                "Select or load a reconstruction scene before attaching a reference.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Phase 8.7C Reference Assembly",
            str(Path.home()),
            "Reference JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            self.load_reconstruction_reference_comparison_json(path)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Reference Comparison", str(exc))

    def _clear_active_reconstruction_reference(self) -> None:
        selection = self._inspection_selection
        session_key = str(selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        if session_key:
            self._reconstruction_reference_descriptors.pop(session_key, None)
        self._refresh_reconstruction_reference_comparison_presentation()

    def _refresh_reconstruction_reference_comparison_presentation(self) -> None:
        selection = self._inspection_selection
        session_key = str(selection.session_id or selection.scene_id).strip() if selection.has_scene else ""
        reference = self._reconstruction_reference_descriptors.get(session_key) if session_key else None
        attached_report = self._reconstruction_inspection_qc_reports.get(session_key) if session_key else None
        presentation = None
        if reference is not None:
            report = attached_report[0] if attached_report is not None else None
            report_source = attached_report[1] if attached_report is not None else "phase8_7c.inspection_report"
            presentation = build_reconstruction_reference_comparison_presentation(
                selection, reference, report, report_source=report_source
            )
        changed = presentation != self._reconstruction_reference_comparison_presentation
        self._reconstruction_reference_comparison_presentation = presentation

        if not hasattr(self, "_reference_comparison_target_value"):
            return
        if presentation is None:
            target, reference_text, structure, registration, deviations, scale, state, authority = (
                empty_reference_comparison_text(selection)
            )
            self._reference_comparison_target_value.setText(target)
            self._reference_comparison_reference_value.setText(reference_text)
            self._reference_comparison_reference_value.setToolTip("")
            self._reference_comparison_structure_value.setText(structure)
            self._reference_comparison_registration_value.setText(registration)
            self._reference_comparison_deviations_value.setText(deviations)
            self._reference_comparison_evidence_value.setText("—")
            self._reference_comparison_scale_value.setText(scale)
            self._reference_comparison_state_value.setText(state)
            self._reference_comparison_authority_value.setText(authority)
        else:
            self._reference_comparison_target_value.setText(presentation.target_text)
            self._reference_comparison_reference_value.setText(presentation.reference.reference_text)
            self._reference_comparison_reference_value.setToolTip(presentation.provenance_tooltip)
            self._reference_comparison_structure_value.setText(presentation.reference.structure_text)
            self._reference_comparison_registration_value.setText(presentation.registration_text)
            self._reference_comparison_deviations_value.setText(presentation.deviations_text)
            self._reference_comparison_evidence_value.setText(presentation.evidence_text)
            self._reference_comparison_scale_value.setText(presentation.scale_text)
            self._reference_comparison_state_value.setText(presentation.state_text)
            self._reference_comparison_authority_value.setText(presentation.authority_text)

        self._reference_comparison_clear_button.setEnabled(reference is not None)
        self._reference_comparison_load_button.setEnabled(selection.has_scene)
        if changed:
            self.reconstruction_reference_comparison_changed.emit(presentation)

    @property
    def appearance_representation_presentation(self) -> ReconstructionAppearancePresentation | None:
        return self._appearance_presentation

    @property
    def selected_appearance_representation(self) -> AppearanceRepresentationEntry | None:
        presentation = self._appearance_presentation
        if presentation is None or not self._selected_appearance_artifact_id:
            return None
        return presentation.entry_by_artifact_id(self._selected_appearance_artifact_id)

    def _on_appearance_representation_selected(self, combo_index: int) -> None:
        artifact_id = str(self._appearance_representation_combo.itemData(combo_index) or "").strip()
        self._selected_appearance_artifact_id = artifact_id
        self._refresh_appearance_representation_values()
        self.appearance_representation_changed.emit(self.selected_appearance_representation)

    def _refresh_appearance_representation_controls(self) -> None:
        previous = self._selected_appearance_artifact_id
        try:
            presentation = build_reconstruction_appearance_presentation(self._persistent_scene)
        except (TypeError, ValueError) as exc:
            self._appearance_presentation = None
            self._selected_appearance_artifact_id = ""
            self._appearance_representation_combo.blockSignals(True)
            self._appearance_representation_combo.clear()
            self._appearance_representation_combo.addItem("Appearance unavailable", "")
            self._appearance_representation_combo.setEnabled(False)
            self._appearance_representation_combo.blockSignals(False)
            values = empty_appearance_representation_text()
            self._appearance_available_value.setText("Appearance metadata rejected")
            self._appearance_available_value.setToolTip(str(exc))
            self._appearance_provider_value.setText(values["provider"])
            self._appearance_artifact_value.setText(values["artifact"])
            self._appearance_render_value.setText(values["render"])
            self._appearance_scale_value.setText(values["scale"])
            self._appearance_state_value.setText("Appearance metadata unavailable")
            self._appearance_state_value.setToolTip(str(exc))
            self._appearance_authority_value.setText(values["authority"])
            self.appearance_representation_changed.emit(None)
            return

        self._appearance_presentation = presentation
        self._appearance_representation_combo.blockSignals(True)
        self._appearance_representation_combo.clear()
        if presentation is None:
            self._selected_appearance_artifact_id = ""
            self._appearance_representation_combo.addItem("No appearance scene", "")
            self._appearance_representation_combo.setEnabled(False)
        else:
            for entry in presentation.entries:
                self._appearance_representation_combo.addItem(entry.label, entry.artifact_id)
            self._appearance_representation_combo.setEnabled(bool(presentation.entries))
            available_ids = {entry.artifact_id for entry in presentation.entries}
            if previous in available_ids:
                self._selected_appearance_artifact_id = previous
            elif presentation.entries:
                self._selected_appearance_artifact_id = presentation.entries[0].artifact_id
            else:
                self._selected_appearance_artifact_id = ""
            if self._selected_appearance_artifact_id:
                index = self._appearance_representation_combo.findData(self._selected_appearance_artifact_id)
                if index >= 0:
                    self._appearance_representation_combo.setCurrentIndex(index)
            elif not presentation.entries:
                self._appearance_representation_combo.addItem("No neural appearance artifacts", "")
        self._appearance_representation_combo.blockSignals(False)
        self._refresh_appearance_representation_values()
        self.appearance_representation_changed.emit(self.selected_appearance_representation)

    def _refresh_appearance_representation_values(self) -> None:
        values = empty_appearance_representation_text()
        presentation = self._appearance_presentation
        selected = self.selected_appearance_representation
        self._appearance_available_value.setText(
            presentation.available_text if presentation is not None else values["available"]
        )
        self._appearance_available_value.setToolTip(
            "Persisted Phase 8.7 Gaussian/NeRF appearance artifacts only; this is not backend runtime probing."
        )
        if selected is None:
            self._appearance_provider_value.setText(values["provider"])
            self._appearance_artifact_value.setText(values["artifact"])
            self._appearance_render_value.setText(values["render"])
            self._appearance_scale_value.setText(values["scale"])
        else:
            self._appearance_provider_value.setText(selected.provider_text)
            self._appearance_artifact_value.setText(selected.artifact_id)
            self._appearance_artifact_value.setToolTip(selected.artifact_id)
            self._appearance_render_value.setText(selected.render_text)
            self._appearance_scale_value.setText(selected.scale_text)
        self._appearance_state_value.setText(
            presentation.state_text if presentation is not None else values["state"]
        )
        self._appearance_authority_value.setText(
            presentation.authority_text if presentation is not None else values["authority"]
        )
        self._appearance_authority_value.setToolTip(
            "F1 inspects persisted appearance metadata only. Gaussian/NeRF do not become canonical geometry, measurement, inspection/QC, reference, or certified-metrology authority."
        )

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("reconstructionSettingsPanel")
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(380)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(10)

        title = QLabel("RECONSTRUCTION SETTINGS")
        title.setObjectName("reconstructionSectionTitle")
        panel_layout.addWidget(title)

        # D4 makes the offline inspection stack taller than the available
        # sidebar height on common 768p/900p/1080p desktops.  Do not let Qt
        # satisfy that pressure by crushing rows into each other.  Keep the
        # panel width fixed and make only the settings body vertically
        # scrollable.  The title remains pinned and no horizontal scrolling
        # is permitted.
        self._settings_scroll = QScrollArea(panel)
        self._settings_scroll.setObjectName("reconstructionSettingsScroll")
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._settings_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._settings_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._settings_scroll.viewport().setAutoFillBackground(False)

        settings_body = QWidget()
        settings_body.setObjectName("reconstructionSettingsScrollBody")
        settings_body.setAutoFillBackground(False)
        layout = QVBoxLayout(settings_body)
        # Some Linux/Qt styles draw the vertical scrollbar over the scroll
        # viewport instead of reserving its full visual width.  Keep a small
        # right-side gutter so right-aligned settings values wrap before the
        # scrollbar and never render underneath it.
        layout.setContentsMargins(0, 0, 14, 0)
        layout.setSpacing(10)

        workflow = self._settings_group("WORKFLOW")
        workflow_layout = workflow.layout()
        self._workflow_input_value = self._settings_value("No source selected")
        self._runtime_prerequisite_value = self._settings_value("Choose a workflow")
        self._runtime_output_value = self._settings_value("Not started")
        self._add_status_row(workflow_layout, "Input", self._workflow_input_value)
        self._add_status_row(workflow_layout, "Runtime", self._runtime_prerequisite_value)
        self._add_status_row(workflow_layout, "Output", self._runtime_output_value)
        self._add_settings_row(workflow_layout, "Profile", "Efficient")
        self._add_settings_row(workflow_layout, "Geometry", "Canonical authority")
        self._add_settings_row(workflow_layout, "Appearance", "Deferred")
        layout.addWidget(workflow)

        session = self._settings_group("SESSION & PERSISTENCE")
        session_layout = session.layout()
        self._session_value = self._settings_value("No active session")
        self._checkpoint_value = self._settings_value("No checkpoint")
        self._scene_manifest_value = self._settings_value("No persistent scene")
        self._telemetry_value = self._settings_value("No runtime telemetry")
        self._add_status_row(session_layout, "Session", self._session_value)
        self._add_status_row(session_layout, "Scene", self._scene_manifest_value)
        self._add_status_row(session_layout, "Checkpoint", self._checkpoint_value)
        self._add_status_row(session_layout, "Telemetry", self._telemetry_value)
        layout.addWidget(session)

        self._offline_inspection_group = self._settings_group("OFFLINE INSPECTION")
        inspection_layout = self._offline_inspection_group.layout()
        selector_row = QHBoxLayout()
        selector_row.setSpacing(6)
        selector_label = QLabel("Camera")
        selector_label.setObjectName("reconstructionRowLabel")
        selector_row.addWidget(selector_label)
        self._offline_camera_combo = QComboBox()
        self._offline_camera_combo.setObjectName("reconstructionOfflineCameraCombo")
        self._offline_camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        # D4 responsive-sidebar hardening: the camera identifier may be long,
        # but it must not impose that text width as a minimum on the entire
        # scroll body.  The combo remains readable at normal sidebar widths
        # and can shrink when the splitter is constrained.
        self._offline_camera_combo.setMinimumContentsLength(4)
        self._offline_camera_combo.setMinimumWidth(0)
        self._offline_camera_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Fixed,
        )
        self._offline_camera_combo.addItem("Select camera…", -1)
        self._offline_camera_combo.setEnabled(False)
        selector_row.addWidget(self._offline_camera_combo, stretch=1)
        inspection_layout.addLayout(selector_row)

        self._offline_camera_position_value = self._settings_value("—")
        self._offline_camera_delta_value = self._settings_value("—")
        self._offline_structure_value = self._settings_value("No offline sparse SfM")
        self._add_status_row(inspection_layout, "Position", self._offline_camera_position_value)
        self._add_status_row(inspection_layout, "Δ previous", self._offline_camera_delta_value)
        self._add_status_row(inspection_layout, "Structure", self._offline_structure_value)

        quality_title = QLabel("SPARSE SFM HEALTH")
        quality_title.setObjectName("reconstructionRowLabel")
        quality_title.setToolTip(
            "Operational reconstruction-health telemetry from the qualified SfM backend; not a metrology score."
        )
        inspection_layout.addWidget(quality_title)
        self._offline_quality_status_value = self._status_value()
        self._offline_quality_registration_value = self._settings_value("—")
        self._offline_quality_reprojection_value = self._settings_value("—")
        self._offline_quality_support_value = self._settings_value("—")
        self._offline_quality_gate_value = self._settings_value(
            "Operational SfM health only • not metrology"
        )
        self._add_status_row(inspection_layout, "Health", self._offline_quality_status_value)
        self._add_status_row(inspection_layout, "Registration", self._offline_quality_registration_value)
        self._add_status_row(inspection_layout, "Reprojection", self._offline_quality_reprojection_value)
        self._add_status_row(inspection_layout, "Support", self._offline_quality_support_value)
        self._add_status_row(inspection_layout, "Gate", self._offline_quality_gate_value)

        spatial_title = QLabel("SPARSE SPATIAL COVERAGE")
        spatial_title.setObjectName("reconstructionRowLabel")
        spatial_title.setToolTip(
            "Presentation-space coverage derived from the bounded sparse-SfM sample; not object dimensions or metric scale."
        )
        inspection_layout.addWidget(spatial_title)
        self._offline_spatial_envelope_value = self._settings_value("—")
        self._offline_spatial_camera_span_value = self._settings_value("—")
        self._offline_spatial_standoff_value = self._settings_value("—")
        self._offline_spatial_basis_value = self._settings_value(
            "Sparse presentation envelope only • not object dimensions"
        )
        self._add_status_row(inspection_layout, "Envelope", self._offline_spatial_envelope_value)
        self._add_status_row(inspection_layout, "Camera span", self._offline_spatial_camera_span_value)
        self._add_status_row(inspection_layout, "Standoff", self._offline_spatial_standoff_value)
        self._add_status_row(inspection_layout, "Basis", self._offline_spatial_basis_value)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        self._offline_prev_camera_button = QPushButton("Prev")
        self._offline_focus_camera_button = QPushButton("Focus")
        self._offline_next_camera_button = QPushButton("Next")
        self._offline_clear_camera_button = QPushButton("Clear")
        for button in (
            self._offline_prev_camera_button,
            self._offline_focus_camera_button,
            self._offline_next_camera_button,
            self._offline_clear_camera_button,
        ):
            button.setEnabled(False)
            # Keep the D2 camera navigation row horizontally shrinkable.
            # QPushButton defaults can otherwise raise the scroll body's
            # minimum width above the narrow settings viewport once D3/D4
            # sections are visible, which clips every right-aligned value
            # even when horizontal scrolling is disabled.
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QSizePolicy.Policy.Ignored,
                QSizePolicy.Policy.Fixed,
            )
            nav_row.addWidget(button)
        inspection_layout.addLayout(nav_row)
        self._offline_camera_combo.currentIndexChanged.connect(self._on_offline_camera_selected)
        self._offline_prev_camera_button.clicked.connect(self._select_previous_offline_camera)
        self._offline_focus_camera_button.clicked.connect(self._focus_selected_offline_camera)
        self._offline_next_camera_button.clicked.connect(self._select_next_offline_camera)
        self._offline_clear_camera_button.clicked.connect(self._clear_offline_camera_selection)
        self._reset_offline_sfm_health()
        self._reset_offline_spatial_coverage()
        self._offline_inspection_group.setVisible(False)
        layout.addWidget(self._offline_inspection_group)

        self._semantic_objects_group = self._settings_group("SEMANTIC / OBJECTS")
        semantic_layout = self._semantic_objects_group.layout()
        semantic_selector = QHBoxLayout()
        semantic_selector.setSpacing(6)
        semantic_label = QLabel("Object")
        semantic_label.setObjectName("reconstructionRowLabel")
        semantic_selector.addWidget(semantic_label)
        self._semantic_object_combo = QComboBox()
        self._semantic_object_combo.setObjectName("reconstructionSemanticObjectCombo")
        self._semantic_object_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._semantic_object_combo.setMinimumContentsLength(4)
        self._semantic_object_combo.setMinimumWidth(0)
        self._semantic_object_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._semantic_object_combo.addItem("Select object…", -1)
        self._semantic_object_combo.setEnabled(False)
        semantic_selector.addWidget(self._semantic_object_combo, stretch=1)
        semantic_layout.addLayout(semantic_selector)

        self._semantic_structure_value = self._settings_value("No semantic context")
        self._semantic_role_value = self._settings_value("—")
        self._semantic_support_value = self._settings_value("—")
        self._semantic_quality_value = self._settings_value("—")
        self._semantic_scale_value = self._settings_value("—")
        self._semantic_authority_value = self._settings_value(
            "Semantic context only • not inspection/QC decision"
        )
        self._add_status_row(semantic_layout, "Structure", self._semantic_structure_value)
        self._add_status_row(semantic_layout, "Role", self._semantic_role_value)
        self._add_status_row(semantic_layout, "Support", self._semantic_support_value)
        self._add_status_row(semantic_layout, "Quality", self._semantic_quality_value)
        self._add_status_row(semantic_layout, "Scale", self._semantic_scale_value)
        self._add_status_row(semantic_layout, "Authority", self._semantic_authority_value)

        semantic_nav = QHBoxLayout()
        semantic_nav.setSpacing(6)
        self._semantic_prev_button = QPushButton("Prev")
        self._semantic_focus_button = QPushButton("Focus")
        self._semantic_next_button = QPushButton("Next")
        self._semantic_clear_button = QPushButton("Clear")
        for button in (
            self._semantic_prev_button,
            self._semantic_focus_button,
            self._semantic_next_button,
            self._semantic_clear_button,
        ):
            button.setEnabled(False)
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            semantic_nav.addWidget(button)
        semantic_layout.addLayout(semantic_nav)
        self._semantic_object_combo.currentIndexChanged.connect(self._on_semantic_object_selected)
        self._semantic_prev_button.clicked.connect(self._select_previous_semantic_object)
        self._semantic_focus_button.clicked.connect(self._focus_selected_semantic_object)
        self._semantic_next_button.clicked.connect(self._select_next_semantic_object)
        self._semantic_clear_button.clicked.connect(self._clear_semantic_object_selection)
        self._semantic_objects_group.setVisible(False)
        layout.addWidget(self._semantic_objects_group)

        # E1 reuses the D5 semantic selector instead of introducing a second
        # object picker.  With no semantic object selected, the active scene is
        # the target; choosing an existing D5 object changes the target scope
        # to that object.  Later E2-E5 consume this generic selection contract.
        self._inspection_target_group = self._settings_group("INSPECTION TARGET")
        inspection_target_layout = self._inspection_target_group.layout()
        self._inspection_scope_value = self._settings_value("—")
        self._inspection_scene_value = self._settings_value("No reconstruction scene")
        self._inspection_object_value = self._settings_value("—")
        self._inspection_geometry_value = self._settings_value("No geometry target")
        self._inspection_state_value = self._settings_value("No inspection target")
        self._inspection_authority_value = self._settings_value(
            "Selection only • measurement/QC not evaluated"
        )
        self._add_status_row(inspection_target_layout, "Scope", self._inspection_scope_value)
        self._add_status_row(inspection_target_layout, "Scene", self._inspection_scene_value)
        self._add_status_row(inspection_target_layout, "Object", self._inspection_object_value)
        self._add_status_row(inspection_target_layout, "Geometry", self._inspection_geometry_value)
        self._add_status_row(inspection_target_layout, "State", self._inspection_state_value)
        self._add_status_row(inspection_target_layout, "Authority", self._inspection_authority_value)
        self._inspection_authority_value.setToolTip(
            "E1 establishes target selection only. Measurement, confidence, inspection/QC, and reference comparison remain E2-E5 responsibilities."
        )
        layout.addWidget(self._inspection_target_group)

        # E2 is deliberately presentation-only. Existing Phase 7.5 measurement
        # evidence can be attached to the active E1 object target; no depth,
        # fitting, measurement, uncertainty, QC, or reference algorithm runs here.
        self._reconstruction_measurement_group = self._settings_group("RECONSTRUCTION MEASUREMENTS")
        measurement_layout = self._reconstruction_measurement_group.layout()
        self._measurement_target_value = self._settings_value("No target")
        self._measurement_source_value = self._settings_value("No measurement evidence")
        self._measurement_values_value = self._settings_value("—")
        self._measurement_state_value = self._settings_value("Waiting for reconstruction target")
        self._measurement_authority_value = self._settings_value(
            "Existing measurements only • E2 does not compute measurements"
        )
        self._add_status_row(measurement_layout, "Target", self._measurement_target_value)
        self._add_status_row(measurement_layout, "Source", self._measurement_source_value)
        self._add_status_row(measurement_layout, "Values", self._measurement_values_value)
        self._add_status_row(measurement_layout, "State", self._measurement_state_value)
        self._add_status_row(measurement_layout, "Authority", self._measurement_authority_value)
        self._measurement_authority_value.setToolTip(
            "E2 presents existing source-authored values only. Uncertainty, reliability, calibration/scale validity, rejection reasons, and confidence are Phase 8.8E3."
        )
        layout.addWidget(self._reconstruction_measurement_group)

        # E3 qualifies the exact E2-bound source observation. It renders only
        # fields authored by the Phase 7.5 measurement result/bundle: status,
        # uncertainty, bounds, reliability, support, scale/calibration context,
        # warnings, and rejection reasons. No thresholds or QC decisions live here.
        self._measurement_validity_group = self._settings_group("MEASUREMENT VALIDITY")
        validity_layout = self._measurement_validity_group.layout()
        self._measurement_validity_qualified_value = self._settings_value("—")
        self._measurement_validity_status_value = self._settings_value("Waiting for reconstruction target")
        self._measurement_validity_reliability_value = self._settings_value("—")
        self._measurement_validity_uncertainty_value = self._settings_value("—")
        self._measurement_validity_scale_value = self._settings_value("—")
        self._measurement_validity_calibration_value = self._settings_value("—")
        self._measurement_validity_support_value = self._settings_value("—")
        self._measurement_validity_reasons_value = self._settings_value("—")
        self._measurement_validity_authority_value = self._settings_value(
            "Existing validity evidence only • E3 does not compute confidence/uncertainty"
        )
        self._add_status_row(validity_layout, "Qualified", self._measurement_validity_qualified_value)
        self._add_status_row(validity_layout, "Validity", self._measurement_validity_status_value)
        self._add_status_row(validity_layout, "Reliability", self._measurement_validity_reliability_value)
        self._add_status_row(validity_layout, "Uncertainty", self._measurement_validity_uncertainty_value)
        self._add_status_row(validity_layout, "Scale", self._measurement_validity_scale_value)
        self._add_status_row(validity_layout, "Calibration", self._measurement_validity_calibration_value)
        self._add_status_row(validity_layout, "Support", self._measurement_validity_support_value)
        self._add_status_row(validity_layout, "Reasons", self._measurement_validity_reasons_value)
        self._add_status_row(validity_layout, "Authority", self._measurement_validity_authority_value)
        self._measurement_validity_authority_value.setToolTip(
            "E3 presents source-authored measurement validity and uncertainty. It does not infer reliability from semantic confidence, run inspection/QC, compare references, or certify metrology."
        )
        layout.addWidget(self._measurement_validity_group)

        # E4 presents an already-produced Phase 8.7C inspection/QC report.
        # The decision and findings remain source-authored. Registration/QC is
        # not rerun here and the reference selection/comparison workflow is E5.
        self._inspection_qc_group = self._settings_group("INSPECTION / QC")
        inspection_qc_layout = self._inspection_qc_group.layout()
        self._inspection_qc_target_value = self._settings_value("No target")
        self._inspection_qc_decision_value = self._settings_value("Not evaluated")
        self._inspection_qc_findings_value = self._settings_value("—")
        self._inspection_qc_coverage_value = self._settings_value("—")
        self._inspection_qc_decisions_value = self._settings_value("—")
        self._inspection_qc_registration_value = self._settings_value("No inspection report attached")
        self._inspection_qc_comparison_value = self._settings_value("—")
        self._inspection_qc_authority_value = self._settings_value(
            "Existing QC only • E4 does not run inspection"
        )
        self._add_status_row(inspection_qc_layout, "Target", self._inspection_qc_target_value)
        self._add_status_row(inspection_qc_layout, "Decision", self._inspection_qc_decision_value)
        self._add_status_row(inspection_qc_layout, "Findings", self._inspection_qc_findings_value)
        self._add_status_row(inspection_qc_layout, "Coverage", self._inspection_qc_coverage_value)
        self._add_status_row(inspection_qc_layout, "Summary", self._inspection_qc_decisions_value)
        self._add_status_row(inspection_qc_layout, "Registration", self._inspection_qc_registration_value)
        self._add_status_row(inspection_qc_layout, "Comparison", self._inspection_qc_comparison_value)
        self._add_status_row(inspection_qc_layout, "Authority", self._inspection_qc_authority_value)
        self._inspection_qc_authority_value.setToolTip(
            "E4 displays existing Phase 8.7C inspection/QC decisions only. It does not infer decisions from semantics/measurements, rerun registration, or implement the E5 reference-comparison workflow."
        )
        layout.addWidget(self._inspection_qc_group)

        # E5 makes reference identity explicit and pairs it with the exact
        # existing Phase 8.7C report for the current scene. It does not parse
        # CAD geometry, rerun registration, apply tolerance rules, or promote
        # an arbitrary reconstruction to metric/certified authority.
        self._reference_comparison_group = self._settings_group("REFERENCE COMPARISON")
        reference_layout = self._reference_comparison_group.layout()
        self._reference_comparison_target_value = self._settings_value("No target")
        self._reference_comparison_reference_value = self._settings_value("No reference attached")
        self._reference_comparison_structure_value = self._settings_value("—")
        self._reference_comparison_registration_value = self._settings_value("—")
        self._reference_comparison_deviations_value = self._settings_value("—")
        self._reference_comparison_evidence_value = self._settings_value("—")
        self._reference_comparison_scale_value = self._settings_value("—")
        self._reference_comparison_state_value = self._settings_value("Waiting for reconstruction target")
        self._reference_comparison_authority_value = self._settings_value(
            "Reference workflow only • E5 does not register/apply tolerances"
        )
        self._add_status_row(reference_layout, "Target", self._reference_comparison_target_value)
        self._add_status_row(reference_layout, "Reference", self._reference_comparison_reference_value)
        self._add_status_row(reference_layout, "Structure", self._reference_comparison_structure_value)
        self._add_status_row(reference_layout, "Registration", self._reference_comparison_registration_value)
        self._add_status_row(reference_layout, "Deviation", self._reference_comparison_deviations_value)
        self._add_status_row(reference_layout, "Evidence", self._reference_comparison_evidence_value)
        self._add_status_row(reference_layout, "Scale", self._reference_comparison_scale_value)
        self._add_status_row(reference_layout, "State", self._reference_comparison_state_value)
        self._add_status_row(reference_layout, "Authority", self._reference_comparison_authority_value)
        self._reference_comparison_authority_value.setToolTip(
            "E5 loads ReferenceAssembly JSON metadata and presents exact matching Phase 8.7C comparison evidence. Registration, deviations, and tolerance decisions are source-authored; E5 does not compute them or certify metrology."
        )
        reference_buttons = QHBoxLayout()
        reference_buttons.setSpacing(6)
        self._reference_comparison_load_button = QPushButton("Load JSON")
        self._reference_comparison_clear_button = QPushButton("Clear")
        for button in (self._reference_comparison_load_button, self._reference_comparison_clear_button):
            button.setMinimumWidth(0)
            button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            reference_buttons.addWidget(button)
        self._reference_comparison_load_button.setEnabled(False)
        self._reference_comparison_clear_button.setEnabled(False)
        self._reference_comparison_load_button.clicked.connect(self._choose_reconstruction_reference_json)
        self._reference_comparison_clear_button.clicked.connect(self._clear_active_reconstruction_reference)
        reference_layout.addLayout(reference_buttons)
        layout.addWidget(self._reference_comparison_group)

        # F1 controls persisted neural appearance metadata only. The selector
        # chooses which appearance artifact metadata is inspected; it does not
        # render or switch the viewport. Gaussian visualization starts in F2,
        # NeRF visualization in F3, and whole-view switching in F4.
        self._appearance_group = self._settings_group("APPEARANCE REPRESENTATIONS")
        appearance_layout = self._appearance_group.layout()
        appearance_selector = QHBoxLayout()
        appearance_selector.setSpacing(6)
        appearance_label = QLabel("Representation")
        appearance_label.setObjectName("reconstructionRowLabel")
        appearance_selector.addWidget(appearance_label)
        self._appearance_representation_combo = QComboBox()
        self._appearance_representation_combo.setObjectName("reconstructionAppearanceRepresentationCombo")
        self._appearance_representation_combo.setMinimumContentsLength(4)
        self._appearance_representation_combo.setMinimumWidth(0)
        self._appearance_representation_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._appearance_representation_combo.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self._appearance_representation_combo.setToolTip(
            "Inspect persisted appearance artifact metadata. F1 does not render or switch the Reconstruction viewport."
        )
        self._appearance_representation_combo.currentIndexChanged.connect(
            self._on_appearance_representation_selected
        )
        appearance_selector.addWidget(self._appearance_representation_combo, stretch=1)
        appearance_layout.addLayout(appearance_selector)
        self._appearance_available_value = self._settings_value("No persistent appearance scene")
        self._appearance_provider_value = self._settings_value("—")
        self._appearance_artifact_value = self._settings_value("—")
        self._appearance_render_value = self._settings_value("—")
        self._appearance_scale_value = self._settings_value("—")
        self._appearance_state_value = self._settings_value("Waiting for persistent reconstruction scene")
        self._appearance_authority_value = self._settings_value(
            "Appearance only • canonical geometry remains authoritative"
        )
        # Keep the complete authority statement visibly readable at the
        # minimum Reconstruction sidebar width. QLabel height-for-width can
        # otherwise under-allocate this final wrapped row inside the scroll
        # body, visually clipping the safety-critical last line.
        self._appearance_authority_value.setMinimumHeight(
            self._appearance_authority_value.fontMetrics().lineSpacing() * 3
        )
        self._add_status_row(appearance_layout, "Available", self._appearance_available_value)
        self._add_status_row(appearance_layout, "Provider", self._appearance_provider_value)
        self._add_status_row(appearance_layout, "Artifact", self._appearance_artifact_value)
        self._add_status_row(appearance_layout, "Render", self._appearance_render_value)
        self._add_status_row(appearance_layout, "Scale", self._appearance_scale_value)
        self._add_status_row(appearance_layout, "State", self._appearance_state_value)
        self._add_status_row(appearance_layout, "Authority", self._appearance_authority_value)
        layout.addWidget(self._appearance_group)
        self._refresh_appearance_representation_controls()

        authority = self._settings_group("AUTHORITY & SAFETY")
        authority_layout = authority.layout()
        self._metric_scale_value = self._settings_value("Evidence-driven")
        self._camera_geometry_value = self._settings_value("Not selected")
        self._metrology_value = self._settings_value("Not certified")
        self._add_status_row(authority_layout, "Metric scale", self._metric_scale_value)
        self._add_status_row(authority_layout, "Calibration", self._camera_geometry_value)
        self._add_status_row(authority_layout, "Metrology", self._metrology_value)
        self._add_settings_row(authority_layout, "Heavy backends", "Isolated workers")
        layout.addWidget(authority)
        layout.addStretch(1)
        self._settings_scroll.setWidget(settings_body)
        panel_layout.addWidget(self._settings_scroll, stretch=1)
        return panel

    @staticmethod
    def _card(title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(13, 11, 13, 11)
        layout.setSpacing(8)
        title = QLabel(title_text)
        title.setObjectName("reconstructionSectionTitle")
        layout.addWidget(title)
        return card

    @staticmethod
    def _settings_group(title_text: str) -> QFrame:
        card = QFrame()
        card.setObjectName("reconstructionSettingsGroup")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 9, 10, 9)
        layout.setSpacing(7)
        title = QLabel(title_text)
        title.setObjectName("reconstructionSettingsGroupTitle")
        layout.addWidget(title)
        return card

    @staticmethod
    def _value_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("reconstructionValue")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _status_value() -> QLabel:
        label = QLabel()
        label.setObjectName("reconstructionStatusValue")
        return label

    @staticmethod
    def _settings_value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("reconstructionSettingsValue")
        label.setWordWrap(True)
        # Right-side values must be allowed to wrap rather than establish a
        # horizontal minimum that forces the settings scroll body under the
        # vertical scrollbar.  Ignored affects only layout negotiation; the
        # label still receives its normal size at ordinary sidebar widths.
        label.setMinimumWidth(0)
        label.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        return label

    @staticmethod
    def _add_status_row(layout: QVBoxLayout, label_text: str, value: QLabel) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("reconstructionRowLabel")
        # Give the value the row's remaining width directly.  A separate
        # expanding spacer used to consume that width first; once D4 made
        # value labels horizontally shrinkable, Qt could collapse them to
        # nearly zero width and word-wrapped labels then requested enormous
        # heights.  That produced the large blank vertical gaps seen in the
        # real Linux preview.
        row.addWidget(label, 0, Qt.AlignmentFlag.AlignTop)
        row.addWidget(value, 1, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)

    def _add_settings_row(self, layout: QVBoxLayout, label_text: str, value_text: str) -> None:
        self._add_status_row(layout, label_text, self._settings_value(value_text))

    def bind_controller(self, controller) -> None:
        """Bind passive controller state. No reconstruction action is triggered here."""

        if self._controller is not None:
            raise RuntimeError("ReconstructionPage controller is already bound.")
        self._controller = controller

        controller.live_state_changed.connect(self.set_live_state)
        controller.live_observation_ready.connect(self.set_live_observation)
        controller.offline_state_changed.connect(self.set_offline_state)
        controller.offline_event_received.connect(self.set_offline_event)
        controller.offline_result_ready.connect(self._on_offline_result_ready)
        controller.offline_run_finished.connect(self._on_offline_run_finished)
        controller.source_selection_changed.connect(self.set_source_selection)
        controller.scene_availability_changed.connect(self.set_scene_availability)
        controller.persistent_scene_changed.connect(self.set_persistent_scene)
        controller.checkpoint_changed.connect(self.set_checkpoint)
        controller.error_occurred.connect(self.set_error)

        self.set_live_state(controller.live_state)
        self.set_offline_state(controller.offline_state)
        self.set_source_selection(controller.source_selection)
        self.set_scene_availability(controller.scene_availability)
        self._refresh_source_controls()
        self._refresh_runtime_controls()
        scene = controller.persistent_scene
        self.set_persistent_scene(scene, str(controller.persistent_scene_manifest or ""))
        self.set_checkpoint(controller.engine_checkpoint, str(controller.engine_checkpoint_manifest or ""))
        self._telemetry_timer.start()
        self._refresh_runtime_telemetry()

    def set_source_selection(self, selection: ReconstructionSourceSelection) -> None:
        if not isinstance(selection, ReconstructionSourceSelection):
            raise TypeError("selection must be ReconstructionSourceSelection.")
        workflow = selection.workflow
        self._source_value.setText(selection.display_name)
        self._source_value.setToolTip(str(selection.path or selection.detail))
        self._workflow_value.setText(self._workflow_label(workflow))
        self._workflow_value.setToolTip(selection.detail)
        if workflow is ReconstructionWorkflow.NONE:
            readiness_text = "Not selected"
            readiness_kind = "neutral"
        elif selection.ready:
            readiness_text = "Ready"
            readiness_kind = "ready"
        else:
            readiness_text = "Needs attention"
            readiness_kind = "warning"
        self._source_readiness_value.setText(readiness_text)
        self._source_readiness_value.setProperty("statusKind", readiness_kind)
        self._refresh_style(self._source_readiness_value)
        for candidate, button in self._source_buttons.items():
            button.setChecked(candidate is workflow)
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            identity = (
                selection.live_source_kind or "",
                selection.display_name,
                str(selection.path or ""),
            )
            if (
                self._last_live_source_identity is not None
                and identity != self._last_live_source_identity
            ):
                self._reset_live_camera_selection()
            self._last_live_source_identity = identity
        elif workflow is not self._last_workflow:
            self._reset_live_camera_selection()
            self._last_live_source_identity = None
        self._last_workflow = workflow
        self._sync_diagnostics_visibility()
        if workflow is ReconstructionWorkflow.IMAGE_SEQUENCE and selection.item_count is not None:
            self._workflow_input_value.setText(f"{selection.item_count} images")
        elif workflow is ReconstructionWorkflow.RECORDED_VIDEO:
            self._workflow_input_value.setText(selection.display_name)
        elif workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._workflow_input_value.setText(selection.display_name)
        else:
            self._workflow_input_value.setText("No source selected")
        self._workflow_input_value.setToolTip(selection.detail)
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._refresh_live_camera_presentation()
        elif workflow in {ReconstructionWorkflow.IMAGE_SEQUENCE, ReconstructionWorkflow.RECORDED_VIDEO}:
            self._runtime_prerequisite_value.setText("Isolated PyCOLMAP worker")
            self._runtime_prerequisite_value.setToolTip(
                "The production depth environment launches PyCOLMAP in its isolated interpreter."
            )
        else:
            self._runtime_prerequisite_value.setText("Choose a workflow")
            self._runtime_prerequisite_value.setToolTip("")
            self._refresh_camera_authority()
        preserve_live_terminal = (
            workflow is ReconstructionWorkflow.LIVE_SOURCE
            and self._live_state in {
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            }
        )
        if (
            (self._controller is None or not self._controller.is_active)
            and not preserve_live_terminal
        ):
            if selection.ready:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Ready to start",
                        "Source validated. Start is available when application resource guards are clear.",
                        "ready",
                    )
                )
            elif workflow is ReconstructionWorkflow.NONE:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Waiting for source",
                        "Choose a Reconstruction workflow to begin.",
                    )
                )
            else:
                self._apply_runtime_presentation(
                    ReconstructionStatusPresentation(
                        "Source needs attention",
                        selection.detail,
                        "warning",
                    )
                )
        self._refresh_runtime_controls()

    def _select_live_source(self) -> None:
        if self._controller is None:
            return
        try:
            self._controller.select_live_source()
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))

    def _choose_image_sequence(self) -> None:
        if self._controller is None:
            return
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Reconstruction Image Sequence",
            str(Path.home()),
        )
        if not selected:
            self.set_source_selection(self._controller.source_selection)
            return
        try:
            self._controller.select_image_sequence(selected)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_source_selection(self._controller.source_selection)
            self.set_error(str(exc))

    def _choose_recorded_video(self) -> None:
        if self._controller is None:
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reconstruction Video",
            str(Path.home()),
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All files (*)",
        )
        if not selected:
            self.set_source_selection(self._controller.source_selection)
            return
        try:
            self._controller.select_recorded_video(selected)
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_source_selection(self._controller.source_selection)
            self.set_error(str(exc))

    def _start_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        selection = controller.source_selection
        if not selection.ready:
            self.set_error(selection.detail or "Select a ready reconstruction source first.")
            self._refresh_runtime_controls()
            return
        self._prepare_runtime_presentation_for_start(selection.workflow)
        try:
            if selection.workflow is ReconstructionWorkflow.LIVE_SOURCE:
                prepared = self._prepare_live_camera_for_start()
                if prepared is None:
                    self._refresh_runtime_controls()
                    return
                camera, policy = prepared
                if policy is CameraIntrinsicsPolicy.CALIBRATED_ONLY:
                    session_id = controller.start_selected_live(camera=camera)
                else:
                    session_id = controller.start_selected_live(
                        camera=camera,
                        camera_intrinsics_policy=policy,
                    )
            elif selection.workflow in {
                ReconstructionWorkflow.IMAGE_SEQUENCE,
                ReconstructionWorkflow.RECORDED_VIDEO,
            }:
                session_id = controller.start_selected_offline()
            else:
                raise RuntimeError("Choose a Reconstruction workflow before starting.")
            self._session_value.setText(session_id)
            output_dir = controller.offline_output_dir
            if output_dir is not None:
                self._runtime_output_value.setText(output_dir.name)
                self._runtime_output_value.setToolTip(str(output_dir))
            elif selection.workflow is ReconstructionWorkflow.LIVE_SOURCE:
                self._runtime_output_value.setText("Live session")
                self._runtime_output_value.setToolTip(
                    "Live reconstruction has no offline output directory yet."
                )
            else:
                self._runtime_output_value.setText("Preparing output")
                self._runtime_output_value.setToolTip("")
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_source_controls()
            self._refresh_runtime_controls()

    def _choose_live_camera_mode(self) -> str | None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Live Reconstruction Camera Geometry")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText("Choose how to provide camera intrinsics for live reconstruction.")
        dialog.setInformativeText(
            "Use a calibration file for calibrated ray geometry. Test Mode uses explicit "
            "estimated intrinsics from the live resolution and an assumed 70° horizontal "
            "field of view. Test Mode is suitable for functional/visual testing only: "
            "geometry remains provisional, scale stays arbitrary, and metrology is not certified."
        )
        calibrated = dialog.addButton(
            "Use calibration file",
            QMessageBox.ButtonRole.ActionRole,
        )
        estimated = dialog.addButton(
            "Run Test Mode",
            QMessageBox.ButtonRole.AcceptRole,
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is calibrated:
            return "calibrated"
        if clicked is estimated:
            return "estimated"
        return None

    def _prepare_live_camera_for_start(
        self,
    ) -> tuple[ReconstructionCamera, CameraIntrinsicsPolicy] | None:
        mode = self._choose_live_camera_mode()
        if mode is None:
            return None
        if mode == "calibrated":
            camera = self._choose_live_camera_calibration()
            if camera is None:
                return None
            self._live_camera_mode = "calibrated"
            self._configure_visualizer_camera(camera)
            self._refresh_live_camera_presentation()
            return camera, CameraIntrinsicsPolicy.CALIBRATED_ONLY
        if mode == "estimated":
            if self._controller is None:
                return None
            camera = self._controller.build_estimated_live_camera()
            self._live_camera = camera
            self._live_calibration_path = None
            self._live_camera_mode = "estimated"
            self._configure_visualizer_camera(camera)
            self._refresh_live_camera_presentation()
            return camera, CameraIntrinsicsPolicy.ALLOW_ESTIMATED
        raise RuntimeError(f"Unsupported live camera mode: {mode}")

    def _configure_visualizer_camera(self, camera: ReconstructionCamera) -> None:
        self._reconstruction_canvas.configure_camera_geometry(
            intrinsics_matrix=camera.intrinsics.matrix,
            image_size=(camera.image_size.width, camera.image_size.height),
        )

    def _choose_live_camera_calibration(self) -> ReconstructionCamera | None:
        if self._live_camera is not None and self._live_calibration_path is not None:
            return self._live_camera
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Live Reconstruction Camera Calibration",
            str(Path.home()),
            "Camera calibration (*.json);;JSON files (*.json);;All files (*)",
        )
        if not selected:
            return None
        path = Path(selected).expanduser().resolve()
        camera = load_live_camera_calibration(path)
        self._live_camera = camera
        self._live_calibration_path = path
        return camera

    def _reset_live_camera_selection(self) -> None:
        self._live_camera = None
        self._live_calibration_path = None
        self._live_camera_mode = None
        if hasattr(self, "_reconstruction_canvas"):
            self._reconstruction_canvas.configure_camera_geometry(
                intrinsics_matrix=None, image_size=None
            )
        self._refresh_camera_authority()

    def _refresh_live_camera_presentation(self) -> None:
        if self._live_camera_mode == "estimated":
            self._runtime_prerequisite_value.setText("Estimated intrinsics • Test mode")
            self._runtime_prerequisite_value.setToolTip(
                "Explicit provisional mode: 70° assumed horizontal FOV, centered principal "
                "point, square pixels, zero distortion. Non-metric; metrology is not certified."
            )
        elif self._live_calibration_path is not None:
            self._runtime_prerequisite_value.setText(self._live_calibration_path.name)
            self._runtime_prerequisite_value.setToolTip(str(self._live_calibration_path))
        else:
            self._runtime_prerequisite_value.setText("Choose calibration or Test mode on Start")
            self._runtime_prerequisite_value.setToolTip(
                "Live Start explicitly offers a calibrated JSON profile or provisional Test Mode."
            )
        self._refresh_camera_authority()

    def _refresh_camera_authority(self) -> None:
        if not hasattr(self, "_camera_geometry_value"):
            return
        if self._live_camera_mode == "estimated":
            self._camera_geometry_value.setText("Estimated / test")
            self._camera_geometry_value.setToolTip(
                "Provisional intrinsics only; camera calibration has not been established."
            )
            self._metric_scale_value.setText("Unavailable (test)")
            self._metric_scale_value.setToolTip(
                "Test Mode does not establish metric scale or calibrated ray geometry."
            )
        elif self._live_camera_mode == "calibrated":
            self._camera_geometry_value.setText("Calibrated")
            self._camera_geometry_value.setToolTip(
                str(self._live_calibration_path or "Explicit calibrated camera profile")
            )
            self._metric_scale_value.setText("Evidence-driven")
            self._metric_scale_value.setToolTip(
                "Calibrated intrinsics do not by themselves establish monocular metric scale."
            )
        else:
            self._camera_geometry_value.setText("Not selected")
            self._camera_geometry_value.setToolTip("")
            self._metric_scale_value.setText("Evidence-driven")
            self._metric_scale_value.setToolTip("")
        self._metrology_value.setText("Not certified")

    def _pause_or_resume_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        try:
            if self._live_state is LiveCaptureRuntimeState.PAUSED:
                if not controller.resume_live():
                    raise RuntimeError("Live reconstruction could not resume from its current state.")
            elif self._live_state is LiveCaptureRuntimeState.RUNNING:
                if not controller.pause_live():
                    raise RuntimeError("Live reconstruction could not pause from its current state.")
            else:
                return
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_runtime_controls()

    def _stop_runtime(self) -> None:
        controller = self._controller
        if controller is None or controller.is_shutdown:
            return
        try:
            if self._offline_state in {
                OfflineReconstructionRunState.STARTING,
                OfflineReconstructionRunState.RUNNING,
            }:
                if not controller.cancel_offline():
                    raise RuntimeError("Offline reconstruction could not be cancelled.")
            elif self._live_state in {
                LiveCaptureRuntimeState.STARTING,
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
                LiveCaptureRuntimeState.RECOVERING,
            }:
                if not controller.request_stop_live():
                    raise RuntimeError("Live reconstruction could not be stopped.")
            else:
                return
            self._clear_warning_if_possible()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            self.set_error(str(exc))
        finally:
            self._refresh_runtime_controls()

    def _prepare_runtime_presentation_for_start(self, workflow: ReconstructionWorkflow) -> None:
        self._runtime_started_at = None
        self._runtime_elapsed_frozen = 0.0
        self._active_runtime_kind = None
        self._last_runtime_error = ""
        self.clear_reconstruction_measurement_evidence()
        self.clear_reconstruction_inspection_qc_evidence()
        self.clear_reconstruction_reference_comparison()
        self._reset_semantic_presentation()
        if workflow is ReconstructionWorkflow.LIVE_SOURCE:
            self._live_pose_count = 0
            self._live_keyframe_count = 0
            self._live_landmark_count = None
            self._telemetry_value.setText("Waiting for live telemetry")
            self._apply_runtime_presentation(
                ReconstructionStatusPresentation(
                    "Preparing live session",
                    "Validating camera geometry and live reconstruction prerequisites.",
                    "running",
                    indeterminate=True,
                )
            )
        else:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            self._visual_pose_timer.stop()
            self._reset_offline_sfm_health()
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            self._reconstruction_canvas.clear_live_state()
            self._viewport_placeholder.setVisible(True)
            self._visualizer_fitted_once = False
            self._offline_visualization_path = None
            self._offline_visualization_error = ""
            self._telemetry_value.setText("Waiting for offline telemetry")
            self._apply_runtime_presentation(
                ReconstructionStatusPresentation(
                    "Preparing offline job",
                    "Preparing the isolated offline reconstruction worker.",
                    "running",
                    indeterminate=True,
                )
            )

    def _begin_runtime_clock(self, kind: str) -> None:
        if self._active_runtime_kind == kind and self._runtime_started_at is not None:
            return
        self._active_runtime_kind = kind
        self._runtime_started_at = monotonic()
        self._runtime_elapsed_frozen = 0.0
        self._refresh_elapsed_label()

    def _freeze_runtime_clock(self) -> None:
        if self._runtime_started_at is not None:
            self._runtime_elapsed_frozen = max(0.0, monotonic() - self._runtime_started_at)
            self._runtime_started_at = None
        self._active_runtime_kind = None
        self._refresh_elapsed_label()

    def _refresh_elapsed_label(self) -> None:
        elapsed = self._runtime_elapsed_frozen
        if self._runtime_started_at is not None:
            elapsed = max(0.0, monotonic() - self._runtime_started_at)
        self._elapsed_value.setText(format_elapsed_seconds(elapsed) if elapsed > 0.0 else "—")

    def _apply_runtime_presentation(self, presentation: ReconstructionStatusPresentation) -> None:
        if not isinstance(presentation, ReconstructionStatusPresentation):
            raise TypeError("presentation must be ReconstructionStatusPresentation.")
        self._stage_value.setText(presentation.stage_label)
        self._stage_value.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._stage_value)
        self._runtime_message.setText(presentation.message)
        self._runtime_message.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._runtime_message)

        if presentation.indeterminate:
            self._progress_bar.setRange(0, 0)
            self._progress_bar.setFormat("Active")
        else:
            self._progress_bar.setRange(0, 100)
            if presentation.progress_fraction is None:
                self._progress_bar.setValue(0)
                self._progress_bar.setFormat(
                    presentation.stage_label if presentation.status_kind in {"error", "warning"} else "—"
                )
            else:
                value = max(0, min(100, round(presentation.progress_fraction * 100.0)))
                self._progress_bar.setValue(value)
                self._progress_bar.setFormat("%p%")
        self._progress_bar.setProperty("statusKind", presentation.status_kind)
        self._refresh_style(self._progress_bar)

    def set_offline_event(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self._apply_runtime_presentation(present_offline_event(payload))
        if self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }:
            self._begin_runtime_clock("offline")

    def _on_offline_result_ready(self, payload: object, _output_dir: str) -> None:
        if not isinstance(payload, dict):
            return
        sfm = payload.get("sfm_summary", {})
        if isinstance(sfm, dict):
            registered = sfm.get("registered_keyframes")
            total = sfm.get("input_keyframes")
            landmarks = sfm.get("landmarks")
            error = sfm.get("mean_reprojection_error_px")
            parts: list[str] = []
            if isinstance(registered, int):
                if isinstance(total, int) and total > 0:
                    parts.append(f"{registered}/{total} keyframes")
                else:
                    parts.append(f"{registered} keyframes")
            if isinstance(landmarks, int):
                parts.append(f"{landmarks:,} landmarks")
            if isinstance(error, (int, float)):
                parts.append(f"{float(error):.2f} px")
            if parts:
                self._telemetry_value.setText(" • ".join(parts))
                self._telemetry_value.setToolTip("Offline sparse-reconstruction result summary.")
            try:
                self._apply_offline_sfm_health(build_offline_sfm_health_presentation(sfm))
            except (TypeError, ValueError) as exc:
                self._apply_offline_sfm_health(unavailable_offline_sfm_health(str(exc)))
        else:
            self._apply_offline_sfm_health(
                unavailable_offline_sfm_health("Offline result does not contain an SfM summary.")
            )

        if str(payload.get("status", "")).strip().lower() != "completed":
            return
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            self._offline_visualization_error = "Offline result does not expose visualization artifacts."
            return
        visualization_path = str(artifacts.get("offline_visualization_json", "")).strip()
        if not visualization_path:
            # Older A3 results remain valid but simply cannot be visualized by D1/D2.
            self._offline_visualization_error = "Offline result predates the D1 visualization artifact."
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            return
        try:
            visual = load_offline_sparse_visualization(visualization_path)
            expected_session = str(payload.get("session_id", "")).strip()
            if expected_session and visual.session_id != expected_session:
                raise ValueError("Offline visualization session identity does not match the worker result.")
            self._reconstruction_canvas.show_offline_sparse_reconstruction(
                session_id=visual.session_id,
                scale_status=visual.scale_status,
                points_reconstruction=visual.landmark_points_reconstruction,
                camera_matrices_reconstruction=visual.camera_matrices_reconstruction,
                camera_ids=visual.camera_ids,
                landmark_colors_rgb=visual.landmark_colors_rgb,
                landmark_total=visual.landmark_total,
                registered_camera_total=visual.registered_camera_total,
            )
            self._offline_visualization_path = Path(visualization_path).expanduser().resolve()
            self._offline_visualization_error = ""
            self._activate_offline_inspection(visual)
            self._apply_offline_spatial_coverage(build_offline_sparse_spatial_coverage(visual))
            self._viewport_state.setToolTip("")
            if self._reconstruction_canvas.offline_visualization_loaded:
                self._viewport_placeholder.setVisible(False)
                self._reconstruction_canvas.fit_scene()
                self._visualizer_fitted_once = True
            self._refresh_viewport_state()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            # A visualization artifact is presentation-only. A completed authoritative
            # offline worker result remains completed even if this optional view fails.
            self._offline_visualization_error = str(exc)
            self._reset_offline_spatial_coverage()
            self._reset_offline_inspection()
            self._viewport_state.setText("Offline visualization unavailable")
            self._viewport_state.setToolTip(str(exc))

    def set_live_observation(self, observation: object) -> None:
        if self._viewport_shutdown:
            return
        diagnostics = self._live_diagnostics.update(observation)
        self._diagnostics_hud.set_presentation(diagnostics)
        visual_update_pending = False
        pose = getattr(observation, "pose", None)
        if pose is not None:
            self._live_pose_count += 1
            matrix = getattr(pose, "T_reconstruction_from_camera", None)
            if matrix is not None:
                value = np.asarray(matrix, dtype=np.float64)
                if value.shape == (4, 4) and np.isfinite(value).all():
                    # Worker-thread observations are already queued onto the Qt/UI
                    # thread by the controller. C1 additionally coalesces visual
                    # updates so a high-rate live stream cannot force one GL update
                    # per observation. C3 coalesces sparse-map samples on the same
                    # presentation timer.
                    self._pending_visual_pose = np.array(value, copy=True)
                    visual_update_pending = True
        if bool(getattr(observation, "is_keyframe", False)):
            self._live_keyframe_count += 1
        snapshot = getattr(observation, "map_snapshot", None)
        if snapshot is not None:
            self._live_keyframe_count = max(
                self._live_keyframe_count,
                int(getattr(snapshot, "keyframe_count", 0)),
            )
            self._live_landmark_count = int(getattr(snapshot, "landmark_count", 0))
            if hasattr(snapshot, "landmarks_reconstruction"):
                raw_points = getattr(snapshot, "landmarks_reconstruction")
                points = np.asarray(raw_points, dtype=np.float64)
                if points.size == 0:
                    points = np.zeros((0, 3), dtype=np.float64)
                if points.ndim == 2 and points.shape[1:] == (3,) and np.isfinite(points).all():
                    self._pending_sparse_landmarks = (
                        np.array(points, copy=True),
                        int(getattr(snapshot, "landmark_count", points.shape[0])),
                        int(getattr(snapshot, "revision", 0)),
                    )
                    visual_update_pending = True
            if hasattr(snapshot, "keyframes_reconstruction"):
                raw_keyframes = tuple(getattr(snapshot, "keyframes_reconstruction"))
                matrices: list[np.ndarray] = []
                keyframe_ids: list[str] = []
                for sample in raw_keyframes:
                    matrix = getattr(sample, "T_reconstruction_from_camera", None)
                    keyframe_id = str(getattr(sample, "keyframe_id", "")).strip()
                    if matrix is None or not keyframe_id:
                        continue
                    value = np.asarray(matrix, dtype=np.float64)
                    if value.shape == (4, 4) and np.isfinite(value).all():
                        matrices.append(np.array(value, copy=True))
                        keyframe_ids.append(keyframe_id)
                keyframe_array = (
                    np.asarray(matrices, dtype=np.float64)
                    if matrices
                    else np.zeros((0, 4, 4), dtype=np.float64)
                )
                self._pending_keyframes = (
                    keyframe_array,
                    tuple(keyframe_ids),
                    int(getattr(snapshot, "keyframe_count", keyframe_array.shape[0])),
                    int(getattr(snapshot, "revision", 0)),
                )
                visual_update_pending = True
        if visual_update_pending and not self._visual_pose_timer.isActive():
            self._visual_pose_timer.start()

    def _flush_pending_visual_pose(self) -> None:
        if self._viewport_shutdown:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            return
        matrix = self._pending_visual_pose
        landmark_update = self._pending_sparse_landmarks
        keyframe_update = self._pending_keyframes
        self._pending_visual_pose = None
        self._pending_sparse_landmarks = None
        self._pending_keyframes = None
        if matrix is None and landmark_update is None and keyframe_update is None:
            return
        try:
            if matrix is not None:
                self._reconstruction_canvas.update_live_pose(matrix)
            if landmark_update is not None:
                points, total_count, revision = landmark_update
                self._reconstruction_canvas.update_sparse_landmarks(
                    points, total_count=total_count, revision=revision
                )
            if keyframe_update is not None:
                matrices, keyframe_ids, total_count, revision = keyframe_update
                self._reconstruction_canvas.update_keyframe_poses(
                    matrices,
                    keyframe_ids=keyframe_ids,
                    total_count=total_count,
                    revision=revision,
                )
            if (
                self._reconstruction_canvas.has_pose
                or self._reconstruction_canvas.has_sparse_map
                or self._reconstruction_canvas.has_keyframes
            ):
                self._viewport_placeholder.setVisible(False)
            if not self._visualizer_fitted_once and (
                self._reconstruction_canvas.has_pose
                or self._reconstruction_canvas.has_sparse_map
                or self._reconstruction_canvas.has_keyframes
            ):
                self._reconstruction_canvas.fit_scene()
                self._visualizer_fitted_once = True
            self._refresh_viewport_state()
        except (TypeError, ValueError) as exc:
            # Presentation errors must never terminate the authoritative live
            # reconstruction runtime. Surface the issue as a UI warning only.
            self.set_error(f"Reconstruction viewport update rejected: {exc}")

    def _refresh_viewport_state(self) -> None:
        if self._scene_availability is ReconstructionSceneAvailability.READY:
            self._viewport_state.setText("Scene ready")
            return
        if self._reconstruction_canvas.offline_visualization_loaded:
            self._viewport_state.setText("Offline sparse reconstruction")
            return
        if self._reconstruction_canvas.semantic_objects_available:
            self._viewport_state.setText("Semantic object context")
            return
        if self._reconstruction_canvas.has_pose:
            suffix = " retained" if self._live_state in {
                LiveCaptureRuntimeState.STOPPED,
                LiveCaptureRuntimeState.FAILED,
            } else ""
            self._viewport_state.setText(f"Live camera pose{suffix}")
            return
        if self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }:
            self._viewport_state.setText("Waiting for live pose")
            return
        self._viewport_state.setText("Awaiting scene")

    def _refresh_runtime_telemetry(self) -> None:
        self._refresh_elapsed_label()
        controller = self._controller
        if controller is None:
            return
        if self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }:
            stats = controller.live_stats
            self._telemetry_value.setText(
                format_live_telemetry(
                    stats,
                    pose_count=self._live_pose_count,
                    keyframe_count=self._live_keyframe_count,
                    landmark_count=self._live_landmark_count,
                )
            )
            self._telemetry_value.setToolTip(
                "Live statistics are presentation-only telemetry from the authoritative live runtime."
            )
            self._apply_runtime_presentation(present_live_state(self._live_state, stats))

    def _on_offline_run_finished(self, _result: object) -> None:
        self._refresh_runtime_telemetry()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def _refresh_source_controls(self) -> None:
        controller = self._controller
        enabled = controller is not None and not controller.is_active and not controller.is_shutdown
        for button in self._source_buttons.values():
            button.setEnabled(enabled)

    def _refresh_runtime_controls(self) -> None:
        controller = self._controller
        if controller is None:
            self._start_button.setEnabled(False)
            self._pause_button.setEnabled(False)
            self._stop_button.setEnabled(False)
            return

        selection = controller.source_selection
        live_busy = self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }
        offline_busy = self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }
        busy = live_busy or offline_busy or controller.is_active
        can_start = (
            not controller.is_shutdown
            and selection.ready
            and selection.workflow is not ReconstructionWorkflow.NONE
            and not busy
        )
        self._start_button.setEnabled(can_start)

        can_pause = (
            not controller.is_shutdown
            and not offline_busy
            and self._live_state in {
                LiveCaptureRuntimeState.RUNNING,
                LiveCaptureRuntimeState.PAUSED,
            }
        )
        self._pause_button.setEnabled(can_pause)
        self._pause_button.setText(
            "Resume" if self._live_state is LiveCaptureRuntimeState.PAUSED else "Pause"
        )

        can_stop_live = self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }
        can_stop_offline = self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
        }
        self._stop_button.setEnabled(
            not controller.is_shutdown and (can_stop_live or can_stop_offline)
        )
        self._stop_button.setText("Cancel" if offline_busy else "Stop")

    def _clear_warning_if_possible(self) -> None:
        if self._controller is None or self._controller.is_shutdown:
            return
        if (
            self._controller.source_selection.workflow is ReconstructionWorkflow.LIVE_SOURCE
            and self._live_camera_mode == "estimated"
        ):
            self._readiness_badge.setText("TEST MODE • NON-METRIC")
            self._readiness_badge.setProperty("statusKind", "warning")
            self._readiness_badge.setToolTip(
                "Live reconstruction is using estimated intrinsics. Visual/relative testing is "
                "allowed; metric geometry and metrology are not certified."
            )
            self._refresh_style(self._readiness_badge)
            return
        self._readiness_badge.setText("CONTROLLER READY")
        self._readiness_badge.setProperty("statusKind", "")
        self._readiness_badge.setToolTip("")
        self._refresh_style(self._readiness_badge)

    @staticmethod
    def _workflow_label(workflow: ReconstructionWorkflow) -> str:
        labels = {
            ReconstructionWorkflow.NONE: "Choose a workflow",
            ReconstructionWorkflow.LIVE_SOURCE: "Live source",
            ReconstructionWorkflow.IMAGE_SEQUENCE: "Image sequence",
            ReconstructionWorkflow.RECORDED_VIDEO: "Recorded video",
        }
        return labels[workflow]

    def set_live_state(self, state: LiveCaptureRuntimeState) -> None:
        previous_state = self._live_state
        self._live_state = state
        if state is LiveCaptureRuntimeState.STARTING and previous_state is not LiveCaptureRuntimeState.STARTING:
            self._pending_visual_pose = None
            self._pending_sparse_landmarks = None
            self._pending_keyframes = None
            self._visual_pose_timer.stop()
            self._reset_offline_sfm_health()
            self._reset_offline_inspection()
            self._reset_semantic_presentation()
            self._reconstruction_canvas.clear_live_state()
            self._viewport_placeholder.setVisible(True)
            self._visualizer_fitted_once = False
            self._diagnostics_hud.set_presentation(self._live_diagnostics.reset_session())
        elif state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED}:
            self._diagnostics_hud.set_presentation(self._live_diagnostics.mark_terminal(True))
        elif state in {
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
        }:
            self._diagnostics_hud.set_presentation(self._live_diagnostics.mark_terminal(False))
        self._sync_diagnostics_visibility()
        self._live_value.setText(self._humanize(state))
        self._live_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._live_value)
        if state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        }:
            self._begin_runtime_clock("live")
        elif state in {LiveCaptureRuntimeState.STOPPED, LiveCaptureRuntimeState.FAILED}:
            self._freeze_runtime_clock()
        stats = self._controller.live_stats if self._controller is not None else None
        presentation = present_live_state(state, stats)
        if state is LiveCaptureRuntimeState.FAILED and self._last_runtime_error:
            presentation = ReconstructionStatusPresentation(
                presentation.stage_label,
                self._last_runtime_error,
                "error",
            )
        self._apply_runtime_presentation(presentation)
        self._refresh_session_label()
        self._refresh_viewport_state()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def set_offline_state(self, state: OfflineReconstructionRunState) -> None:
        self._offline_state = state
        self._offline_value.setText(self._humanize(state))
        self._offline_value.setProperty("statusKind", self._state_kind(state.value))
        self._refresh_style(self._offline_value)
        if state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        }:
            self._begin_runtime_clock("offline")
        elif state in {
            OfflineReconstructionRunState.COMPLETED,
            OfflineReconstructionRunState.FAILED,
            OfflineReconstructionRunState.CANCELLED,
        }:
            self._freeze_runtime_clock()
        # Preserve a richer worker event while RUNNING; state transitions own terminal presentation.
        if state is not OfflineReconstructionRunState.RUNNING:
            presentation = present_offline_state(state)
            if state is OfflineReconstructionRunState.FAILED and self._last_runtime_error:
                presentation = ReconstructionStatusPresentation(
                    presentation.stage_label,
                    self._last_runtime_error,
                    "error",
                )
            self._apply_runtime_presentation(presentation)
        self._refresh_session_label()
        self._refresh_source_controls()
        self._refresh_runtime_controls()

    def set_scene_availability(self, availability: ReconstructionSceneAvailability) -> None:
        self._scene_availability = availability
        self._scene_value.setText(self._humanize(availability))
        kind = {
            ReconstructionSceneAvailability.READY: "ready",
            ReconstructionSceneAvailability.DEGRADED: "warning",
            ReconstructionSceneAvailability.PARTIAL: "running",
            ReconstructionSceneAvailability.NONE: "neutral",
        }[availability]
        self._scene_value.setProperty("statusKind", kind)
        self._refresh_style(self._scene_value)
        self._refresh_inspection_selection()
        self._refresh_viewport_state()

    def set_persistent_scene(self, scene, manifest_path: str) -> None:
        self._persistent_scene = scene
        self._refresh_appearance_representation_controls()
        if scene is None:
            self._scene_manifest_value.setText("No persistent scene")
            self.clear_reconstruction_measurement_evidence()
            self.clear_reconstruction_inspection_qc_evidence()
            self.clear_reconstruction_reference_comparison()
            self._reset_semantic_presentation()
            self._refresh_inspection_selection()
            return
        path = Path(manifest_path).name if manifest_path else "Persistent scene ready"
        self._scene_manifest_value.setText(path)
        try:
            presentation = semantic_presentation_from_persistent_scene(scene)
        except (TypeError, ValueError) as exc:
            self._reset_semantic_presentation(hide=False)
            self._semantic_structure_value.setText("Semantic context unavailable")
            self._semantic_structure_value.setToolTip(str(exc))
            return
        if presentation is None:
            self._reset_semantic_presentation()
            return
        self._apply_semantic_presentation(presentation)

    def set_checkpoint(self, checkpoint, manifest_path: str) -> None:
        if checkpoint is None:
            self._checkpoint_value.setText("No checkpoint")
            return
        path = Path(manifest_path).name if manifest_path else "Checkpoint ready"
        self._checkpoint_value.setText(path)

    def set_error(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self._last_runtime_error = text
        self._readiness_badge.setText("CONTROLLER WARNING")
        self._readiness_badge.setProperty("statusKind", "warning")
        self._readiness_badge.setToolTip(text)
        self._refresh_style(self._readiness_badge)
        self._runtime_message.setText(text)
        self._runtime_message.setProperty("statusKind", "error")
        self._refresh_style(self._runtime_message)

    def _refresh_session_label(self) -> None:
        controller = self._controller
        if controller is None:
            return
        workflow = controller.source_selection.workflow
        if self._offline_state in {
            OfflineReconstructionRunState.STARTING,
            OfflineReconstructionRunState.RUNNING,
            OfflineReconstructionRunState.CANCELLING,
        } or workflow in {ReconstructionWorkflow.IMAGE_SEQUENCE, ReconstructionWorkflow.RECORDED_VIDEO}:
            session_id = controller.offline_session_id
        elif self._live_state in {
            LiveCaptureRuntimeState.STARTING,
            LiveCaptureRuntimeState.RUNNING,
            LiveCaptureRuntimeState.PAUSED,
            LiveCaptureRuntimeState.RECOVERING,
            LiveCaptureRuntimeState.STOPPING,
        } or workflow is ReconstructionWorkflow.LIVE_SOURCE:
            session_id = controller.live_session_id
        else:
            session_id = controller.offline_session_id or controller.live_session_id
        self._session_value.setText(session_id or "No active session")
        output_dir = controller.offline_output_dir
        if output_dir is not None:
            self._runtime_output_value.setText(output_dir.name)
            self._runtime_output_value.setToolTip(str(output_dir))

    def shutdown_viewport(self) -> None:
        """Release the dedicated reconstruction GL canvas during app shutdown."""

        if self._viewport_shutdown:
            return
        self._viewport_shutdown = True
        self._visual_pose_timer.stop()
        self._pending_visual_pose = None
        self._pending_sparse_landmarks = None
        self._pending_keyframes = None
        if hasattr(self, "_orientation_gizmo"):
            self._orientation_gizmo.shutdown()
        self._reconstruction_canvas.close()

    @staticmethod
    def _humanize(value) -> str:
        raw = getattr(value, "value", value)
        return str(raw).replace("_", " ").strip().title() or "—"

    @staticmethod
    def _state_kind(value: str) -> str:
        normalized = str(value).strip().lower()
        if normalized in {"running", "starting", "recovering", "cancelling"}:
            return "running"
        if normalized in {"failed"}:
            return "error"
        if normalized in {"paused", "degraded"}:
            return "warning"
        if normalized in {"completed", "ready", "stopped"}:
            return "ready"
        return "neutral"

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)
