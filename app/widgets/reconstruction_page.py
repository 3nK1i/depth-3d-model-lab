"""Reconstruction workspace through Phase 8.8C1.

B5 enabled primary navigation. C1 replaces the placeholder with a real VisPy
reconstruction viewport foundation while preserving the authoritative B3/B4
runtime, status, scale, and calibration contracts.
"""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.offline_reconstruction_controller import OfflineReconstructionRunState
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
from app.widgets.viewport.vispy_canvas import ReconstructionCanvas
from core.reconstruction import (
    CameraIntrinsicsPolicy,
    LiveCaptureRuntimeState,
    ReconstructionCamera,
    load_live_camera_calibration,
)


class ReconstructionPage(QWidget):
    """Professional Reconstruction workspace shell with passive state binding."""

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
        self._viewport_shutdown = False
        self._pending_visual_pose: np.ndarray | None = None
        self._visualizer_fitted_once = False
        self._viewport_tool_buttons: dict[str, IconButton] = {}
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
            "World axes, grid, and current camera pose are active. Live trajectory, sparse/canonical geometry, and neural appearance are introduced progressively from Phase 8.8C2 onward."
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

        grid.addWidget(
            tools,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
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
        grid.setRowStretch(0, 1)
        grid.setColumnStretch(0, 1)
        layout.addWidget(host, stretch=1)
        return card

    def _set_viewport_tool(self, mode: str) -> None:
        self._reconstruction_canvas.set_interaction_mode(mode)

    def _build_settings_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("reconstructionSettingsPanel")
        panel.setMinimumWidth(290)
        panel.setMaximumWidth(380)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("RECONSTRUCTION SETTINGS")
        title.setObjectName("reconstructionSectionTitle")
        layout.addWidget(title)

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
        return label

    @staticmethod
    def _add_status_row(layout: QVBoxLayout, label_text: str, value: QLabel) -> None:
        row = QHBoxLayout()
        row.setSpacing(8)
        label = QLabel(label_text)
        label.setObjectName("reconstructionRowLabel")
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value)
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
        if not isinstance(sfm, dict):
            return
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

    def set_live_observation(self, observation: object) -> None:
        if self._viewport_shutdown:
            return
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
                    # per observation.
                    self._pending_visual_pose = np.array(value, copy=True)
                    if not self._visual_pose_timer.isActive():
                        self._visual_pose_timer.start()
        if bool(getattr(observation, "is_keyframe", False)):
            self._live_keyframe_count += 1
        snapshot = getattr(observation, "map_snapshot", None)
        if snapshot is not None:
            self._live_keyframe_count = max(
                self._live_keyframe_count,
                int(getattr(snapshot, "keyframe_count", 0)),
            )
            self._live_landmark_count = int(getattr(snapshot, "landmark_count", 0))

    def _flush_pending_visual_pose(self) -> None:
        if self._viewport_shutdown:
            self._pending_visual_pose = None
            return
        matrix = self._pending_visual_pose
        self._pending_visual_pose = None
        if matrix is None:
            return
        try:
            self._reconstruction_canvas.update_live_pose(matrix)
            self._viewport_placeholder.setVisible(False)
            if not self._visualizer_fitted_once:
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
            self._visual_pose_timer.stop()
            self._reconstruction_canvas.clear_live_state()
            self._viewport_placeholder.setVisible(True)
            self._visualizer_fitted_once = False
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
        self._refresh_viewport_state()

    def set_persistent_scene(self, scene, manifest_path: str) -> None:
        if scene is None:
            self._scene_manifest_value.setText("No persistent scene")
            return
        path = Path(manifest_path).name if manifest_path else "Persistent scene ready"
        self._scene_manifest_value.setText(path)

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
