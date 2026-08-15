"""Professional Measure & Inspect page for Phase 7.5 Step 8."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.measurement_inspection_controller import InspectionControllerState
from app.measurement_canonical_registration_presentation import (
    MeasurementCanonicalRegistrationPresentation,
)
from app.widgets.inspection_3d_canvas import Inspection3DCanvas
from app.widgets.inspection_canvas import InspectionImageCanvas
from core.inspection import (
    InspectionBoxMethod,
    InspectionBundleSummary,
    InspectionObjectSummary,
    InspectionPointMode,
    MaskEditMode,
)
from core.measurement.contracts import ReliabilityLevel
from core.reconstruction.calibration_profile import CameraCalibrationProfile
from core.perception.metadata import PerceptionTask
from core.perception.registry import PERCEPTION_MODEL_REGISTRY
from core.perception.reviewed_analysis import ReviewedSemanticSelection


class InspectionPage(QWidget):
    capture_requested = pyqtSignal()
    analysis_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()
    recompute_requested = pyqtSignal(dict)
    export_requested = pyqtSignal(str)
    calibration_requested = pyqtSignal()
    auto_segment_requested = pyqtSignal()
    selected_object_changed = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("inspectionPage")
        self._captured_bgr: np.ndarray | None = None
        self._summary: InspectionBundleSummary | None = None
        self._arrays: dict[str, np.ndarray] = {}
        self._selected_id = ""
        self._manual_dirty = False
        self._reset_manual_requested = False
        self._locked_ids: set[str] = set()
        self._manual_revisions: dict[str, int] = {}
        self._depth_model_id = ""
        self._depth_model_label = "No depth model selected"
        self._source_available = False
        self._state = InspectionControllerState.IDLE
        self._export_running = False
        self._camera_calibration_profile: CameraCalibrationProfile | None = None
        self._reviewed_semantics: ReviewedSemanticSelection | None = None
        self._reviewed_semantic_source_label = ""
        self._canonical_registration_presentation: MeasurementCanonicalRegistrationPresentation | None = None
        self._build_ui()
        self._populate_perception_models()
        self.set_controller_state(InspectionControllerState.IDLE)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 12)
        root.setSpacing(10)
        root.addWidget(self._build_toolbar())
        root.addWidget(self._build_status_row())
        root.addWidget(self._build_body(), stretch=1)

    def _build_toolbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("inspectionToolbar")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)

        title = QLabel("MEASURE & INSPECT")
        title.setObjectName("inspectionTitle")
        layout.addWidget(title)
        self._depth_model = QLabel(self._depth_model_label)
        self._depth_model.setObjectName("inspectionModelBadge")
        layout.addWidget(self._depth_model)
        layout.addSpacing(8)

        layout.addWidget(QLabel("Perception"))
        self._perception_combo = QComboBox()
        self._perception_combo.setObjectName("inspectionCombo")
        self._perception_combo.setMinimumWidth(190)
        layout.addWidget(self._perception_combo)

        layout.addWidget(QLabel("Profile"))
        self._profile_combo = QComboBox()
        self._profile_combo.setObjectName("inspectionCombo")
        self._profile_combo.addItems(["Efficient", "Balanced", "Ultra"])
        layout.addWidget(self._profile_combo)

        layout.addWidget(QLabel("Objects"))
        self._max_objects = QSpinBox()
        self._max_objects.setObjectName("inspectionSpin")
        self._max_objects.setRange(1, 20)
        self._max_objects.setValue(5)
        layout.addWidget(self._max_objects)
        layout.addStretch(1)

        self._capture_btn = QPushButton("Capture Current Frame")
        self._capture_btn.setObjectName("inspectionSecondaryButton")
        self._capture_btn.clicked.connect(self.capture_requested.emit)
        layout.addWidget(self._capture_btn)

        self._analyze_btn = QPushButton("Analyze Frame")
        self._analyze_btn.setObjectName("inspectionPrimaryButton")
        self._analyze_btn.clicked.connect(self._emit_analysis)
        layout.addWidget(self._analyze_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("inspectionDangerButton")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self._cancel_btn)
        return frame

    def _build_status_row(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("inspectionStatusRow")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(10)
        self._status_badge = QLabel("IDLE")
        self._status_badge.setObjectName("inspectionStatusBadge")
        self._status_badge.setProperty("statusKind", "ready")
        layout.addWidget(self._status_badge)
        self._status_text = QLabel("Capture a frame from Live 3D to begin.")
        self._status_text.setObjectName("inspectionStatusText")
        layout.addWidget(self._status_text, stretch=1)
        self._progress = QProgressBar()
        self._progress.setObjectName("inspectionProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setProperty("statusKind", "ready")
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(220)
        layout.addWidget(self._progress)
        return frame

    def _build_body(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("inspectionMainSplitter")
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        visual_splitter = QSplitter(Qt.Orientation.Horizontal)
        visual_splitter.setChildrenCollapsible(False)
        visual_splitter.setHandleWidth(10)
        visual_splitter.addWidget(self._build_rgb_panel())
        visual_splitter.addWidget(self._build_3d_panel())
        visual_splitter.setStretchFactor(0, 1)
        visual_splitter.setStretchFactor(1, 1)
        visual_splitter.setSizes([720, 620])
        splitter.addWidget(visual_splitter)
        splitter.addWidget(self._build_sidebar())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1090, 360])
        return splitter

    def _panel_header(self, title: str, subtitle: str) -> QVBoxLayout:
        layout = QVBoxLayout()
        row = QHBoxLayout()
        label = QLabel(title)
        label.setObjectName("inspectionPanelTitle")
        row.addWidget(label)
        row.addStretch(1)
        layout.addLayout(row)
        description = QLabel(subtitle)
        description.setObjectName("inspectionMuted")
        description.setWordWrap(True)
        layout.addWidget(description)
        return layout

    def _build_rgb_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("inspectionPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)
        layout.addLayout(self._panel_header("ALIGNED RGB & MASK INSPECTION", "Click an object to select it. Include/exclude brush edits are non-destructive until recompute."))
        self._image_canvas = InspectionImageCanvas()
        self._image_canvas.object_clicked.connect(self.select_object)
        self._image_canvas.mask_edited.connect(self._on_mask_edited)
        layout.addWidget(self._image_canvas, stretch=1)
        return panel

    def _build_3d_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("inspectionPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)
        header = QHBoxLayout()
        title = QLabel("OBJECT POINT CLOUD & 3D BOX")
        title.setObjectName("inspectionPanelTitle")
        header.addWidget(title)
        header.addStretch(1)
        fit = QPushButton("Fit")
        fit.setObjectName("inspectionMiniButton")
        fit.clicked.connect(lambda: self._inspection_3d.fit_scene())
        reset = QPushButton("Reset View")
        reset.setObjectName("inspectionMiniButton")
        reset.clicked.connect(lambda: self._inspection_3d.reset_view())
        header.addWidget(fit)
        header.addWidget(reset)
        layout.addLayout(header)
        subtitle = QLabel("Inspect raw or filtered object-only points. Box rendering follows the selected measurement method.")
        subtitle.setObjectName("inspectionMuted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self._inspection_3d = Inspection3DCanvas(self)
        layout.addWidget(self._inspection_3d.native, stretch=1)
        return panel

    def _build_sidebar(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("inspectionSidebar")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_object_card())
        layout.addWidget(self._build_semantic_workspace_card())
        layout.addWidget(self._build_display_card())
        layout.addWidget(self._build_correction_card())
        layout.addWidget(self._build_calibration_card())
        layout.addWidget(self._build_measurement_card())
        layout.addWidget(self._build_canonical_registration_card())
        layout.addWidget(self._build_export_card())
        layout.addWidget(self._build_details_card())
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("inspectionCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(11, 10, 11, 11)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setObjectName("inspectionCardTitle")
        layout.addWidget(label)
        return frame, layout

    def _build_object_card(self) -> QFrame:
        frame, layout = self._card("OBJECTS")
        self._object_list = QListWidget()
        self._object_list.setObjectName("inspectionObjectList")
        self._object_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._object_list.currentItemChanged.connect(self._on_list_selection)
        self._object_list.setMinimumHeight(130)
        layout.addWidget(self._object_list)
        return frame

    def _build_semantic_workspace_card(self) -> QFrame:
        frame, layout = self._card("SEMANTIC WORKSPACE")
        detail = QLabel(
            "Review SAM2 masks and stable project classes before 3D analysis. "
            "Reviewed masks define object membership only; depth/calibration/scale remain separate authorities."
        )
        detail.setObjectName("inspectionMuted")
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self._reviewed_semantic_status = QLabel(
            "No accepted reviewed masks — Analyze Frame uses the selected perception model."
        )
        self._reviewed_semantic_status.setObjectName("inspectionMuted")
        self._reviewed_semantic_status.setWordWrap(True)
        layout.addWidget(self._reviewed_semantic_status)
        self._use_reviewed_semantics = QCheckBox("Use reviewed masks for Analyze Frame")
        self._use_reviewed_semantics.setChecked(False)
        self._use_reviewed_semantics.setEnabled(False)
        self._use_reviewed_semantics.toggled.connect(lambda _checked: self._apply_state())
        layout.addWidget(self._use_reviewed_semantics)
        button = QPushButton("Open Auto Segment")
        button.setObjectName("inspectionSecondaryButton")
        button.clicked.connect(self.auto_segment_requested.emit)
        layout.addWidget(button)
        return frame

    def set_reviewed_semantic_context(
        self,
        selection: ReviewedSemanticSelection | None,
        *,
        source_label: str = "",
    ) -> None:
        if selection is not None and not isinstance(selection, ReviewedSemanticSelection):
            raise TypeError("selection must be ReviewedSemanticSelection or None.")
        self._reviewed_semantics = selection
        self._reviewed_semantic_source_label = str(source_label).strip()
        if selection is None:
            self._use_reviewed_semantics.blockSignals(True)
            self._use_reviewed_semantics.setChecked(False)
            self._use_reviewed_semantics.setEnabled(False)
            self._use_reviewed_semantics.blockSignals(False)
            self._reviewed_semantic_status.setText(
                "No accepted reviewed masks — Analyze Frame uses the selected perception model."
            )
            self._apply_state()
            return
        count = len(selection.objects)
        if count <= self._max_objects.maximum() and self._max_objects.value() < count:
            self._max_objects.setValue(count)
        source = f" • {self._reviewed_semantic_source_label}" if self._reviewed_semantic_source_label else ""
        limit_note = (
            "" if count <= self._max_objects.maximum()
            else f" • reduce accepted objects to ≤ {self._max_objects.maximum()} before analysis"
        )
        self._reviewed_semantic_status.setText(
            f"{count} accepted reviewed object(s){source}. Analyze Frame can use these masks directly and skip YOLO/perception inference{limit_note}."
        )
        self._use_reviewed_semantics.blockSignals(True)
        self._use_reviewed_semantics.setEnabled(True)
        self._use_reviewed_semantics.setChecked(True)
        self._use_reviewed_semantics.blockSignals(False)
        self._apply_state()

    @property
    def use_reviewed_semantics_for_analysis(self) -> bool:
        return bool(
            self._reviewed_semantics is not None
            and self._use_reviewed_semantics.isEnabled()
            and self._use_reviewed_semantics.isChecked()
        )

    def _build_display_card(self) -> QFrame:
        frame, layout = self._card("DISPLAY")
        toggles = QGridLayout()
        self._show_mask = QCheckBox("Masks")
        self._show_mask.setChecked(True)
        self._show_boxes = QCheckBox("3D boxes")
        self._show_boxes.setChecked(True)
        self._show_labels = QCheckBox("Labels")
        self._show_labels.setChecked(True)
        self._show_uncertainty = QCheckBox("Uncertainty")
        self._show_uncertainty.setChecked(True)
        for index, widget in enumerate((self._show_mask, self._show_boxes, self._show_labels, self._show_uncertainty)):
            widget.toggled.connect(self._refresh_visuals)
            toggles.addWidget(widget, index // 2, index % 2)
        layout.addLayout(toggles)

        row = QGridLayout()
        row.addWidget(QLabel("Points"), 0, 0)
        self._point_mode = QComboBox()
        self._point_mode.setObjectName("inspectionCombo")
        self._point_mode.addItem("Filtered", InspectionPointMode.FILTERED)
        self._point_mode.addItem("Raw", InspectionPointMode.RAW)
        self._point_mode.currentIndexChanged.connect(self._refresh_visuals)
        row.addWidget(self._point_mode, 0, 1)
        row.addWidget(QLabel("Box method"), 1, 0)
        self._box_method = QComboBox()
        self._box_method.setObjectName("inspectionCombo")
        for text, method in (
            ("Auto — preferred", InspectionBoxMethod.AUTO),
            ("Axis-aligned", InspectionBoxMethod.AXIS_ALIGNED),
            ("Camera-aligned", InspectionBoxMethod.CAMERA_ALIGNED),
            ("Oriented", InspectionBoxMethod.ORIENTED),
        ):
            self._box_method.addItem(text, method)
        self._box_method.currentIndexChanged.connect(self._on_box_method_changed)
        row.addWidget(self._box_method, 1, 1)
        layout.addLayout(row)
        return frame

    def _build_correction_card(self) -> QFrame:
        frame, layout = self._card("MASK CORRECTION")
        mode_row = QHBoxLayout()
        self._edit_group = QButtonGroup(self)
        self._edit_group.setExclusive(True)
        self._select_tool = QPushButton("Select")
        self._include_tool = QPushButton("Include +")
        self._exclude_tool = QPushButton("Exclude −")
        for button, mode in (
            (self._select_tool, MaskEditMode.SELECT),
            (self._include_tool, MaskEditMode.INCLUDE),
            (self._exclude_tool, MaskEditMode.EXCLUDE),
        ):
            button.setObjectName("inspectionToolButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked, value=mode: self._set_edit_mode(value))
            self._edit_group.addButton(button)
            mode_row.addWidget(button)
        self._select_tool.setChecked(True)
        layout.addLayout(mode_row)

        brush = QHBoxLayout()
        brush.addWidget(QLabel("Brush"))
        self._brush = QSlider(Qt.Orientation.Horizontal)
        self._brush.setRange(3, 80)
        self._brush.setValue(16)
        self._brush.valueChanged.connect(self._image_canvas.set_brush_radius)
        brush.addWidget(self._brush, stretch=1)
        self._brush_value = QLabel("16 px")
        self._brush.valueChanged.connect(lambda value: self._brush_value.setText(f"{value} px"))
        brush.addWidget(self._brush_value)
        layout.addLayout(brush)

        self._correction_status = QLabel("Automatic mask — no edits")
        self._correction_status.setObjectName("inspectionMuted")
        self._correction_status.setWordWrap(True)
        layout.addWidget(self._correction_status)

        buttons = QGridLayout()
        self._recompute_btn = QPushButton("Apply & Recompute")
        self._recompute_btn.setObjectName("inspectionPrimaryButton")
        self._recompute_btn.clicked.connect(self._emit_recompute)
        self._reset_mask_btn = QPushButton("Reset Mask")
        self._reset_mask_btn.setObjectName("inspectionSecondaryButton")
        self._reset_mask_btn.clicked.connect(self._reset_mask)
        self._lock_btn = QPushButton("Lock Approved Result")
        self._lock_btn.setObjectName("inspectionSecondaryButton")
        self._lock_btn.clicked.connect(self._toggle_lock)
        buttons.addWidget(self._recompute_btn, 0, 0, 1, 2)
        buttons.addWidget(self._reset_mask_btn, 1, 0)
        buttons.addWidget(self._lock_btn, 1, 1)
        layout.addLayout(buttons)
        return frame

    def _build_calibration_card(self) -> QFrame:
        frame, layout = self._card("CAMERA CALIBRATION")
        self._calibration_state = QLabel("Not selected")
        self._calibration_state.setObjectName("inspectionWarningText")
        self._calibration_state.setWordWrap(True)
        layout.addWidget(self._calibration_state)
        self._calibration_detail = QLabel(
            "Model/estimated intrinsics will be used. Metric scale remains a separate authority."
        )
        self._calibration_detail.setObjectName("inspectionMuted")
        self._calibration_detail.setWordWrap(True)
        layout.addWidget(self._calibration_detail)
        button = QPushButton("Manage Calibration")
        button.setObjectName("inspectionSecondaryButton")
        button.clicked.connect(self.calibration_requested.emit)
        layout.addWidget(button)
        return frame

    def set_camera_calibration_context(
        self, profile: CameraCalibrationProfile | None
    ) -> None:
        if profile is not None and not isinstance(profile, CameraCalibrationProfile):
            raise TypeError("profile must be CameraCalibrationProfile or None.")
        self._camera_calibration_profile = profile
        if profile is None:
            self._calibration_state.setText("Not selected")
            self._calibration_detail.setText(
                "Model/estimated intrinsics will be used. Metric scale remains a separate authority."
            )
            return
        state = profile.validation_state.value.replace("_", " ").title()
        self._calibration_state.setText(f"{state} • {profile.profile_id}")
        size = profile.calibration_image_size
        self._calibration_detail.setText(
            f"{size.width} × {size.height} • {profile.method.value.replace('_', ' ').title()} • "
            "applied safely at analysis time. Metric scale remains separate."
        )

    def _build_measurement_card(self) -> QFrame:
        frame, layout = self._card("MEASUREMENTS")
        self._uncertainty_notice = QLabel("Estimated Uncertainty — Not Calibrated")
        self._uncertainty_notice.setObjectName("inspectionWarningText")
        self._uncertainty_notice.setWordWrap(True)
        layout.addWidget(self._uncertainty_notice)
        self._measurement_table = QTableWidget(0, 4)
        self._measurement_table.setObjectName("inspectionMeasurementTable")
        self._measurement_table.setHorizontalHeaderLabels(["Measurement", "Value", "Reliability", "Status"])
        self._measurement_table.verticalHeader().setVisible(False)
        self._measurement_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._measurement_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._measurement_table.horizontalHeader().setStretchLastSection(True)
        self._measurement_table.setMinimumHeight(250)
        layout.addWidget(self._measurement_table)
        return frame

    def _build_canonical_registration_card(self) -> QFrame:
        frame, layout = self._card("CANONICAL REGISTRATION")
        notice = QLabel(
            "Optional Digital Twin placement. Missing calibration/scale/registration never blocks the original measurement workflow."
        )
        notice.setObjectName("inspectionMuted")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self._canonical_registration_state = QLabel("Unavailable")
        self._canonical_registration_state.setObjectName("inspectionWarningText")
        self._canonical_registration_state.setWordWrap(True)
        layout.addWidget(self._canonical_registration_state)
        self._canonical_registration_binding = QLabel("No canonical scene/object/camera binding")
        self._canonical_registration_binding.setObjectName("inspectionMuted")
        self._canonical_registration_binding.setWordWrap(True)
        layout.addWidget(self._canonical_registration_binding)
        self._canonical_registration_dimensions = QLabel("No canonical dimension geometry")
        self._canonical_registration_dimensions.setObjectName("inspectionMuted")
        self._canonical_registration_dimensions.setWordWrap(True)
        layout.addWidget(self._canonical_registration_dimensions)
        self._canonical_registration_uncertainty = QLabel("No canonical dimension placement uncertainty")
        self._canonical_registration_uncertainty.setObjectName("inspectionMuted")
        self._canonical_registration_uncertainty.setWordWrap(True)
        layout.addWidget(self._canonical_registration_uncertainty)
        self._canonical_registration_reason = QLabel("No qualified measurement→canonical registration")
        self._canonical_registration_reason.setObjectName("inspectionMuted")
        self._canonical_registration_reason.setWordWrap(True)
        layout.addWidget(self._canonical_registration_reason)
        return frame

    @property
    def bundle_summary(self) -> InspectionBundleSummary | None:
        return self._summary

    @property
    def canonical_registration_presentation(self) -> MeasurementCanonicalRegistrationPresentation | None:
        return self._canonical_registration_presentation

    def set_canonical_registration_presentation(
        self,
        presentation: MeasurementCanonicalRegistrationPresentation | None,
    ) -> None:
        if presentation is not None and not isinstance(
            presentation, MeasurementCanonicalRegistrationPresentation
        ):
            raise TypeError(
                "presentation must be MeasurementCanonicalRegistrationPresentation or None."
            )
        self._canonical_registration_presentation = presentation
        if presentation is None:
            self._canonical_registration_state.setText("Unavailable")
            self._canonical_registration_binding.setText("No canonical scene/object/camera binding")
            self._canonical_registration_dimensions.setText("No canonical dimension geometry")
            self._canonical_registration_uncertainty.setText(
                "No canonical dimension placement uncertainty"
            )
            self._canonical_registration_reason.setText(
                "No qualified measurement→canonical registration"
            )
            return
        self._canonical_registration_state.setText(presentation.status_text)
        self._canonical_registration_binding.setText(presentation.binding_text)
        self._canonical_registration_dimensions.setText(presentation.dimensions_text)
        self._canonical_registration_uncertainty.setText(presentation.uncertainty_text)
        self._canonical_registration_reason.setText(presentation.reason_text)
        self._canonical_registration_reason.setToolTip(presentation.authority_text)

    def _build_export_card(self) -> QFrame:
        frame, layout = self._card("EXPORT & REPORT")
        notice = QLabel("Exports the current approved inspection bundle without rerunning depth or perception.")
        notice.setObjectName("inspectionMuted")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        row = QHBoxLayout()
        self._export_selected_btn = QPushButton("Export Selected")
        self._export_selected_btn.setObjectName("inspectionSecondaryButton")
        self._export_selected_btn.clicked.connect(lambda: self.export_requested.emit("selected"))
        self._export_all_btn = QPushButton("Export All")
        self._export_all_btn.setObjectName("inspectionPrimaryButton")
        self._export_all_btn.clicked.connect(lambda: self.export_requested.emit("all"))
        row.addWidget(self._export_selected_btn)
        row.addWidget(self._export_all_btn)
        layout.addLayout(row)
        self._export_status = QLabel("No export created in this session.")
        self._export_status.setObjectName("inspectionMuted")
        self._export_status.setWordWrap(True)
        layout.addWidget(self._export_status)
        return frame

    def _build_details_card(self) -> QFrame:
        frame, layout = self._card("ADVANCED DETAILS")
        self._details = QTextEdit()
        self._details.setObjectName("inspectionDetails")
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(190)
        layout.addWidget(self._details)
        return frame

    def _populate_perception_models(self) -> None:
        self._perception_combo.clear()
        for descriptor in PERCEPTION_MODEL_REGISTRY.descriptors():
            if PerceptionTask.INSTANCE_SEGMENTATION not in descriptor.capabilities.tasks:
                continue
            if descriptor.development_only:
                continue
            self._perception_combo.addItem(descriptor.display_name, descriptor.model_id)
        if not self._perception_combo.count():
            self._perception_combo.addItem("Development Perception Mock", "development-perception-mock")

    def set_context(self, *, source_available: bool, depth_model_id: str, depth_model_label: str) -> None:
        self._source_available = bool(source_available)
        self._depth_model_id = str(depth_model_id)
        self._depth_model_label = str(depth_model_label)
        self._depth_model.setText(self._depth_model_label)
        self._apply_state()

    def set_captured_frame(self, image_bgr: np.ndarray) -> None:
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        self._captured_bgr = np.array(image, copy=True)
        rgb = image[..., ::-1]
        self._summary = None
        self._arrays = {}
        self.set_canonical_registration_presentation(None)
        self._object_list.clear()
        self._image_canvas.set_bundle(rgb, (), {})
        self._inspection_3d.clear()
        self._status_text.setText(f"Captured {image.shape[1]} × {image.shape[0]} frame. Ready to analyze.")
        self._progress.setValue(0)
        self.set_controller_state(InspectionControllerState.READY)

    def load_bundle(self, summary: InspectionBundleSummary, arrays: Mapping[str, np.ndarray]) -> None:
        previous = self._selected_id
        self._summary = summary
        self._arrays = {name: np.array(value, copy=True) for name, value in arrays.items()}
        if summary.calibration_status == "calibrated":
            self._uncertainty_notice.setText(
                "Estimated Uncertainty — Calibrated camera geometry; depth/scale uncertainty still applies"
            )
        else:
            self._uncertainty_notice.setText("Estimated Uncertainty — Not Calibrated")
        self._object_list.blockSignals(True)
        self._object_list.clear()
        for obj in summary.objects:
            item = QListWidgetItem(self._object_item_text(obj))
            item.setData(Qt.ItemDataRole.UserRole, obj.observation_id)
            self._object_list.addItem(item)
        self._object_list.blockSignals(False)
        selected = previous if any(value.observation_id == previous for value in summary.objects) else (summary.objects[0].observation_id if summary.objects else "")
        self._image_canvas.set_bundle(self._arrays["aligned_rgb"], summary.objects, self._arrays)
        if selected:
            self.select_object(selected)
        self._progress.setValue(100)
        self._status_text.setText(
            f"{len(summary.objects)} object(s) analyzed in {summary.elapsed_seconds:.2f}s. Select an object to inspect or correct."
        )
        self.set_controller_state(InspectionControllerState.COMPLETED)

    def _object_item_text(self, obj: InspectionObjectSummary) -> str:
        confidence = f"{obj.detection_confidence:.2f}" if obj.detection_confidence is not None else "—"
        track = f" · {obj.track_id}" if obj.track_id else ""
        identity = obj.instance_name or obj.class_name
        class_note = f" · {obj.class_name}" if obj.instance_name else ""
        return f"{obj.object_index + 1}. {identity}{class_note}  ·  {confidence}{track}\n{obj.measurement_status.upper()} · {obj.reliability_level.value.upper()}"

    def select_object(self, observation_id: str) -> None:
        if self._summary is None:
            return
        obj = next((value for value in self._summary.objects if value.observation_id == observation_id), None)
        if obj is None:
            return
        self._selected_id = observation_id
        self._image_canvas.set_selected_object(observation_id)
        for index in range(self._object_list.count()):
            item = self._object_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == observation_id:
                self._object_list.setCurrentItem(item)
                break
        self._manual_dirty = False
        self._reset_manual_requested = False
        self._update_object_panels(obj)
        self._refresh_visuals()
        self._apply_state()
        self.selected_object_changed.emit(obj)

    def _on_list_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.select_object(str(current.data(Qt.ItemDataRole.UserRole)))

    @property
    def selected_object(self) -> InspectionObjectSummary | None:
        if self._summary is None:
            return None
        return next((value for value in self._summary.objects if value.observation_id == self._selected_id), None)

    def _update_object_panels(self, obj: InspectionObjectSummary) -> None:
        self._populate_measurements(obj)
        diagnostics = dict(obj.diagnostics)
        details = [
            f"Object ID: {obj.observation_id}",
            f"Class: {obj.class_name} ({obj.class_id})",
            f"Instance: {obj.instance_name or 'not assigned'}",
            f"Semantic source: {obj.semantic_source}",
            f"Semantic review: {obj.semantic_review_status or 'not reviewed'}",
            f"Detection confidence: {obj.detection_confidence if obj.detection_confidence is not None else 'unavailable'}",
            f"Mask: {obj.mask_source} · revision {obj.mask_revision} · {obj.approval_status}",
            f"Extraction: {obj.extraction_status}",
            f"Fitting: {obj.fitting_status}",
            f"Preferred box: {obj.preferred_box_reason or obj.preferred_box_id}",
            f"Depth model: {self._summary.depth_model_name if self._summary else ''}",
            f"Perception model: {self._summary.perception_model_name if self._summary else ''}",
            f"Intrinsics: {self._summary.intrinsics_source if self._summary else ''}",
            f"Calibration: {self._summary.calibration_status if self._summary else ''}",
            f"Calibration profile: {self._summary.calibration_profile_id if self._summary and self._summary.calibration_profile_id else 'none'}",
            f"Calibration record: {self._summary.calibration_record_id if self._summary and self._summary.calibration_record_id else 'none'}",
            f"Calibration method: {self._summary.calibration_method if self._summary and self._summary.calibration_method else 'none'}",
            f"Calibration application: {self._summary.calibration_application_mode if self._summary and self._summary.calibration_application_mode else 'none'}",
            f"Distortion rectified: {'yes' if self._summary and self._summary.calibration_rectified else 'no'}",
            f"Depth scale: {self._summary.depth_scale if self._summary else ''}",
        ]
        counts = diagnostics.get("extraction_stage_counts", {})
        if counts:
            details.append("Point counts: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
        if obj.warnings:
            details.append("Warnings: " + "; ".join(obj.warnings))
        self._details.setPlainText("\n".join(details))
        locked = obj.observation_id in self._locked_ids
        self._correction_status.setText(
            "Approved result locked — unlock to edit."
            if locked else (
                f"Manual revision {obj.mask_revision} approved."
                if obj.mask_source == "manual_correction" else "Automatic mask — no edits"
            )
        )
        self._lock_btn.setText("Unlock Result" if locked else "Lock Approved Result")

    def _populate_measurements(self, obj: InspectionObjectSummary) -> None:
        self._measurement_table.setRowCount(len(obj.measurements))
        for row, measurement in enumerate(obj.measurements):
            kind = str(measurement.get("kind", ""))
            status = str(measurement.get("status", "unavailable"))
            value = self._format_measurement(measurement)
            reliability = str(measurement.get("reliability", "unavailable"))
            reasons = ", ".join(measurement.get("rejection_reasons", ()))
            values = [kind.replace("_", " ").title(), value, reliability.title(), status.title() if not reasons else f"{status.title()}: {reasons}"]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 2:
                    item.setForeground(self._reliability_color(reliability))
                if status != "available":
                    item.setForeground(QColor("#e8a83c"))
                self._measurement_table.setItem(row, column, item)
        self._measurement_table.resizeColumnsToContents()

    @staticmethod
    def _format_measurement(measurement: Mapping[str, object]) -> str:
        if measurement.get("status") != "available":
            return "—"
        value = float(measurement.get("value_si", 0.0))
        uncertainty = float(measurement.get("expanded_uncertainty_si", 0.0))
        kind = str(measurement.get("kind", ""))
        if kind in {"width", "height", "depth"}:
            return f"{value * 1000:.1f} ± {uncertainty * 1000:.1f} mm"
        if kind.startswith("orientation_"):
            return f"{math.degrees(value):.2f}° ± {math.degrees(uncertainty):.2f}°"
        if kind == "volume":
            return f"{value * 1000:.2f} ± {uncertainty * 1000:.2f} L"
        return f"{value:.3f} ± {uncertainty:.3f} m"

    @staticmethod
    def _reliability_color(value: str) -> QColor:
        return {
            ReliabilityLevel.HIGH.value: QColor("#2ecc71"),
            ReliabilityLevel.MEDIUM.value: QColor("#e8a83c"),
            ReliabilityLevel.LOW.value: QColor("#e0293e"),
        }.get(value, QColor("#9a9ba3"))

    def _emit_analysis(self) -> None:
        if self._captured_bgr is None:
            return
        self.analysis_requested.emit({
            "image_bgr": np.array(self._captured_bgr, copy=True),
            "depth_model_id": self._depth_model_id,
            "perception_model_id": str(self._perception_combo.currentData()),
            "depth_profile": self._profile_combo.currentText().lower(),
            "perception_profile": self._profile_combo.currentText().lower(),
            "max_objects": self._max_objects.value(),
            "box_method": self.current_box_method.value,
            "use_reviewed_semantics": self.use_reviewed_semantics_for_analysis,
        })

    def _emit_recompute(self) -> None:
        obj = self.selected_object
        if self._summary is None or obj is None or obj.observation_id in self._locked_ids:
            return
        mask = self._image_canvas.editable_mask if self._manual_dirty and not self._reset_manual_requested else None
        revision = max(obj.mask_revision, self._manual_revisions.get(obj.observation_id, obj.mask_revision)) + (1 if mask is not None else 0)
        if mask is not None:
            self._manual_revisions[obj.observation_id] = revision
        self.recompute_requested.emit({
            "base_state_path": self._summary.state_path,
            "observation_id": obj.observation_id,
            "manual_mask": mask,
            "manual_revision": revision,
            "box_method": self.current_box_method.value,
            "reset_manual": self._reset_manual_requested,
        })

    @property
    def current_box_method(self) -> InspectionBoxMethod:
        return self._box_method.currentData()

    @property
    def current_point_mode(self) -> InspectionPointMode:
        return self._point_mode.currentData()

    def _on_box_method_changed(self, _index: int = -1) -> None:
        self._refresh_visuals()
        self._recompute_btn.setText(
            "Apply & Recompute"
            if self.current_box_method is InspectionBoxMethod.AUTO
            else "Recompute with Selected Box"
        )
        self._apply_state()

    def _set_edit_mode(self, mode: MaskEditMode) -> None:
        self._image_canvas.set_edit_mode(mode)

    def _on_mask_edited(self, observation_id: str) -> None:
        if observation_id == self._selected_id:
            self._manual_dirty = True
            self._reset_manual_requested = False
            self._correction_status.setText("Manual mask edited — apply and recompute to make it authoritative.")
            self._apply_state()

    def _reset_mask(self) -> None:
        if self.selected_object is None:
            return
        self._image_canvas.reset_selected_mask()
        self._manual_dirty = False
        self._reset_manual_requested = self.selected_object.mask_source == "manual_correction"
        self._correction_status.setText(
            "Automatic mask restored — apply and recompute to discard the approved manual revision."
            if self._reset_manual_requested
            else "Automatic mask restored — no pending correction."
        )
        self._apply_state()

    def _toggle_lock(self) -> None:
        obj = self.selected_object
        if obj is None:
            return
        if obj.observation_id in self._locked_ids:
            self._locked_ids.remove(obj.observation_id)
        else:
            self._locked_ids.add(obj.observation_id)
        self._update_object_panels(obj)
        self._apply_state()

    def _refresh_visuals(self, *_args) -> None:
        self._image_canvas.set_overlay_options(
            masks=self._show_mask.isChecked(),
            boxes=self._show_boxes.isChecked(),
            labels=self._show_labels.isChecked(),
            uncertainty=self._show_uncertainty.isChecked(),
        )
        self._image_canvas.set_box_method(self.current_box_method)
        self._inspection_3d.set_object(
            self.selected_object,
            self._arrays,
            point_mode=self.current_point_mode,
            box_method=self.current_box_method,
            show_boxes=self._show_boxes.isChecked(),
        )

    def _set_status_visuals(self, text: str, kind: str) -> None:
        self._status_badge.setText(str(text))
        for widget in (self._status_badge, self._progress):
            widget.setProperty("statusKind", str(kind))
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def set_controller_state(self, state: InspectionControllerState) -> None:
        self._state = state
        labels = {
            InspectionControllerState.IDLE: ("READY", "ready"),
            InspectionControllerState.READY: ("READY", "ready"),
            InspectionControllerState.RUNNING: ("RUNNING", "running"),
            InspectionControllerState.RECOMPUTING: ("RECOMPUTING", "running"),
            InspectionControllerState.CANCELLING: ("CANCELLING", "warning"),
            InspectionControllerState.COMPLETED: ("COMPLETE", "ready"),
            InspectionControllerState.FAILED: ("FAILED", "error"),
            InspectionControllerState.CANCELLED: ("CANCELLED", "warning"),
        }
        text, kind = labels[state]
        self._set_status_visuals(text, kind)
        self._apply_state()

    def set_progress(self, progress: int, stage: str, message: str) -> None:
        self._progress.setValue(max(0, min(100, int(progress))))
        stage_text = str(stage).strip().upper() or "RUNNING"
        if stage_text in {"FAILED", "ERROR"}:
            kind = "error"
        elif stage_text in {"CANCELLED", "CANCELLING", "WARNING"}:
            kind = "warning"
        elif stage_text in {"COMPLETE", "COMPLETED", "SUCCESS"}:
            kind = "ready"
            stage_text = "COMPLETE"
        else:
            kind = "running"
        self._set_status_visuals(stage_text, kind)
        self._status_text.setText(message)

    def append_job_log(self, line: str) -> None:
        text = str(line).strip()
        if text:
            self._details.append(text)

    def show_failure(self, message: str) -> None:
        self._status_text.setText(message)
        self._details.append(f"ERROR: {message}")
        self.set_controller_state(InspectionControllerState.FAILED)

    def export_context(self, scope: str) -> dict[str, object]:
        if self._summary is None:
            raise RuntimeError("Analyze a frame before exporting.")
        observation_ids: tuple[str, ...]
        if str(scope) == "selected":
            if not self._selected_id:
                raise RuntimeError("Select an object before exporting it.")
            observation_ids = (self._selected_id,)
        else:
            observation_ids = tuple(obj.observation_id for obj in self._summary.objects)
        return {
            "summary": self._summary,
            "arrays": self._arrays,
            "observation_ids": observation_ids,
            "locked_ids": tuple(sorted(self._locked_ids)),
        }

    def set_export_running(self, running: bool) -> None:
        self._export_running = bool(running)
        if running:
            self._export_status.setText("Exporting approved inspection results...")
        self._apply_state()

    def set_export_progress(self, progress: int, stage: str, message: str) -> None:
        self._export_status.setText(f"{str(stage).replace('_', ' ').title()} ({int(progress)}%): {message}")

    def show_export_success(self, path: str, object_count: int, file_count: int) -> None:
        self._export_status.setText(
            f"Export complete: {object_count} object(s), {file_count} files. {path}"
        )

    def show_export_failure(self, message: str) -> None:
        self._export_status.setText(f"Export failed: {message}")
        self._details.append(f"EXPORT ERROR: {message}")

    def _apply_state(self) -> None:
        running = self._state in {
            InspectionControllerState.RUNNING,
            InspectionControllerState.RECOMPUTING,
            InspectionControllerState.CANCELLING,
        }
        has_capture = self._captured_bgr is not None
        has_result = self._summary is not None and self.selected_object is not None
        locked = bool(self.selected_object and self.selected_object.observation_id in self._locked_ids)
        self._capture_btn.setEnabled(self._source_available and not running)
        self._analyze_btn.setEnabled(has_capture and bool(self._depth_model_id) and not running)
        self._cancel_btn.setEnabled(running)
        use_reviewed = self.use_reviewed_semantics_for_analysis
        self._perception_combo.setEnabled(not running and not use_reviewed)
        self._profile_combo.setEnabled(not running)
        self._max_objects.setEnabled(not running)
        self._object_list.setEnabled(has_result and not running)
        self._recompute_btn.setEnabled(has_result and not running and not locked and (self._manual_dirty or self._reset_manual_requested or self.current_box_method is not InspectionBoxMethod.AUTO))
        self._reset_mask_btn.setEnabled(has_result and not running and not locked)
        self._lock_btn.setEnabled(has_result and not running)
        self._export_selected_btn.setEnabled(has_result and not running and not self._export_running)
        self._export_all_btn.setEnabled(bool(self._summary and self._summary.objects) and not running and not self._export_running)
        editing = has_result and not running and not locked
        self._include_tool.setEnabled(editing)
        self._exclude_tool.setEnabled(editing)
        self._brush.setEnabled(editing)
        self._image_canvas.set_editing_enabled(editing)
        if running:
            self._status_badge.setText("RUNNING")

    def close(self) -> None:
        self._inspection_3d.close()
