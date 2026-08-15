"""Professional Phase 9.2E Auto Segment and SAM2 prompt-refinement workspace."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QStandardPaths, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QToolButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.auto_segment_controller import AutoSegmentControllerState
from app.dialogs.manage_segmentation_classes_dialog import ManageSegmentationClassesDialog
from app.widgets.auto_segment_canvas import AutoSegmentCanvas
from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.models.metadata import ConfidenceKind
from core.perception.result import DenseMaskPayload, InstanceMask, MaskSource
from core.perception.semantic_project import (
    SegmentationProject,
    SegmentationProjectStore,
    add_project_class,
    create_segmentation_project,
    merge_project_catalog_into_workspace,
)
from core.perception.semantic_workspace_persistence import SegmentationWorkspaceStore
from core.perception.reviewed_analysis import source_image_sha256
from core.perception.semantic_workspace import (
    SemanticReviewStatus,
    SemanticWorkspace,
    append_workspace_mask_proposal,
    assign_workspace_class,
    delete_workspace_object,
    merge_workspace_objects,
    replace_workspace_object_mask,
    set_workspace_review_status,
    workspace_from_mask_proposals,
)


class _ProposalListRow(QWidget):
    """Compact proposal row with merge selection and persistent delete affordance."""

    activated = pyqtSignal(str)
    delete_requested = pyqtSignal(str)
    merge_toggled = pyqtSignal(str, bool)

    def __init__(
        self,
        object_id: str,
        label: str,
        *,
        merge_checked: bool,
        delete_enabled: bool,
        delete_tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._object_id = str(object_id).strip()
        self.setObjectName("autoSegmentProposalRow")
        layout = QHBoxLayout(self)
        self.setMinimumHeight(30)
        layout.setContentsMargins(8, 3, 6, 3)
        layout.setSpacing(8)

        self.merge_check = QCheckBox()
        self.merge_check.setObjectName("autoSegmentProposalMergeCheck")
        self.merge_check.setToolTip("Select proposal for mask merge")
        self.merge_check.setChecked(bool(merge_checked))
        self.merge_check.toggled.connect(
            lambda checked: self.merge_toggled.emit(self._object_id, bool(checked))
        )
        layout.addWidget(self.merge_check)

        self.label = QLabel(str(label))
        self.label.setObjectName("autoSegmentProposalRowLabel")
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.label, stretch=1)

        self.delete_button = QToolButton()
        self.delete_button.setObjectName("autoSegmentProposalDeleteButton")
        self.delete_button.setText("×")
        self.delete_button.setAutoRaise(True)
        self.delete_button.setFixedSize(24, 24)
        self.delete_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete_button.setEnabled(bool(delete_enabled))
        self.delete_button.setToolTip(str(delete_tooltip))
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self._object_id))
        layout.addWidget(self.delete_button)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self.activated.emit(self._object_id)
        super().mousePressEvent(event)

    def set_merge_checked(self, checked: bool) -> None:
        self.merge_check.blockSignals(True)
        try:
            self.merge_check.setChecked(bool(checked))
        finally:
            self.merge_check.blockSignals(False)


def default_segmentation_project_root() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not str(base).strip():
        base = str(Path.home() / ".local" / "share" / "eia-depth-3d-model-lab")
    return Path(base).expanduser().resolve() / "segmentation_projects"


class AutoSegmentWorkspaceDialog(QDialog):
    """Large modal semantic-review workspace; no geometry/metric authority is granted here."""

    run_requested = pyqtSignal(dict)
    prompt_requested = pyqtSignal(dict)
    cancel_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        project_store: SegmentationProjectStore | None = None,
        initial_bgr: np.ndarray | None = None,
        initial_source_label: str = "",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("autoSegmentWorkspaceDialog")
        self.setWindowTitle("Auto Segment Workspace")
        self.setModal(True)
        self.setMinimumSize(900, 620)
        self.resize(1180, 700)

        self._store = project_store or SegmentationProjectStore(default_segmentation_project_root())
        self._workspace_store = SegmentationWorkspaceStore(self._store.root.parent / "segmentation_workspaces")
        self._project: SegmentationProject | None = None
        self._workspace: SemanticWorkspace | None = None
        self._image_bgr: np.ndarray | None = None
        self._source_label = "No image selected"
        self._selected_id = ""
        self._running = False
        self._prompt_mode = "select"
        self._prompt_positive: list[tuple[float, float]] = []
        self._prompt_negative: list[tuple[float, float]] = []
        self._prompt_target_id = ""
        self._merge_selected_ids: set[str] = set()

        self._build_ui()
        self._refresh_projects()
        if initial_bgr is not None:
            self.set_source_image(initial_bgr, label=initial_source_label or "Captured Measure & Inspect frame")
        self._refresh_all()

    @property
    def workspace(self) -> SemanticWorkspace | None:
        return self._workspace

    @property
    def project(self) -> SegmentationProject | None:
        return self._project

    @property
    def source_image_bgr(self) -> np.ndarray | None:
        return None if self._image_bgr is None else self._image_bgr.copy()

    @property
    def source_label(self) -> str:
        return self._source_label

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(9)
        outer.addWidget(self._build_header())
        outer.addWidget(self._build_status())
        outer.addWidget(self._build_body(), stretch=1)
        outer.addWidget(self._build_footer())

    def _build_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("autoSegmentHeader")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)
        title_box = QVBoxLayout()
        title = QLabel("AUTO SEGMENT WORKSPACE")
        title.setObjectName("autoSegmentTitle")
        subtitle = QLabel("Generate unlabeled SAM2 proposals, then review and assign stable project classes.")
        subtitle.setObjectName("autoSegmentMuted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, stretch=1)

        layout.addWidget(QLabel("Project"))
        self._project_combo = QComboBox()
        self._project_combo.setObjectName("autoSegmentCombo")
        self._project_combo.setMinimumWidth(190)
        self._project_combo.currentIndexChanged.connect(self._on_project_changed)
        layout.addWidget(self._project_combo)
        new_project = QPushButton("New Project")
        new_project.setObjectName("autoSegmentSecondaryButton")
        new_project.clicked.connect(self._create_project)
        layout.addWidget(new_project)
        manage_classes = QPushButton("Manage Classes")
        manage_classes.setObjectName("autoSegmentSecondaryButton")
        manage_classes.clicked.connect(self._manage_classes)
        layout.addWidget(manage_classes)
        self._restore_workspace = QPushButton("Restore Saved")
        self._restore_workspace.setObjectName("autoSegmentSecondaryButton")
        self._restore_workspace.clicked.connect(self._restore_saved_workspace)
        layout.addWidget(self._restore_workspace)
        return frame

    def _build_status(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("autoSegmentStatusRow")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(8)
        self._status_badge = QLabel("READY")
        self._status_badge.setObjectName("autoSegmentStatusBadge")
        self._status_badge.setProperty("statusKind", "ready")
        layout.addWidget(self._status_badge)
        self._status_text = QLabel("Open an image or use the captured frame.")
        self._status_text.setObjectName("autoSegmentMuted")
        layout.addWidget(self._status_text, stretch=1)
        self._progress = QProgressBar()
        self._progress.setObjectName("autoSegmentProgress")
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(190)
        layout.addWidget(self._progress)
        return frame

    def _build_body(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(9)

        visual = QFrame()
        visual.setObjectName("autoSegmentPanel")
        visual_layout = QVBoxLayout(visual)
        visual_layout.setContentsMargins(10, 9, 10, 10)
        visual_layout.setSpacing(7)
        controls = QHBoxLayout()
        self._source_text = QLabel("No image selected")
        self._source_text.setObjectName("autoSegmentMuted")
        controls.addWidget(self._source_text, stretch=1)
        open_image = QPushButton("Open Image")
        open_image.setObjectName("autoSegmentSecondaryButton")
        open_image.clicked.connect(self._open_image)
        controls.addWidget(open_image)
        self._show_masks = QCheckBox("Masks")
        self._show_masks.setChecked(True)
        self._show_labels = QCheckBox("Labels")
        self._show_labels.setChecked(True)
        self._show_masks.toggled.connect(self._sync_overlay_options)
        self._show_labels.toggled.connect(self._sync_overlay_options)
        controls.addWidget(self._show_masks)
        controls.addWidget(self._show_labels)
        visual_layout.addLayout(controls)
        self._canvas = AutoSegmentCanvas()
        self._canvas.object_clicked.connect(self.select_object)
        self._canvas.prompt_point_clicked.connect(self._on_prompt_point)
        visual_layout.addWidget(self._canvas, stretch=1)
        splitter.addWidget(visual)

        sidebar = QScrollArea()
        sidebar.setObjectName("autoSegmentSidebar")
        sidebar.setWidgetResizable(True)
        sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar.setMinimumWidth(330)
        sidebar.setMaximumWidth(390)
        content = QWidget()
        side = QVBoxLayout(content)
        side.setContentsMargins(8, 8, 8, 8)
        side.setSpacing(8)
        side.addWidget(self._build_generation_card())
        side.addWidget(self._build_prompt_card())
        side.addWidget(self._build_merge_card())
        side.addWidget(self._build_objects_card(), stretch=1)
        side.addWidget(self._build_selected_card())
        side.addWidget(self._build_summary_card())
        side.addStretch(1)
        sidebar.setWidget(content)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([840, 340])
        return splitter

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("autoSegmentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)
        label = QLabel(title)
        label.setObjectName("autoSegmentCardTitle")
        layout.addWidget(label)
        return card, layout

    def _build_generation_card(self) -> QFrame:
        card, layout = self._card("AUTO SEGMENT")
        grid = QGridLayout()
        grid.setHorizontalSpacing(7)
        grid.setVerticalSpacing(6)
        grid.addWidget(QLabel("Profile"), 0, 0)
        self._profile = QComboBox()
        self._profile.setObjectName("autoSegmentCombo")
        self._profile.addItems(["Efficient", "Balanced", "Ultra"])
        grid.addWidget(self._profile, 0, 1)
        grid.addWidget(QLabel("Min pixels"), 1, 0)
        self._min_pixels = QSpinBox()
        self._min_pixels.setObjectName("autoSegmentSpin")
        self._min_pixels.setRange(16, 1000000)
        self._min_pixels.setValue(64)
        grid.addWidget(self._min_pixels, 1, 1)
        grid.addWidget(QLabel("Max proposals"), 2, 0)
        self._max_proposals = QSpinBox()
        self._max_proposals.setObjectName("autoSegmentSpin")
        self._max_proposals.setRange(1, 256)
        self._max_proposals.setValue(64)
        grid.addWidget(self._max_proposals, 2, 1)
        layout.addLayout(grid)
        row = QHBoxLayout()
        self._run = QPushButton("Run Auto Segment")
        self._run.setObjectName("autoSegmentPrimaryButton")
        self._run.clicked.connect(self._emit_run)
        row.addWidget(self._run, stretch=1)
        self._cancel = QPushButton("Cancel")
        self._cancel.setObjectName("autoSegmentDangerButton")
        self._cancel.clicked.connect(self.cancel_requested.emit)
        row.addWidget(self._cancel)
        layout.addLayout(row)
        return card

    def _build_prompt_card(self) -> QFrame:
        card, layout = self._card("INTERACTIVE SAM2")
        help_text = QLabel(
            "Segment a missed object with one click, or refine the selected mask with include/exclude points."
        )
        help_text.setObjectName("autoSegmentMuted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self._segment_object = QPushButton("+ Segment Object")
        self._segment_object.setObjectName("autoSegmentPrimaryButton")
        self._segment_object.clicked.connect(self._begin_segment_object)
        layout.addWidget(self._segment_object)

        row = QHBoxLayout()
        self._refine_positive = QPushButton("+ Include")
        self._refine_positive.setObjectName("autoSegmentSecondaryButton")
        self._refine_positive.clicked.connect(lambda: self._set_prompt_mode("refine_positive"))
        row.addWidget(self._refine_positive)
        self._refine_negative = QPushButton("− Exclude")
        self._refine_negative.setObjectName("autoSegmentSecondaryButton")
        self._refine_negative.clicked.connect(lambda: self._set_prompt_mode("refine_negative"))
        row.addWidget(self._refine_negative)
        layout.addLayout(row)

        self._prompt_summary = QLabel("Select an object to refine, or click + Segment Object.")
        self._prompt_summary.setObjectName("autoSegmentMuted")
        self._prompt_summary.setWordWrap(True)
        layout.addWidget(self._prompt_summary)

        apply_row = QHBoxLayout()
        self._apply_refine = QPushButton("Apply Refinement")
        self._apply_refine.setObjectName("autoSegmentPrimaryButton")
        self._apply_refine.clicked.connect(self._apply_refinement)
        apply_row.addWidget(self._apply_refine, stretch=1)
        self._clear_prompts = QPushButton("Clear")
        self._clear_prompts.setObjectName("autoSegmentMiniButton")
        self._clear_prompts.clicked.connect(self._clear_prompt_points)
        apply_row.addWidget(self._clear_prompts)
        layout.addLayout(apply_row)
        return card

    def _build_merge_card(self) -> QFrame:
        card, layout = self._card("MASK EDITING")
        help_text = QLabel("Check two or more proposals below, then merge them into one deterministic semantic mask.")
        help_text.setObjectName("autoSegmentMuted")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        self._merge_summary = QLabel("0 selected for merge")
        self._merge_summary.setObjectName("autoSegmentMuted")
        layout.addWidget(self._merge_summary)
        row = QHBoxLayout()
        self._merge_masks = QPushButton("Merge Selected")
        self._merge_masks.setObjectName("autoSegmentPrimaryButton")
        self._merge_masks.clicked.connect(self._merge_checked_objects)
        row.addWidget(self._merge_masks, stretch=1)
        self._clear_merge = QPushButton("Clear")
        self._clear_merge.setObjectName("autoSegmentMiniButton")
        self._clear_merge.clicked.connect(self._clear_merge_selection)
        row.addWidget(self._clear_merge)
        layout.addLayout(row)
        return card

    def _build_objects_card(self) -> QFrame:
        card, layout = self._card("PROPOSALS")
        self._object_list = QListWidget()
        self._object_list.setObjectName("autoSegmentObjectList")
        self._object_list.setMinimumHeight(150)
        self._object_list.currentItemChanged.connect(self._on_list_selection)
        self._object_list.itemChanged.connect(self._on_merge_item_changed)
        layout.addWidget(self._object_list)
        return card

    def _build_selected_card(self) -> QFrame:
        card, layout = self._card("SELECTED OBJECT")
        self._selected_status = QLabel("Select a proposal in the image or list.")
        self._selected_status.setObjectName("autoSegmentMuted")
        self._selected_status.setWordWrap(True)
        layout.addWidget(self._selected_status)
        layout.addWidget(QLabel("Class"))
        class_row = QHBoxLayout()
        self._class_combo = QComboBox()
        self._class_combo.setObjectName("autoSegmentCombo")
        self._class_combo.currentIndexChanged.connect(self._suggest_instance_name)
        class_row.addWidget(self._class_combo, stretch=1)
        new_class = QPushButton("+ New")
        new_class.setObjectName("autoSegmentMiniButton")
        new_class.clicked.connect(self._create_class)
        class_row.addWidget(new_class)
        layout.addLayout(class_row)
        layout.addWidget(QLabel("Instance"))
        self._instance_name = QLineEdit()
        self._instance_name.setObjectName("autoSegmentLineEdit")
        self._instance_name.setPlaceholderText("e.g. robot_arm_001")
        layout.addWidget(self._instance_name)
        self._assign = QPushButton("Assign Class")
        self._assign.setObjectName("autoSegmentSecondaryButton")
        self._assign.clicked.connect(self._assign_class)
        layout.addWidget(self._assign)
        review = QHBoxLayout()
        self._accept = QPushButton("Accept")
        self._accept.setObjectName("autoSegmentPrimaryButton")
        self._accept.clicked.connect(lambda: self._set_review(SemanticReviewStatus.ACCEPTED))
        review.addWidget(self._accept)
        self._pending = QPushButton("Pending")
        self._pending.setObjectName("autoSegmentSecondaryButton")
        self._pending.clicked.connect(lambda: self._set_review(SemanticReviewStatus.PENDING))
        review.addWidget(self._pending)
        self._reject = QPushButton("Reject")
        self._reject.setObjectName("autoSegmentDangerButton")
        self._reject.clicked.connect(lambda: self._set_review(SemanticReviewStatus.REJECTED))
        review.addWidget(self._reject)
        layout.addLayout(review)
        return card

    def _build_summary_card(self) -> QFrame:
        card, layout = self._card("REVIEW SUMMARY")
        self._summary = QLabel("No proposals")
        self._summary.setObjectName("autoSegmentMuted")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        note = QLabel("Semantic review only — labels and masks do not establish geometry, metric scale, or measurement authority.")
        note.setObjectName("autoSegmentWarningText")
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _build_footer(self) -> QFrame:
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 2, 0, 0)
        self._footer = QLabel("Project classes and reviewed masks persist per project. Semantic state never establishes geometry, metric, or measurement authority.")
        self._footer.setObjectName("autoSegmentMuted")
        layout.addWidget(self._footer, stretch=1)
        self._done = QPushButton("Done")
        self._done.setObjectName("autoSegmentSecondaryButton")
        self._done.clicked.connect(self.accept)
        layout.addWidget(self._done)
        return frame

    def accept(self) -> None:
        if self._running:
            QMessageBox.information(self, "Auto Segment", "Cancel the active Auto Segment job before closing the workspace.")
            return
        self._persist_workspace()
        super().accept()

    def reject(self) -> None:
        if self._running:
            QMessageBox.information(self, "Auto Segment", "Cancel the active Auto Segment job before closing the workspace.")
            return
        super().reject()

    def set_source_image(self, image_bgr: np.ndarray, *, label: str) -> None:
        image = np.asarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Auto Segment source must be an HxWx3 BGR image.")
        self._image_bgr = np.ascontiguousarray(image.copy())
        self._source_label = str(label).strip() or "Image"
        self._workspace = None
        self._selected_id = ""
        self._merge_selected_ids.clear()
        self._clear_prompt_points()
        self._canvas.set_image(self._image_bgr)
        self._canvas.set_workspace(None)
        self._source_text.setText(f"{self._source_label} · {image.shape[1]} × {image.shape[0]}")
        restored = self._restore_saved_workspace_if_exact_source()
        if not restored:
            self._status_text.setText("Ready to generate unlabeled SAM2 proposals.")
        self._refresh_all()

    def _persist_workspace(self) -> None:
        if self._project is None or self._workspace is None or self._image_bgr is None:
            return
        try:
            self._store.save(self._project)
            self._workspace_store.save(
                self._project,
                self._workspace,
                self._image_bgr,
                source_label=self._source_label,
            )
        except (ContractValidationError, OSError, ValueError) as exc:
            self._status_text.setText(f"Workspace persistence failed: {exc}")

    def _adopt_persisted_workspace(self, persisted) -> None:
        self._image_bgr = np.ascontiguousarray(persisted.image_bgr.copy())
        self._source_label = persisted.source_label
        self._workspace = persisted.workspace
        self._selected_id = ""
        self._merge_selected_ids.clear()
        self._clear_prompt_points()
        self._canvas.set_image(self._image_bgr)
        self._canvas.set_workspace(self._workspace)
        self._canvas.set_merge_selection(())
        self._source_text.setText(
            f"{self._source_label} · {self._image_bgr.shape[1]} × {self._image_bgr.shape[0]}"
        )
        self._status_text.setText(
            f"Restored saved workspace · {len(self._workspace.objects)} proposal(s) · revision {self._workspace.revision}."
        )

    def _restore_saved_workspace_if_exact_source(self) -> bool:
        if self._project is None or self._image_bgr is None or not self._workspace_store.contains(self._project.project_id):
            return False
        try:
            persisted = self._workspace_store.load(self._project)
        except (ContractValidationError, OSError, ValueError):
            return False
        if persisted.source_image_sha256 != source_image_sha256(self._image_bgr):
            return False
        self._adopt_persisted_workspace(persisted)
        return True

    def _restore_saved_workspace(self) -> None:
        if self._project is None:
            return
        if not self._workspace_store.contains(self._project.project_id):
            QMessageBox.information(self, "Restore workspace", "This project has no saved Auto Segment workspace yet.")
            return
        if self._workspace is not None and self._workspace.objects:
            answer = QMessageBox.question(
                self,
                "Restore workspace",
                "Restoring replaces the current in-dialog proposal review with the saved project workspace. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            persisted = self._workspace_store.load(self._project)
            self._adopt_persisted_workspace(persisted)
            self._refresh_all()
        except (ContractValidationError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Restore workspace", str(exc))

    def _manage_classes(self) -> None:
        if self._project is None:
            return
        dialog = ManageSegmentationClassesDialog(self._project, self._workspace, self)
        dialog.exec()
        project = dialog.project
        workspace = dialog.workspace
        if project == self._project and workspace == self._workspace:
            return
        try:
            self._store.save(project)
            self._project = project
            self._workspace = workspace
            if self._workspace is not None:
                self._canvas.set_workspace(self._workspace)
            self._refresh_projects(project.project_id)
            self._persist_workspace()
            self._refresh_all(keep_selection=True)
            self._status_text.setText("Project classes updated and persisted.")
        except (ContractValidationError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Manage classes", str(exc))

    def _open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image for Auto Segment", str(Path.home()), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Auto Segment", "The selected image could not be read.")
            return
        self.set_source_image(image, label=Path(path).name)

    def _refresh_projects(self, select_id: str = "") -> None:
        projects = list(self._store.list_projects())
        if not projects:
            project = create_segmentation_project("Auto Segment Project")
            self._store.save(project)
            projects = [project]
        wanted = select_id or (self._project.project_id if self._project else projects[0].project_id)
        self._project_combo.blockSignals(True)
        self._project_combo.clear()
        target_index = 0
        for index, project in enumerate(projects):
            self._project_combo.addItem(project.project_name, project.project_id)
            if project.project_id == wanted:
                target_index = index
        self._project_combo.setCurrentIndex(target_index)
        self._project_combo.blockSignals(False)
        self._project = projects[target_index]
        self._refresh_class_combo()

    def _on_project_changed(self, index: int) -> None:
        if index < 0:
            return
        project_id = str(self._project_combo.itemData(index) or "")
        if not project_id:
            return
        if self._workspace is not None and self._workspace.objects:
            answer = QMessageBox.question(self, "Switch segmentation project", "Switching projects clears the current proposal review. Continue?")
            if answer != QMessageBox.StandardButton.Yes:
                self._refresh_projects(self._project.project_id if self._project else "")
                return
            self._workspace = None
            self._selected_id = ""
            self._clear_prompt_points()
            self._canvas.set_workspace(None)
        self._project = self._store.load(project_id)
        self._refresh_class_combo()
        self._refresh_all()

    def _create_project(self) -> None:
        name, ok = QInputDialog.getText(self, "New segmentation project", "Project name")
        if not ok or not str(name).strip():
            return
        project = create_segmentation_project(str(name).strip())
        self._store.save(project)
        self._project = project
        self._refresh_projects(project.project_id)
        self._refresh_all()

    def _create_class(self) -> None:
        if self._project is None:
            return
        name, ok = QInputDialog.getText(self, "New class", "Class name")
        if not ok or not str(name).strip():
            return
        try:
            project, definition = add_project_class(self._project, str(name).strip())
            self._store.save(project)
            self._project = project
            if self._workspace is not None:
                self._workspace = merge_project_catalog_into_workspace(self._workspace, project)
                self._canvas.set_workspace(self._workspace)
            self._refresh_projects(project.project_id)
            self._select_class_id(definition.class_id)
            self._suggest_instance_name()
            self._persist_workspace()
        except (ContractValidationError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "New class", str(exc))

    def _refresh_class_combo(self) -> None:
        current = self._class_combo.currentData() if hasattr(self, "_class_combo") else None
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItem("Select class...", None)
        if self._project is not None:
            for definition in self._project.class_catalog.classes:
                self._class_combo.addItem(definition.class_name, definition.class_id)
        if current is not None:
            self._select_class_id(int(current), blocked=True)
        self._class_combo.blockSignals(False)

    def _select_class_id(self, class_id: int, *, blocked: bool = False) -> None:
        for index in range(self._class_combo.count()):
            if self._class_combo.itemData(index) == class_id:
                if blocked:
                    self._class_combo.setCurrentIndex(index)
                else:
                    self._class_combo.setCurrentIndex(index)
                return

    def _emit_run(self) -> None:
        if self._image_bgr is None:
            QMessageBox.information(self, "Auto Segment", "Open an image or use a captured frame first.")
            return
        self.run_requested.emit(
            {
                "image_bgr": self._image_bgr.copy(),
                "model_id": "ultralytics-sam2-prompted",
                "profile": self._profile.currentText().strip().casefold(),
                "minimum_mask_pixels": self._min_pixels.value(),
                "max_proposals": self._max_proposals.value(),
            }
        )

    def set_controller_state(self, state: AutoSegmentControllerState) -> None:
        self._running = state in {AutoSegmentControllerState.RUNNING, AutoSegmentControllerState.CANCELLING}
        kinds = {
            AutoSegmentControllerState.RUNNING: ("RUNNING", "running"),
            AutoSegmentControllerState.CANCELLING: ("CANCELLING", "warning"),
            AutoSegmentControllerState.COMPLETED: ("READY", "ready"),
            AutoSegmentControllerState.FAILED: ("FAILED", "error"),
            AutoSegmentControllerState.CANCELLED: ("CANCELLED", "warning"),
            AutoSegmentControllerState.IDLE: ("READY", "ready"),
        }
        text, kind = kinds[state]
        self._status_badge.setText(text)
        self._status_badge.setProperty("statusKind", kind)
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)
        self._run.setEnabled(not self._running and self._image_bgr is not None)
        self._cancel.setEnabled(self._running)
        if hasattr(self, "_restore_workspace"):
            self._restore_workspace.setEnabled(
                not self._running
                and self._project is not None
                and self._workspace_store.contains(self._project.project_id)
            )
        self._project_combo.setEnabled(not self._running)
        self._done.setEnabled(not self._running)
        self._refresh_prompt_controls()
        self._sync_merge_selection()
        self._sync_proposal_delete_controls()

    def set_progress(self, progress: int, stage: str, message: str) -> None:
        self._progress.setValue(max(0, min(100, int(progress))))
        self._status_text.setText(str(message).strip() or str(stage).replace("_", " ").title())

    def show_failure(self, message: str) -> None:
        self._status_text.setText(str(message))
        QMessageBox.warning(self, "Auto Segment", str(message))

    def load_job_result(self, result_dir: str | Path) -> None:
        if self._image_bgr is None:
            raise RuntimeError("Auto Segment result arrived without a source image.")
        root = Path(result_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        arrays = np.load(root / "masks.npz", allow_pickle=False)
        height, width = self._image_bgr.shape[:2]
        frame = FrameReference(0, 0, 0.0, ImageSize(width, height))
        space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
        provenance = ModelProvenance(
            model_id=str(manifest.get("model_id", "auto-segment")),
            display_name="SAM2 Auto Segment",
            task="automatic_segmentation",
        )
        masks: list[InstanceMask] = []
        for proposal in manifest.get("proposals", []):
            data = np.asarray(arrays[str(proposal["mask_key"])], dtype=bool)
            kind = ConfidenceKind(str(proposal.get("confidence_kind", ConfidenceKind.NONE.value)))
            confidence = proposal.get("confidence")
            masks.append(
                InstanceMask(
                    mask_id=str(proposal["mask_id"]),
                    frame_reference=frame,
                    pixel_space=space,
                    payload=DenseMaskPayload(data),
                    source=MaskSource.NATIVE_INSTANCE,
                    model_provenance=provenance,
                    confidence=None if confidence is None else float(confidence),
                    confidence_kind=kind,
                    foreground_valid=bool(np.any(data)),
                )
            )
        catalog = self._project.class_catalog if self._project is not None else None
        if masks:
            workspace = workspace_from_mask_proposals(
                masks,
                workspace_id=f"auto-segment-ui-{uuid4()}",
                class_catalog=catalog,
            )
            manifest_object_ids = tuple(
                str(proposal.get("object_id", "")).strip()
                for proposal in manifest.get("proposals", [])
            )
            if any(not object_id for object_id in manifest_object_ids):
                raise ContractValidationError(
                    "Auto Segment result manifest contains an empty proposal object_id."
                )
            if len(manifest_object_ids) != len(workspace.objects):
                raise ContractValidationError(
                    "Auto Segment result manifest proposal count does not match loaded masks."
                )
            self._workspace = SemanticWorkspace(
                workspace_id=workspace.workspace_id,
                frame_reference=workspace.frame_reference,
                class_catalog=workspace.class_catalog,
                objects=tuple(
                    replace(item, object_id=object_id)
                    for item, object_id in zip(workspace.objects, manifest_object_ids, strict=True)
                ),
                revision=workspace.revision,
            )
        else:
            self._workspace = SemanticWorkspace(
                workspace_id=f"auto-segment-ui-{uuid4()}",
                frame_reference=frame,
                class_catalog=catalog if catalog is not None else self._empty_catalog(),
                objects=(),
            )
        self._selected_id = ""
        self._merge_selected_ids.clear()
        self._canvas.set_workspace(self._workspace)
        self._canvas.set_merge_selection(())
        self._status_text.setText(f"Generated {len(self._workspace.objects)} unlabeled proposal(s). Click a mask to review it.")
        self._progress.setValue(100)
        self._persist_workspace()
        self._refresh_all()

    @staticmethod
    def _empty_catalog():
        from core.perception.semantic_workspace import SemanticClassCatalog
        return SemanticClassCatalog()

    def select_object(self, object_id: str) -> None:
        if self._workspace is None or self._workspace.object_by_id(object_id) is None:
            return
        key = str(object_id)
        if self._prompt_target_id and self._prompt_target_id != key:
            self._clear_prompt_points()
        self._selected_id = key
        self._canvas.set_selected_object(self._selected_id)
        for row in range(self._object_list.count()):
            item = self._object_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == self._selected_id:
                self._object_list.blockSignals(True)
                self._object_list.setCurrentRow(row)
                self._object_list.blockSignals(False)
                break
        obj = self._workspace.object_by_id(self._selected_id)
        assert obj is not None
        if obj.class_id is not None:
            self._select_class_id(obj.class_id)
        else:
            self._class_combo.setCurrentIndex(0)
        self._instance_name.setText(obj.instance_name)
        self._refresh_selected()

    def _on_list_selection(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        object_id = str(current.data(Qt.ItemDataRole.UserRole) or "")
        if object_id and object_id != self._selected_id:
            self.select_object(object_id)

    def _suggest_instance_name(self) -> None:
        if self._project is None or not self._selected_id:
            return
        class_id = self._class_combo.currentData()
        if class_id is None:
            return
        definition = self._project.class_by_id(int(class_id))
        if definition is None:
            return
        existing = tuple(
            item.instance_name for item in (self._workspace.objects if self._workspace else ())
            if item.instance_name and item.object_id != self._selected_id
        )
        current = self._workspace.object_by_id(self._selected_id) if self._workspace else None
        if current is None or not current.instance_name or current.class_id != definition.class_id:
            self._instance_name.setText(self._project.naming_policy.suggest(definition.class_name, existing_names=existing))

    def _assign_class(self) -> None:
        if self._workspace is None or self._project is None or not self._selected_id:
            return
        class_id = self._class_combo.currentData()
        if class_id is None:
            QMessageBox.information(self, "Assign class", "Choose an existing class or create a new one first.")
            return
        definition = self._project.class_by_id(int(class_id))
        if definition is None:
            return
        try:
            self._workspace = assign_workspace_class(
                self._workspace,
                self._selected_id,
                definition,
                instance_name=self._instance_name.text().strip(),
            )
            self._canvas.set_workspace(self._workspace)
            self._persist_workspace()
            self._refresh_all(keep_selection=True)
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Assign class", str(exc))

    def _set_review(self, status: SemanticReviewStatus) -> None:
        if self._workspace is None or not self._selected_id:
            return
        obj = self._workspace.object_by_id(self._selected_id)
        if status is SemanticReviewStatus.ACCEPTED and (obj is None or obj.class_id is None):
            QMessageBox.information(self, "Accept proposal", "Assign a real project class before accepting this proposal.")
            return
        try:
            self._workspace = set_workspace_review_status(self._workspace, self._selected_id, status)
            self._canvas.set_workspace(self._workspace)
            self._persist_workspace()
            self._refresh_all(keep_selection=True)
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Review proposal", str(exc))

    @staticmethod
    def _proposal_delete_policy(obj) -> tuple[bool, str]:
        if obj.review_status is SemanticReviewStatus.ACCEPTED:
            return False, "Accepted objects are protected. Return to Pending before deleting."
        if obj.merged_from_object_ids or obj.merged_from_mask_ids:
            return False, "Merged semantic results are protected to preserve review provenance."
        return True, "Delete proposal permanently from this saved workspace"

    def _on_row_merge_toggled(self, object_id: str, checked: bool) -> None:
        key = str(object_id).strip()
        if not key or self._workspace is None or self._workspace.object_by_id(key) is None:
            return
        # Keep the legacy QListWidgetItem check-state contract synchronized with
        # the visible custom-row checkbox. Existing Phase 9.2F callers/tests may
        # still toggle the backing item directly. The row widget is opaque, so Qt
        # does not visually paint a duplicate checkbox underneath it.
        for row in range(self._object_list.count()):
            item = self._object_list.item(row)
            if str(item.data(Qt.ItemDataRole.UserRole) or "").strip() != key:
                continue
            desired = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            if item.checkState() != desired:
                self._object_list.blockSignals(True)
                try:
                    item.setCheckState(desired)
                finally:
                    self._object_list.blockSignals(False)
            break
        if checked:
            self._merge_selected_ids.add(key)
        else:
            self._merge_selected_ids.discard(key)
        self._sync_merge_selection()

    def _delete_proposal(self, object_id: str) -> None:
        if self._workspace is None or self._running:
            return
        key = str(object_id).strip()
        obj = self._workspace.object_by_id(key)
        if obj is None:
            return
        allowed, reason = self._proposal_delete_policy(obj)
        if not allowed:
            QMessageBox.information(self, "Delete proposal", reason)
            return
        if obj.class_id is not None:
            name = obj.instance_name or obj.class_name or "this labeled proposal"
            answer = QMessageBox.question(
                self,
                "Delete labeled proposal",
                f"Delete {name!r} from this Auto Segment workspace?\n\n"
                "Its proposal mask and saved review state will be removed.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self._workspace = delete_workspace_object(self._workspace, key)
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Delete proposal", str(exc))
            return
        self._merge_selected_ids.discard(key)
        if self._selected_id == key:
            self._selected_id = ""
            self._clear_prompt_points()
            self._canvas.set_selected_object("")
        self._canvas.set_workspace(self._workspace)
        self._status_text.setText(
            "Proposal deleted from this workspace. Reopening the saved image will keep it removed; "
            "running Auto Segment again may generate a new proposal."
        )
        self._persist_workspace()
        self._refresh_all()

    def _on_merge_item_changed(self, item: QListWidgetItem) -> None:
        object_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not object_id:
            return
        checked = item.checkState() == Qt.CheckState.Checked
        row_widget = self._object_list.itemWidget(item)
        if isinstance(row_widget, _ProposalListRow):
            row_widget.set_merge_checked(checked)
        if checked:
            self._merge_selected_ids.add(object_id)
        else:
            self._merge_selected_ids.discard(object_id)
        self._sync_merge_selection()

    def _sync_proposal_delete_controls(self) -> None:
        if self._workspace is None or not hasattr(self, "_object_list"):
            return
        for row in range(self._object_list.count()):
            item = self._object_list.item(row)
            object_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            obj = self._workspace.object_by_id(object_id) if object_id else None
            row_widget = self._object_list.itemWidget(item)
            if obj is None or not isinstance(row_widget, _ProposalListRow):
                continue
            allowed, tooltip = self._proposal_delete_policy(obj)
            row_widget.delete_button.setEnabled(allowed and not self._running)
            row_widget.delete_button.setToolTip(tooltip)

    def _sync_merge_selection(self) -> None:
        if self._workspace is not None:
            valid = {item.object_id for item in self._workspace.objects}
            self._merge_selected_ids.intersection_update(valid)
            ordered = tuple(
                item.object_id for item in self._workspace.objects
                if item.object_id in self._merge_selected_ids
            )
        else:
            self._merge_selected_ids.clear()
            ordered = ()
        if hasattr(self, "_canvas"):
            self._canvas.set_merge_selection(ordered)
        if hasattr(self, "_merge_summary"):
            count = len(ordered)
            self._merge_summary.setText(f"{count} selected for merge")
            self._merge_masks.setEnabled(count >= 2 and not self._running)
            self._clear_merge.setEnabled(count > 0 and not self._running)

    def _clear_merge_selection(self) -> None:
        self._merge_selected_ids.clear()
        if hasattr(self, "_object_list"):
            self._object_list.blockSignals(True)
            try:
                for row in range(self._object_list.count()):
                    item = self._object_list.item(row)
                    item.setCheckState(Qt.CheckState.Unchecked)
                    row_widget = self._object_list.itemWidget(item)
                    if isinstance(row_widget, _ProposalListRow):
                        row_widget.set_merge_checked(False)
            finally:
                self._object_list.blockSignals(False)
        self._sync_merge_selection()

    def _merge_checked_objects(self) -> None:
        if self._workspace is None or self._running:
            return
        ordered = tuple(
            item.object_id for item in self._workspace.objects
            if item.object_id in self._merge_selected_ids
        )
        if len(ordered) < 2:
            QMessageBox.information(self, "Merge masks", "Check at least two proposal masks to merge.")
            return
        primary_id = self._selected_id if self._selected_id in self._merge_selected_ids else ordered[0]
        primary = self._workspace.object_by_id(primary_id)
        identity = (
            primary.instance_name or primary.class_name
            if primary is not None and primary.class_id is not None
            else "unlabeled merged proposal"
        )
        answer = QMessageBox.question(
            self,
            "Merge masks",
            f"Merge {len(ordered)} proposal masks into one {identity}?\n\n"
            "The source proposals will be replaced and the merged result will return to Pending review.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._workspace, merged_id = merge_workspace_objects(
                self._workspace,
                ordered,
                primary_object_id=primary_id,
            )
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Merge masks", str(exc))
            return
        self._merge_selected_ids.clear()
        self._selected_id = merged_id
        self._canvas.set_workspace(self._workspace)
        self._canvas.set_merge_selection(())
        self._clear_prompt_points()
        self._status_text.setText(
            f"Merged {len(ordered)} masks. Review the combined result and accept it again when correct."
        )
        self._persist_workspace()
        self._refresh_all(keep_selection=True)
        self.select_object(merged_id)

    def _begin_segment_object(self) -> None:
        if self._image_bgr is None:
            QMessageBox.information(self, "Segment object", "Open an image or use a captured frame first.")
            return
        if self._running:
            return
        self._clear_prompt_points()
        self._set_prompt_mode("segment_new")
        self._status_text.setText("Click the object you want SAM2 to segment.")

    def _set_prompt_mode(self, mode: str) -> None:
        if self._running:
            return
        if mode in {"refine_positive", "refine_negative"}:
            obj = self._workspace.object_by_id(self._selected_id) if self._workspace and self._selected_id else None
            if obj is None or obj.proposal_mask is None:
                QMessageBox.information(self, "Refine mask", "Select a mask proposal before adding refinement points.")
                return
            if self._prompt_target_id and self._prompt_target_id != self._selected_id:
                self._clear_prompt_points()
            self._prompt_target_id = self._selected_id
        self._prompt_mode = mode
        self._canvas.set_interaction_mode(mode)
        self._refresh_prompt_controls()

    def _on_prompt_point(self, x: int, y: int) -> None:
        if self._running or self._image_bgr is None:
            return
        point = (float(x), float(y))
        if self._prompt_mode == "segment_new":
            object_id = f"interactive:{uuid4().hex[:12]}"
            self._prompt_positive = [point]
            self._prompt_negative = []
            self._canvas.set_prompt_points(positive=tuple(self._prompt_positive), negative=())
            self.prompt_requested.emit(
                self._prompt_configuration(
                    operation="create",
                    object_id=object_id,
                    parent_mask_id="",
                    revision=0,
                    positive_points=tuple(self._prompt_positive),
                    negative_points=(),
                )
            )
            self._set_prompt_mode("select")
            return
        if self._prompt_mode == "refine_positive":
            self._prompt_positive.append(point)
        elif self._prompt_mode == "refine_negative":
            self._prompt_negative.append(point)
        else:
            return
        self._canvas.set_prompt_points(
            positive=tuple(self._prompt_positive),
            negative=tuple(self._prompt_negative),
        )
        self._refresh_prompt_controls()

    def _prompt_configuration(
        self,
        *,
        operation: str,
        object_id: str,
        parent_mask_id: str,
        revision: int,
        positive_points: tuple[tuple[float, float], ...],
        negative_points: tuple[tuple[float, float], ...],
    ) -> dict[str, object]:
        assert self._image_bgr is not None
        return {
            "image_bgr": self._image_bgr.copy(),
            "model_id": "ultralytics-sam2-prompted",
            "profile": self._profile.currentText().strip().casefold(),
            "minimum_mask_pixels": self._min_pixels.value(),
            "prompt_spec": {
                "operation": operation,
                "object_id": object_id,
                "parent_mask_id": parent_mask_id,
                "revision": int(revision),
                "positive_points": [list(point) for point in positive_points],
                "negative_points": [list(point) for point in negative_points],
            },
        }

    @staticmethod
    def _mask_anchor(mask: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            raise ContractValidationError("Cannot refine an empty mask.")
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        index = int(np.argmin((xs.astype(float) - cx) ** 2 + (ys.astype(float) - cy) ** 2))
        return float(xs[index]), float(ys[index])

    def _apply_refinement(self) -> None:
        if self._running or self._workspace is None or not self._selected_id:
            return
        obj = self._workspace.object_by_id(self._selected_id)
        if obj is None or obj.proposal_mask is None:
            return
        mask = np.asarray(obj.proposal_mask.payload.data, dtype=bool)
        anchor = self._mask_anchor(mask)
        positives = (anchor,) + tuple(self._prompt_positive)
        negatives = tuple(self._prompt_negative)
        self._prompt_target_id = self._selected_id
        self.prompt_requested.emit(
            self._prompt_configuration(
                operation="refine",
                object_id=self._selected_id,
                parent_mask_id=obj.proposal_mask.mask_id,
                revision=obj.proposal_mask.revision + 1,
                positive_points=positives,
                negative_points=negatives,
            )
        )
        self._set_prompt_mode("select")

    def _clear_prompt_points(self) -> None:
        self._prompt_positive = []
        self._prompt_negative = []
        self._prompt_target_id = ""
        self._prompt_mode = "select"
        if hasattr(self, "_canvas"):
            self._canvas.set_prompt_points(positive=(), negative=())
            self._canvas.set_interaction_mode("select")
        if hasattr(self, "_prompt_summary"):
            self._refresh_prompt_controls()

    def _refresh_prompt_controls(self) -> None:
        if not hasattr(self, "_segment_object"):
            return
        has_image = self._image_bgr is not None
        obj = self._workspace.object_by_id(self._selected_id) if self._workspace and self._selected_id else None
        can_refine = obj is not None and obj.proposal_mask is not None
        self._segment_object.setEnabled(has_image and not self._running)
        self._refine_positive.setEnabled(can_refine and not self._running)
        self._refine_negative.setEnabled(can_refine and not self._running)
        has_edits = bool(self._prompt_positive or self._prompt_negative)
        target_matches = bool(self._selected_id) and self._prompt_target_id == self._selected_id
        self._apply_refine.setEnabled(can_refine and target_matches and has_edits and not self._running)
        self._clear_prompts.setEnabled(has_edits and not self._running)
        if self._prompt_mode == "segment_new":
            text = "Click one point on the object to create a new SAM2 proposal."
        elif self._prompt_mode == "refine_positive":
            text = f"Include mode · {len(self._prompt_positive)} include · {len(self._prompt_negative)} exclude"
        elif self._prompt_mode == "refine_negative":
            text = f"Exclude mode · {len(self._prompt_positive)} include · {len(self._prompt_negative)} exclude"
        elif has_edits:
            text = f"{len(self._prompt_positive)} include · {len(self._prompt_negative)} exclude · Apply when ready"
        elif can_refine:
            text = "Selected mask ready for refinement. Add include/exclude points."
        else:
            text = "Select an object to refine, or click + Segment Object."
        self._prompt_summary.setText(text)

    def load_prompt_result(self, result_dir: str | Path) -> None:
        if self._image_bgr is None:
            raise RuntimeError("Prompted SAM2 result arrived without a source image.")
        root = Path(result_dir)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        data = np.asarray(np.load(root / "mask.npy", allow_pickle=False), dtype=bool)
        height, width = self._image_bgr.shape[:2]
        if data.shape != (height, width):
            raise ContractValidationError("Prompted SAM2 result mask does not match the source image.")
        operation = str(manifest.get("operation", "")).strip().lower()
        object_id = str(manifest.get("object_id", "")).strip()
        parent_mask_id = str(manifest.get("parent_mask_id", "")).strip()
        revision = int(manifest.get("revision", 0))
        if operation not in {"create", "refine"} or not object_id:
            raise ContractValidationError("Prompted SAM2 result manifest is invalid.")

        if self._workspace is not None:
            frame = self._workspace.frame_reference
            current = self._workspace.object_by_id(object_id)
            if operation == "refine" and (current is None or current.proposal_mask is None):
                raise ContractValidationError("Prompted refinement target no longer exists in the workspace.")
            space = (
                current.proposal_mask.pixel_space
                if current is not None and current.proposal_mask is not None
                else PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
            )
        else:
            frame = FrameReference(0, 0, 0.0, ImageSize(width, height))
            space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
            catalog = self._project.class_catalog if self._project is not None else self._empty_catalog()
            self._workspace = SemanticWorkspace(
                workspace_id=f"auto-segment-ui-{uuid4()}",
                frame_reference=frame,
                class_catalog=catalog,
                objects=(),
            )

        provenance = ModelProvenance(
            model_id=str(manifest.get("model_id", "ultralytics-sam2-prompted")),
            display_name="SAM2 Interactive Segmentation",
            task="prompted_segmentation",
        )
        confidence = manifest.get("confidence")
        kind = ConfidenceKind(str(manifest.get("confidence_kind", ConfidenceKind.NONE.value)))
        mask = InstanceMask(
            mask_id=str(manifest.get("mask_id", f"{object_id}:sam2:r{revision}")),
            frame_reference=frame,
            pixel_space=space,
            payload=DenseMaskPayload(data),
            source=MaskSource.PROMPTED_REFINEMENT,
            model_provenance=provenance,
            revision=revision,
            parent_mask_id=parent_mask_id,
            confidence=None if confidence is None else float(confidence),
            confidence_kind=kind,
            foreground_valid=bool(np.any(data)),
        )
        assert self._workspace is not None
        if operation == "create":
            self._workspace = append_workspace_mask_proposal(self._workspace, mask, object_id=object_id)
            self._status_text.setText("Created a new SAM2 proposal. Assign a class when the mask is correct.")
        else:
            self._workspace = replace_workspace_object_mask(self._workspace, object_id, mask)
            self._status_text.setText("Refined mask applied. Review and accept it again when correct.")
        self._selected_id = object_id
        self._canvas.set_workspace(self._workspace)
        self._clear_prompt_points()
        self._progress.setValue(100)
        self._persist_workspace()
        self._refresh_all(keep_selection=True)
        self.select_object(object_id)

    def _refresh_all(self, *, keep_selection: bool = False) -> None:
        self._refresh_class_combo()
        self._refresh_object_list()
        self._refresh_selected()
        self._refresh_summary()
        self._sync_overlay_options()
        self._run.setEnabled(not self._running and self._image_bgr is not None)
        self._cancel.setEnabled(self._running)
        if hasattr(self, "_restore_workspace"):
            self._restore_workspace.setEnabled(
                not self._running
                and self._project is not None
                and self._workspace_store.contains(self._project.project_id)
            )
        self._refresh_prompt_controls()
        self._sync_merge_selection()
        self._sync_proposal_delete_controls()
        if keep_selection and self._selected_id:
            self._canvas.set_selected_object(self._selected_id)

    def _refresh_object_list(self) -> None:
        self._object_list.blockSignals(True)
        self._object_list.clear()
        if self._workspace is not None:
            for index, obj in enumerate(self._workspace.objects, start=1):
                state = {
                    SemanticReviewStatus.PENDING: "○",
                    SemanticReviewStatus.ACCEPTED: "✓",
                    SemanticReviewStatus.REJECTED: "×",
                }[obj.review_status]
                name = obj.instance_name or obj.class_name or f"Proposal {index:02d}"
                display = f"{state}  {name}"
                # Keep the backing item checkable for the established Phase 9.2F
                # merge-selection contract, but leave its text empty. The opaque custom
                # row covers Qt's backing-item checkbox so only one checkbox is visible.
                item = QListWidgetItem()
                item.setData(Qt.ItemDataRole.UserRole, obj.object_id)
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                merge_checked = obj.object_id in self._merge_selected_ids
                item.setCheckState(
                    Qt.CheckState.Checked if merge_checked else Qt.CheckState.Unchecked
                )
                self._object_list.addItem(item)
                delete_enabled, delete_tooltip = self._proposal_delete_policy(obj)
                row_widget = _ProposalListRow(
                    obj.object_id,
                    display,
                    merge_checked=merge_checked,
                    delete_enabled=delete_enabled and not self._running,
                    delete_tooltip=delete_tooltip,
                    parent=self._object_list,
                )
                row_widget.activated.connect(self.select_object)
                row_widget.merge_toggled.connect(self._on_row_merge_toggled)
                row_widget.delete_requested.connect(self._delete_proposal)
                item.setSizeHint(row_widget.sizeHint())
                self._object_list.setItemWidget(item, row_widget)
                if obj.object_id == self._selected_id:
                    self._object_list.setCurrentItem(item)
        self._object_list.blockSignals(False)

    def _refresh_selected(self) -> None:
        obj = self._workspace.object_by_id(self._selected_id) if self._workspace and self._selected_id else None
        enabled = obj is not None
        for widget in (self._class_combo, self._instance_name, self._assign, self._accept, self._pending, self._reject):
            widget.setEnabled(enabled and not self._running)
        if obj is None:
            self._selected_status.setText("Select a proposal in the image or list.")
            return
        label = obj.class_name or "Unlabeled"
        instance = obj.instance_name or "No instance name"
        self._selected_status.setText(f"{label} · {instance} · {obj.review_status.value.title()}")
        self._refresh_prompt_controls()

    def _refresh_summary(self) -> None:
        if self._workspace is None:
            self._summary.setText("No proposals")
            return
        total = len(self._workspace.objects)
        accepted = sum(item.review_status is SemanticReviewStatus.ACCEPTED for item in self._workspace.objects)
        rejected = sum(item.review_status is SemanticReviewStatus.REJECTED for item in self._workspace.objects)
        pending = total - accepted - rejected
        unlabeled = sum(item.class_id is None for item in self._workspace.objects)
        self._summary.setText(f"{total} proposals · {accepted} accepted · {pending} pending · {rejected} rejected · {unlabeled} unlabeled")

    def _sync_overlay_options(self) -> None:
        self._canvas.set_overlay_options(masks=self._show_masks.isChecked(), labels=self._show_labels.isChecked())
