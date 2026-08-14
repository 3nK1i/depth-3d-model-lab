"""Professional safe class-management dialog for Phase 9.2H."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.contracts.validation import ContractValidationError
from core.perception.semantic_project import (
    SegmentationProject,
    rebind_project_catalog_to_workspace,
    rename_project_class,
    retire_project_class,
)
from core.perception.semantic_workspace import (
    SemanticClassOrigin,
    SemanticWorkspace,
    clear_workspace_class_assignment,
)


class ManageSegmentationClassesDialog(QDialog):
    """Rename/delete only user-owned classes and resolve active usage explicitly."""

    def __init__(
        self,
        project: SegmentationProject,
        workspace: SemanticWorkspace | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage Classes")
        self.setModal(True)
        self.resize(650, 430)
        self._project = project
        self._workspace = workspace

        layout = QVBoxLayout(self)
        title = QLabel("Project Classes")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)
        note = QLabel(
            "Model-origin classes are protected. Custom classes can be renamed or retired; "
            "used classes must first return their objects to Unlabeled. Retired IDs are never reused."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Class", "Source", "Usage", "Stable ID"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            self._table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._sync_actions)
        layout.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        self._rename = QPushButton("Rename")
        self._rename.clicked.connect(self._rename_selected)
        actions.addWidget(self._rename)
        self._delete = QPushButton("Delete")
        self._delete.clicked.connect(self._delete_selected)
        actions.addWidget(self._delete)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

        self._refresh()

    @property
    def project(self) -> SegmentationProject:
        return self._project

    @property
    def workspace(self) -> SemanticWorkspace | None:
        return self._workspace

    def _usage_count(self, class_id: int) -> int:
        if self._workspace is None:
            return 0
        return sum(item.class_id == class_id for item in self._workspace.objects)

    def _selected_class_id(self) -> int | None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self._table.item(rows[0].row(), 0)
        value = item.data(Qt.ItemDataRole.UserRole)
        return None if value is None else int(value)

    def _refresh(self, select_class_id: int | None = None) -> None:
        self._table.setRowCount(0)
        selected_row = -1
        for row, definition in enumerate(self._project.class_catalog.classes):
            self._table.insertRow(row)
            name = QTableWidgetItem(definition.class_name)
            name.setData(Qt.ItemDataRole.UserRole, definition.class_id)
            source = "Model" if definition.origin is SemanticClassOrigin.MODEL else definition.origin.value.title()
            usage = self._usage_count(definition.class_id)
            self._table.setItem(row, 0, name)
            self._table.setItem(row, 1, QTableWidgetItem(source))
            self._table.setItem(row, 2, QTableWidgetItem(str(usage)))
            self._table.setItem(row, 3, QTableWidgetItem(str(definition.class_id)))
            if definition.class_id == select_class_id:
                selected_row = row
        if selected_row >= 0:
            self._table.selectRow(selected_row)
        elif self._table.rowCount():
            self._table.selectRow(0)
        self._sync_actions()

    def _sync_actions(self) -> None:
        class_id = self._selected_class_id()
        definition = self._project.class_by_id(class_id) if class_id is not None else None
        editable = definition is not None and definition.origin is not SemanticClassOrigin.MODEL
        self._rename.setEnabled(editable)
        self._delete.setEnabled(editable)
        if definition is not None and definition.origin is SemanticClassOrigin.MODEL:
            self._delete.setToolTip("Model-origin classes are protected.")
            self._rename.setToolTip("Model-origin classes are protected.")
        else:
            self._delete.setToolTip("")
            self._rename.setToolTip("")

    def _rename_selected(self) -> None:
        class_id = self._selected_class_id()
        definition = self._project.class_by_id(class_id) if class_id is not None else None
        if definition is None or definition.origin is SemanticClassOrigin.MODEL:
            return
        name, ok = QInputDialog.getText(self, "Rename class", "Class name", text=definition.class_name)
        if not ok or not str(name).strip() or str(name).strip() == definition.class_name:
            return
        try:
            project = rename_project_class(self._project, definition.class_id, str(name).strip())
            workspace = self._workspace
            if workspace is not None:
                workspace = rebind_project_catalog_to_workspace(workspace, project)
            self._project = project
            self._workspace = workspace
            self._refresh(definition.class_id)
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Rename class", str(exc))

    def _delete_selected(self) -> None:
        class_id = self._selected_class_id()
        definition = self._project.class_by_id(class_id) if class_id is not None else None
        if definition is None:
            return
        if definition.origin is SemanticClassOrigin.MODEL:
            QMessageBox.information(self, "Delete class", "Model-origin classes are protected and cannot be deleted.")
            return

        usage = self._usage_count(definition.class_id)
        if usage:
            answer = QMessageBox.question(
                self,
                "Delete used class",
                f"{definition.class_name!r} is assigned to {usage} workspace object(s).\n\n"
                "Deleting it will return those objects to Unlabeled / Pending so no stale class identity survives. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        else:
            answer = QMessageBox.question(
                self,
                "Delete class",
                f"Delete custom class {definition.class_name!r}? Its stable ID {definition.class_id} will be permanently retired and never reused.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        try:
            workspace = self._workspace
            if workspace is not None and usage:
                for item in tuple(workspace.objects):
                    if item.class_id == definition.class_id:
                        workspace = clear_workspace_class_assignment(workspace, item.object_id)
            project = retire_project_class(self._project, definition.class_id)
            if workspace is not None:
                workspace = rebind_project_catalog_to_workspace(workspace, project)
            self._project = project
            self._workspace = workspace
            self._refresh()
        except ContractValidationError as exc:
            QMessageBox.warning(self, "Delete class", str(exc))
