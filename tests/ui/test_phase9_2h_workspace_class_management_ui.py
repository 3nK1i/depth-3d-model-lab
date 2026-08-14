from __future__ import annotations

from pathlib import Path

import numpy as np

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from app.dialogs.manage_segmentation_classes_dialog import ManageSegmentationClassesDialog
from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SegmentationProjectStore,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    add_project_class,
    assign_workspace_class,
    create_segmentation_project,
    create_segmentation_project_from_model_names,
    set_workspace_review_status,
)


def _reviewed(project, image: np.ndarray, class_id: int) -> SemanticWorkspace:
    definition = project.class_by_id(class_id)
    assert definition is not None
    h, w = image.shape[:2]
    frame = FrameReference(0, 0, 0.0, ImageSize(w, h))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    dense = np.zeros((h, w), dtype=bool)
    dense[8 : h - 8, 10 : w // 2] = True
    mask = InstanceMask(
        "mask-1", frame, space, DenseMaskPayload(dense), MaskSource.NATIVE_INSTANCE,
        ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )
    workspace = SemanticWorkspace(
        "ui-workspace", frame, project.class_catalog,
        (SemanticWorkspaceObject("proposal:1", proposal_mask=mask),),
    )
    workspace = assign_workspace_class(workspace, "proposal:1", definition, instance_name=f"{definition.class_name}_001")
    return set_workspace_review_status(workspace, "proposal:1", SemanticReviewStatus.ACCEPTED)


def test_9_2h_model_classes_are_visibly_protected(qtbot) -> None:
    project = create_segmentation_project_from_model_names(
        "Factory", {0: "person"}, source_model_id="yolo", project_id="factory"
    )
    dialog = ManageSegmentationClassesDialog(project)
    qtbot.addWidget(dialog)
    dialog._table.selectRow(0)
    dialog._sync_actions()
    assert not dialog._rename.isEnabled()
    assert not dialog._delete.isEnabled()
    assert "protected" in dialog._delete.toolTip().lower()


def test_9_2h_used_custom_class_delete_returns_objects_to_unlabeled(qtbot, monkeypatch) -> None:
    from PyQt6.QtWidgets import QMessageBox

    image = np.zeros((60, 90, 3), dtype=np.uint8)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    workspace = _reviewed(project, image, plant.class_id)
    dialog = ManageSegmentationClassesDialog(project, workspace)
    qtbot.addWidget(dialog)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    dialog._table.selectRow(0)
    dialog._delete_selected()

    assert dialog.project.class_by_id(plant.class_id) is None
    assert plant.class_id in dialog.project.retired_class_ids
    obj = dialog.workspace.objects[0]
    assert obj.class_id is None
    assert obj.review_status is SemanticReviewStatus.PENDING


def test_9_2h_dialog_auto_restores_exact_saved_source_after_restart(qtbot, tmp_path: Path) -> None:
    image = np.zeros((60, 90, 3), dtype=np.uint8)
    image[12:45, 20:50] = (30, 100, 170)
    store = SegmentationProjectStore(tmp_path / "projects")
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    store.save(project)

    first = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=image, initial_source_label="factory.png")
    qtbot.addWidget(first)
    first._project = project
    first._workspace = _reviewed(project, image, plant.class_id)
    first._canvas.set_workspace(first._workspace)
    first._persist_workspace()

    second = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=image, initial_source_label="factory.png")
    qtbot.addWidget(second)
    assert second.workspace is not None
    assert len(second.workspace.accepted_objects) == 1
    assert second.workspace.accepted_objects[0].instance_name == "plant_001"
    assert "Restored saved workspace" in second._status_text.text()
    assert second._restore_workspace.isEnabled()
