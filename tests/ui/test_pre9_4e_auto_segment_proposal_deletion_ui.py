from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog, _ProposalListRow
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project
from core.perception.semantic_workspace import SemanticReviewStatus


def _dialog(qtbot, tmp_path: Path, *, image: np.ndarray | None = None) -> AutoSegmentWorkspaceDialog:
    store = SegmentationProjectStore(tmp_path / "projects")
    if not store.contains("factory"):
        project = create_segmentation_project("Factory", project_id="factory")
        project, _ = add_project_class(project, "plant")
        store.save(project)
    source = np.zeros((120, 180, 3), dtype=np.uint8) if image is None else image
    dialog = AutoSegmentWorkspaceDialog(
        project_store=store,
        initial_bgr=source,
        initial_source_label="factory.png",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def _load_proposals(dialog: AutoSegmentWorkspaceDialog, tmp_path: Path) -> None:
    result = tmp_path / "result"
    result.mkdir(exist_ok=True)
    masks = []
    for index in range(3):
        data = np.zeros((120, 180), dtype=bool)
        data[15 + index * 18 : 55 + index * 18, 15 + index * 45 : 55 + index * 45] = True
        masks.append(data)
    np.savez_compressed(result / "masks.npz", **{f"mask_{i:04d}": mask for i, mask in enumerate(masks)})
    (result / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "development-perception-mock",
                "proposals": [
                    {
                        "object_id": f"proposal:{i}",
                        "mask_id": f"mask-{i}",
                        "mask_key": f"mask_{i:04d}",
                        "confidence": None,
                        "confidence_kind": "none",
                    }
                    for i in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    dialog.load_job_result(result)


def _row_for(dialog: AutoSegmentWorkspaceDialog, object_id: str) -> tuple[object, _ProposalListRow]:
    for index in range(dialog._object_list.count()):
        item = dialog._object_list.item(index)
        if item.data(Qt.ItemDataRole.UserRole) == object_id:
            widget = dialog._object_list.itemWidget(item)
            assert isinstance(widget, _ProposalListRow)
            return item, widget
    raise AssertionError(f"Missing proposal row: {object_id}")


def test_pre9_4e_proposal_rows_have_compact_delete_control_and_keep_merge_check(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    _load_proposals(dialog, tmp_path)

    item, row = _row_for(dialog, "proposal:0")

    assert item.text() == ""
    assert bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert row.delete_button.text() == "×"
    assert row.delete_button.isEnabled()
    assert "Delete proposal" in row.delete_button.toolTip()
    row.merge_check.click()
    assert row.merge_check.isChecked()
    assert item.checkState() == Qt.CheckState.Checked
    assert "proposal:0" in dialog._merge_selected_ids

    item.setCheckState(Qt.CheckState.Unchecked)
    assert not row.merge_check.isChecked()
    assert "proposal:0" not in dialog._merge_selected_ids


def test_pre9_4e_rejected_selected_proposal_delete_clears_state_and_persists(qtbot, tmp_path: Path) -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    dialog = _dialog(qtbot, tmp_path, image=image)
    _load_proposals(dialog, tmp_path)
    dialog.select_object("proposal:1")
    dialog._reject.click()
    assert dialog.workspace.object_by_id("proposal:1").review_status is SemanticReviewStatus.REJECTED

    _item, row = _row_for(dialog, "proposal:1")
    row.delete_button.click()

    assert dialog.workspace.object_by_id("proposal:1") is None
    assert dialog._selected_id == ""
    assert dialog._object_list.count() == 2
    assert "Proposal deleted from this workspace" in dialog._status_text.text()

    dialog.close()
    restored = _dialog(qtbot, tmp_path, image=image)
    assert restored.workspace is not None
    assert restored.workspace.object_by_id("proposal:1") is None
    assert tuple(item.object_id for item in restored.workspace.objects) == ("proposal:0", "proposal:2")


def test_pre9_4e_accepted_proposal_delete_control_is_protected(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    _load_proposals(dialog, tmp_path)
    dialog.select_object("proposal:0")
    plant_index = next(i for i in range(dialog._class_combo.count()) if dialog._class_combo.itemText(i) == "plant")
    dialog._class_combo.setCurrentIndex(plant_index)
    dialog._assign.click()
    dialog._accept.click()

    _item, row = _row_for(dialog, "proposal:0")

    assert not row.delete_button.isEnabled()
    assert "protected" in row.delete_button.toolTip().lower()
    assert dialog.workspace.object_by_id("proposal:0") is not None
