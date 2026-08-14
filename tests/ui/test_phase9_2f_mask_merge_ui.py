from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMessageBox

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project
from core.perception.semantic_workspace import SemanticReviewStatus


def _dialog(qtbot, tmp_path: Path) -> AutoSegmentWorkspaceDialog:
    store = SegmentationProjectStore(tmp_path / "projects")
    project = create_segmentation_project("Factory", project_id="factory")
    project, _ = add_project_class(project, "plant")
    store.save(project)
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    dialog = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=image, initial_source_label="Captured frame")
    qtbot.addWidget(dialog)
    dialog.show()
    result = tmp_path / "result"
    result.mkdir()
    masks = []
    a = np.zeros((120, 180), dtype=bool); a[15:80, 15:70] = True; masks.append(a)
    b = np.zeros((120, 180), dtype=bool); b[70:110, 25:80] = True; masks.append(b)
    c = np.zeros((120, 180), dtype=bool); c[30:100, 110:165] = True; masks.append(c)
    np.savez_compressed(result / "masks.npz", **{f"mask_{i:04d}": m for i, m in enumerate(masks)})
    (result / "manifest.json").write_text(json.dumps({
        "model_id": "development-perception-mock",
        "proposals": [
            {"object_id": f"proposal:{i}", "mask_id": f"mask-{i}", "mask_key": f"mask_{i:04d}", "confidence": None, "confidence_kind": "none"}
            for i in range(3)
        ],
    }), encoding="utf-8")
    dialog.load_job_result(result)
    return dialog


def _check_object(dialog: AutoSegmentWorkspaceDialog, object_id: str) -> None:
    for row in range(dialog._object_list.count()):
        item = dialog._object_list.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == object_id:
            item.setCheckState(Qt.CheckState.Checked)
            return
    raise AssertionError(f"missing object {object_id}")


def test_9_2f_merge_selected_masks_preserves_primary_identity_and_returns_pending(qtbot, tmp_path: Path, monkeypatch) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.select_object("proposal:0")
    plant_index = next(i for i in range(dialog._class_combo.count()) if dialog._class_combo.itemText(i) == "plant")
    dialog._class_combo.setCurrentIndex(plant_index)
    dialog._assign.click()
    dialog._accept.click()
    assert dialog.workspace.object_by_id("proposal:0").review_status is SemanticReviewStatus.ACCEPTED

    _check_object(dialog, "proposal:0")
    _check_object(dialog, "proposal:1")
    assert dialog._merge_masks.isEnabled()
    assert "2 selected" in dialog._merge_summary.text()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes)
    dialog._merge_masks.click()

    assert len(dialog.workspace.objects) == 2
    assert dialog.workspace.object_by_id("proposal:0") is None
    assert dialog.workspace.object_by_id("proposal:1") is None
    merged = dialog.workspace.object_by_id(dialog._selected_id)
    assert merged is not None
    assert merged.class_name == "plant"
    assert merged.instance_name == "plant_001"
    assert merged.review_status is SemanticReviewStatus.PENDING
    assert merged.merged_from_object_ids == ("proposal:0", "proposal:1")
    assert dialog._merge_selected_ids == set()
    assert "Merged 2 masks" in dialog._status_text.text()


def test_9_2f_clear_merge_selection_unchecks_items(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    _check_object(dialog, "proposal:0")
    _check_object(dialog, "proposal:2")
    assert len(dialog._merge_selected_ids) == 2
    dialog._clear_merge.click()
    assert dialog._merge_selected_ids == set()
    assert dialog._merge_summary.text() == "0 selected for merge"
    assert all(dialog._object_list.item(i).checkState() == Qt.CheckState.Unchecked for i in range(dialog._object_list.count()))
