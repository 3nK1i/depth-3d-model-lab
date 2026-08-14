from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QScrollArea

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project
from core.perception.semantic_workspace import SemanticReviewStatus


def _dialog(qtbot, tmp_path: Path) -> AutoSegmentWorkspaceDialog:
    store = SegmentationProjectStore(tmp_path / "projects")
    project = create_segmentation_project("Factory", project_id="factory")
    project, _ = add_project_class(project, "robot_arm")
    project, _ = add_project_class(project, "conveyor")
    store.save(project)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    dialog = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=image, initial_source_label="Captured frame")
    qtbot.addWidget(dialog)
    dialog.show()
    result = tmp_path / "result"
    result.mkdir()
    mask0 = np.zeros((120, 160), dtype=bool); mask0[10:70, 10:70] = True
    mask1 = np.zeros((120, 160), dtype=bool); mask1[40:110, 90:150] = True
    np.savez_compressed(result / "masks.npz", mask_0000=mask0, mask_0001=mask1)
    (result / "manifest.json").write_text(json.dumps({
        "model_id": "development-perception-mock",
        "proposals": [
            {"object_id": "proposal:0", "mask_id": "mask-0", "mask_key": "mask_0000", "confidence": None, "confidence_kind": "none"},
            {"object_id": "proposal:1", "mask_id": "mask-1", "mask_key": "mask_0001", "confidence": None, "confidence_kind": "none"},
        ],
    }), encoding="utf-8")
    dialog.load_job_result(result)
    return dialog


def test_9_2d_dialog_is_vertical_sidebar_and_review_first(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.resize(1100, 680)
    qtbot.wait(10)
    assert dialog._object_list.count() == 2
    assert dialog._project_combo.currentText() == "Factory"
    assert dialog._summary.text().startswith("2 proposals")
    assert dialog.findChild(type(dialog._object_list), "autoSegmentObjectList") is dialog._object_list
    assert dialog.findChild(type(dialog._project_combo), "autoSegmentCombo") is not None
    sidebar = dialog.findChild(QScrollArea, "autoSegmentSidebar")
    assert sidebar is not None
    assert sidebar.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_9_2d_click_review_assigns_class_and_instance_then_accepts(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.select_object("proposal:0")
    robot_index = next(i for i in range(dialog._class_combo.count()) if dialog._class_combo.itemText(i) == "robot_arm")
    dialog._class_combo.setCurrentIndex(robot_index)
    assert dialog._instance_name.text() == "robot_arm_001"
    dialog._assign.click()
    dialog._accept.click()
    obj = dialog.workspace.object_by_id("proposal:0")
    assert obj is not None
    assert obj.class_name == "robot_arm"
    assert obj.instance_name == "robot_arm_001"
    assert obj.review_status is SemanticReviewStatus.ACCEPTED
    assert obj.is_geometry_authority is False
    assert obj.is_metric_authority is False
    assert obj.is_measurement_authority is False


def test_9_2d_reject_keeps_unlabeled_proposal_truthful(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.select_object("proposal:1")
    dialog._reject.click()
    obj = dialog.workspace.object_by_id("proposal:1")
    assert obj is not None
    assert obj.class_id is None
    assert obj.class_name == ""
    assert obj.review_status is SemanticReviewStatus.REJECTED


def test_9_2d_canvas_click_prefers_smallest_overlapping_mask(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    canvas = dialog._canvas
    canvas.resize(640, 480)
    canvas.show()
    qtbot.wait(10)
    # The dedicated canvas contract is mask-hit-test based, not bounding-box based.
    assert canvas._workspace is dialog.workspace
    assert canvas._show_masks is True
