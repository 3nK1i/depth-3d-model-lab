from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project
from core.perception.semantic_workspace import SemanticReviewStatus


def _dialog(qtbot, tmp_path: Path) -> AutoSegmentWorkspaceDialog:
    store = SegmentationProjectStore(tmp_path / "projects")
    project = create_segmentation_project("Factory", project_id="factory")
    project, _ = add_project_class(project, "robot_arm")
    store.save(project)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    dialog = AutoSegmentWorkspaceDialog(
        project_store=store,
        initial_bgr=image,
        initial_source_label="Captured frame",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    result = tmp_path / "auto-result"
    result.mkdir()
    mask = np.zeros((120, 160), dtype=bool)
    mask[20:95, 30:125] = True
    np.savez_compressed(result / "masks.npz", mask_0000=mask)
    (result / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": "development-perception-mock",
                "proposals": [
                    {
                        "object_id": "proposal:0",
                        "mask_id": "mask-r0",
                        "mask_key": "mask_0000",
                        "confidence": None,
                        "confidence_kind": "none",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dialog.load_job_result(result)
    return dialog


def _prompt_result(
    root: Path,
    *,
    operation: str,
    object_id: str,
    parent_mask_id: str,
    revision: int,
) -> Path:
    root.mkdir()
    mask = np.zeros((120, 160), dtype=bool)
    mask[25:90, 40:120] = True
    np.save(root / "mask.npy", mask)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "operation": operation,
                "object_id": object_id,
                "mask_id": f"{object_id}:sam2:r{revision}",
                "parent_mask_id": parent_mask_id,
                "revision": revision,
                "model_id": "development-perception-mock",
                "confidence": 0.8,
                "confidence_kind": "confidence",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_9_2e_one_click_segment_emits_parentless_new_object_prompt(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    payloads: list[dict[str, object]] = []
    dialog.prompt_requested.connect(payloads.append)
    dialog._begin_segment_object()
    assert dialog._canvas.interaction_mode == "segment_new"
    dialog._on_prompt_point(70, 55)
    assert len(payloads) == 1
    spec = payloads[0]["prompt_spec"]
    assert spec["operation"] == "create"
    assert str(spec["object_id"]).startswith("interactive:")
    assert spec["parent_mask_id"] == ""
    assert spec["revision"] == 0
    assert spec["positive_points"] == [[70.0, 55.0]]
    assert spec["negative_points"] == []


def test_9_2e_new_prompt_result_becomes_selected_unlabeled_pending_proposal(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    result = _prompt_result(
        tmp_path / "prompt-create",
        operation="create",
        object_id="interactive:new",
        parent_mask_id="",
        revision=0,
    )
    dialog.load_prompt_result(result)
    obj = dialog.workspace.object_by_id("interactive:new")
    assert obj is not None
    assert obj.class_id is None
    assert obj.class_name == ""
    assert obj.review_status is SemanticReviewStatus.PENDING
    assert dialog._selected_id == "interactive:new"
    assert obj.is_geometry_authority is False
    assert obj.is_metric_authority is False
    assert obj.is_measurement_authority is False


def test_9_2e_refinement_uses_mask_anchor_and_returns_revised_object_to_pending(qtbot, tmp_path: Path) -> None:
    dialog = _dialog(qtbot, tmp_path)
    dialog.select_object("proposal:0")
    robot_index = next(
        i for i in range(dialog._class_combo.count())
        if dialog._class_combo.itemText(i) == "robot_arm"
    )
    dialog._class_combo.setCurrentIndex(robot_index)
    dialog._assign.click()
    dialog._accept.click()
    accepted = dialog.workspace.object_by_id("proposal:0")
    assert accepted is not None
    assert accepted.review_status is SemanticReviewStatus.ACCEPTED

    payloads: list[dict[str, object]] = []
    dialog.prompt_requested.connect(payloads.append)
    dialog._set_prompt_mode("refine_positive")
    dialog._on_prompt_point(75, 55)
    dialog._set_prompt_mode("refine_negative")
    dialog._on_prompt_point(118, 80)
    dialog._apply_refinement()
    assert len(payloads) == 1
    spec = payloads[0]["prompt_spec"]
    assert spec["operation"] == "refine"
    assert spec["object_id"] == "proposal:0"
    assert spec["parent_mask_id"] == "mask-r0"
    assert spec["revision"] == 1
    assert len(spec["positive_points"]) == 2  # mask anchor + user include
    assert len(spec["negative_points"]) == 1

    result = _prompt_result(
        tmp_path / "prompt-refine",
        operation="refine",
        object_id="proposal:0",
        parent_mask_id="mask-r0",
        revision=1,
    )
    dialog.load_prompt_result(result)
    refined = dialog.workspace.object_by_id("proposal:0")
    assert refined is not None
    assert refined.class_name == "robot_arm"
    assert refined.instance_name == "robot_arm_001"
    assert refined.review_status is SemanticReviewStatus.PENDING
    assert refined.proposal_mask.parent_mask_id == "mask-r0"
    assert refined.proposal_mask.revision == 1
