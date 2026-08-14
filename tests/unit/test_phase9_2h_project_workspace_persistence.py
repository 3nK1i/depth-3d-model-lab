from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.contracts.validation import ContractValidationError
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SegmentationWorkspaceStore,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    add_project_class,
    assign_workspace_class,
    clear_workspace_class_assignment,
    create_segmentation_project,
    create_segmentation_project_from_model_names,
    rebind_project_catalog_to_workspace,
    retire_project_class,
    set_workspace_review_status,
)


def _workspace(project, image: np.ndarray) -> SemanticWorkspace:
    h, w = image.shape[:2]
    frame = FrameReference(0, 0, 0.0, ImageSize(w, h))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    dense = np.zeros((h, w), dtype=bool)
    dense[4 : h - 4, 5 : w // 2] = True
    mask = InstanceMask(
        "mask-1",
        frame,
        space,
        DenseMaskPayload(dense),
        MaskSource.NATIVE_INSTANCE,
        ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )
    return SemanticWorkspace(
        "workspace-1",
        frame,
        project.class_catalog,
        (SemanticWorkspaceObject("proposal:1", proposal_mask=mask),),
    )


def test_9_2h_retired_class_id_is_never_reused() -> None:
    project = create_segmentation_project("Factory", project_id="factory")
    project, pallet = add_project_class(project, "pallet")
    project = retire_project_class(project, pallet.class_id)
    project, robot = add_project_class(project, "robot")
    assert pallet.class_id in project.retired_class_ids
    assert robot.class_id > pallet.class_id
    assert robot.class_id not in project.retired_class_ids


def test_9_2h_model_class_is_protected_from_retirement() -> None:
    project = create_segmentation_project_from_model_names(
        "Factory", {0: "person"}, source_model_id="yolo", project_id="factory"
    )
    with pytest.raises(ContractValidationError, match="protected"):
        retire_project_class(project, 0)


def test_9_2h_used_class_can_be_returned_to_unlabeled_before_retirement() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    workspace = _workspace(project, image)
    workspace = assign_workspace_class(workspace, "proposal:1", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:1", SemanticReviewStatus.ACCEPTED)
    mask_id = workspace.objects[0].selected_mask_id

    workspace = clear_workspace_class_assignment(workspace, "proposal:1")
    project = retire_project_class(project, plant.class_id)
    workspace = rebind_project_catalog_to_workspace(workspace, project)

    obj = workspace.objects[0]
    assert obj.class_id is None
    assert obj.class_name == ""
    assert obj.instance_name == ""
    assert obj.review_status is SemanticReviewStatus.PENDING
    assert obj.selected_mask_id == mask_id
    assert project.class_by_id(plant.class_id) is None
    assert plant.class_id in project.retired_class_ids


def test_9_2h_rename_rebind_preserves_object_identity_and_review() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    workspace = _workspace(project, image)
    workspace = assign_workspace_class(workspace, "proposal:1", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:1", SemanticReviewStatus.ACCEPTED)

    from core.perception import rename_project_class
    renamed = rename_project_class(project, plant.class_id, "potted_plant")
    rebound = rebind_project_catalog_to_workspace(workspace, renamed)
    obj = rebound.objects[0]
    assert obj.class_id == plant.class_id
    assert obj.class_name == "potted_plant"
    assert obj.instance_name == "plant_001"
    assert obj.review_status is SemanticReviewStatus.ACCEPTED


def test_9_2h_workspace_round_trip_preserves_review_masks_and_authority(tmp_path: Path) -> None:
    image = np.zeros((48, 72, 3), dtype=np.uint8)
    image[5:20, 8:30] = (10, 90, 180)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    workspace = _workspace(project, image)
    workspace = assign_workspace_class(workspace, "proposal:1", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:1", SemanticReviewStatus.ACCEPTED)

    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    path = store.save(project, workspace, image, source_label="factory.png")
    assert path.is_dir()
    assert store.contains(project.project_id)
    persisted = store.load(project)

    assert persisted.project_id == project.project_id
    assert persisted.source_label == "factory.png"
    assert np.array_equal(persisted.image_bgr, image)
    assert persisted.workspace.workspace_id == workspace.workspace_id
    assert persisted.workspace.objects[0].class_name == "plant"
    assert persisted.workspace.objects[0].instance_name == "plant_001"
    assert persisted.workspace.objects[0].review_status is SemanticReviewStatus.ACCEPTED
    assert np.array_equal(
        persisted.workspace.objects[0].proposal_mask.payload.data,
        workspace.objects[0].proposal_mask.payload.data,
    )
    assert persisted.is_geometry_authority is False
    assert persisted.is_metric_authority is False
    assert persisted.is_measurement_authority is False


def test_9_2h_persistence_rejects_workspace_with_retired_class(tmp_path: Path) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    workspace = _workspace(project, image)
    workspace = assign_workspace_class(workspace, "proposal:1", plant, instance_name="plant_001")
    retired_project = retire_project_class(project, plant.class_id)
    with pytest.raises(ContractValidationError, match="not active"):
        SegmentationWorkspaceStore(tmp_path / "workspaces").save(
            retired_project, workspace, image, source_label="factory.png"
        )
