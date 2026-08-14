from __future__ import annotations

from pathlib import Path

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
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
    create_segmentation_project,
    reviewed_selection_from_workspace,
    set_workspace_review_status,
)


def test_9_2h_persisted_review_remains_compatible_with_9_2g_measure_inspect(tmp_path: Path) -> None:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[8:52, 10:40] = (20, 80, 160)
    project = create_segmentation_project("Factory", project_id="factory")
    project, plant = add_project_class(project, "plant")
    frame = FrameReference(0, 0, 0.0, ImageSize(96, 64))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    dense = np.zeros((64, 96), dtype=bool)
    dense[8:52, 10:40] = True
    mask = InstanceMask(
        "plant-mask",
        frame,
        space,
        DenseMaskPayload(dense),
        MaskSource.NATIVE_INSTANCE,
        ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )
    workspace = SemanticWorkspace(
        "factory-review",
        frame,
        project.class_catalog,
        (SemanticWorkspaceObject("proposal:plant", proposal_mask=mask),),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:plant", SemanticReviewStatus.ACCEPTED)

    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    store.save(project, workspace, image, source_label="factory.png")
    resumed = store.load(project)
    selection = reviewed_selection_from_workspace(resumed.workspace, resumed.image_bgr)

    assert selection.workspace_id == "factory-review"
    assert selection.source_image_sha256 == resumed.source_image_sha256
    assert len(selection.objects) == 1
    assert selection.objects[0].class_name == "plant"
    assert selection.objects[0].instance_name == "plant_001"
    assert selection.objects[0].mask_source is MaskSource.NATIVE_INSTANCE
    assert selection.is_geometry_authority is False
    assert selection.is_metric_authority is False
    assert selection.is_measurement_authority is False
