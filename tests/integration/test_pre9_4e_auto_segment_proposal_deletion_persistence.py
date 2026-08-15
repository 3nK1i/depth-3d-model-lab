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
    create_segmentation_project,
    delete_workspace_object,
    set_workspace_review_status,
)


def _workspace(image: np.ndarray) -> SemanticWorkspace:
    height, width = image.shape[:2]
    frame = FrameReference(0, 0, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    objects = []
    for index in range(3):
        data = np.zeros((height, width), dtype=bool)
        data[5 + index * 8 : 18 + index * 8, 7 + index * 12 : 25 + index * 12] = True
        mask = InstanceMask(
            f"mask-{index}",
            frame,
            space,
            DenseMaskPayload(data),
            MaskSource.NATIVE_INSTANCE,
            ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
        )
        objects.append(SemanticWorkspaceObject(f"proposal:{index}", proposal_mask=mask))
    return SemanticWorkspace("workspace-delete-persist", frame, create_segmentation_project("x").class_catalog, tuple(objects))


def test_pre9_4e_deleted_rejected_proposal_stays_gone_after_persistent_reload(tmp_path: Path) -> None:
    image = np.zeros((48, 72, 3), dtype=np.uint8)
    image[4:20, 6:28] = (30, 80, 160)
    project = create_segmentation_project("Factory", project_id="factory")
    workspace = _workspace(image)
    target = "proposal:1"
    workspace = set_workspace_review_status(workspace, target, SemanticReviewStatus.REJECTED)
    revision_before_delete = workspace.revision
    workspace = delete_workspace_object(workspace, target)

    store = SegmentationWorkspaceStore(tmp_path / "workspaces")
    store.save(project, workspace, image, source_label="factory.png")
    restored = store.load(project)

    assert restored.workspace.object_by_id(target) is None
    assert tuple(item.object_id for item in restored.workspace.objects) == ("proposal:0", "proposal:2")
    assert restored.workspace.revision == revision_before_delete + 1
    assert restored.is_geometry_authority is False
    assert restored.is_metric_authority is False
    assert restored.is_measurement_authority is False
