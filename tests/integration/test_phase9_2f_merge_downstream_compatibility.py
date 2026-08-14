from __future__ import annotations

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    accepted_observations,
    assign_workspace_class,
    merge_workspace_objects,
    set_workspace_review_status,
)


def test_9_2f_merged_reviewed_mask_stays_compatible_with_existing_observation_pipeline() -> None:
    frame = FrameReference(700, 1, 0.0, ImageSize(20, 16))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    provenance = ModelProvenance("sam2", "SAM2", "automatic_segmentation")
    masks = []
    for index, bounds in enumerate(((2, 12, 2, 9), (9, 15, 4, 11))):
        y0, y1, x0, x1 = bounds
        data = np.zeros((16, 20), dtype=bool)
        data[y0:y1, x0:x1] = True
        masks.append(InstanceMask(
            f"mask-{index}", frame, space, DenseMaskPayload(data), MaskSource.NATIVE_INSTANCE, provenance
        ))
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    ws = SemanticWorkspace(
        "integration-merge",
        frame,
        catalog,
        tuple(SemanticWorkspaceObject(f"proposal:{i}", proposal_mask=mask) for i, mask in enumerate(masks)),
    )
    ws = assign_workspace_class(ws, "proposal:0", plant, instance_name="plant_001")
    ws, merged_id = merge_workspace_objects(ws, ("proposal:0", "proposal:1"), primary_object_id="proposal:0")
    ws = set_workspace_review_status(ws, merged_id, SemanticReviewStatus.ACCEPTED)
    obs = accepted_observations(ws)[0]
    assert obs.observation_id == merged_id
    assert obs.class_name == "plant"
    assert obs.selected_mask is not None
    assert obs.selected_mask.foreground_valid is True
    assert obs.selected_mask.source is MaskSource.SEMANTIC_MERGE
    assert np.count_nonzero(obs.selected_mask.payload.data) > 0
