#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    assign_workspace_class,
    merge_workspace_objects,
    set_workspace_review_status,
)


def main() -> int:
    frame = FrameReference(920, 1, 0.0, ImageSize(32, 24))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    prov = ModelProvenance("sam2", "SAM2", "automatic_segmentation")
    plant = np.zeros((24, 32), dtype=bool); plant[3:16, 4:14] = True
    pot = np.zeros((24, 32), dtype=bool); pot[14:22, 7:16] = True
    masks = (
        InstanceMask("plant-mask", frame, space, DenseMaskPayload(plant), MaskSource.NATIVE_INSTANCE, prov),
        InstanceMask("pot-mask", frame, space, DenseMaskPayload(pot), MaskSource.NATIVE_INSTANCE, prov),
    )
    catalog, plant_class = SemanticClassCatalog().with_custom_class("plant")
    ws = SemanticWorkspace(
        "phase9-2f-smoke",
        frame,
        catalog,
        tuple(SemanticWorkspaceObject(f"proposal:{i}", proposal_mask=mask) for i, mask in enumerate(masks)),
    )
    ws = assign_workspace_class(ws, "proposal:0", plant_class, instance_name="plant_001")
    ws = set_workspace_review_status(ws, "proposal:0", SemanticReviewStatus.ACCEPTED)
    merged, merged_id = merge_workspace_objects(ws, ("proposal:0", "proposal:1"), primary_object_id="proposal:0")
    obj = merged.object_by_id(merged_id)
    assert obj is not None and obj.proposal_mask is not None
    payload = {
        "phase": "9.2F",
        "source_object_count": 2,
        "workspace_object_count_after_merge": len(merged.objects),
        "merged_object_id": merged_id,
        "merged_class": obj.class_name,
        "merged_instance": obj.instance_name,
        "review_state": obj.review_status.value,
        "merged_from_object_ids": list(obj.merged_from_object_ids),
        "merged_from_mask_ids": list(obj.merged_from_mask_ids),
        "union_pixels": int(np.count_nonzero(obj.proposal_mask.payload.data)),
        "mask_source": obj.proposal_mask.source.value,
        "geometry_authority": obj.is_geometry_authority,
        "metric_authority": obj.is_metric_authority,
        "measurement_authority": obj.is_measurement_authority,
        "status": "PASS",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
