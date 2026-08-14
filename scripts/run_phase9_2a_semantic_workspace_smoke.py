#!/usr/bin/env python3
"""Dependency-light Phase 9.2A semantic-workspace smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    ObjectObservation,
    PerceptionFrameResult,
    SemanticReviewStatus,
    SemanticWorkspace,
    accepted_observations,
    assign_workspace_class,
    set_workspace_review_status,
    workspace_from_perception_result,
)


def main() -> int:
    frame = FrameReference(92, 1, 0.0, ImageSize(64, 48))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    provenance = ModelProvenance("ultralytics-yolo-seg", "YOLO Seg", "instance_segmentation")
    mask_data = np.zeros((48, 64), dtype=bool)
    mask_data[10:38, 15:45] = True
    mask = InstanceMask(
        "mask-1",
        frame,
        space,
        DenseMaskPayload(mask_data),
        MaskSource.NATIVE_INSTANCE,
        provenance,
        confidence=0.94,
        confidence_kind=ConfidenceKind.CONFIDENCE,
    )
    observation = ObjectObservation("obs-1", frame, 0, "person", masks=(mask,))
    result = PerceptionFrameResult(frame, (observation,), (provenance,))
    workspace = workspace_from_perception_result(result, workspace_id="phase9-2a-smoke")
    catalog, robot = workspace.class_catalog.with_custom_class("robot_arm")
    workspace = SemanticWorkspace(
        workspace.workspace_id,
        workspace.frame_reference,
        catalog,
        workspace.objects,
        workspace.revision,
    )
    workspace = assign_workspace_class(
        workspace,
        "obs-1",
        robot,
        instance_name="robot_arm_001",
    )
    workspace = set_workspace_review_status(workspace, "obs-1", SemanticReviewStatus.ACCEPTED)
    accepted = accepted_observations(workspace)
    obj = workspace.object_by_id("obs-1")
    payload = {
        "status": "PASS",
        "source_class": observation.class_name,
        "assigned_class": accepted[0].class_name,
        "assigned_class_id": accepted[0].class_id,
        "instance_name": obj.instance_name,
        "mask_preserved": accepted[0].selected_mask_id == observation.selected_mask_id,
        "accepted_object_count": len(accepted),
        "geometry_authority": obj.is_geometry_authority,
        "metric_authority": obj.is_metric_authority,
        "measurement_authority": obj.is_measurement_authority,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
