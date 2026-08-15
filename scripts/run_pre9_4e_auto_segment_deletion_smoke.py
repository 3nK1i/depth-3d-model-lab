#!/usr/bin/env python3
"""Deterministic backend-free smoke for persistent Auto Segment proposal deletion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        data[4 + index * 7 : 16 + index * 7, 5 + index * 11 : 23 + index * 11] = True
        mask = InstanceMask(
            f"mask-{index}",
            frame,
            space,
            DenseMaskPayload(data),
            MaskSource.NATIVE_INSTANCE,
            ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
        )
        objects.append(SemanticWorkspaceObject(f"proposal:{index}", proposal_mask=mask))
    return SemanticWorkspace(
        "pre9-4e-delete-smoke",
        frame,
        create_segmentation_project("Smoke").class_catalog,
        tuple(objects),
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="pre9_4e_auto_segment_delete_") as tmp:
        root = Path(tmp)
        image = np.zeros((48, 72, 3), dtype=np.uint8)
        project = create_segmentation_project("Smoke", project_id="smoke")
        workspace = _workspace(image)
        target = "proposal:1"
        workspace = set_workspace_review_status(workspace, target, SemanticReviewStatus.REJECTED)
        revision_before_delete = workspace.revision
        workspace = delete_workspace_object(workspace, target)

        store = SegmentationWorkspaceStore(root / "workspaces")
        store.save(project, workspace, image, source_label="smoke.png")
        restored = store.load(project)
        restored_ids = tuple(item.object_id for item in restored.workspace.objects)
        payload = {
            "status": "PASS",
            "step": "pre-9.4E-auto-segment-proposal-management",
            "deleted_object_id": target,
            "deleted_stays_deleted_after_reload": restored.workspace.object_by_id(target) is None,
            "remaining_object_ids": list(restored_ids),
            "workspace_revision_incremented": restored.workspace.revision == revision_before_delete + 1,
            "segmentation_rerun": False,
            "geometry_authoritative": restored.is_geometry_authority,
            "metric_authoritative": restored.is_metric_authority,
            "measurement_authoritative": restored.is_measurement_authority,
        }
        required = (
            payload["deleted_stays_deleted_after_reload"] is True
            and restored_ids == ("proposal:0", "proposal:2")
            and payload["workspace_revision_incremented"] is True
            and payload["segmentation_rerun"] is False
            and payload["geometry_authoritative"] is False
            and payload["metric_authoritative"] is False
            and payload["measurement_authoritative"] is False
        )
        if not required:
            payload["status"] = "FAIL"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
