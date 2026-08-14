#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    add_project_class,
    assign_workspace_class,
    clear_workspace_class_assignment,
    create_segmentation_project,
    rebind_project_catalog_to_workspace,
    retire_project_class,
    reviewed_selection_from_workspace,
    set_workspace_review_status,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/eia_phase9_2h_workspace")
    args = parser.parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    image = np.zeros((64, 96, 3), dtype=np.uint8)
    image[8:52, 10:42] = (20, 100, 180)
    project = create_segmentation_project("Phase 9.2H Smoke", project_id="phase9-2h-smoke")
    project, plant = add_project_class(project, "plant")
    frame = FrameReference(0, 0, 0.0, ImageSize(96, 64))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    dense = np.zeros((64, 96), dtype=bool)
    dense[8:52, 10:42] = True
    mask = InstanceMask(
        "plant-mask", frame, space, DenseMaskPayload(dense), MaskSource.NATIVE_INSTANCE,
        ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
    )
    workspace = SemanticWorkspace(
        "phase9-2h-workspace", frame, project.class_catalog,
        (SemanticWorkspaceObject("proposal:plant", proposal_mask=mask),),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:plant", SemanticReviewStatus.ACCEPTED)

    store = SegmentationWorkspaceStore(root / "workspaces")
    store.save(project, workspace, image, source_label="smoke.png")
    resumed = store.load(project)
    selection = reviewed_selection_from_workspace(resumed.workspace, resumed.image_bgr)

    # Exercise safe retirement using a second class so the accepted reviewed object
    # remains intact for the 9.2G compatibility proof.
    project, obsolete = add_project_class(project, "obsolete_fixture")
    obsolete_id = obsolete.class_id
    project = retire_project_class(project, obsolete_id)
    project, replacement = add_project_class(project, "replacement_fixture")

    # Exercise used-class retirement on a separate workspace copy.
    cleanup_workspace = clear_workspace_class_assignment(resumed.workspace, "proposal:plant")
    cleanup_project = retire_project_class(project, plant.class_id)
    cleanup_workspace = rebind_project_catalog_to_workspace(cleanup_workspace, cleanup_project)

    result = {
        "phase": "9.2H",
        "status": "PASS",
        "workspace_persisted": store.contains("phase9-2h-smoke"),
        "workspace_resumed": resumed.workspace.workspace_id == workspace.workspace_id,
        "accepted_instance": selection.objects[0].instance_name,
        "measure_inspect_review_compatible": len(selection.objects) == 1,
        "used_class_returns_to_unlabeled": cleanup_workspace.objects[0].class_id is None,
        "used_class_returns_to_pending": cleanup_workspace.objects[0].review_status is SemanticReviewStatus.PENDING,
        "retired_class_id": obsolete_id,
        "replacement_class_id": replacement.class_id,
        "retired_id_reused": replacement.class_id == obsolete_id,
        "reviewed_masks_are_geometry_authority": False,
        "reviewed_masks_are_metric_authority": False,
        "reviewed_masks_are_measurement_authority": False,
    }
    if result["retired_id_reused"]:
        result["status"] = "FAIL"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
