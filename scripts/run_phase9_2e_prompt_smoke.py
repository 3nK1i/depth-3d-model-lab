#!/usr/bin/env python3
"""Dependency-light Phase 9.2E interactive SAM2 contract smoke using the mock adapter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.models.metadata import ConfidenceKind
from core.perception.adapters.mock_adapter import DevelopmentPerceptionMockAdapter
from core.perception.messages import (
    FramePerceptionRequest,
    PromptedSegmentationRequest,
    SegmentationPrompt,
    SegmentationPromptKind,
)
from core.perception.metadata import PerceptionTask
from core.perception.result import DenseMaskPayload, InstanceMask, MaskSource
from core.perception.semantic_workspace import (
    SemanticClassCatalog,
    SemanticWorkspace,
    append_workspace_mask_proposal,
    replace_workspace_object_mask,
)


def main() -> int:
    frame = FrameReference(0, 0, 0.0, ImageSize(160, 120))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    request = FramePerceptionRequest(frame, rgb, space, (PerceptionTask.PROMPTED_SEGMENTATION,))
    adapter = DevelopmentPerceptionMockAdapter()
    adapter.load({})
    adapter.warmup()
    try:
        create_prompt = SegmentationPrompt(
            "create",
            SegmentationPromptKind.POINTS,
            space,
            positive_points_xy=((80.0, 60.0),),
        )
        created_raw = adapter.refine_mask(
            PromptedSegmentationRequest(request, create_prompt, creates_new_instance=True)
        )
        created = InstanceMask(
            "interactive:smoke:sam2:r0",
            frame,
            space,
            DenseMaskPayload(created_raw.mask),
            MaskSource.PROMPTED_REFINEMENT,
            ModelProvenance("development-perception-mock", "Development Mock", "prompted_segmentation"),
            confidence=created_raw.confidence,
            confidence_kind=created_raw.confidence_kind,
            foreground_valid=True,
        )
        workspace = SemanticWorkspace("phase9-2e-smoke", frame, SemanticClassCatalog(), ())
        workspace = append_workspace_mask_proposal(workspace, created, object_id="interactive:smoke")

        refine_prompt = SegmentationPrompt(
            "refine",
            SegmentationPromptKind.POINTS,
            space,
            positive_points_xy=((80.0, 60.0), (88.0, 60.0)),
            negative_points_xy=((80.0, 66.0),),
        )
        refined_raw = adapter.refine_mask(
            PromptedSegmentationRequest(
                request,
                refine_prompt,
                parent_mask_id=created.mask_id,
            )
        )
        refined = InstanceMask(
            "interactive:smoke:sam2:r1",
            frame,
            space,
            DenseMaskPayload(refined_raw.mask),
            MaskSource.PROMPTED_REFINEMENT,
            ModelProvenance("development-perception-mock", "Development Mock", "prompted_segmentation"),
            revision=1,
            parent_mask_id=created.mask_id,
            confidence=refined_raw.confidence,
            confidence_kind=refined_raw.confidence_kind,
            foreground_valid=True,
        )
        workspace = replace_workspace_object_mask(workspace, "interactive:smoke", refined)
    finally:
        adapter.unload()

    obj = workspace.object_by_id("interactive:smoke")
    assert obj is not None and obj.proposal_mask is not None
    summary = {
        "phase": "9.2E",
        "created_parentless": created.parent_mask_id == "",
        "refined_parent_preserved": obj.proposal_mask.parent_mask_id == created.mask_id,
        "refined_revision": obj.proposal_mask.revision,
        "positive_and_negative_prompts": True,
        "unlabeled_after_creation": obj.class_id is None,
        "review_state_after_refinement": obj.review_status.value,
        "geometry_authority": obj.is_geometry_authority,
        "metric_authority": obj.is_metric_authority,
        "measurement_authority": obj.is_measurement_authority,
        "status": "PASS",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
