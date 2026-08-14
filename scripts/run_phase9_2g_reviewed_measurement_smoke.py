#!/usr/bin/env python3
"""Dependency-light Phase 9.2G reviewed-mask → Measure & Inspect smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contracts import FrameReference, ImageSize, ModelProvenance, PixelCoordinateSpace, PixelSpaceRole
from core.inspection import load_inspection_bundle
from core.perception import (
    DenseMaskPayload,
    InstanceMask,
    MaskSource,
    SemanticClassCatalog,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    assign_workspace_class,
    set_workspace_review_status,
)
from core.perception.reviewed_analysis import reviewed_selection_from_workspace, save_reviewed_semantic_selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="/tmp/eia_phase9_2g_reviewed_measurement")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    if root.exists():
        import shutil
        shutil.rmtree(root)
    root.mkdir(parents=True)

    height, width = 72, 96
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(20, 180, width, dtype=np.uint8)
    image[..., 1] = 100
    image_path = root / "frame.npy"
    np.save(image_path, image)

    frame = FrameReference(2, 3, 0.0, ImageSize(width, height))
    space = PixelCoordinateSpace("rgb", frame, frame.source_size, PixelSpaceRole.RGB)
    catalog, plant = SemanticClassCatalog().with_custom_class("plant")
    catalog, suitcase = catalog.with_custom_class("suitcase")

    def mask(mask_id: str, bounds: tuple[int, int, int, int]) -> InstanceMask:
        y0, y1, x0, x1 = bounds
        dense = np.zeros((height, width), dtype=bool)
        dense[y0:y1, x0:x1] = True
        return InstanceMask(
            mask_id=mask_id,
            frame_reference=frame,
            pixel_space=space,
            payload=DenseMaskPayload(dense),
            source=MaskSource.NATIVE_INSTANCE,
            model_provenance=ModelProvenance("sam2", "SAM2", "automatic_segmentation"),
        )

    workspace = SemanticWorkspace(
        "phase9-2g-smoke-workspace",
        frame,
        catalog,
        (
            SemanticWorkspaceObject("proposal:plant", proposal_mask=mask("plant-mask", (8, 62, 7, 41))),
            SemanticWorkspaceObject("proposal:suitcase", proposal_mask=mask("suitcase-mask", (14, 67, 55, 90))),
        ),
    )
    workspace = assign_workspace_class(workspace, "proposal:plant", plant, instance_name="plant_001")
    workspace = set_workspace_review_status(workspace, "proposal:plant", SemanticReviewStatus.ACCEPTED)
    workspace = assign_workspace_class(workspace, "proposal:suitcase", suitcase, instance_name="suitcase_001")
    workspace = set_workspace_review_status(workspace, "proposal:suitcase", SemanticReviewStatus.ACCEPTED)
    selection = reviewed_selection_from_workspace(workspace, image)
    reviewed_path, _ = save_reviewed_semantic_selection(selection, root / "reviewed")
    depth_config = root / "depth.json"
    depth_config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}), encoding="utf-8")
    output = root / "result"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(image_path),
            "--output", str(output),
            "--depth-model", "development-mock",
            "--depth-config", str(depth_config),
            "--perception-model", "invalid-model-proves-bypass",
            "--reviewed-semantics", str(reviewed_path),
            "--max-objects", "4",
            "--minimum-mask-pixels", "64",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    summary, _arrays = load_inspection_bundle(output)
    payload = {
        "phase": "9.2G",
        "status": "PASS",
        "semantic_source": summary.semantic_source,
        "perception_model_id": summary.perception_model_id,
        "perception_inference_skipped": "perception model inference skipped" in completed.stdout,
        "workspace_id": summary.semantic_workspace_id,
        "reviewed_object_count": len(summary.objects),
        "instances": [obj.instance_name for obj in summary.objects],
        "reviewed_masks_are_geometry_authority": False,
        "reviewed_masks_are_metric_authority": False,
        "reviewed_masks_are_measurement_authority": False,
        "depth_model_still_executes": summary.depth_model_id == "development-mock",
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
