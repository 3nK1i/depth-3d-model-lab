#!/usr/bin/env python3
"""Dependency-light Phase 8.7E3 real-scene validation contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    NerfHeldOutQualityEvaluation,
    NerfHeldOutRender,
    NerfHeldOutRenderManifest,
    NerfHeldOutViewMetrics,
    NerfImageQualityMetrics,
    NerfRealSceneValidationSpec,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    aggregate_nerf_held_out_metrics,
    deterministic_nerf_validation_split,
    write_nerf_held_out_render_manifest,
    write_nerf_real_scene_validation_spec,
)

OUTPUT = Path("/tmp/phase8_7e3_nerf_contracts")


def _dataset() -> NeuralSceneDataset:
    frames = []
    for index in range(12):
        path = (OUTPUT / "images" / f"frame-{index:02d}.png").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((72, 96, 3), 30 + index * 10, dtype=np.uint8), mode="RGB").save(path)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index:02d}",
                image_path=str(path),
                width=96,
                height=72,
                fx=80.0,
                fy=80.0,
                cx=48.0,
                cy=36.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="e3-contract-smoke",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="camera",
        scale_profile_id="arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    dataset = _dataset()
    train, heldout = deterministic_nerf_validation_split(
        tuple(frame.frame_id for frame in dataset.frames), validation_count=3
    )
    spec = NerfRealSceneValidationSpec(
        validation_id="e3-contract-smoke",
        source_label="synthetic-contract-only",
        dataset=dataset,
        training_frame_ids=train,
        validation_frame_ids=heldout,
        training_steps=128,
    )
    spec_path = write_nerf_real_scene_validation_spec(spec, OUTPUT / "validation_spec.json")

    renders = []
    views = []
    by_id = {frame.frame_id: frame for frame in dataset.frames}
    for index, frame_id in enumerate(heldout):
        frame = by_id[frame_id]
        rendered = (OUTPUT / "renders" / f"{frame_id}.png").resolve()
        rendered.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frame.image_path, rendered)
        renders.append(
            NerfHeldOutRender(
                frame_id=frame_id,
                reference_image_path=frame.image_path,
                rendered_image_path=str(rendered),
                width=frame.width,
                height=frame.height,
            )
        )
        views.append(NerfHeldOutViewMetrics(frame_id=frame_id, psnr_db=25.0 + index, ssim=0.9, lpips=0.1))
    manifest = NerfHeldOutRenderManifest(
        validation_fingerprint_sha256=spec.validation_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=heldout,
        renders=tuple(renders),
    )
    render_path = write_nerf_held_out_render_manifest(manifest, OUTPUT / "heldout_render_manifest.json")
    aggregate = aggregate_nerf_held_out_metrics(views)
    quality = NerfHeldOutQualityEvaluation(
        validation_fingerprint_sha256=spec.validation_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=heldout,
        views=tuple(views),
        aggregate=aggregate,
    )
    assert quality.passes(spec.quality_policy)
    summary = {
        "phase": "8.7E3",
        "pass": True,
        "spec": str(spec_path),
        "render_manifest": str(render_path),
        "training_frame_count": len(train),
        "validation_frame_count": len(heldout),
        "aggregate": quality.aggregate.as_dict(),
        "shared_metrics_worker_reused": True,
        "scene_role": "appearance_only",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "performance_hardening_deferred_to": "8.7E4",
    }
    path = OUTPUT / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Phase 8.7E3 NeRF real-scene contracts: PASS")
    print(f"train={len(train)} heldout={len(heldout)} metrics=PSNR/SSIM/LPIPS")
    print("shared_metrics_worker_reused=True performance_hardening_deferred_to=8.7E4")
    print("scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
