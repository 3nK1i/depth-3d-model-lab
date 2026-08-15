from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.contracts.validation import metadata_as_dict
from core.reconstruction import (
    persistent_scene_from_pipeline_payload,
    GAUSSIAN_SPLATTING_POSE_CONVENTION,
    GaussianBackendKind,
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
    MetricGeometryStatus,
    gaussian_backend_provenance,
)
from core.reconstruction.orchestration import (
    ReconstructionArtifactKind,
    ReconstructionPipelineStage,
)
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene




def _without_neural_appearance(scene):
    neural_stages = {
        ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
        ReconstructionPipelineStage.NERF_APPEARANCE,
    }
    artifacts = [item for item in scene.artifacts if item.stage not in neural_stages]
    stages = []
    fallback = {item.stage: item for item in scene.fallback_evidence if item.stage not in neural_stages}
    for artifact in artifacts:
        evidence = fallback.get(artifact.stage)
        stages.append(
            {
                "stage": artifact.stage.value,
                "state": "completed",
                "required": True,
                "provider_attempts": list(evidence.provider_attempts) if evidence else [artifact.provider_id],
                "provider_id": artifact.provider_id,
                "fallback_used": bool(evidence.fallback_used) if evidence else False,
                "reason": evidence.reason if evidence else "",
                "artifact": artifact.as_dict(),
            }
        )
    for stage, evidence in fallback.items():
        if stage in {item.stage for item in artifacts}:
            continue
        stages.append(
            {
                "stage": stage.value,
                "state": evidence.state.value,
                "required": True,
                "provider_attempts": list(evidence.provider_attempts),
                "provider_id": "",
                "fallback_used": evidence.fallback_used,
                "reason": evidence.reason,
                "artifact": None,
            }
        )
    order = {stage: index for index, stage in enumerate(ReconstructionPipelineStage)}
    stages.sort(key=lambda item: order[ReconstructionPipelineStage(item["stage"])])
    provenance = [
        item for item in scene.provider_provenance
        if not any(stage in neural_stages for stage in item.stages)
    ]
    snapshots = {
        item.kind.value: metadata_as_dict(item.payload_snapshot)
        for item in artifacts
    }
    return persistent_scene_from_pipeline_payload(
        {
            "request_id": scene.request_id,
            "session_id": scene.session_id,
            "status": scene.pipeline_status.value,
            "canonical_geometry_artifact_id": scene.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
            "stages": stages,
            "provider_provenance": [item.as_dict() for item in provenance],
        },
        payload_snapshots_by_kind=snapshots,
    )

def make_phase9_4d_source(tmp_path: Path):
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "base")
    scene = _without_neural_appearance(scene)
    camera = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY)
    camera_snapshot = metadata_as_dict(camera.payload_snapshot)  # type: ignore[arg-type]
    frame_id = str(camera_snapshot["reconstruction_frame_id"])
    surface = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.SURFACE_MESH)
    surface_snapshot = metadata_as_dict(surface.payload_snapshot)  # type: ignore[arg-type]
    metric_status = MetricGeometryStatus(str(surface_snapshot["metric_geometry_status"]))
    unit_to_metre = surface_snapshot.get("unit_to_metre")

    artifact_dir = (tmp_path / "source-gaussian").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scene_path = artifact_dir / "scene.pt"
    scene_path.write_bytes(b"phase9-4d-fixture-gaussian-scene\n")
    preview_path = artifact_dir / "preview.png"
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 0] = 40
    image[..., 1] = np.arange(64, dtype=np.uint8)[None, :]
    image[..., 2] = 180
    assert cv2.imwrite(str(preview_path), image)

    artifact = GaussianSceneArtifact(
        request_id="phase9-4d-gaussian-request",
        backend_id="gsplat",
        backend_version="1.5.3",
        backend_provenance=gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL),
        optimization_state=GaussianOptimizationState.OPTIMIZED,
        scene_role=GaussianSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_dir),
        scene_path=str(scene_path),
        dataset_fingerprint_sha256="a" * 64,
        input_frame_count=3,
        gaussian_count=64,
        training_steps=24,
        training_duration_s=1.25,
        render_capable=True,
        device="NVIDIA GeForce RTX 2060",
        peak_vram_mib=128.0,
        pose_convention=GAUSSIAN_SPLATTING_POSE_CONVENTION,
        reconstruction_frame_id=frame_id,
        metric_geometry_status=metric_status,
        unit_to_metre=unit_to_metre,
        relative_scale_uncertainty=None,
    )
    artifact_manifest = artifact_dir / "artifact_manifest.json"
    artifact_manifest.write_text(json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = artifact_dir / "qualification_metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "backend_kind": "gsplat_external",
                "real_cuda_execution": True,
                "optimization_completed": True,
                "render_written": True,
                "training_steps": 24,
                "preview_path": str(preview_path),
                "initial_l1": 0.30,
                "final_l1": 0.23,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = artifact_dir / "phase9_4c_real_optimization_summary.json"
    summary.write_text(
        json.dumps(
            {
                "phase": "9.4C",
                "status": "PASS",
                "scene_id": scene.scene_id,
                "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
                "scene_path": artifact.scene_path,
                "preview_path": str(preview_path),
                "artifact_manifest": str(artifact_manifest),
                "qualification_metrics": str(metrics),
                "canonical_geometry_authoritative": True,
                "gaussian_canonical_geometry_authoritative": False,
                "certified_metrology": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return scene, artifact, preview_path, metrics, artifact_manifest, summary
