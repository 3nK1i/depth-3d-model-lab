from __future__ import annotations

import json
from pathlib import Path

from core.contracts.validation import metadata_as_dict
from core.reconstruction import (
    MetricGeometryStatus,
    NerfBackendKind,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    nerf_backend_provenance,
    persistent_scene_from_pipeline_payload,
)
from core.reconstruction.orchestration import ReconstructionArtifactKind, ReconstructionPipelineStage
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def _without_neural_appearance(scene):
    neural_stages = {
        ReconstructionPipelineStage.GAUSSIAN_APPEARANCE,
        ReconstructionPipelineStage.NERF_APPEARANCE,
    }
    artifacts = [item for item in scene.artifacts if item.stage not in neural_stages]
    fallback = {item.stage: item for item in scene.fallback_evidence if item.stage not in neural_stages}
    stages = []
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
    snapshots = {item.kind.value: metadata_as_dict(item.payload_snapshot) for item in artifacts}
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


def make_phase9_5d_source(tmp_path: Path):
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "base")
    scene = _without_neural_appearance(scene)
    camera = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY)
    camera_snapshot = metadata_as_dict(camera.payload_snapshot)
    frame_id = str(camera_snapshot["reconstruction_frame_id"])
    surface = next(item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.SURFACE_MESH)
    surface_snapshot = metadata_as_dict(surface.payload_snapshot)

    artifact_root = (tmp_path / "source-nerf").resolve()
    dataset = artifact_root / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "0000.png").write_bytes(b"fixture-image-0")
    (images / "0001.png").write_bytes(b"fixture-image-1")
    (dataset / "transforms.json").write_text(
        json.dumps({"camera_model": "OPENCV", "frames": [{"file_path": "images/0000.png"}]}, indent=2) + "\n",
        encoding="utf-8",
    )
    config = artifact_root / "nerfstudio-output/eia-phase8-7e2/nerfacto/qualification/config.yml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(f"data: {dataset}\noutput_dir: {artifact_root / 'nerfstudio-output'}\n", encoding="utf-8")
    checkpoint = config.parent / "nerfstudio_models/step-000000063.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fixture-nerf-checkpoint")
    preview = artifact_root / "preview.png"
    preview.write_bytes(b"fixture-preview")
    runtime_evidence = artifact_root / "e2_runtime_evidence.json"
    runtime_evidence.write_text(
        json.dumps(
            {
                "nerfstudio_revision": "6b60855003011b2ca23c2fe3f8e2ca6314c69924",
                "tiny_cuda_nn_revision": "b3473c81396fe927293bdfd5a6be32df8769927c",
                "training_completed_without_oom": True,
                "preview_path": str(preview),
                "config_path": str(config),
                "checkpoint_path": str(checkpoint),
                "gpu_name": "NVIDIA GeForce RTX 2060",
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    artifact = NerfSceneArtifact(
        request_id="phase9-5d-nerf-request",
        backend_id="nerfstudio-nerfacto",
        backend_version="1.1.5",
        backend_revision="6b60855003011b2ca23c2fe3f8e2ca6314c69924",
        backend_provenance=nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL),
        method_name="nerfacto",
        training_state=NerfTrainingState.TRAINED,
        scene_role=NerfSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_root),
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        dataset_fingerprint_sha256="d" * 64,
        input_frame_count=3,
        training_steps=64,
        training_duration_s=5.2,
        render_capable=True,
        device="cuda:0 NVIDIA GeForce RTX 2060",
        pose_convention="T_reconstruction_from_camera",
        reconstruction_frame_id=frame_id,
        metric_geometry_status=MetricGeometryStatus(str(surface_snapshot["metric_geometry_status"])),
        unit_to_metre=surface_snapshot.get("unit_to_metre"),
        relative_scale_uncertainty=surface_snapshot.get("relative_scale_uncertainty"),
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )

    from hashlib import sha256
    def file_hash(path: Path) -> str:
        return sha256(path.read_bytes()).hexdigest()

    optimization_manifest = tmp_path / "phase9_5c_optimization_manifest.json"
    optimization_manifest.write_text(
        json.dumps(
            {
                "phase": "9.5C",
                "request_id": artifact.request_id,
                "dataset_fingerprint_sha256": artifact.dataset_fingerprint_sha256,
                "render_capable": True,
                "device": artifact.device,
                "config_sha256": file_hash(config),
                "checkpoint_sha256": file_hash(checkpoint),
                "preview_sha256": file_hash(preview),
                "runtime_evidence_sha256": file_hash(runtime_evidence),
                "canonical_geometry_authoritative": False,
                "metric_scale_authoritative": False,
                "measurement_authoritative": False,
                "qc_authoritative": False,
                "certified_metrology": False,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "phase9_5c_real_optimization_summary.json"
    summary.write_text(json.dumps({"phase": "9.5C", "status": "PASS"}, indent=2) + "\n", encoding="utf-8")
    console = tmp_path / "nerfstudio_worker_console.log"
    console.write_text("fixture worker completed\n", encoding="utf-8")
    return scene, artifact, preview, runtime_evidence, optimization_manifest, summary, console
