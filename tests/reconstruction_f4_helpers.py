from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    GaussianBackendKind,
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
    NerfBackendKind,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    gaussian_backend_provenance,
    nerf_backend_provenance,
)


def fake_gaussian_artifact(context, bundle, surface, root: Path) -> GaussianSceneArtifact:
    out = (root / "gaussian").resolve(); out.mkdir(parents=True, exist_ok=True)
    scene = out / "scene.npz"; scene.write_bytes(b"f4-gaussian")
    return GaussianSceneArtifact(
        request_id=context.request.request_id,
        backend_id="gsplat",
        backend_version="1.5.3",
        backend_provenance=gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL),
        optimization_state=GaussianOptimizationState.OPTIMIZED,
        scene_role=GaussianSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(out),
        scene_path=str(scene),
        dataset_fingerprint_sha256=bundle.dataset_fingerprint_sha256,
        input_frame_count=bundle.dataset.frame_count,
        gaussian_count=max(1, surface.mesh.vertex_count),
        training_steps=24,
        training_duration_s=0.25,
        render_capable=True,
        device="cuda:0 test",
        pose_convention=bundle.dataset.pose_convention,
        reconstruction_frame_id=bundle.dataset.reconstruction_frame_id,
        metric_geometry_status=bundle.dataset.metric_geometry_status,
        unit_to_metre=bundle.dataset.unit_to_metre,
        relative_scale_uncertainty=bundle.dataset.relative_scale_uncertainty,
        peak_vram_mib=128.0,
    )


def fake_nerf_artifact(context, bundle, root: Path) -> NerfSceneArtifact:
    out = (root / "nerf").resolve(); out.mkdir(parents=True, exist_ok=True)
    config = out / "config.yml"; config.write_text("f4\n")
    ckpt = out / "model.ckpt"; ckpt.write_bytes(b"f4-nerf")
    return NerfSceneArtifact(
        request_id=context.request.request_id,
        backend_id="nerfstudio-nerfacto",
        backend_version="1.1.5",
        backend_revision="6b60855003011b2ca23c2fe3f8e2ca6314c69924",
        backend_provenance=nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL),
        method_name="nerfacto",
        training_state=NerfTrainingState.TRAINED,
        scene_role=NerfSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(out),
        config_path=str(config),
        checkpoint_path=str(ckpt),
        dataset_fingerprint_sha256=bundle.dataset_fingerprint_sha256,
        input_frame_count=bundle.dataset.frame_count,
        training_steps=16,
        training_duration_s=0.5,
        render_capable=True,
        device="cuda:0 test",
        pose_convention=bundle.dataset.pose_convention,
        reconstruction_frame_id=bundle.dataset.reconstruction_frame_id,
        metric_geometry_status=bundle.dataset.metric_geometry_status,
        unit_to_metre=bundle.dataset.unit_to_metre,
        relative_scale_uncertainty=bundle.dataset.relative_scale_uncertainty,
    )
