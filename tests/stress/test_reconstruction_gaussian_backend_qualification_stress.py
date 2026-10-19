from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    GSPLAT_QUALIFIED_VERSION,
    GaussianBackendKind,
    GaussianBackendQualificationStatus,
    GaussianEnvironmentProbe,
    GaussianOptimizationState,
    GaussianRuntimeEvidence,
    GaussianSceneArtifact,
    GaussianSceneRole,
    GaussianWorkerQualificationEvidence,
    MetricGeometryStatus,
    assess_gaussian_backend_qualification,
    gaussian_backend_provenance,
)


def test_repeated_real_backend_qualification_contract_is_bounded_and_deterministic(tmp_path: Path) -> None:
    python = str((tmp_path / "isolated/bin/python").resolve())
    artifact_dir = (tmp_path / "artifact").resolve()
    artifact_dir.mkdir(parents=True)
    scene = artifact_dir / "scene.pt"
    preview = artifact_dir / "preview.png"
    metrics = artifact_dir / "qualification_metrics.json"
    scene.write_bytes(b"scene")
    preview.write_bytes(b"preview")
    metrics.write_text("{}", encoding="utf-8")

    probe = GaussianEnvironmentProbe(
        python_executable=python,
        available=True,
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        torch_version="2.7.1+cu128",
        gsplat_version=GSPLAT_QUALIFIED_VERSION,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("Synthetic CUDA GPU",),
    )
    runtime = GaussianRuntimeEvidence(
        python_executable=python,
        python_version="3.10.18",
        torch_version="2.7.1+cu128",
        torch_cuda_version="12.8",
        cuda_available=True,
        device_names=("Synthetic CUDA GPU",),
        device_capabilities=((7, 5),),
    )
    artifact = GaussianSceneArtifact(
        request_id="stress",
        backend_id="depth3d-gsplat-external",
        backend_version=GSPLAT_QUALIFIED_VERSION,
        backend_provenance=gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL),
        optimization_state=GaussianOptimizationState.OPTIMIZED,
        scene_role=GaussianSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_dir),
        scene_path=str(scene),
        dataset_fingerprint_sha256="c" * 64,
        input_frame_count=4,
        gaussian_count=512,
        training_steps=24,
        training_duration_s=1.0,
        render_capable=True,
        device="cuda:0 Synthetic CUDA GPU",
        pose_convention="T_reconstruction_from_camera",
        reconstruction_frame_id="reconstruction",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
    )
    evidence = GaussianWorkerQualificationEvidence(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        real_cuda_execution=True,
        optimization_completed=True,
        render_written=True,
        training_steps=24,
        source_revision=GSPLAT_QUALIFIED_VERSION,
        preview_path=str(preview),
        metrics_path=str(metrics),
    )

    for _ in range(500):
        result = assess_gaussian_backend_qualification(
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            environment_probe=probe,
            runtime_evidence=runtime,
            artifact=artifact,
            worker_evidence=evidence,
            expected_dataset_fingerprint_sha256="c" * 64,
        )
        assert result.status is GaussianBackendQualificationStatus.QUALIFIED
        assert result.reasons == ()
