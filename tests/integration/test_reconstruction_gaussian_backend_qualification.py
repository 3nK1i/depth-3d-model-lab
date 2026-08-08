from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction import (
    GaussianBackendKind,
    GaussianBackendQualificationStatus,
    GaussianEnvironmentProbe,
    GaussianRuntimeEvidence,
    GaussianSceneArtifact,
    GaussianWorkerQualificationEvidence,
    assess_gaussian_backend_qualification,
)
from scripts.run_gaussian_splatting_foundation_smoke import OUTPUT, main as run_d1_smoke


def test_d1_development_artifact_cannot_be_misqualified_as_real_gsplat(tmp_path: Path) -> None:
    assert run_d1_smoke() == 0
    manifest = OUTPUT / "worker-output/development-mock/artifact_manifest.json"
    artifact = GaussianSceneArtifact.from_payload(json.loads(manifest.read_text(encoding="utf-8")))
    assert artifact.backend_provenance.backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK

    isolated_python = str((tmp_path / "isolated/bin/python").resolve())
    probe = GaussianEnvironmentProbe(
        python_executable=isolated_python,
        available=True,
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        torch_version="2.7.1+cu128",
        gsplat_version="1.5.3",
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("Synthetic CUDA GPU",),
    )
    runtime = GaussianRuntimeEvidence(
        python_executable=isolated_python,
        python_version="3.10.18",
        torch_version="2.7.1+cu128",
        torch_cuda_version="12.8",
        cuda_available=True,
        device_names=("Synthetic CUDA GPU",),
        device_capabilities=((7, 5),),
    )
    preview = (tmp_path / "preview.png").resolve()
    preview.write_bytes(b"preview")
    metrics = (tmp_path / "metrics.json").resolve()
    metrics.write_text("{}", encoding="utf-8")
    worker_evidence = GaussianWorkerQualificationEvidence(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        real_cuda_execution=True,
        optimization_completed=True,
        render_written=True,
        training_steps=24,
        source_revision="1.5.3",
        preview_path=str(preview),
        metrics_path=str(metrics),
    )
    qualification = assess_gaussian_backend_qualification(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        environment_probe=probe,
        runtime_evidence=runtime,
        artifact=artifact,
        worker_evidence=worker_evidence,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
    )
    assert qualification.status is GaussianBackendQualificationStatus.REJECTED
    assert "artifact_backend_mismatch" in qualification.reasons
    assert "artifact_not_optimized" in qualification.reasons
    assert "artifact_not_render_capable" in qualification.reasons
