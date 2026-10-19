from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import re
from pathlib import Path
import sys

import pytest

from core.reconstruction import (
    GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION,
    GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY,
    GAUSSIAN_QUALIFIED_CUDA,
    GAUSSIAN_QUALIFIED_PYTHON,
    GAUSSIAN_QUALIFIED_TORCH,
    GRAPHDECO_QUALIFIED_COMPATIBILITY_PATCH,
    GRAPHDECO_HELDOUT_SPLIT_COMPATIBILITY_PATCH,
    GRAPHDECO_QUALIFIED_REVISION,
    GSPLAT_QUALIFIED_TAG,
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
    ReconstructionValidationError,
    assess_gaussian_backend_qualification,
    gaussian_backend_provenance,
    gaussian_backend_runtime_pin,
    not_ready_gaussian_backend_qualification,
    write_gaussian_backend_qualification_json,
)

ROOT = Path(__file__).resolve().parents[2]


def _probe(tmp_path: Path, backend_kind: GaussianBackendKind) -> GaussianEnvironmentProbe:
    kwargs = dict(
        python_executable=str((tmp_path / "isolated" / "bin" / "python").resolve()),
        available=True,
        backend_kind=backend_kind,
        torch_version="2.7.1+cu128",
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("Synthetic CUDA GPU",),
    )
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        return GaussianEnvironmentProbe(**kwargs, gsplat_version="1.5.3")
    return GaussianEnvironmentProbe(
        **kwargs,
        graphdeco_root=str((tmp_path / "graphdeco").resolve()),
        graphdeco_license_present=True,
        graphdeco_runtime_ready=True,
    )


def _runtime(tmp_path: Path) -> GaussianRuntimeEvidence:
    return GaussianRuntimeEvidence(
        python_executable=str((tmp_path / "isolated" / "bin" / "python").resolve()),
        python_version="3.10.18",
        torch_version="2.7.1+cu128",
        torch_cuda_version="12.8",
        cuda_available=True,
        device_names=("Synthetic CUDA GPU",),
        device_capabilities=((7, 5),),
    )


def _artifact(tmp_path: Path, backend_kind: GaussianBackendKind) -> GaussianSceneArtifact:
    artifact_dir = (tmp_path / backend_kind.value).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scene_path = artifact_dir / "scene.bin"
    scene_path.write_bytes(b"scene")
    backend_version = (
        GSPLAT_QUALIFIED_VERSION
        if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL
        else GRAPHDECO_QUALIFIED_REVISION
    )
    return GaussianSceneArtifact(
        request_id="request",
        backend_id="real-backend",
        backend_version=backend_version,
        backend_provenance=gaussian_backend_provenance(backend_kind),
        optimization_state=GaussianOptimizationState.OPTIMIZED,
        scene_role=GaussianSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_dir),
        scene_path=str(scene_path),
        dataset_fingerprint_sha256="a" * 64,
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
        peak_vram_mib=100.0,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )


def _worker_evidence(tmp_path: Path, backend_kind: GaussianBackendKind) -> GaussianWorkerQualificationEvidence:
    artifact_dir = (tmp_path / backend_kind.value).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preview = artifact_dir / "preview.png"
    preview.write_bytes(b"png")
    source_revision = (
        GSPLAT_QUALIFIED_VERSION
        if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL
        else GRAPHDECO_QUALIFIED_REVISION
    )
    metrics = artifact_dir / "qualification_metrics.json"
    metrics.write_text("{}", encoding="utf-8")
    return GaussianWorkerQualificationEvidence(
        backend_kind=backend_kind,
        real_cuda_execution=True,
        optimization_completed=True,
        render_written=True,
        training_steps=24,
        source_revision=source_revision,
        preview_path=str(preview),
        metrics_path=str(metrics),
    )


def test_runtime_pins_are_explicit_and_stable() -> None:
    gsplat = gaussian_backend_runtime_pin(GaussianBackendKind.GSPLAT_EXTERNAL)
    graphdeco = gaussian_backend_runtime_pin(GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL)
    assert gsplat.backend_version == GSPLAT_QUALIFIED_VERSION == "1.5.3"
    assert gsplat.upstream_revision == GSPLAT_QUALIFIED_TAG == "v1.5.3"
    assert graphdeco.backend_version == GRAPHDECO_QUALIFIED_REVISION
    assert gsplat.compatibility_patches == ()
    assert graphdeco.compatibility_patches == (
        GRAPHDECO_QUALIFIED_COMPATIBILITY_PATCH,
        GRAPHDECO_HELDOUT_SPLIT_COMPATIBILITY_PATCH,
    )
    assert len(GRAPHDECO_QUALIFIED_REVISION) == 40
    assert gsplat.python_version == graphdeco.python_version == GAUSSIAN_QUALIFIED_PYTHON == "3.10"
    assert gsplat.torch_version == graphdeco.torch_version == GAUSSIAN_QUALIFIED_TORCH == "2.7.1"
    assert gsplat.cuda_version == graphdeco.cuda_version == GAUSSIAN_QUALIFIED_CUDA == "12.8"
    assert GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY == (7, 0)


def test_runtime_pin_rejects_development_backend() -> None:
    with pytest.raises(ReconstructionValidationError):
        gaussian_backend_runtime_pin(GaussianBackendKind.DEVELOPMENT_MOCK)


@pytest.mark.parametrize(
    "backend_kind",
    (GaussianBackendKind.GSPLAT_EXTERNAL, GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL),
)
def test_qualification_accepts_real_cuda_optimized_renderable_artifact(tmp_path: Path, backend_kind) -> None:
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend_kind,
        environment_probe=_probe(tmp_path, backend_kind),
        runtime_evidence=_runtime(tmp_path),
        artifact=_artifact(tmp_path, backend_kind),
        worker_evidence=_worker_evidence(tmp_path, backend_kind),
        expected_dataset_fingerprint_sha256="a" * 64,
        graphdeco_revision=(
            GRAPHDECO_QUALIFIED_REVISION
            if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
            else ""
        ),
    )
    assert qualification.status is GaussianBackendQualificationStatus.QUALIFIED
    assert qualification.qualified is True
    assert qualification.reasons == ()
    assert qualification.schema_version == GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION == 1


def test_qualification_rejects_seed_only_artifact(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    artifact = _artifact(tmp_path, backend)
    seed_scene = Path(artifact.scene_path)
    seed_only = GaussianSceneArtifact(
        request_id=artifact.request_id,
        backend_id=artifact.backend_id,
        backend_version=artifact.backend_version,
        backend_provenance=artifact.backend_provenance,
        optimization_state=GaussianOptimizationState.SEED_ONLY,
        scene_role=artifact.scene_role,
        artifact_dir=artifact.artifact_dir,
        scene_path=str(seed_scene),
        dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        input_frame_count=artifact.input_frame_count,
        gaussian_count=artifact.gaussian_count,
        training_steps=0,
        training_duration_s=0.0,
        render_capable=False,
        device="cuda:0",
        pose_convention=artifact.pose_convention,
        reconstruction_frame_id=artifact.reconstruction_frame_id,
        metric_geometry_status=artifact.metric_geometry_status,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
    )
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=_runtime(tmp_path),
        artifact=seed_only,
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
    )
    assert qualification.qualified is False
    assert "artifact_not_optimized" in qualification.reasons
    assert "artifact_not_render_capable" in qualification.reasons


def test_qualification_rejects_wrong_gsplat_version(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    probe = replace(_probe(tmp_path, backend), gsplat_version="1.6.0")
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=probe,
        runtime_evidence=_runtime(tmp_path),
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
    )
    assert "gsplat_version_mismatch" in qualification.reasons


def test_qualification_rejects_wrong_graphdeco_revision(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=_runtime(tmp_path),
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
        graphdeco_revision="b" * 40,
    )
    assert "graphdeco_revision_mismatch" in qualification.reasons


def test_qualification_rejects_wrong_torch_cuda_runtime(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    runtime = replace(_runtime(tmp_path), torch_cuda_version="12.4")
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=runtime,
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
    )
    assert "torch_cuda_version_mismatch" in qualification.reasons


def test_qualification_rejects_below_minimum_compute_capability(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    runtime = replace(_runtime(tmp_path), device_capabilities=((6, 1),))
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=runtime,
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
    )
    assert "compute_capability_below_minimum" in qualification.reasons


def test_qualification_rejects_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=_runtime(tmp_path),
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="b" * 64,
    )
    assert "dataset_fingerprint_mismatch" in qualification.reasons


def test_not_ready_qualification_does_not_fake_artifact(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    probe = replace(_probe(tmp_path, backend), available=False, cuda_available=False, cuda_device_count=0, cuda_device_names=())
    runtime = replace(_runtime(tmp_path), cuda_available=False, device_names=(), device_capabilities=())
    qualification = not_ready_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=probe,
        runtime_evidence=runtime,
        reason="backend_environment_not_ready",
    )
    assert qualification.status is GaussianBackendQualificationStatus.NOT_READY
    assert qualification.artifact is None
    assert qualification.worker_evidence is None


def test_worker_qualification_evidence_loads_from_external_json(tmp_path: Path) -> None:
    preview = (tmp_path / "preview.png").resolve()
    preview.write_bytes(b"png")
    metrics = (tmp_path / "qualification_metrics.json").resolve()
    metrics.write_text(
        json.dumps(
            {
                "backend_kind": "gsplat_external",
                "real_cuda_execution": True,
                "optimization_completed": True,
                "render_written": True,
                "training_steps": 24,
                "source_revision": "1.5.3",
                "preview_path": str(preview),
            }
        ),
        encoding="utf-8",
    )
    evidence = GaussianWorkerQualificationEvidence.from_file(metrics)
    assert evidence.backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL
    assert evidence.real_cuda_execution is True
    assert evidence.preview_path == str(preview)


def test_qualification_report_round_trip_json(tmp_path: Path) -> None:
    backend = GaussianBackendKind.GSPLAT_EXTERNAL
    qualification = assess_gaussian_backend_qualification(
        backend_kind=backend,
        environment_probe=_probe(tmp_path, backend),
        runtime_evidence=_runtime(tmp_path),
        artifact=_artifact(tmp_path, backend),
        worker_evidence=_worker_evidence(tmp_path, backend),
        expected_dataset_fingerprint_sha256="a" * 64,
    )
    path = write_gaussian_backend_qualification_json(qualification, (tmp_path / "qualification.json").resolve())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["qualified"] is True
    assert payload["runtime_pin"]["backend_version"] == "1.5.3"


def test_production_core_has_no_real_backend_imports() -> None:
    forbidden = (
        "import gsplat",
        "from gsplat",
        "import gaussian_renderer",
        "from gaussian_renderer",
        "import diff_gaussian_rasterization",
        "from diff_gaussian_rasterization",
        "import simple_knn",
        "from simple_knn",
    )
    for path in (ROOT / "core" / "reconstruction").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_external_workers_contain_backend_specific_imports() -> None:
    gsplat_text = (ROOT / "scripts/gaussian_workers/gsplat_worker.py").read_text(encoding="utf-8")
    graphdeco_text = (ROOT / "scripts/gaussian_workers/graphdeco_worker.py").read_text(encoding="utf-8")
    assert "from gsplat.rendering import rasterization" in gsplat_text
    assert "DEPTH3D_GRAPHDECO_ROOT" in graphdeco_text
    assert "train.py" in graphdeco_text
    assert "render.py" in graphdeco_text


def test_setup_script_pins_versions_and_never_pip_installs_into_depth() -> None:
    text = (ROOT / "scripts/setup_gaussian_backends.sh").read_text(encoding="utf-8")
    assert 'GSPLAT_TAG="v1.5.3"' in text
    assert f'GRAPHDECO_REVISION="{GRAPHDECO_QUALIFIED_REVISION}"' in text
    assert 'TORCH_VERSION="2.7.1"' in text
    assert 'CUDA_VERSION="12.8"' in text
    assert "conda run -n" in text
    assert "pip install gsplat" not in text
    assert "conda activate depth" not in text


def test_runtime_evidence_requires_matching_device_vectors(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError):
        GaussianRuntimeEvidence(
            python_executable=str((tmp_path / "python").resolve()),
            python_version="3.10.0",
            torch_version="2.7.1",
            torch_cuda_version="12.8",
            cuda_available=True,
            device_names=("GPU",),
            device_capabilities=(),
        )


def test_real_backend_probe_path_is_not_current_interpreter(tmp_path: Path) -> None:
    probe = _probe(tmp_path, GaussianBackendKind.GSPLAT_EXTERNAL)
    assert Path(probe.python_executable).resolve() != Path(sys.executable).resolve()
    assert probe.interpreter_is_isolated is True


def test_setup_script_verifies_exact_gsplat_release_commit() -> None:
    text = (ROOT / "scripts/setup_gaussian_backends.sh").read_text(encoding="utf-8")
    assert 'GSPLAT_REVISION_SHORT="937e299"' in text
    assert 'git -C "$checkout" rev-parse --short=7 HEAD' in text


def test_setup_script_applies_audited_graphdeco_cstdint_compatibility_patch() -> None:
    setup_text = (ROOT / "scripts/setup_gaussian_backends.sh").read_text(encoding="utf-8")
    patch_path = ROOT / "scripts/gaussian_patches/graphdeco_diff_gaussian_rasterization_cstdint.patch"
    patch_text = patch_path.read_text(encoding="utf-8")
    assert 'GRAPHDECO_CSTDINT_PATCH_ID="diff-gaussian-rasterization-pr49-cstdint"' in setup_text
    assert 'sha256sum "$GRAPHDECO_CSTDINT_PATCH"' in setup_text
    assert 'git -C "$rasterizer" apply --unidiff-zero --check "$GRAPHDECO_CSTDINT_PATCH"' in setup_text
    assert 'git -C "$rasterizer" apply --unidiff-zero --reverse --check "$GRAPHDECO_CSTDINT_PATCH"' in setup_text
    assert "+#include <cstdint>" in patch_text
    match = re.search(r'GRAPHDECO_CSTDINT_PATCH_SHA256="([0-9a-f]{64})"', setup_text)
    assert match is not None
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == match.group(1)


def test_setup_script_applies_audited_graphdeco_heldout_split_patch() -> None:
    setup_text = (ROOT / "scripts/setup_gaussian_backends.sh").read_text(encoding="utf-8")
    patch_path = ROOT / "scripts/gaussian_patches/graphdeco_explicit_test_split.patch"
    patch_text = patch_path.read_text(encoding="utf-8")
    assert 'GRAPHDECO_TEST_SPLIT_PATCH_ID="explicit-test-txt-heldout-split"' in setup_text
    assert 'sha256sum "$GRAPHDECO_TEST_SPLIT_PATCH"' in setup_text
    assert 'git -C "$checkout" apply --unidiff-zero --check "$GRAPHDECO_TEST_SPLIT_PATCH"' in setup_text
    assert 'git -C "$checkout" apply --unidiff-zero --reverse --check "$GRAPHDECO_TEST_SPLIT_PATCH"' in setup_text
    assert 'llffhold=0' in patch_text
    match = re.search(r'GRAPHDECO_TEST_SPLIT_PATCH_SHA256="([0-9a-f]{64})"', setup_text)
    assert match is not None
    assert hashlib.sha256(patch_path.read_bytes()).hexdigest() == match.group(1)


def test_graphdeco_worker_rejects_off_center_principal_point() -> None:
    text = (ROOT / "scripts/gaussian_workers/graphdeco_worker.py").read_text(encoding="utf-8")
    assert "requires a centered principal point" in text
    assert 'f"{camera_id} PINHOLE' in text
    assert 'sparse_dir / "points3D.ply"' in text
    assert '"--densify_until_iter",' in text
    assert '"0",' in text


def test_real_qualification_runner_scopes_external_conda_cuda_environment() -> None:
    text = (ROOT / "scripts/run_gaussian_backend_qualification.py").read_text(encoding="utf-8")
    assert "def _isolated_backend_environment" in text
    assert 'os.environ["CONDA_PREFIX"] = str(prefix)' in text
    assert 'os.environ["PYTHONNOUSERSITE"] = "1"' in text
    assert 'os.environ["CUDA_HOME"] = str(prefix)' in text
    assert 'os.environ["CUDACXX"] = str(nvcc)' in text
    assert "finally:" in text
