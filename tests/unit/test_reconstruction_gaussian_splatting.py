from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBackendProvenance,
    GaussianEnvironmentProbe,
    GaussianOptimizationProfile,
    GaussianOptimizationState,
    GaussianSceneArtifact,
    GaussianSceneRole,
    GaussianUsageScope,
    GaussianWorkerExecutionError,
    GaussianWorkerProcessSpec,
    GaussianWorkerRequest,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    ReconstructionValidationError,
    build_gaussian_seed,
    gaussian_backend_provenance,
    generate_quality_aware_surface_mesh,
    neural_scene_dataset_fingerprint,
    probe_gaussian_environment,
    write_gaussian_seed_npz,
    write_neural_scene_dataset_manifest,
)
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera, _metric_surface


def _surface(metric: bool = False):
    camera = _camera(96, 72)
    dense, quality, scale = (_metric_surface if metric else _arbitrary_surface)(96, 72, camera)
    return generate_quality_aware_surface_mesh(dense, quality), scale


def _image(tmp_path: Path, name: str = "frame.ppm") -> Path:
    path = (tmp_path / name).resolve()
    path.write_bytes(b"P6\n2 2\n255\n" + bytes([120, 100, 80]) * 4)
    return path


def _frame(tmp_path: Path, *, camera_profile_id: str, index: int = 0) -> NeuralSceneFrame:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = index * 0.1
    return NeuralSceneFrame(
        frame_id=f"frame-{index}",
        image_path=str(_image(tmp_path, f"frame-{index}.ppm")),
        width=2,
        height=2,
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
        T_reconstruction_from_camera=pose,
        pose_layer="optimized_keyframe",
        camera_profile_id=camera_profile_id,
        timestamp_ns=index,
    )


def _dataset(tmp_path: Path, *, metric: bool = False, frame_count: int = 2) -> NeuralSceneDataset:
    surface, _ = _surface(metric=metric)
    return NeuralSceneDataset(
        session_id="unit-neural-scene",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        camera_profile_id=surface.camera_profile_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        scale_status=surface.scale_status,
        world_alignment_status=surface.world_alignment_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        frames=tuple(_frame(tmp_path, camera_profile_id=surface.camera_profile_id, index=i) for i in range(frame_count)),
    )


def test_neural_scene_frame_preserves_camera_to_reconstruction_pose(tmp_path: Path) -> None:
    frame = _frame(tmp_path, camera_profile_id="camera", index=2)
    assert frame.T_reconstruction_from_camera[0, 3] == pytest.approx(0.2)
    assert frame.pose_layer == "optimized_keyframe"


def test_neural_scene_frame_requires_absolute_image_path(tmp_path: Path) -> None:
    frame = _frame(tmp_path, camera_profile_id="camera")
    with pytest.raises(ReconstructionValidationError):
        replace(frame, image_path="relative.ppm")


def test_neural_scene_frame_rejects_nonrigid_pose(tmp_path: Path) -> None:
    frame = _frame(tmp_path, camera_profile_id="camera")
    bad = frame.T_reconstruction_from_camera.copy()
    bad[0, 0] = 2.0
    with pytest.raises(ReconstructionValidationError):
        replace(frame, T_reconstruction_from_camera=bad)


def test_arbitrary_dataset_has_no_metric_scale(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    assert dataset.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert dataset.unit_to_metre is None
    assert dataset.relative_scale_uncertainty is None
    assert dataset.certified_metrology is False


def test_metric_dataset_preserves_provisional_scale(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=True)
    assert dataset.metric_geometry_status is MetricGeometryStatus.PROVISIONAL_METRIC
    assert dataset.unit_to_metre == pytest.approx(0.01)
    assert dataset.relative_scale_uncertainty is not None


def test_arbitrary_dataset_rejects_unit_to_metre(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    with pytest.raises(ReconstructionValidationError):
        replace(dataset, unit_to_metre=0.01)


def test_dataset_rejects_mixed_camera_profiles(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    bad_frame = replace(dataset.frames[0], camera_profile_id="other")
    with pytest.raises(ReconstructionValidationError):
        replace(dataset, frames=(bad_frame, dataset.frames[1]))


def test_dataset_fingerprint_is_deterministic(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    assert neural_scene_dataset_fingerprint(dataset) == neural_scene_dataset_fingerprint(dataset)
    assert len(neural_scene_dataset_fingerprint(dataset)) == 64


def test_dataset_fingerprint_changes_with_pose(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    frame = dataset.frames[0]
    pose = frame.T_reconstruction_from_camera.copy()
    pose[0, 3] = 0.25
    changed = replace(dataset, frames=(replace(frame, T_reconstruction_from_camera=pose), dataset.frames[1]))
    assert neural_scene_dataset_fingerprint(changed) != neural_scene_dataset_fingerprint(dataset)


def test_manifest_contains_dataset_fingerprint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    path = write_neural_scene_dataset_manifest(dataset, (tmp_path / "dataset.json").resolve())
    payload = json.loads(path.read_text())
    assert payload["dataset_fingerprint_sha256"] == neural_scene_dataset_fingerprint(dataset)
    assert payload["pose_convention"] == "T_reconstruction_from_camera"


def test_gaussian_seed_preserves_mesh_vertex_positions() -> None:
    surface, _ = _surface(metric=False)
    seed = build_gaussian_seed(surface, max_gaussians=256, minimum_vertex_quality=0.5)
    expected = surface.mesh.vertices_reconstruction[seed.source_vertex_indices].astype(np.float32)
    assert np.array_equal(seed.positions_reconstruction, expected)


def test_gaussian_seed_is_bounded() -> None:
    surface, _ = _surface(metric=False)
    seed = build_gaussian_seed(surface, max_gaussians=64)
    assert seed.gaussian_count == 64


def test_gaussian_seed_quality_gate_is_enforced() -> None:
    surface, _ = _surface(metric=False)
    seed = build_gaussian_seed(surface, max_gaussians=10_000, minimum_vertex_quality=0.88)
    assert float(seed.quality_scores.min()) >= 0.88


def test_gaussian_seed_rejects_empty_quality_selection() -> None:
    surface, _ = _surface(metric=False)
    with pytest.raises(ReconstructionValidationError):
        build_gaussian_seed(surface, minimum_vertex_quality=1.0)


def test_gaussian_seed_opacity_stays_strictly_bounded() -> None:
    surface, _ = _surface(metric=False)
    seed = build_gaussian_seed(surface, max_gaussians=128)
    assert np.all(seed.opacities > 0.0)
    assert np.all(seed.opacities < 1.0)


def test_gaussian_seed_npz_round_trip(tmp_path: Path) -> None:
    surface, _ = _surface(metric=False)
    seed = build_gaussian_seed(surface, max_gaussians=32)
    path = write_gaussian_seed_npz(seed, (tmp_path / "seed.npz").resolve())
    with np.load(path, allow_pickle=False) as payload:
        assert payload["positions_reconstruction"].shape == (32, 3)
        assert np.array_equal(payload["source_vertex_indices"], seed.source_vertex_indices)


def test_worker_request_uses_dataset_fingerprint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    request = GaussianWorkerRequest.from_dataset(
        request_id="request",
        backend_kind=GaussianBackendKind.DEVELOPMENT_MOCK,
        profile=GaussianOptimizationProfile.EFFICIENT,
        dataset=dataset,
        dataset_manifest_path=(tmp_path / "dataset.json").resolve(),
        seed_path=(tmp_path / "seed.npz").resolve(),
        output_dir=(tmp_path / "out").resolve(),
        maximum_input_frames=8,
    )
    assert request.dataset_fingerprint_sha256 == neural_scene_dataset_fingerprint(dataset)
    assert request.maximum_input_frames == 8


def _artifact_payload(
    tmp_path: Path,
    *,
    optimized: bool = False,
    backend_kind: GaussianBackendKind = GaussianBackendKind.DEVELOPMENT_MOCK,
) -> dict:
    return {
        "schema_version": 1,
        "request_id": "request",
        "backend_id": "backend",
        "backend_version": "1",
        "backend_provenance": gaussian_backend_provenance(backend_kind).as_dict(),
        "optimization_state": "optimized" if optimized else "seed_only",
        "scene_role": "appearance_only",
        "artifact_dir": str((tmp_path / "artifact").resolve()),
        "scene_path": str((tmp_path / "artifact/scene.npz").resolve()),
        "dataset_fingerprint_sha256": "a" * 64,
        "input_frame_count": 4,
        "gaussian_count": 100,
        "training_steps": 100 if optimized else 0,
        "training_duration_s": 1.0,
        "render_capable": optimized,
        "device": "cuda:0" if optimized else "mock",
        "pose_convention": "T_reconstruction_from_camera",
        "reconstruction_frame_id": "reconstruction",
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "relative_scale_uncertainty": None,
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }


def test_seed_only_artifact_cannot_claim_render_capability(tmp_path: Path) -> None:
    payload = _artifact_payload(tmp_path)
    payload["render_capable"] = True
    with pytest.raises(GaussianWorkerExecutionError):
        GaussianSceneArtifact.from_payload(payload)


def test_optimized_artifact_requires_training_steps(tmp_path: Path) -> None:
    payload = _artifact_payload(tmp_path, optimized=True)
    payload["training_steps"] = 0
    with pytest.raises(GaussianWorkerExecutionError):
        GaussianSceneArtifact.from_payload(payload)


def test_gaussian_artifact_is_appearance_only(tmp_path: Path) -> None:
    artifact = GaussianSceneArtifact.from_payload(_artifact_payload(tmp_path))
    assert artifact.scene_role is GaussianSceneRole.APPEARANCE_ONLY
    assert artifact.canonical_geometry_authoritative is False
    assert artifact.certified_metrology is False


def test_gaussian_artifact_rejects_canonical_geometry_claim(tmp_path: Path) -> None:
    payload = _artifact_payload(tmp_path)
    payload["canonical_geometry_authoritative"] = True
    with pytest.raises(GaussianWorkerExecutionError):
        GaussianSceneArtifact.from_payload(payload)


def test_worker_spec_defaults_to_isolation() -> None:
    spec = GaussianWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((Path(__file__).resolve().parents[2] / "scripts/run_gaussian_worker_mock.py").resolve()),
    )
    assert spec.require_isolated_interpreter is True
    assert spec.interpreter_is_isolated is False


def test_environment_probe_missing_python_is_safe(tmp_path: Path) -> None:
    probe = probe_gaussian_environment((tmp_path / "missing-python").resolve())
    assert probe.available is False
    assert probe.backend_ready is False


def test_environment_probe_contract_requires_matching_cuda_names() -> None:
    with pytest.raises(ReconstructionValidationError):
        GaussianEnvironmentProbe(
            python_executable=str(Path(sys.executable).resolve()),
            available=True,
            torch_version="1",
            gsplat_version="1",
            cuda_available=True,
            cuda_device_count=1,
            cuda_device_names=(),
        )


def test_environment_probe_reports_current_interpreter_not_isolated() -> None:
    probe = GaussianEnvironmentProbe(
        python_executable=str(Path(sys.executable).resolve()),
        available=False,
    )
    assert probe.interpreter_is_isolated is False


def test_metric_artifact_preserves_scale_uncertainty(tmp_path: Path) -> None:
    payload = _artifact_payload(tmp_path)
    payload["metric_geometry_status"] = "provisional_metric"
    payload["unit_to_metre"] = 0.01
    payload["relative_scale_uncertainty"] = 0.02
    artifact = GaussianSceneArtifact.from_payload(payload)
    assert artifact.unit_to_metre == pytest.approx(0.01)
    assert artifact.relative_scale_uncertainty == pytest.approx(0.02)


def test_arbitrary_artifact_rejects_metric_scale(tmp_path: Path) -> None:
    payload = _artifact_payload(tmp_path)
    payload["unit_to_metre"] = 0.01
    with pytest.raises(GaussianWorkerExecutionError):
        GaussianSceneArtifact.from_payload(payload)


def test_gsplat_backend_provenance_is_permissive_general_use() -> None:
    provenance = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    assert provenance.usage_scope is GaussianUsageScope.PERMISSIVE_GENERAL_USE
    assert provenance.license_name == "Apache-2.0"
    assert provenance.commercial_use_permitted_by_backend_license is True
    assert provenance.explicit_commercial_consent_required is False


def test_graphdeco_backend_provenance_is_research_only() -> None:
    provenance = gaussian_backend_provenance(
        GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
    )
    assert provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    assert provenance.license_name == "Gaussian-Splatting License"
    assert provenance.commercial_use_permitted_by_backend_license is False
    assert provenance.explicit_commercial_consent_required is True
    assert provenance.redistribution_license_notice_required is True
    assert provenance.research_publication_supported is True


def test_graphdeco_worker_request_serializes_research_scope(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, metric=False)
    request = GaussianWorkerRequest.from_dataset(
        request_id="graphdeco-request",
        backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        profile=GaussianOptimizationProfile.BALANCED,
        dataset=dataset,
        dataset_manifest_path=(tmp_path / "dataset.json").resolve(),
        seed_path=(tmp_path / "seed.npz").resolve(),
        output_dir=(tmp_path / "out").resolve(),
        maximum_input_frames=32,
    )
    payload = request.as_dict()
    assert payload["backend_kind"] == "graphdeco_reference_external"
    assert payload["backend_provenance"]["usage_scope"] == "research_evaluation_only"
    assert payload["backend_provenance"]["commercial_use_permitted_by_backend_license"] is False
    assert payload["backend_provenance"]["explicit_commercial_consent_required"] is True


def test_artifact_rejects_tampered_graphdeco_license_scope(tmp_path: Path) -> None:
    payload = _artifact_payload(
        tmp_path,
        backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
    )
    payload["backend_provenance"]["usage_scope"] = "permissive_general_use"
    payload["backend_provenance"]["commercial_use_permitted_by_backend_license"] = True
    with pytest.raises(GaussianWorkerExecutionError):
        GaussianSceneArtifact.from_payload(payload)


def test_artifact_accepts_canonical_graphdeco_research_provenance(tmp_path: Path) -> None:
    artifact = GaussianSceneArtifact.from_payload(
        _artifact_payload(
            tmp_path,
            backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        )
    )
    assert isinstance(artifact.backend_provenance, GaussianBackendProvenance)
    assert artifact.backend_provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    assert artifact.backend_provenance.explicit_commercial_consent_required is True


def test_graphdeco_probe_requires_checkout_root() -> None:
    with pytest.raises(ReconstructionValidationError):
        probe_gaussian_environment(
            Path(sys.executable).resolve(),
            backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        )


def test_graphdeco_probe_rejects_checkout_without_license(tmp_path: Path) -> None:
    root = (tmp_path / "graphdeco").resolve()
    (root / "gaussian_renderer").mkdir(parents=True)
    (root / "scene").mkdir(parents=True)
    (root / "train.py").write_text("# test\n", encoding="utf-8")
    (root / "gaussian_renderer" / "__init__.py").write_text("# test\n", encoding="utf-8")
    (root / "scene" / "gaussian_model.py").write_text("# test\n", encoding="utf-8")
    probe = probe_gaussian_environment(
        Path(sys.executable).resolve(),
        backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        graphdeco_root=root,
    )
    assert probe.available is False
    assert probe.graphdeco_license_present is False
    assert "missing_graphdeco_files" in probe.error


def test_graphdeco_probe_records_license_before_runtime_qualification(tmp_path: Path) -> None:
    root = (tmp_path / "graphdeco").resolve()
    (root / "gaussian_renderer").mkdir(parents=True)
    (root / "scene").mkdir(parents=True)
    (root / "train.py").write_text("# test\n", encoding="utf-8")
    (root / "gaussian_renderer" / "__init__.py").write_text("# test\n", encoding="utf-8")
    (root / "scene" / "gaussian_model.py").write_text("# test\n", encoding="utf-8")
    (root / "LICENSE.md").write_text(
        "research users academic and industrial; non-commercial research; explicit consent for other use\n",
        encoding="utf-8",
    )
    probe = probe_gaussian_environment(
        Path(sys.executable).resolve(),
        backend_kind=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        graphdeco_root=root,
    )
    assert probe.graphdeco_license_present is True
    assert probe.backend_provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
    # The test environment intentionally lacks the Graphdeco CUDA extensions.
    assert probe.backend_ready is False
