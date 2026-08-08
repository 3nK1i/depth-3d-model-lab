from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianOptimizationState,
    GaussianUsageScope,
    GaussianWorkerProcessSpec,
    GaussianWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    build_gaussian_seed,
    generate_quality_aware_surface_mesh,
    run_gaussian_external_worker,
    write_gaussian_seed_npz,
    write_neural_scene_dataset_manifest,
)
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera

ROOT = Path(__file__).resolve().parents[2]


def test_surface_to_external_gaussian_seed_pipeline(tmp_path: Path) -> None:
    camera = _camera(96, 72)
    dense, quality, _scale = _arbitrary_surface(96, 72, camera)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    seed = build_gaussian_seed(surface, max_gaussians=256, minimum_vertex_quality=0.75)
    frames = []
    for index in range(3):
        image = (tmp_path / f"frame-{index}.ppm").resolve()
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([100 + index, 90, 80]) * 4)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.05 * index
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index}",
                image_path=str(image),
                width=2,
                height=2,
                fx=2.0,
                fy=2.0,
                cx=1.0,
                cy=1.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id=surface.camera_profile_id,
            )
        )
    dataset = NeuralSceneDataset(
        session_id="integration-gaussian",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        camera_profile_id=surface.camera_profile_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        scale_status=surface.scale_status,
        world_alignment_status=surface.world_alignment_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        frames=tuple(frames),
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (tmp_path / "dataset.json").resolve())
    seed_path = write_gaussian_seed_npz(seed, (tmp_path / "seed.npz").resolve())
    request = GaussianWorkerRequest.from_dataset(
        request_id="integration-request",
        backend_kind=GaussianBackendKind.DEVELOPMENT_MOCK,
        profile=GaussianOptimizationProfile.EFFICIENT,
        dataset=dataset,
        dataset_manifest_path=manifest,
        seed_path=seed_path,
        output_dir=(tmp_path / "worker").resolve(),
        maximum_input_frames=16,
    )
    spec = GaussianWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((ROOT / "scripts/run_gaussian_worker_mock.py").resolve()),
        timeout_s=30.0,
        require_isolated_interpreter=False,
    )
    artifact = run_gaussian_external_worker(spec, request)
    assert artifact.optimization_state is GaussianOptimizationState.SEED_ONLY
    assert artifact.gaussian_count == seed.gaussian_count
    assert artifact.input_frame_count == 3
    assert artifact.render_capable is False
    assert artifact.backend_provenance.backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK
    assert artifact.backend_provenance.usage_scope is GaussianUsageScope.DEVELOPMENT_ONLY
    assert artifact.canonical_geometry_authoritative is False
    assert artifact.certified_metrology is False
