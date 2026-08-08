from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    build_gaussian_seed,
    generate_quality_aware_surface_mesh,
    neural_scene_dataset_fingerprint,
)
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera


def test_gaussian_seed_and_dataset_remain_bounded_under_repetition(tmp_path: Path) -> None:
    camera = _camera(96, 72)
    dense, quality, _scale = _arbitrary_surface(96, 72, camera)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    original = surface.mesh.vertices_reconstruction.copy()
    for _ in range(50):
        seed = build_gaussian_seed(surface, max_gaussians=128, minimum_vertex_quality=0.70)
        assert seed.gaussian_count == 128
        assert np.array_equal(surface.mesh.vertices_reconstruction, original)
    image = (tmp_path / "frame.ppm").resolve()
    image.write_bytes(b"P6\n2 2\n255\n" + bytes([100, 90, 80]) * 4)
    frames = []
    for index in range(500):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.001
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
                pose_layer="optimized_propagated",
                camera_profile_id=surface.camera_profile_id,
                timestamp_ns=index,
            )
        )
    dataset = NeuralSceneDataset(
        session_id="stress-gaussian",
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
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
    assert dataset.frame_count == 500
    assert len(neural_scene_dataset_fingerprint(dataset)) == 64
