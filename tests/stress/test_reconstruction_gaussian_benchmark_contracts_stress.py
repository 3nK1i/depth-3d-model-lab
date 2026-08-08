from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkHardwareProfile,
    GaussianOptimizationProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkSpec,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus


def test_benchmark_fingerprint_is_stable_under_repeated_access(tmp_path: Path) -> None:
    frames = []
    for index in range(8):
        image = (tmp_path / f"frame-{index}.ppm").resolve()
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([index, 1, 2]) * 4)
        pose = np.eye(4)
        pose[1, 3] = index * 0.01
        frames.append(
            NeuralSceneFrame(
                frame_id=f"f{index}",
                image_path=str(image),
                width=2,
                height=2,
                fx=2.0,
                fy=2.0,
                cx=1.0,
                cy=1.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    dataset = NeuralSceneDataset(
        session_id="stress",
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="camera",
        scale_profile_id="scale",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )
    spec = RealSceneGaussianBenchmarkSpec(
        benchmark_id="stress-benchmark",
        source_label="stress",
        dataset=dataset,
        training_frame_ids=("f0", "f1", "f2", "f3", "f4", "f5"),
        validation_frame_ids=("f6", "f7"),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL, GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
    )
    expected = spec.benchmark_fingerprint_sha256
    for _ in range(1000):
        assert spec.benchmark_fingerprint_sha256 == expected
