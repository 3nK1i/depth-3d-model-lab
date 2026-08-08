from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkHardwareProfile,
    GaussianOptimizationProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
    write_real_scene_gaussian_benchmark_report,
    write_real_scene_gaussian_benchmark_spec,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus


def test_dataset_to_held_out_benchmark_manifest_pipeline(tmp_path: Path) -> None:
    frames = []
    for index in range(6):
        image = (tmp_path / f"view-{index}.ppm").resolve()
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([40 + index, 50, 60]) * 4)
        pose = np.eye(4)
        pose[0, 3] = 0.03 * index
        frames.append(
            NeuralSceneFrame(
                frame_id=f"view-{index}",
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
        session_id="real-scene-contract-integration",
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
        benchmark_id="integration-real-scene",
        source_label="future-real-sequence",
        dataset=dataset,
        training_frame_ids=("view-0", "view-1", "view-2", "view-3"),
        validation_frame_ids=("view-4", "view-5"),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL, GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
    )
    spec_path = write_real_scene_gaussian_benchmark_spec(spec, tmp_path / "benchmark_spec.json")
    report_path = write_real_scene_gaussian_benchmark_report(
        RealSceneGaussianBenchmarkReport(spec=spec),
        tmp_path / "benchmark_report.json",
    )
    spec_payload = json.loads(spec_path.read_text())
    report_payload = json.loads(report_path.read_text())
    assert spec_payload["dataset_fingerprint_sha256"] == spec.dataset_fingerprint_sha256
    assert report_payload["benchmark_fingerprint_sha256"] == spec.benchmark_fingerprint_sha256
    assert report_payload["comparison_ready"] is False
    assert report_payload["results"] == []
