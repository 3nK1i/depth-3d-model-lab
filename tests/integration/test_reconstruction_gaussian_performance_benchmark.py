from __future__ import annotations

from pathlib import Path
import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkBackendResult,
    GaussianBenchmarkHardwareProfile,
    GaussianBenchmarkRunState,
    GaussianHardwareEvidence,
    GaussianImageQualityMetrics,
    GaussianLifecycleEvidence,
    GaussianOptimizationProfile,
    GaussianPerformanceMetrics,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus


def _spec(tmp_path: Path) -> RealSceneGaussianBenchmarkSpec:
    frames = []
    for index in range(4):
        image = (tmp_path / f"frame-{index}.png").resolve()
        image.write_bytes(b"x")
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        frames.append(NeuralSceneFrame(
            frame_id=f"frame-{index}", image_path=str(image), width=96, height=72,
            fx=80.0, fy=80.0, cx=48.0, cy=36.0,
            T_reconstruction_from_camera=pose, pose_layer="optimized_keyframe",
            camera_profile_id="camera",
        ))
    dataset = NeuralSceneDataset(
        session_id="performance-integration", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera", scale_profile_id="scale",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY, scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY, unit_to_metre=None,
        relative_scale_uncertainty=None, frames=tuple(frames),
    )
    return RealSceneGaussianBenchmarkSpec(
        benchmark_id="performance-integration", source_label="integration", dataset=dataset,
        training_frame_ids=("frame-0", "frame-1", "frame-2"), validation_frame_ids=("frame-3",),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL,),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
    )


def test_step3_detailed_performance_makes_quality_report_comparison_ready(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    result = GaussianBenchmarkBackendResult(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        optimization_profile=spec.optimization_profile,
        hardware_profile=spec.hardware_profile,
        training_view_count=3,
        validation_view_count=1,
        quality=GaussianImageQualityMetrics(psnr_db=25.0, ssim=0.85, lpips=0.2),
        performance=GaussianPerformanceMetrics(
            peak_training_vram_mib=2200.0,
            peak_render_vram_mib=1700.0,
            training_duration_s=3.5,
            render_duration_s=1.0,
            render_frame_count=30,
            render_repeats=30,
            render_fps=30.0,
            measurement_scope="backend_process_tree_nvidia_smi",
            render_measurement_mode="backend_native_warm",
        ),
        hardware=GaussianHardwareEvidence(
            gpu_name="NVIDIA GeForce RTX 2060", gpu_uuid="GPU-abc", total_vram_mib=6144.0,
            driver_version="610.43.02", cuda_runtime_version="12.8", compute_capability=(7, 5),
            profile_compatible=True,
        ),
        lifecycle=GaussianLifecycleEvidence(state=GaussianBenchmarkRunState.COMPLETED, worker_exit_code=0),
    )
    report = RealSceneGaussianBenchmarkReport(spec=spec, results=(result,))
    assert result.performance.detailed_measured is True
    assert result.hardware.measured is True
    assert result.comparison_ready is True
    assert report.comparison_ready is True
    assert report.as_dict()["results"][0]["hardware"]["profile_compatible"] is True
