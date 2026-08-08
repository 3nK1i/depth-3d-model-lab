from __future__ import annotations

from pathlib import Path

import numpy as np

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkBackendResult,
    GaussianBenchmarkHardwareProfile,
    GaussianBenchmarkRunState,
    GaussianHeldOutQualityEvaluation,
    GaussianHeldOutViewMetrics,
    GaussianLifecycleEvidence,
    GaussianOptimizationProfile,
    GaussianPerformanceMetrics,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
    aggregate_gaussian_held_out_metrics,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus


def test_step2_quality_is_ready_but_full_comparison_waits_for_step3(tmp_path: Path) -> None:
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
        session_id="integration", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera", scale_profile_id="scale",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY, scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY, unit_to_metre=None,
        relative_scale_uncertainty=None, frames=tuple(frames),
    )
    spec = RealSceneGaussianBenchmarkSpec(
        benchmark_id="integration", source_label="integration", dataset=dataset,
        training_frame_ids=("frame-0", "frame-1", "frame-2"), validation_frame_ids=("frame-3",),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL,),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
    )
    view = GaussianHeldOutViewMetrics(frame_id="frame-3", psnr_db=24.0, ssim=0.82, lpips=0.24)
    quality = GaussianHeldOutQualityEvaluation(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=spec.validation_frame_ids,
        views=(view,), aggregate=aggregate_gaussian_held_out_metrics((view,)),
    )
    result = GaussianBenchmarkBackendResult(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        optimization_profile=spec.optimization_profile,
        hardware_profile=spec.hardware_profile,
        training_view_count=3, validation_view_count=1,
        quality=quality.aggregate,
        performance=GaussianPerformanceMetrics(training_duration_s=3.0),
        lifecycle=GaussianLifecycleEvidence(state=GaussianBenchmarkRunState.COMPLETED, worker_exit_code=0),
    )
    report = RealSceneGaussianBenchmarkReport(spec=spec, results=(result,))
    assert result.quality.measured is True
    assert result.performance.measured is False
    assert result.comparison_ready is False
    assert report.comparison_ready is False
