from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkBackendResult,
    GaussianBenchmarkHardwareProfile,
    GaussianBenchmarkRunState,
    GaussianImageQualityMetrics,
    GaussianLifecycleEvidence,
    GaussianOptimizationProfile,
    GaussianPerformanceMetrics,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
    ReconstructionValidationError,
    write_real_scene_gaussian_benchmark_report,
    write_real_scene_gaussian_benchmark_spec,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus


def _dataset(tmp_path: Path, count: int = 5) -> NeuralSceneDataset:
    frames = []
    for index in range(count):
        image = (tmp_path / f"frame-{index}.ppm").resolve()
        image.write_bytes(b"P6\n2 2\n255\n" + bytes([index + 1, 20, 30]) * 4)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.1
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
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="benchmark-test",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
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


def _spec(tmp_path: Path) -> RealSceneGaussianBenchmarkSpec:
    return RealSceneGaussianBenchmarkSpec(
        benchmark_id="office-sequence-01",
        source_label="real-office-walkaround",
        dataset=_dataset(tmp_path),
        training_frame_ids=("frame-0", "frame-1", "frame-2", "frame-3"),
        validation_frame_ids=("frame-4",),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL, GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL),
        optimization_profile=GaussianOptimizationProfile.BALANCED,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        hardware_label="RTX 2060 6 GB",
    )


def test_real_scene_split_is_disjoint_complete_and_fingerprinted(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert len(spec.benchmark_fingerprint_sha256) == 64
    assert set(spec.training_frame_ids).isdisjoint(spec.validation_frame_ids)
    assert set(spec.training_frame_ids) | set(spec.validation_frame_ids) == {
        frame.frame_id for frame in spec.dataset.frames
    }


def test_training_and_validation_overlap_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="disjoint"):
        RealSceneGaussianBenchmarkSpec(
            benchmark_id="bad",
            source_label="bad",
            dataset=dataset,
            training_frame_ids=("frame-0", "frame-1", "frame-2", "frame-4"),
            validation_frame_ids=("frame-4", "frame-3"),
            backends=(GaussianBackendKind.GSPLAT_EXTERNAL,),
            optimization_profile=GaussianOptimizationProfile.EFFICIENT,
            hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        )


def test_unassigned_dataset_view_is_rejected(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="partition"):
        RealSceneGaussianBenchmarkSpec(
            benchmark_id="bad",
            source_label="bad",
            dataset=dataset,
            training_frame_ids=("frame-0", "frame-1", "frame-2"),
            validation_frame_ids=("frame-4",),
            backends=(GaussianBackendKind.GSPLAT_EXTERNAL,),
            optimization_profile=GaussianOptimizationProfile.EFFICIENT,
            hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        )


def test_development_mock_is_not_a_d3_benchmark_backend(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="real external"):
        RealSceneGaussianBenchmarkSpec(
            benchmark_id="bad",
            source_label="bad",
            dataset=dataset,
            training_frame_ids=("frame-0", "frame-1", "frame-2", "frame-3"),
            validation_frame_ids=("frame-4",),
            backends=(GaussianBackendKind.DEVELOPMENT_MOCK,),
            optimization_profile=GaussianOptimizationProfile.EFFICIENT,
            hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        )


def test_quality_metrics_are_all_or_none() -> None:
    assert GaussianImageQualityMetrics().measured is False
    measured = GaussianImageQualityMetrics(psnr_db=28.5, ssim=0.91, lpips=0.12)
    assert measured.measured is True
    with pytest.raises(ReconstructionValidationError, match="measured together"):
        GaussianImageQualityMetrics(psnr_db=28.5)


def test_performance_metrics_can_be_incrementally_unmeasured() -> None:
    assert GaussianPerformanceMetrics().measured is False
    partial = GaussianPerformanceMetrics(training_duration_s=10.0)
    assert partial.measured is False
    complete = GaussianPerformanceMetrics(peak_vram_mib=4100.0, training_duration_s=10.0, render_fps=45.0)
    assert complete.measured is True


def test_lifecycle_evidence_has_fail_closed_semantics() -> None:
    with pytest.raises(ReconstructionValidationError, match="requires an error"):
        GaussianLifecycleEvidence(state=GaussianBenchmarkRunState.CRASHED)
    with pytest.raises(ReconstructionValidationError, match="cancellation_requested"):
        GaussianLifecycleEvidence(state=GaussianBenchmarkRunState.CANCELLED)
    recovered = GaussianLifecycleEvidence(
        state=GaussianBenchmarkRunState.RECOVERED,
        recovery_attempted=True,
        recovery_succeeded=True,
    )
    assert recovered.recovery_succeeded is True


def test_result_never_claims_canonical_geometry_or_certified_metrology(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="canonical geometry"):
        GaussianBenchmarkBackendResult(
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
            dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
            optimization_profile=spec.optimization_profile,
            hardware_profile=spec.hardware_profile,
            training_view_count=4,
            validation_view_count=1,
            canonical_geometry_authoritative=True,
        )


def test_completed_measured_results_make_report_comparison_ready(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    results = []
    for backend in spec.backends:
        results.append(
            GaussianBenchmarkBackendResult(
                backend_kind=backend,
                benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
                dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
                optimization_profile=spec.optimization_profile,
                hardware_profile=spec.hardware_profile,
                training_view_count=4,
                validation_view_count=1,
                quality=GaussianImageQualityMetrics(psnr_db=25.0, ssim=0.9, lpips=0.2),
                performance=GaussianPerformanceMetrics(
                    peak_vram_mib=4000.0,
                    training_duration_s=12.0,
                    render_fps=30.0,
                ),
                lifecycle=GaussianLifecycleEvidence(state=GaussianBenchmarkRunState.COMPLETED, worker_exit_code=0),
            )
        )
    report = RealSceneGaussianBenchmarkReport(spec=spec, results=tuple(results))
    assert report.comparison_ready is True


def test_json_writers_preserve_split_and_no_fake_metrics(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec_path = write_real_scene_gaussian_benchmark_spec(spec, tmp_path / "spec.json")
    report_path = write_real_scene_gaussian_benchmark_report(
        RealSceneGaussianBenchmarkReport(spec=spec),
        tmp_path / "report.json",
    )
    spec_payload = json.loads(spec_path.read_text())
    report_payload = json.loads(report_path.read_text())
    assert spec_payload["validation_frame_ids"] == ["frame-4"]
    assert report_payload["results"] == []
    assert report_payload["comparison_ready"] is False
    assert report_payload["canonical_geometry_authoritative"] is False
    assert report_payload["certified_metrology"] is False
