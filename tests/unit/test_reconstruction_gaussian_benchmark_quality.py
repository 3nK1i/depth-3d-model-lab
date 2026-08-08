from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianBenchmarkHardwareProfile,
    GaussianHeldOutQualityEvaluation,
    GaussianHeldOutRender,
    GaussianHeldOutRenderManifest,
    GaussianHeldOutViewMetrics,
    GaussianImageQualityMetrics,
    GaussianOptimizationProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    RealSceneGaussianBenchmarkSpec,
    aggregate_gaussian_held_out_metrics,
    neural_scene_dataset_fingerprint,
    read_gaussian_held_out_quality_evaluation,
    read_gaussian_held_out_render_manifest,
    read_neural_scene_dataset_manifest,
    write_gaussian_held_out_quality_evaluation,
    write_gaussian_held_out_render_manifest,
    write_neural_scene_dataset_manifest,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus

ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path) -> NeuralSceneDataset:
    frames = []
    for index in range(4):
        image = (tmp_path / f"frame-{index}.png").resolve()
        image.write_bytes(b"not-decoded-by-contract-tests")
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.1
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index}",
                image_path=str(image),
                width=96,
                height=72,
                fx=80.0,
                fy=80.0,
                cx=48.0,
                cy=36.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="quality-test",
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
        benchmark_id="heldout-quality-test",
        source_label="unit-test",
        dataset=_dataset(tmp_path),
        training_frame_ids=("frame-0", "frame-1", "frame-2"),
        validation_frame_ids=("frame-3",),
        backends=(GaussianBackendKind.GSPLAT_EXTERNAL,),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.RTX_2060_6GB,
        hardware_label="RTX 2060 6 GB",
    )


def test_neural_scene_manifest_round_trip_verifies_fingerprint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    path = write_neural_scene_dataset_manifest(dataset, (tmp_path / "dataset.json").resolve())
    loaded = read_neural_scene_dataset_manifest(path)
    assert loaded.as_dict() == dataset.as_dict()
    assert neural_scene_dataset_fingerprint(loaded) == neural_scene_dataset_fingerprint(dataset)

    payload = json.loads(path.read_text())
    payload["dataset_fingerprint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(Exception, match="fingerprint"):
        read_neural_scene_dataset_manifest(path)


def test_held_out_render_manifest_requires_exact_validation_coverage(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    render = GaussianHeldOutRender(
        frame_id="frame-3",
        reference_image_path=str((tmp_path / "frame-3.png").resolve()),
        rendered_image_path=str((tmp_path / "render-3.png").resolve()),
        width=96,
        height=72,
    )
    manifest = GaussianHeldOutRenderManifest(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=spec.validation_frame_ids,
        renders=(render,),
    )
    path = write_gaussian_held_out_render_manifest(manifest, tmp_path / "render_manifest.json")
    assert read_gaussian_held_out_render_manifest(path) == manifest

    with pytest.raises(Exception, match="cover every validation frame"):
        GaussianHeldOutRenderManifest(
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
            dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
            validation_frame_ids=("frame-2", "frame-3"),
            renders=(render,),
        )


def test_quality_evaluation_is_auditable_and_aggregate_is_recomputed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    views = (
        GaussianHeldOutViewMetrics(frame_id="frame-3", psnr_db=28.0, ssim=0.91, lpips=0.12),
    )
    aggregate = aggregate_gaussian_held_out_metrics(views)
    evaluation = GaussianHeldOutQualityEvaluation(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=spec.validation_frame_ids,
        views=views,
        aggregate=aggregate,
    )
    path = write_gaussian_held_out_quality_evaluation(evaluation, tmp_path / "quality.json")
    loaded = read_gaussian_held_out_quality_evaluation(path)
    assert loaded == evaluation
    assert loaded.aggregate == GaussianImageQualityMetrics(psnr_db=28.0, ssim=0.91, lpips=0.12)
    assert loaded.lpips_network == "alex"
    assert loaded.lpips_version == "0.1"
    assert loaded.lpips_package_version == "0.1.4"

    with pytest.raises(Exception, match="do not match"):
        GaussianHeldOutQualityEvaluation(
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
            dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
            validation_frame_ids=spec.validation_frame_ids,
            views=views,
            aggregate=GaussianImageQualityMetrics(psnr_db=10.0, ssim=0.5, lpips=0.9),
        )


def test_psnr_and_ssim_implementation_has_expected_identity_behavior() -> None:
    worker = ROOT / "scripts" / "gaussian_benchmark_workers" / "metrics_worker.py"
    spec = importlib.util.spec_from_file_location("eia_metrics_worker_test", worker)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    image = torch.linspace(0.0, 1.0, 3 * 72 * 96, dtype=torch.float32).reshape(1, 3, 72, 96)
    assert module._psnr(image, image) == pytest.approx(120.0)
    assert module._ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    perturbed = image.clone()
    perturbed[..., 20:30, 20:30] = 0.0
    assert module._psnr(image, perturbed) < 120.0
    assert 0.0 <= module._ssim(image, perturbed) < 1.0


def test_graphdeco_materialized_dataset_places_explicit_test_list_in_sparse_model(tmp_path: Path) -> None:
    worker_path = ROOT / "scripts" / "gaussian_benchmark_workers" / "graphdeco_worker.py"
    worker_spec = importlib.util.spec_from_file_location("eia_graphdeco_benchmark_worker_test", worker_path)
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)

    from PIL import Image

    frames = []
    for index in range(3):
        image_path = (tmp_path / f"source-{index}.png").resolve()
        Image.fromarray(np.full((24, 32, 3), 32 + index * 64, dtype=np.uint8), mode="RGB").save(image_path)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.1
        frames.append(
            {
                "frame_id": f"frame-{index}",
                "image_path": str(image_path),
                "width": 32,
                "height": 24,
                "fx": 30.0,
                "fy": 30.0,
                "cx": 16.0,
                "cy": 12.0,
                "T_reconstruction_from_camera": pose.tolist(),
            }
        )

    seed = {
        "positions_reconstruction": np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        "colors_rgb": np.asarray([[128, 128, 128]], dtype=np.uint8),
    }
    target = (tmp_path / "graphdeco-dataset").resolve()
    generated, validation_names = worker._materialize_benchmark_dataset(
        {"frames": frames},
        seed,
        target,
        validation_frame_ids=("frame-2",),
    )

    expected_name = generated["frame-2"]
    assert validation_names == [expected_name]
    split_path = target / "sparse" / "0" / "test.txt"
    assert split_path.read_text(encoding="utf-8") == expected_name + "\n"
    assert not (target / "test.txt").exists()
