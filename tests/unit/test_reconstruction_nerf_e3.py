from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    NerfHeldOutQualityEvaluation,
    NerfHeldOutRender,
    NerfHeldOutRenderManifest,
    NerfHeldOutViewMetrics,
    NerfImageQualityMetrics,
    NerfOptimizationProfile,
    NerfQualitySanityPolicy,
    NerfRealSceneValidationSpec,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    aggregate_nerf_held_out_metrics,
    deterministic_nerf_validation_split,
    depth3d_pose_from_nerfstudio_camera_to_world,
    nerfstudio_camera_to_world_from_depth3d_pose,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset

ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path, count: int = 12) -> NeuralSceneDataset:
    frames = []
    for index in range(count):
        path = (tmp_path / f"frame-{index:02d}.png").resolve()
        Image.fromarray(np.full((72, 96, 3), 20 + index * 8, dtype=np.uint8), mode="RGB").save(path)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index:02d}",
                image_path=str(path),
                width=96,
                height=72,
                fx=80.0,
                fy=81.0,
                cx=48.0,
                cy=36.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="e3-unit",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="camera",
        scale_profile_id="arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


def test_nerfstudio_pose_conversion_is_exactly_invertible() -> None:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    ns_pose = np.asarray(nerfstudio_camera_to_world_from_depth3d_pose(pose))
    restored = np.asarray(depth3d_pose_from_nerfstudio_camera_to_world(ns_pose))
    assert np.allclose(restored, pose)
    assert np.allclose(ns_pose[:3, 0], pose[:3, 0])
    assert np.allclose(ns_pose[:3, 1:3], -pose[:3, 1:3])


def test_deterministic_validation_split_is_spread_and_reproducible() -> None:
    ids = tuple(f"f{i:02d}" for i in range(24))
    train1, val1 = deterministic_nerf_validation_split(ids, validation_count=4)
    train2, val2 = deterministic_nerf_validation_split(ids, validation_count=4)
    assert (train1, val1) == (train2, val2)
    assert len(train1) == 20
    assert len(val1) == 4
    assert not set(train1) & set(val1)
    assert set(train1) | set(val1) == set(ids)
    indices = [ids.index(value) for value in val1]
    assert indices == sorted(indices)
    assert min(indices) > 0 and max(indices) < len(ids) - 1


def test_split_rejects_too_small_real_scene() -> None:
    with pytest.raises(Exception, match="training views"):
        deterministic_nerf_validation_split(tuple(f"f{i}" for i in range(10)), validation_count=3)
    with pytest.raises(Exception, match="at least 3 held-out"):
        deterministic_nerf_validation_split(tuple(f"f{i}" for i in range(20)), validation_count=2)


def test_real_scene_spec_covers_dataset_exactly(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in dataset.frames), validation_count=3)
    spec = NerfRealSceneValidationSpec(
        validation_id="e3",
        source_label="real-scene",
        dataset=dataset,
        training_frame_ids=train,
        validation_frame_ids=heldout,
        optimization_profile=NerfOptimizationProfile.EFFICIENT,
        training_steps=128,
    )
    assert len(spec.validation_fingerprint_sha256) == 64
    assert spec.as_dict()["performance_hardening_deferred_to"] == "8.7E4"
    with pytest.raises(Exception, match="cover the dataset exactly"):
        NerfRealSceneValidationSpec(
            validation_id="bad",
            source_label="real",
            dataset=dataset,
            training_frame_ids=train[:-1],
            validation_frame_ids=heldout,
            training_steps=128,
        )


def test_explicit_export_preserves_e2_default_and_supports_e3_split(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    default_path = export_nerfstudio_dataset(dataset, tmp_path / "default")
    default_payload = json.loads(default_path.read_text())
    assert len(default_payload["train_filenames"]) == 11
    assert len(default_payload["test_filenames"]) == 1
    assert default_payload["depth3d_backend_qualification_only"] is True

    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in dataset.frames), validation_count=3)
    explicit_path = export_nerfstudio_dataset(
        dataset,
        tmp_path / "explicit",
        copy_images=False,
        training_frame_ids=train,
        validation_frame_ids=heldout,
    )
    payload = json.loads(explicit_path.read_text())
    assert len(payload["train_filenames"]) == 9
    assert len(payload["test_filenames"]) == 3
    assert payload["depth3d_explicit_heldout_split"] is True
    assert {item["depth3d_frame_id"] for item in payload["frames"]} == {frame.frame_id for frame in dataset.frames}


def test_explicit_export_rejects_overlap_or_partial_coverage(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    ids = tuple(frame.frame_id for frame in dataset.frames)
    with pytest.raises(ValueError, match="disjoint"):
        export_nerfstudio_dataset(dataset, tmp_path / "bad1", training_frame_ids=ids[:-2], validation_frame_ids=ids[-3:])
    with pytest.raises(ValueError, match="cover"):
        export_nerfstudio_dataset(dataset, tmp_path / "bad2", training_frame_ids=ids[:-3], validation_frame_ids=ids[-2:])


def test_heldout_render_manifest_exposes_metrics_worker_adapter_alias(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in dataset.frames), validation_count=3)
    spec = NerfRealSceneValidationSpec(
        validation_id="e3", source_label="real", dataset=dataset,
        training_frame_ids=train, validation_frame_ids=heldout, training_steps=128,
    )
    by_id = {frame.frame_id: frame for frame in dataset.frames}
    renders = []
    for frame_id in heldout:
        frame = by_id[frame_id]
        rendered = (tmp_path / "renders" / f"{frame_id}.png").resolve()
        rendered.parent.mkdir(exist_ok=True)
        rendered.write_bytes(Path(frame.image_path).read_bytes())
        renders.append(NerfHeldOutRender(frame_id, frame.image_path, str(rendered), frame.width, frame.height))
    manifest = NerfHeldOutRenderManifest(
        validation_fingerprint_sha256=spec.validation_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        validation_frame_ids=heldout,
        renders=tuple(renders),
    )
    payload = manifest.as_dict()
    assert payload["benchmark_fingerprint_sha256"] == spec.validation_fingerprint_sha256
    assert payload["canonical_geometry_authoritative"] is False


def test_quality_evaluation_recomputes_aggregate_and_applies_sanity_policy() -> None:
    views = tuple(
        NerfHeldOutViewMetrics(frame_id=f"f{i}", psnr_db=20.0 + i, ssim=0.7, lpips=0.2)
        for i in range(3)
    )
    aggregate = aggregate_nerf_held_out_metrics(views)
    evaluation = NerfHeldOutQualityEvaluation(
        validation_fingerprint_sha256="1" * 64,
        dataset_fingerprint_sha256="2" * 64,
        validation_frame_ids=tuple(value.frame_id for value in views),
        views=views,
        aggregate=aggregate,
    )
    assert evaluation.passes(NerfQualitySanityPolicy())
    assert not evaluation.passes(NerfQualitySanityPolicy(minimum_psnr_db=40.0))
    with pytest.raises(Exception, match="do not match"):
        NerfHeldOutQualityEvaluation(
            validation_fingerprint_sha256="1" * 64,
            dataset_fingerprint_sha256="2" * 64,
            validation_frame_ids=tuple(value.frame_id for value in views),
            views=views,
            aggregate=NerfImageQualityMetrics(psnr_db=10.0, ssim=0.2, lpips=0.8),
        )


def test_quality_parser_accepts_existing_backend_neutral_d3_metric_payload() -> None:
    payload = {
        "schema_version": 1,
        "backend_kind": "nerfstudio_nerfacto_external",
        "benchmark_fingerprint_sha256": "a" * 64,
        "dataset_fingerprint_sha256": "b" * 64,
        "validation_frame_ids": ["a", "b", "c"],
        "views": [
            {"frame_id": "a", "psnr_db": 20.0, "ssim": 0.8, "lpips": 0.2},
            {"frame_id": "b", "psnr_db": 21.0, "ssim": 0.8, "lpips": 0.2},
            {"frame_id": "c", "psnr_db": 22.0, "ssim": 0.8, "lpips": 0.2},
        ],
        "aggregate": {"measured": True, "psnr_db": 21.0, "ssim": 0.8, "lpips": 0.2},
        "evaluator": {
            "name": "depth3d-heldout-quality",
            "psnr_definition": "rgb_mse_unit_range_mse_floor_1e-12",
            "ssim_definition": "gaussian_11_sigma_1.5_rgb_unit_range",
            "lpips_network": "alex",
            "lpips_version": "0.1",
            "lpips_package_version": "0.1.4",
        },
    }
    evaluation = NerfHeldOutQualityEvaluation.from_metrics_worker_payload(payload)
    assert evaluation.aggregate.psnr_db == pytest.approx(21.0)
    assert evaluation.validation_fingerprint_sha256 == "a" * 64


def test_e3_worker_has_no_top_level_nerfstudio_import() -> None:
    path = ROOT / "scripts/nerf_workers/nerfstudio_e3_worker.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    top = [line for line in lines if line.startswith("import nerfstudio") or line.startswith("from nerfstudio")]
    assert top == []


def test_prepare_script_undistorts_downscales_and_writes_canonical_split(tmp_path: Path) -> None:
    module_path = ROOT / "scripts/prepare_nerf_e3_real_scene.py"
    spec = importlib.util.spec_from_file_location("e3_prepare_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    source = tmp_path / "source"
    images = source / "images"
    images.mkdir(parents=True)
    frames = []
    for index in range(12):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.circle(image, (80, 60), 15 + index, (30 + index * 10, 120, 200), -1)
        path = images / f"frame_{index:05d}.png"
        assert cv2.imwrite(str(path), image)
        pose = np.eye(4)
        pose[0, 3] = index * 0.1
        frames.append({"file_path": f"./images/{path.name}", "transform_matrix": pose.tolist()})
    transforms = {
        "camera_model": "OPENCV",
        "fl_x": 130.0, "fl_y": 132.0, "cx": 80.0, "cy": 60.0,
        "w": 160, "h": 120,
        "k1": -0.01, "k2": 0.002, "p1": 0.0, "p2": 0.0,
        "frames": frames,
    }
    (source / "transforms.json").write_text(json.dumps(transforms))
    result = module.prepare(source, tmp_path / "prepared", max_frames=12, heldout_count=3, max_long_edge=96)
    assert result["pass"] is True
    split = json.loads((tmp_path / "prepared/split.json").read_text())
    assert len(split["training_frame_ids"]) == 9
    assert len(split["validation_frame_ids"]) == 3
    dataset = json.loads((tmp_path / "prepared/dataset.json").read_text())
    assert dataset["pose_convention"] == "T_reconstruction_from_camera"
    assert max(dataset["frames"][0]["width"], dataset["frames"][0]["height"]) == 96
    assert Path(dataset["frames"][0]["image_path"]).is_file()
