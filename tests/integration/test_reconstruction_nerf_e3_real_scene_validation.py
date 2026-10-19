from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    NerfHeldOutRenderManifest,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    deterministic_nerf_validation_split,
    read_nerf_held_out_render_manifest,
    write_neural_scene_dataset_manifest,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset

ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path, count: int = 12) -> NeuralSceneDataset:
    frames = []
    for index in range(count):
        image = (tmp_path / "images" / f"frame-{index:02d}.png").resolve()
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((72, 96, 3), 20 + index * 10, dtype=np.uint8), mode="RGB").save(image)
        pose = np.eye(4)
        pose[0, 3] = index * 0.04
        frames.append(
            NeuralSceneFrame(
                frame_id=f"frame-{index:02d}", image_path=str(image), width=96, height=72,
                fx=80.0, fy=80.0, cx=48.0, cy=36.0,
                T_reconstruction_from_camera=pose, pose_layer="optimized_keyframe",
                camera_profile_id="camera",
            )
        )
    return NeuralSceneDataset(
        session_id="e3-integration", scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction", camera_profile_id="camera",
        scale_profile_id="arbitrary", metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY, world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None, relative_scale_uncertainty=None, frames=tuple(frames),
    )


def test_e3_manifest_split_and_export_are_consistent(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    manifest = write_neural_scene_dataset_manifest(dataset, (tmp_path / "dataset.json").resolve())
    assert manifest.is_file()
    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in dataset.frames), validation_count=3)
    transforms = export_nerfstudio_dataset(
        dataset,
        tmp_path / "nerfstudio",
        copy_images=False,
        training_frame_ids=train,
        validation_frame_ids=heldout,
    )
    payload = json.loads(transforms.read_text())
    path_by_id = {item["depth3d_frame_id"]: item["file_path"] for item in payload["frames"]}
    assert payload["train_filenames"] == [path_by_id[value] for value in train]
    assert payload["test_filenames"] == [path_by_id[value] for value in heldout]
    assert all((transforms.parent / relative).exists() for relative in payload["train_filenames"] + payload["test_filenames"])


def test_e3_worker_maps_multiple_dataset_render_outputs_to_exact_frame_ids(tmp_path: Path) -> None:
    worker_path = ROOT / "scripts/nerf_workers/nerfstudio_e3_worker.py"
    spec = importlib.util.spec_from_file_location("e3_worker_integration", worker_path)
    assert spec is not None and spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    dataset = _dataset(tmp_path)
    train, heldout = deterministic_nerf_validation_split(tuple(frame.frame_id for frame in dataset.frames), validation_count=3)
    dataset_dir = tmp_path / "dataset-export"
    transforms_path = export_nerfstudio_dataset(
        dataset, dataset_dir, copy_images=False, training_frame_ids=train, validation_frame_ids=heldout
    )
    transforms = json.loads(transforms_path.read_text())
    by_id = {item["depth3d_frame_id"]: Path(item["file_path"]).stem for item in transforms["frames"]}
    render_root = tmp_path / "rendered"
    rgb = render_root / "test" / "rgb"
    rgb.mkdir(parents=True)
    for frame_id in heldout:
        frame = next(value for value in dataset.frames if value.frame_id == frame_id)
        Image.open(frame.image_path).save(rgb / f"{by_id[frame_id]}.png")
    renders = worker._map_rendered_outputs(
        render_root=render_root,
        transforms_path=transforms_path,
        validation_ids=heldout,
        dataset=dataset,
    )
    assert tuple(value.frame_id for value in renders) == heldout
    manifest = NerfHeldOutRenderManifest(
        validation_fingerprint_sha256="a" * 64,
        dataset_fingerprint_sha256="b" * 64,
        validation_frame_ids=heldout,
        renders=renders,
    )
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(manifest.as_dict(), indent=2))
    loaded = read_nerf_held_out_render_manifest(path)
    assert loaded == manifest
