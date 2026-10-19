#!/usr/bin/env python3
"""Dependency-light helpers shared by Phase 8.7E2 NeRF qualification tools."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from core.reconstruction import (
    NeuralSceneDataset,
    NerfOptimizationProfile,
    nerfstudio_camera_to_world_from_eia_pose,
)


def profile_settings(profile: NerfOptimizationProfile) -> dict[str, int]:
    """Small bounded E2 profiles; E4 owns performance tuning."""

    if profile is NerfOptimizationProfile.EFFICIENT:
        return {
            "train_rays_per_batch": 512,
            "eval_rays_per_batch": 512,
            "eval_num_rays_per_chunk": 2048,
        }
    if profile is NerfOptimizationProfile.BALANCED:
        return {
            "train_rays_per_batch": 1024,
            "eval_rays_per_batch": 1024,
            "eval_num_rays_per_chunk": 4096,
        }
    return {
        "train_rays_per_batch": 2048,
        "eval_rays_per_batch": 2048,
        "eval_num_rays_per_chunk": 8192,
    }


def _safe_frame_token(frame_id: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(frame_id).strip())
    return token or "frame"


def export_nerfstudio_dataset(
    dataset: NeuralSceneDataset,
    output_dir: str | Path,
    *,
    copy_images: bool = True,
    training_frame_ids: Sequence[str] | None = None,
    validation_frame_ids: Sequence[str] | None = None,
) -> Path:
    """Export a canonical neural-scene dataset to Nerfstudio ``transforms.json``.

    E2 keeps its historical default split when explicit IDs are omitted. E3 can
    provide exact train/held-out frame IDs; those IDs must be disjoint and cover
    the dataset exactly. The canonical reconstruction pose is OpenCV-style camera-to-world
    (X right, Y down, Z forward), while Nerfstudio consumes OpenGL-style c2w.
    """

    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    root = Path(output_dir).expanduser().resolve()
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = tuple(frame.frame_id for frame in dataset.frames)
    explicit_split = training_frame_ids is not None or validation_frame_ids is not None
    if explicit_split:
        if training_frame_ids is None or validation_frame_ids is None:
            raise ValueError("training_frame_ids and validation_frame_ids must be provided together")
        training_ids = tuple(str(value).strip() for value in training_frame_ids)
        validation_ids = tuple(str(value).strip() for value in validation_frame_ids)
        if not training_ids or not validation_ids:
            raise ValueError("explicit Nerfstudio train/validation splits must be non-empty")
        if any(not value for value in training_ids + validation_ids):
            raise ValueError("explicit Nerfstudio split frame IDs must be non-empty")
        if len(set(training_ids)) != len(training_ids) or len(set(validation_ids)) != len(validation_ids):
            raise ValueError("explicit Nerfstudio split frame IDs must be unique")
        if set(training_ids) & set(validation_ids):
            raise ValueError("explicit Nerfstudio train/validation splits must be disjoint")
        if set(training_ids) | set(validation_ids) != set(dataset_ids):
            raise ValueError("explicit Nerfstudio split must cover the dataset exactly")
    else:
        if len(dataset_ids) >= 2:
            training_ids = dataset_ids[:-1]
            validation_ids = (dataset_ids[-1],)
        else:
            training_ids = dataset_ids
            validation_ids = dataset_ids

    frames: list[dict[str, Any]] = []
    relative_by_id: dict[str, str] = {}
    for index, frame in enumerate(dataset.frames):
        source = Path(frame.image_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"NeRF input image is missing: {source}")
        suffix = source.suffix.lower() or ".png"
        target = images_dir / f"{index:05d}_{_safe_frame_token(frame.frame_id)}{suffix}"
        if copy_images:
            shutil.copy2(source, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
        relative = target.relative_to(root).as_posix()
        relative_by_id[frame.frame_id] = relative
        frames.append(
            {
                "file_path": relative,
                "depth3d_frame_id": frame.frame_id,
                "fl_x": float(frame.fx),
                "fl_y": float(frame.fy),
                "cx": float(frame.cx),
                "cy": float(frame.cy),
                "w": int(frame.width),
                "h": int(frame.height),
                "transform_matrix": nerfstudio_camera_to_world_from_eia_pose(
                    frame.T_reconstruction_from_camera
                ),
            }
        )

    train_paths = [relative_by_id[value] for value in training_ids]
    eval_paths = [relative_by_id[value] for value in validation_ids]
    payload = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": frames,
        "train_filenames": train_paths,
        "val_filenames": eval_paths,
        "test_filenames": eval_paths,
        "depth3d_pose_source_convention": "T_reconstruction_from_camera",
        "depth3d_nerfstudio_camera_convention": "OpenGL_c2w",
        "depth3d_explicit_heldout_split": bool(explicit_split),
        "depth3d_backend_qualification_only": not bool(explicit_split),
    }
    transforms = root / "transforms.json"
    transforms.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return transforms
