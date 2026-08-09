#!/usr/bin/env python3
"""Dependency-light helpers shared by Phase 8.7E2 NeRF qualification tools."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

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


def export_nerfstudio_dataset(
    dataset: NeuralSceneDataset,
    output_dir: str | Path,
    *,
    copy_images: bool = True,
) -> Path:
    """Export an EIA neural-scene dataset to Nerfstudio ``transforms.json``.

    The canonical EIA pose is OpenCV-style camera-to-world (X right, Y down,
    Z forward).  Nerfstudio's dataset format consumes OpenGL-style c2w poses;
    the conversion flips camera-axis Y/Z while preserving the reconstruction
    world frame.
    """

    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    root = Path(output_dir).expanduser().resolve()
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    relative_paths: list[str] = []
    for index, frame in enumerate(dataset.frames):
        source = Path(frame.image_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"NeRF input image is missing: {source}")
        suffix = source.suffix.lower() or ".png"
        target = images_dir / f"{index:05d}_{frame.frame_id}{suffix}"
        if copy_images:
            shutil.copy2(source, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(source)
        relative = target.relative_to(root).as_posix()
        relative_paths.append(relative)
        frames.append(
            {
                "file_path": relative,
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

    # E2 qualifies backend execution, not held-out quality.  We nevertheless
    # provide an explicit deterministic validation camera so rendering reloads
    # exercise Nerfstudio's test/val path.  E3 owns real held-out evaluation.
    if len(relative_paths) >= 2:
        train_paths = relative_paths[:-1]
        eval_paths = [relative_paths[-1]]
    else:
        train_paths = list(relative_paths)
        eval_paths = list(relative_paths)

    payload = {
        "camera_model": "OPENCV",
        "orientation_override": "none",
        "frames": frames,
        "train_filenames": train_paths,
        "val_filenames": eval_paths,
        "test_filenames": eval_paths,
        "eia_pose_source_convention": "T_reconstruction_from_camera",
        "eia_nerfstudio_camera_convention": "OpenGL_c2w",
        "eia_backend_qualification_only": True,
    }
    transforms = root / "transforms.json"
    transforms.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return transforms
