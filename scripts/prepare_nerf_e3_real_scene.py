#!/usr/bin/env python3
"""Prepare a real Nerfstudio-format capture for Phase 8.7E3 validation.

The helper is intended for the official Nerfstudio ``poster`` capture or an
equivalent Nerfstudio-format real scene.  Selected images are undistorted and
optionally downscaled into a project-local pinhole dataset, camera poses are
converted into ``T_reconstruction_from_camera``, and a deterministic held-out
split is written alongside the canonical NeuralSceneDataset manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    deterministic_nerf_validation_split,
    eia_pose_from_nerfstudio_camera_to_world,
    neural_scene_dataset_fingerprint,
    write_neural_scene_dataset_manifest,
)


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(frame: dict, root: dict, key: str, default: float | int | None = None):
    if key in frame:
        return frame[key]
    if key in root:
        return root[key]
    if default is not None:
        return default
    raise ValueError(f"Nerfstudio transforms is missing required camera field: {key}")


def _select_evenly(frames: list[dict], maximum: int) -> list[dict]:
    if maximum < 12:
        raise ValueError("E3 real-scene preparation requires --max-frames >= 12")
    if len(frames) <= maximum:
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, num=maximum, dtype=np.int64)
    unique = []
    seen: set[int] = set()
    for raw in indices.tolist():
        value = int(raw)
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return [frames[index] for index in unique]


def _undistort_and_resize(
    source: Path,
    target: Path,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
    k3: float,
    max_long_edge: int,
) -> tuple[int, int, float, float, float, float]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode source image: {source}")
    height, width = image.shape[:2]
    camera = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    distortion = np.asarray([k1, k2, p1, p2, k3], dtype=np.float64)
    if float(np.max(np.abs(distortion))) > 1e-12:
        image = cv2.undistort(image, camera, distortion, None, camera)

    scale = min(1.0, float(max_long_edge) / float(max(width, height)))
    out_width = max(64, int(round(width * scale)))
    out_height = max(64, int(round(height * scale)))
    if (out_width, out_height) != (width, height):
        image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
    sx = float(out_width) / float(width)
    sy = float(out_height) / float(height)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Could not write prepared image: {target}")
    return out_width, out_height, fx * sx, fy * sy, cx * sx, cy * sy


def prepare(source_dir: Path, output_dir: Path, *, max_frames: int, heldout_count: int, max_long_edge: int) -> dict:
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    transforms = source_dir / "transforms.json"
    if not transforms.is_file():
        raise FileNotFoundError(f"Nerfstudio transforms.json is missing: {transforms}")
    payload = json.loads(transforms.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
        raise ValueError("Nerfstudio transforms.json must contain a frames array")
    camera_model = str(payload.get("camera_model", "OPENCV")).upper()
    if camera_model not in {"OPENCV", "PERSPECTIVE"}:
        raise ValueError(f"E3 preparation currently supports OPENCV/PERSPECTIVE captures, got {camera_model}")

    source_frames = sorted(payload["frames"], key=lambda item: _natural_key(str(item.get("file_path", ""))))
    selected = _select_evenly(source_frames, int(max_frames))
    if len(selected) - int(heldout_count) < 8:
        raise ValueError("Selected scene does not leave at least eight training views")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frames: list[NeuralSceneFrame] = []
    source_records: list[dict] = []
    for index, item in enumerate(selected):
        relative = str(item.get("file_path", "")).strip()
        if not relative:
            raise ValueError("Nerfstudio frame is missing file_path")
        source = (source_dir / relative).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source frame is missing: {source}")
        fx = float(_value(item, payload, "fl_x"))
        fy = float(_value(item, payload, "fl_y"))
        cx = float(_value(item, payload, "cx"))
        cy = float(_value(item, payload, "cy"))
        k1 = float(_value(item, payload, "k1", 0.0))
        k2 = float(_value(item, payload, "k2", 0.0))
        p1 = float(_value(item, payload, "p1", 0.0))
        p2 = float(_value(item, payload, "p2", 0.0))
        k3 = float(_value(item, payload, "k3", 0.0))
        source_stem = source.stem
        frame_id = f"real-{index:03d}-{source_stem}"
        target = (images_dir / f"{frame_id}.png").resolve()
        width, height, out_fx, out_fy, out_cx, out_cy = _undistort_and_resize(
            source,
            target,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            k1=k1,
            k2=k2,
            p1=p1,
            p2=p2,
            k3=k3,
            max_long_edge=int(max_long_edge),
        )
        pose = np.asarray(
            eia_pose_from_nerfstudio_camera_to_world(item["transform_matrix"]), dtype=np.float64
        )
        frames.append(
            NeuralSceneFrame(
                frame_id=frame_id,
                image_path=str(target),
                width=width,
                height=height,
                fx=out_fx,
                fy=out_fy,
                cx=out_cx,
                cy=out_cy,
                T_reconstruction_from_camera=pose,
                pose_layer="external_reference_nerfstudio",
                camera_profile_id="e3-undistorted-pinhole",
            )
        )
        source_records.append(
            {
                "frame_id": frame_id,
                "source_file_path": str(source),
                "prepared_image_path": str(target),
            }
        )

    dataset = NeuralSceneDataset(
        session_id="phase8-7e3-real-scene",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="e3-undistorted-pinhole",
        scale_profile_id="external-arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )
    training_ids, validation_ids = deterministic_nerf_validation_split(
        tuple(frame.frame_id for frame in frames), validation_count=int(heldout_count)
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (output_dir / "dataset.json").resolve())
    split_payload = {
        "phase": "8.7E3",
        "source_format": "nerfstudio_transforms_json",
        "source_dir": str(source_dir),
        "source_transforms_path": str(transforms.resolve()),
        "source_transforms_sha256": _sha256_file(transforms),
        "camera_model": camera_model,
        "preparation": {
            "selected_frame_count": len(frames),
            "maximum_frames": int(max_frames),
            "heldout_count": int(heldout_count),
            "max_long_edge": int(max_long_edge),
            "distortion_policy": "opencv_undistort_to_eia_pinhole",
            "sampling_policy": "natural_filename_order_even_coverage",
            "heldout_policy": "deterministic_interior_trajectory_spread",
        },
        "dataset_manifest_path": str(manifest),
        "dataset_fingerprint_sha256": neural_scene_dataset_fingerprint(dataset),
        "training_frame_ids": list(training_ids),
        "validation_frame_ids": list(validation_ids),
        "source_records": source_records,
        "pose_convention": "T_reconstruction_from_camera",
        "metric_geometry_status": "arbitrary",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    split_path = output_dir / "split.json"
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "pass": True,
        "dataset_manifest_path": str(manifest),
        "split_manifest_path": str(split_path),
        "selected_frame_count": len(frames),
        "training_frame_count": len(training_ids),
        "validation_frame_count": len(validation_ids),
        "dataset_fingerprint_sha256": neural_scene_dataset_fingerprint(dataset),
    }
    (output_dir / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--heldout-count", type=int, default=4)
    parser.add_argument("--max-long-edge", type=int, default=720)
    args = parser.parse_args()
    summary = prepare(
        args.source_dir,
        args.output_dir,
        max_frames=args.max_frames,
        heldout_count=args.heldout_count,
        max_long_edge=args.max_long_edge,
    )
    print("Phase 8.7E3 real-scene preparation: PASS")
    print(
        f"frames={summary['selected_frame_count']} train={summary['training_frame_count']} "
        f"heldout={summary['validation_frame_count']}"
    )
    print(f"dataset={summary['dataset_manifest_path']}")
    print(f"split={summary['split_manifest_path']}")
    print("metric_geometry_status=arbitrary canonical_geometry_authoritative=False certified_metrology=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
