#!/usr/bin/env python3
"""Real CUDA held-out-render/quality smoke for Phase 8.7D3 Step 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    NeuralSceneDataset,
    NeuralSceneFrame,
    build_gaussian_seed,
    generate_quality_aware_surface_mesh,
    write_gaussian_seed_npz,
    write_neural_scene_dataset_manifest,
)
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera

DEFAULT_OUTPUT = Path("/tmp/phase8_7d3_heldout_quality_smoke")


def _default_env_python(env_name: str) -> Path:
    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 3 and executable.parent.name == "bin":
        envs_dir = executable.parents[2]
        if envs_dir.name == "envs":
            return envs_dir / env_name / "bin" / "python"
    return Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python"


def _write_image(path: Path, index: int) -> None:
    width, height = 96, 72
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(35 + xx * 2 + index * 9, 0, 255)
    image[..., 1] = np.clip(45 + yy * 2 + index * 7, 0, 255)
    image[..., 2] = np.clip(80 + ((xx + yy + index * 11) % 120), 0, 255)
    x0 = 18 + index * 3
    image[20:48, x0:x0 + 24] = np.array([220, 170 - index * 8, 50 + index * 12], dtype=np.uint8)
    Image.fromarray(image, mode="RGB").save(path)


def prepare_heldout_smoke_inputs(output: Path) -> tuple[Path, Path]:
    output = output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    input_dir = output / "input"
    images_dir = input_dir / "images"
    images_dir.mkdir(parents=True)

    camera = _camera(96, 72)
    dense, quality, _scale = _arbitrary_surface(96, 72, camera)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    seed = build_gaussian_seed(surface, max_gaussians=512, minimum_vertex_quality=0.75)
    frames = []
    for index in range(5):
        image_path = images_dir / f"frame_{index:03d}.png"
        _write_image(image_path, index)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = (index - 2.0) * 0.07
        frames.append(
            NeuralSceneFrame(
                frame_id=f"heldout-smoke-{index}",
                image_path=str(image_path.resolve()),
                width=96, height=72, fx=82.0, fy=82.0, cx=48.0, cy=36.0,
                T_reconstruction_from_camera=pose,
                pose_layer="optimized_keyframe",
                camera_profile_id=surface.camera_profile_id,
                timestamp_ns=index * 33_333_333,
            )
        )
    dataset = NeuralSceneDataset(
        session_id="phase8-7d3-heldout-quality-smoke",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        camera_profile_id=surface.camera_profile_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        scale_status=surface.scale_status,
        world_alignment_status=surface.world_alignment_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        frames=tuple(frames),
    )
    manifest = write_neural_scene_dataset_manifest(dataset, (input_dir / "dataset.json").resolve())
    seed_path = write_gaussian_seed_npz(seed, (input_dir / "gaussian_seed.npz").resolve())
    return manifest, seed_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gsplat", "graphdeco", "both"), default="both")
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("depth3d-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("depth3d-gaussian-graphdeco"))
    parser.add_argument("--metrics-python", type=Path, default=_default_env_python("depth3d-gaussian-metrics"))
    parser.add_argument(
        "--graphdeco-root", type=Path,
        default=Path.home() / "Desktop" / "depth3d-gaussian-backends" / "gaussian-splatting",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-steps", type=int, default=64)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    manifest, seed_path = prepare_heldout_smoke_inputs(output)
    runner = ROOT / "scripts" / "run_gaussian_real_scene_benchmark.py"
    benchmark_output = output / "benchmark"
    command = [
        sys.executable, str(runner),
        "--dataset-manifest", str(manifest),
        "--seed", str(seed_path),
        "--validation-frame-id", "heldout-smoke-4",
        "--benchmark-id", "phase8-7d3-heldout-quality-smoke",
        "--source-label", "deterministic-heldout-smoke",
        "--backend", args.backend,
        "--profile", "efficient",
        "--training-steps", str(args.training_steps),
        "--gsplat-python", str(args.gsplat_python.expanduser().resolve()),
        "--graphdeco-python", str(args.graphdeco_python.expanduser().resolve()),
        "--metrics-python", str(args.metrics_python.expanduser().resolve()),
        "--graphdeco-root", str(args.graphdeco_root.expanduser().resolve()),
        "--metrics-device", "cpu",
        "--output-dir", str(benchmark_output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    print("Phase 8.7D3 held-out quality smoke: PASS")
    print(f"summary={benchmark_output / 'step2_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
