#!/usr/bin/env python3
"""Real isolated Graphdeco reference-backend qualification worker for 8.7D2.

The worker adapts the EIA neural-scene manifest into a minimal COLMAP-text
workspace, initializes Graphdeco from the trusted EIA Gaussian seed, runs a
bounded real CUDA optimization, and verifies that upstream render.py can render
the resulting model.  Full-quality paper-style training belongs to 8.7D3.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch

from common import (
    load_json,
    ply_vertex_count,
    rotation_matrix_to_quaternion_wxyz,
    validate_request,
    write_ascii_ply,
    write_qualification_metrics,
    write_worker_result,
)

BACKEND_KIND = "graphdeco_reference_external"
QUALIFIED_REVISION = "54c035f7834b564019656c3e3fcc3646292f727d"
PROFILE_STEPS = {"efficient": 64, "balanced": 128, "high_quality": 256}


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else ""


def _validate_graphdeco_root(root: Path) -> None:
    required = (
        root / "train.py",
        root / "render.py",
        root / "LICENSE.md",
        root / "gaussian_renderer" / "__init__.py",
        root / "scene" / "gaussian_model.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Graphdeco checkout is incomplete: " + ", ".join(missing))
    revision = _git_revision(root)
    if revision != QUALIFIED_REVISION:
        raise RuntimeError(
            f"Phase 8.7D2 requires Graphdeco revision {QUALIFIED_REVISION}; found {revision or 'unknown'}"
        )
    license_text = (root / "LICENSE.md").read_text(encoding="utf-8", errors="replace").lower()
    if not all(term in license_text for term in ("research users", "non-commercial", "explicit consent")):
        raise RuntimeError("Graphdeco LICENSE.md does not match the qualified research-only policy markers")


def _materialize_colmap_dataset(dataset: dict, seed: dict[str, np.ndarray], target: Path) -> None:
    images_dir = target / "images"
    sparse_dir = target / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    cameras_lines = ["# EIA Phase 8.7D2 generated camera intrinsics"]
    images_lines = ["# EIA Phase 8.7D2 generated camera poses"]
    used_names: set[str] = set()

    for index, frame in enumerate(dataset["frames"], start=1):
        source = Path(str(frame["image_path"]))
        if not source.is_file():
            raise RuntimeError(f"input image missing: {source}")
        name = f"frame_{index:05d}.png"
        if name in used_names:
            raise RuntimeError("generated image-name collision")
        used_names.add(name)
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            expected_size = (int(frame["width"]), int(frame["height"]))
            if rgb.size != expected_size:
                raise RuntimeError(f"image size mismatch: {source} {rgb.size} != {expected_size}")
            rgb.save(images_dir / name)

        camera_id = index
        width = int(frame["width"])
        height = int(frame["height"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])
        # The qualified Graphdeco loader accepts PINHOLE intrinsics but builds
        # its projection from focal lengths/FOV only; it does not propagate an
        # off-centre principal point.  Reject such cameras instead of silently
        # changing their geometry.  D3 may add an explicit rectification path.
        if abs(cx - width / 2.0) > 1e-6 or abs(cy - height / 2.0) > 1e-6:
            raise RuntimeError(
                "Graphdeco D2 qualification requires a centered principal point; "
                f"received cx={cx}, cy={cy}, expected {width / 2.0}, {height / 2.0}"
            )
        cameras_lines.append(
            f"{camera_id} PINHOLE {width} {height} "
            f"{float(frame['fx']):.12g} {float(frame['fy']):.12g} "
            f"{cx:.12g} {cy:.12g}"
        )

        c2w = np.asarray(frame["T_reconstruction_from_camera"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise RuntimeError("invalid T_reconstruction_from_camera in neural-scene dataset")
        w2c = np.linalg.inv(c2w)
        q = rotation_matrix_to_quaternion_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]
        images_lines.append(
            f"{index} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {camera_id} {name}"
        )
        images_lines.append("")  # COLMAP's second per-image line: no sparse 2D observations required.

    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    # Graphdeco accepts an existing points3D.ply and therefore does not need a
    # points3D.bin/text track structure for this bounded qualification dataset.
    write_ascii_ply(
        sparse_dir / "points3D.ply",
        np.asarray(seed["positions_reconstruction"], dtype=np.float32),
        np.asarray(seed["colors_rgb"], dtype=np.uint8),
    )


def _run_logged(command: list[str], *, cwd: Path, log_path: Path, timeout_s: float) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"Graphdeco command failed with exit {completed.returncode}: {' '.join(command)}\n{tail}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = load_json(request_path)
    dataset, seed = validate_request(request, BACKEND_KIND)

    root_text = os.environ.get("EIA_GRAPHDECO_ROOT", "").strip()
    if not root_text:
        raise RuntimeError("EIA_GRAPHDECO_ROOT is required for the Graphdeco external worker")
    graphdeco_root = Path(root_text).resolve()
    _validate_graphdeco_root(graphdeco_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Graphdeco qualification requires a CUDA GPU")

    artifact_dir = Path(str(request["output_dir"])).resolve() / "graphdeco-reference"
    dataset_dir = artifact_dir / "qualification-dataset"
    model_dir = artifact_dir / "model"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _materialize_colmap_dataset(dataset, seed, dataset_dir)

    steps = PROFILE_STEPS[str(request["profile"])]
    started = time.monotonic()
    train_command = [
        sys.executable,
        str(graphdeco_root / "train.py"),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(steps),
        "--save_iterations",
        str(steps),
        "--test_iterations",
        "-1",
        "--sh_degree",
        "0",
        "--data_device",
        "cpu",
        "--resolution",
        "1",
        "--densify_until_iter",
        "0",
        "--disable_viewer",
        "--quiet",
    ]
    _run_logged(
        train_command,
        cwd=graphdeco_root,
        log_path=artifact_dir / "train.log",
        timeout_s=1800.0,
    )

    point_cloud = model_dir / "point_cloud" / f"iteration_{steps}" / "point_cloud.ply"
    if not point_cloud.is_file():
        candidates = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
        if not candidates:
            raise RuntimeError("Graphdeco training completed without a saved point_cloud.ply")
        point_cloud = candidates[-1]
    gaussian_count = ply_vertex_count(point_cloud)
    if gaussian_count < 1:
        raise RuntimeError("Graphdeco saved an empty Gaussian point cloud")

    render_command = [
        sys.executable,
        str(graphdeco_root / "render.py"),
        "-m",
        str(model_dir),
        "-s",
        str(dataset_dir),
        "--skip_test",
        "--quiet",
    ]
    _run_logged(
        render_command,
        cwd=graphdeco_root,
        log_path=artifact_dir / "render.log",
        timeout_s=900.0,
    )
    render_candidates = sorted(model_dir.glob("train/ours_*/renders/*.png"))
    if not render_candidates:
        raise RuntimeError("Graphdeco render.py completed without a training-set render")
    preview_path = artifact_dir / "preview.png"
    shutil.copy2(render_candidates[0], preview_path)
    if not preview_path.is_file() or preview_path.stat().st_size == 0:
        raise RuntimeError("Graphdeco preview render is missing or empty")

    duration_s = float(time.monotonic() - started)
    device_name = str(torch.cuda.get_device_name(0))
    revision = _git_revision(graphdeco_root)
    write_qualification_metrics(
        artifact_dir=artifact_dir,
        backend_kind=BACKEND_KIND,
        source_revision=revision,
        training_steps=steps,
        preview_path=preview_path,
        real_cuda_execution=True,
        optimization_completed=True,
        extra={
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "device": device_name,
            "device_capability": list(torch.cuda.get_device_capability(0)),
            "initial_seed_gaussians": int(len(seed["positions_reconstruction"])),
            "saved_gaussians": int(gaussian_count),
            "graphdeco_root": str(graphdeco_root),
            "research_evaluation_only": True,
            "commercial_use_permitted_by_backend_license": False,
        },
    )
    write_worker_result(
        result_path=result_path,
        request=request,
        dataset=dataset,
        backend_id="eia-graphdeco-reference-external",
        backend_version=revision,
        artifact_dir=artifact_dir,
        scene_path=point_cloud,
        gaussian_count=gaussian_count,
        training_steps=steps,
        training_duration_s=duration_s,
        device=f"cuda:0 {device_name}",
        peak_vram_mib=None,
    )
    print(
        f"EIA Graphdeco qualification: PASS steps={steps} gaussians={gaussian_count} "
        f"revision={revision[:12]} device={device_name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
