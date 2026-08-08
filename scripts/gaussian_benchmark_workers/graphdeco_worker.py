#!/usr/bin/env python3
"""Isolated Graphdeco held-out rendering worker for Phase 8.7D3 Step 2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
D2_WORKERS = HERE.parent / "gaussian_workers"

import importlib.util
_BENCH_COMMON_SPEC = importlib.util.spec_from_file_location(
    "eia_d3_gaussian_benchmark_worker_common", HERE / "common.py"
)
if _BENCH_COMMON_SPEC is None or _BENCH_COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load D3 Gaussian benchmark worker helpers")
_BENCH_COMMON = importlib.util.module_from_spec(_BENCH_COMMON_SPEC)
_BENCH_COMMON_SPEC.loader.exec_module(_BENCH_COMMON)
frame_map = _BENCH_COMMON.frame_map
load_benchmark_request = _BENCH_COMMON.load_benchmark_request
write_backend_result = _BENCH_COMMON.write_backend_result
write_render_manifest = _BENCH_COMMON.write_render_manifest

if str(D2_WORKERS) not in sys.path:
    sys.path.insert(0, str(D2_WORKERS))
from graphdeco_worker import QUALIFIED_REVISION, _git_revision, _run_logged, _validate_graphdeco_root

# Load D2 common helpers explicitly by file path for quaternion/PLY utilities.
_D2_COMMON_SPEC = importlib.util.spec_from_file_location(
    "eia_d2_gaussian_worker_common",
    D2_WORKERS / "common.py",
)
if _D2_COMMON_SPEC is None or _D2_COMMON_SPEC.loader is None:
    raise RuntimeError("cannot load D2 Gaussian worker common helpers")
_D2_COMMON = importlib.util.module_from_spec(_D2_COMMON_SPEC)
_D2_COMMON_SPEC.loader.exec_module(_D2_COMMON)
rotation_matrix_to_quaternion_wxyz = _D2_COMMON.rotation_matrix_to_quaternion_wxyz
write_ascii_ply = _D2_COMMON.write_ascii_ply
ply_vertex_count = _D2_COMMON.ply_vertex_count

BACKEND_KIND = "graphdeco_reference_external"
PROFILE_DEFAULT_STEPS = {"efficient": 128, "balanced": 256, "high_quality": 512}


def _require_explicit_test_split_support(root: Path) -> None:
    dataset_reader = root / "scene" / "dataset_readers.py"
    scene_init = root / "scene" / "__init__.py"
    if not dataset_reader.is_file() or not scene_init.is_file():
        raise RuntimeError("Graphdeco checkout is missing split-support source files")
    reader_text = dataset_reader.read_text(encoding="utf-8", errors="replace")
    scene_text = scene_init.read_text(encoding="utf-8", errors="replace")
    reader_supports_explicit_test_file = (
        "test.txt" in reader_text
        and "llffhold=8" in reader_text
        and "if llffhold" in reader_text
    )
    checkout_forces_explicit_eval_split = (
        'sceneLoadTypeCallbacks["Colmap"]' in scene_text
        and "llffhold=0" in scene_text
    )
    if not reader_supports_explicit_test_file or not checkout_forces_explicit_eval_split:
        raise RuntimeError(
            "Pinned Graphdeco checkout is missing the audited explicit test.txt held-out split "
            "compatibility patch; held-out validation would risk training-view leakage"
        )


def _materialize_benchmark_dataset(
    dataset: dict,
    seed: dict[str, np.ndarray],
    target: Path,
    *,
    validation_frame_ids: tuple[str, ...],
) -> tuple[dict[str, str], list[str]]:
    images_dir = target / "images"
    sparse_dir = target / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    validation_set = set(validation_frame_ids)
    cameras_lines = ["# EIA Phase 8.7D3 generated camera intrinsics"]
    images_lines = ["# EIA Phase 8.7D3 generated camera poses"]
    generated_by_frame: dict[str, str] = {}

    for index, frame in enumerate(dataset["frames"], start=1):
        frame_id = str(frame["frame_id"])
        source = Path(str(frame["image_path"])).resolve()
        if not source.is_file():
            raise RuntimeError(f"input image missing: {source}")
        name = f"frame_{index:05d}.png"
        generated_by_frame[frame_id] = name
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            expected = (int(frame["width"]), int(frame["height"]))
            if rgb.size != expected:
                raise RuntimeError(f"image size mismatch: {source} {rgb.size} != {expected}")
            rgb.save(images_dir / name)

        width = int(frame["width"])
        height = int(frame["height"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])
        if abs(cx - width / 2.0) > 1e-6 or abs(cy - height / 2.0) > 1e-6:
            raise RuntimeError(
                "Graphdeco D3 held-out benchmark requires a centered principal point; "
                f"received cx={cx}, cy={cy}, expected {width / 2.0}, {height / 2.0}"
            )
        cameras_lines.append(
            f"{index} PINHOLE {width} {height} "
            f"{float(frame['fx']):.12g} {float(frame['fy']):.12g} {cx:.12g} {cy:.12g}"
        )
        c2w = np.asarray(frame["T_reconstruction_from_camera"], dtype=np.float64)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise RuntimeError("invalid T_reconstruction_from_camera in neural-scene dataset")
        w2c = np.linalg.inv(c2w)
        q = rotation_matrix_to_quaternion_wxyz(w2c[:3, :3])
        t = w2c[:3, 3]
        images_lines.append(
            f"{index} {q[0]:.17g} {q[1]:.17g} {q[2]:.17g} {q[3]:.17g} "
            f"{t[0]:.17g} {t[1]:.17g} {t[2]:.17g} {index} {name}"
        )
        images_lines.append("")

    validation_names = sorted(generated_by_frame[value] for value in validation_frame_ids)
    if len(validation_names) != len(validation_set):
        raise RuntimeError("Graphdeco validation split contains duplicate frame IDs")
    (sparse_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")
    (sparse_dir / "test.txt").write_text("\n".join(validation_names) + "\n", encoding="utf-8")
    write_ascii_ply(
        sparse_dir / "points3D.ply",
        np.asarray(seed["positions_reconstruction"], dtype=np.float32),
        np.asarray(seed["colors_rgb"], dtype=np.uint8),
    )
    return generated_by_frame, validation_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request, dataset, seed = load_benchmark_request(
        request_path,
        expected_backend_kind=BACKEND_KIND,
    )

    root_text = os.environ.get("EIA_GRAPHDECO_ROOT", "").strip()
    if not root_text:
        raise RuntimeError("EIA_GRAPHDECO_ROOT is required for the Graphdeco benchmark worker")
    graphdeco_root = Path(root_text).resolve()
    _validate_graphdeco_root(graphdeco_root)
    _require_explicit_test_split_support(graphdeco_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Graphdeco held-out benchmark requires a CUDA GPU")

    steps = int(request.get("training_steps", PROFILE_DEFAULT_STEPS[str(request["profile"])]))
    if steps < 1 or steps > 100_000:
        raise RuntimeError("training_steps must lie in [1, 100000]")

    artifact_dir = Path(str(request["output_dir"])).resolve() / "graphdeco-reference"
    dataset_dir = artifact_dir / "benchmark-dataset"
    model_dir = artifact_dir / "model"
    render_dir = artifact_dir / "held_out_renders"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    generated_by_frame, validation_names = _materialize_benchmark_dataset(
        dataset,
        seed,
        dataset_dir,
        validation_frame_ids=tuple(str(value) for value in request["validation_frame_ids"]),
    )

    started = time.monotonic()
    train_command = [
        sys.executable,
        str(graphdeco_root / "train.py"),
        "-s", str(dataset_dir),
        "-m", str(model_dir),
        "--eval",
        "--iterations", str(steps),
        "--save_iterations", str(steps),
        "--test_iterations", "-1",
        "--sh_degree", "0",
        "--data_device", "cpu",
        "--resolution", "1",
        "--densify_until_iter", "0",
        "--disable_viewer",
        "--quiet",
    ]
    _run_logged(train_command, cwd=graphdeco_root, log_path=artifact_dir / "train.log", timeout_s=3600.0)
    torch.cuda.synchronize()
    training_duration_s = float(time.monotonic() - started)

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
        "-m", str(model_dir),
        "-s", str(dataset_dir),
        "--eval",
        "--skip_train",
        "--quiet",
    ]
    _run_logged(render_command, cwd=graphdeco_root, log_path=artifact_dir / "render.log", timeout_s=1800.0)
    candidates = sorted(model_dir.glob("test/ours_*/renders/*.png"))
    if len(candidates) != len(validation_names):
        raise RuntimeError(
            f"Graphdeco held-out render count mismatch: {len(candidates)} != {len(validation_names)}"
        )

    frames_by_id = frame_map(dataset)
    frame_by_generated = {name: frame_id for frame_id, name in generated_by_frame.items()}
    renders: list[dict[str, object]] = []
    render_dir.mkdir(parents=True, exist_ok=True)
    for generated_name, source_render in zip(validation_names, candidates, strict=True):
        frame_id = frame_by_generated[generated_name]
        frame = frames_by_id[frame_id]
        destination = render_dir / f"{frame_id}.png"
        shutil.copy2(source_render, destination)
        with Image.open(destination) as image:
            if image.convert("RGB").size != (int(frame["width"]), int(frame["height"])):
                raise RuntimeError(f"Graphdeco held-out render size mismatch for {frame_id}")
        renders.append(
            {
                "frame_id": frame_id,
                "reference_image_path": str(Path(str(frame["image_path"])).resolve()),
                "rendered_image_path": str(destination.resolve()),
                "width": int(frame["width"]),
                "height": int(frame["height"]),
            }
        )

    render_manifest = write_render_manifest(
        path=artifact_dir / "held_out_render_manifest.json",
        request=request,
        renders=renders,
    )
    revision = _git_revision(graphdeco_root)
    write_backend_result(
        path=result_path,
        request=request,
        backend_version=revision,
        scene_path=point_cloud,
        render_manifest_path=render_manifest,
        gaussian_count=gaussian_count,
        training_steps=steps,
        training_duration_s=training_duration_s,
        device=f"cuda:0 {torch.cuda.get_device_name(0)}",
    )
    print(
        f"EIA Graphdeco held-out benchmark: PASS training={len(request['training_frame_ids'])} "
        f"validation={len(request['validation_frame_ids'])} steps={steps} revision={revision[:12]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
