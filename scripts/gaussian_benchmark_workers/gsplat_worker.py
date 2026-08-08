#!/usr/bin/env python3
"""Isolated gsplat held-out rendering worker for Phase 8.7D3 Step 2."""

from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
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
from gsplat_worker import _base_version, _load_target, _render

BACKEND_KIND = "gsplat_external"
QUALIFIED_VERSION = "1.5.3"


def _save_render(path: Path, image: torch.Tensor) -> None:
    if image.ndim != 3 or image.shape[-1] != 3 or not torch.isfinite(image).all():
        raise RuntimeError("gsplat held-out render is invalid or non-finite")
    array = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


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

    installed_version = metadata.version("gsplat")
    if _base_version(installed_version) != QUALIFIED_VERSION:
        raise RuntimeError(
            f"Phase 8.7D3 requires gsplat {QUALIFIED_VERSION}; found {installed_version}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("gsplat held-out benchmark requires a CUDA GPU")

    steps = int(request.get("training_steps", 0))
    if steps < 1 or steps > 100_000:
        raise RuntimeError("training_steps must lie in [1, 100000]")

    frames_by_id = frame_map(dataset)
    training_frames = [frames_by_id[value] for value in request["training_frame_ids"]]
    validation_frames = [frames_by_id[value] for value in request["validation_frame_ids"]]

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    positions_np = np.asarray(seed["positions_reconstruction"], dtype=np.float32)
    colors_np = np.asarray(seed["colors_rgb"], dtype=np.float32) / 255.0
    log_scales_np = np.asarray(seed["log_scales_reconstruction"], dtype=np.float32)
    opacities_np = np.asarray(seed["opacities"], dtype=np.float32)

    means = torch.from_numpy(positions_np).to(device=device)
    quats = torch.zeros((len(positions_np), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    scales = torch.exp(torch.from_numpy(log_scales_np).to(device=device)).clamp_min(1e-5)
    initial_colors = torch.from_numpy(colors_np).to(device=device).clamp(1e-4, 1.0 - 1e-4)
    initial_opacities = torch.from_numpy(opacities_np).to(device=device).clamp(1e-4, 1.0 - 1e-4)
    color_logits = torch.nn.Parameter(torch.logit(initial_colors))
    opacity_logits = torch.nn.Parameter(torch.logit(initial_opacities))
    optimizer = torch.optim.Adam([color_logits, opacity_logits], lr=0.03)
    training_targets = [_load_target(frame, device) for frame in training_frames]

    started = time.monotonic()
    for step in range(steps):
        index = step % len(training_frames)
        rendered = _render(
            means=means,
            quats=quats,
            scales=scales,
            opacities=torch.sigmoid(opacity_logits),
            colors=torch.sigmoid(color_logits),
            frame=training_frames[index],
            device=device,
        )
        loss = torch.nn.functional.l1_loss(rendered, training_targets[index])
        if not torch.isfinite(loss):
            raise RuntimeError("gsplat held-out training produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device)
    training_duration_s = float(time.monotonic() - started)
    final_colors = torch.sigmoid(color_logits).detach()
    final_opacities = torch.sigmoid(opacity_logits).detach()

    artifact_dir = Path(str(request["output_dir"])).resolve() / "gsplat"
    render_dir = artifact_dir / "held_out_renders"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    renders: list[dict[str, object]] = []
    with torch.no_grad():
        for frame in validation_frames:
            rendered = _render(
                means=means,
                quats=quats,
                scales=scales,
                opacities=final_opacities,
                colors=final_colors,
                frame=frame,
                device=device,
            ).detach()
            render_path = render_dir / f"{frame['frame_id']}.png"
            _save_render(render_path, rendered)
            renders.append(
                {
                    "frame_id": str(frame["frame_id"]),
                    "reference_image_path": str(Path(str(frame["image_path"])).resolve()),
                    "rendered_image_path": str(render_path.resolve()),
                    "width": int(frame["width"]),
                    "height": int(frame["height"]),
                }
            )
    torch.cuda.synchronize(device)

    scene_path = artifact_dir / "scene.pt"
    torch.save(
        {
            "schema_version": 1,
            "backend": BACKEND_KIND,
            "backend_version": QUALIFIED_VERSION,
            "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
            "benchmark_fingerprint_sha256": request["benchmark_fingerprint_sha256"],
            "training_frame_ids": list(request["training_frame_ids"]),
            "validation_frame_ids": list(request["validation_frame_ids"]),
            "means_reconstruction": means.detach().cpu(),
            "quats_wxyz": quats.detach().cpu(),
            "scales_reconstruction": scales.detach().cpu(),
            "opacities": final_opacities.cpu(),
            "colors_rgb": final_colors.cpu(),
            "pose_convention": dataset["pose_convention"],
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        scene_path,
    )
    render_manifest = write_render_manifest(
        path=artifact_dir / "held_out_render_manifest.json",
        request=request,
        renders=renders,
    )
    write_backend_result(
        path=result_path,
        request=request,
        backend_version=QUALIFIED_VERSION,
        scene_path=scene_path,
        render_manifest_path=render_manifest,
        gaussian_count=len(positions_np),
        training_steps=steps,
        training_duration_s=training_duration_s,
        device=f"cuda:0 {torch.cuda.get_device_name(device)}",
    )
    print(
        f"EIA gsplat held-out benchmark: PASS training={len(training_frames)} "
        f"validation={len(validation_frames)} steps={steps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
