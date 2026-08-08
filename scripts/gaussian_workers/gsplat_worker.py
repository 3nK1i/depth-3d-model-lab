#!/usr/bin/env python3
"""Real isolated gsplat qualification worker for Phase 8.7D2.

This script intentionally lives outside the production core.  It exercises a
real CUDA gsplat forward/backward optimization and writes a renderable artifact.
Full-quality scene training and benchmarking belong to Phase 8.7D3.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from gsplat.rendering import rasterization

from common import load_json, validate_request, write_qualification_metrics, write_worker_result

BACKEND_KIND = "gsplat_external"
QUALIFIED_VERSION = "1.5.3"
PROFILE_STEPS = {"efficient": 24, "balanced": 48, "high_quality": 96}


def _base_version(value: str) -> str:
    return str(value).split("+")[0]


def _load_target(frame: dict, device: torch.device) -> torch.Tensor:
    image_path = Path(str(frame["image_path"]))
    if not image_path.is_file():
        raise ValueError(f"input image missing: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        expected = (int(frame["width"]), int(frame["height"]))
        if rgb.size != expected:
            raise ValueError(f"image size mismatch for {image_path}: {rgb.size} != {expected}")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).to(device=device, dtype=torch.float32)


def _camera_tensors(frame: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = np.asarray(frame["T_reconstruction_from_camera"], dtype=np.float64)
    if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError("invalid T_reconstruction_from_camera")
    view = np.linalg.inv(c2w).astype(np.float32)
    k = np.array(
        [
            [float(frame["fx"]), 0.0, float(frame["cx"])],
            [0.0, float(frame["fy"]), float(frame["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (
        torch.from_numpy(view).to(device=device)[None],
        torch.from_numpy(k).to(device=device)[None],
    )


def _render(
    *,
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    frame: dict,
    device: torch.device,
) -> torch.Tensor:
    viewmats, ks = _camera_tensors(frame, device)
    rendered, _alphas, _meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=ks,
        width=int(frame["width"]),
        height=int(frame["height"]),
        sh_degree=None,
        packed=True,
        render_mode="RGB",
    )
    if rendered.ndim != 4 or rendered.shape[0] != 1 or rendered.shape[-1] != 3:
        raise RuntimeError(f"unexpected gsplat render shape: {tuple(rendered.shape)}")
    return rendered[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = load_json(request_path)
    dataset, seed = validate_request(request, BACKEND_KIND)

    installed_version = metadata.version("gsplat")
    if _base_version(installed_version) != QUALIFIED_VERSION:
        raise RuntimeError(
            f"Phase 8.7D2 requires gsplat {QUALIFIED_VERSION}; found {installed_version}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("gsplat qualification requires a CUDA GPU")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    positions_np = np.asarray(seed["positions_reconstruction"], dtype=np.float32)
    colors_np = np.asarray(seed["colors_rgb"], dtype=np.float32) / 255.0
    log_scales_np = np.asarray(seed["log_scales_reconstruction"], dtype=np.float32)
    opacities_np = np.asarray(seed["opacities"], dtype=np.float32)

    means = torch.from_numpy(positions_np).to(device=device)
    quats = torch.zeros((len(positions_np), 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # wxyz, matching gsplat's documented convention.
    scales = torch.exp(torch.from_numpy(log_scales_np).to(device=device)).clamp_min(1e-5)

    initial_colors = torch.from_numpy(colors_np).to(device=device).clamp(1e-4, 1.0 - 1e-4)
    initial_opacities = torch.from_numpy(opacities_np).to(device=device).clamp(1e-4, 1.0 - 1e-4)
    color_logits = torch.nn.Parameter(torch.logit(initial_colors))
    opacity_logits = torch.nn.Parameter(torch.logit(initial_opacities))
    optimizer = torch.optim.Adam([color_logits, opacity_logits], lr=0.03)

    frames = list(dataset["frames"])
    targets = [_load_target(frame, device) for frame in frames]
    steps = PROFILE_STEPS[str(request["profile"])]
    initial_loss: float | None = None
    final_loss: float | None = None

    started = time.monotonic()
    for step in range(steps):
        frame_index = step % len(frames)
        frame = frames[frame_index]
        target = targets[frame_index]
        colors = torch.sigmoid(color_logits)
        opacities = torch.sigmoid(opacity_logits)
        rendered = _render(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            frame=frame,
            device=device,
        )
        loss = torch.nn.functional.l1_loss(rendered, target)
        if not torch.isfinite(loss):
            raise RuntimeError("gsplat qualification produced a non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().cpu())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    torch.cuda.synchronize(device)
    duration_s = float(time.monotonic() - started)

    final_colors = torch.sigmoid(color_logits).detach()
    final_opacities = torch.sigmoid(opacity_logits).detach()
    preview = _render(
        means=means,
        quats=quats,
        scales=scales,
        opacities=final_opacities,
        colors=final_colors,
        frame=frames[0],
        device=device,
    ).detach()
    torch.cuda.synchronize(device)
    if not torch.isfinite(preview).all():
        raise RuntimeError("gsplat qualification preview contains non-finite values")

    artifact_dir = Path(str(request["output_dir"])).resolve() / "gsplat"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preview_path = artifact_dir / "preview.png"
    preview_u8 = (
        preview.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    )
    Image.fromarray(preview_u8, mode="RGB").save(preview_path)

    scene_path = artifact_dir / "scene.pt"
    torch.save(
        {
            "schema_version": 1,
            "backend": BACKEND_KIND,
            "backend_version": QUALIFIED_VERSION,
            "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
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

    peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    write_qualification_metrics(
        artifact_dir=artifact_dir,
        backend_kind=BACKEND_KIND,
        source_revision=QUALIFIED_VERSION,
        training_steps=steps,
        preview_path=preview_path,
        real_cuda_execution=True,
        optimization_completed=True,
        extra={
            "initial_l1": initial_loss,
            "final_l1": final_loss,
            "finite_loss": bool(
                initial_loss is not None
                and final_loss is not None
                and math.isfinite(initial_loss)
                and math.isfinite(final_loss)
            ),
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "device": str(torch.cuda.get_device_name(device)),
            "device_capability": list(torch.cuda.get_device_capability(device)),
            "peak_vram_mib": peak_vram_mib,
        },
    )
    write_worker_result(
        result_path=result_path,
        request=request,
        dataset=dataset,
        backend_id="eia-gsplat-external",
        backend_version=QUALIFIED_VERSION,
        artifact_dir=artifact_dir,
        scene_path=scene_path,
        gaussian_count=len(positions_np),
        training_steps=steps,
        training_duration_s=duration_s,
        device=f"cuda:0 {torch.cuda.get_device_name(device)}",
        peak_vram_mib=peak_vram_mib,
    )
    print(
        f"EIA gsplat qualification: PASS steps={steps} gaussians={len(positions_np)} "
        f"device={torch.cuda.get_device_name(device)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
