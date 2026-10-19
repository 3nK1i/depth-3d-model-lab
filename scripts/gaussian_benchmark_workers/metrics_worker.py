#!/usr/bin/env python3
"""Backend-neutral PSNR/SSIM/LPIPS evaluator for Phase 8.7D3 Step 2.

This worker intentionally runs outside the production ``depth`` environment.
It consumes only RGB reference/render pairs and therefore cannot influence
canonical reconstruction geometry or metrology state.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

LPIPS_PACKAGE_VERSION = "0.1.4"
LPIPS_METRIC_VERSION = "0.1"
LPIPS_NETWORK = "alex"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_rgb(path: Path, *, width: int, height: int) -> torch.Tensor:
    if not path.is_file():
        raise ValueError(f"image missing: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            raise ValueError(f"image size mismatch: {path} {rgb.size} != {(width, height)}")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def _psnr(reference: torch.Tensor, rendered: torch.Tensor) -> float:
    mse = float(torch.mean((reference - rendered) ** 2).item())
    return float(-10.0 * math.log10(max(mse, 1e-12)))


def _gaussian_window(size: int, sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=dtype, device=device) - float(size // 2)
    weights = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    weights /= weights.sum()
    kernel = weights[:, None] @ weights[None, :]
    return kernel.unsqueeze(0).unsqueeze(0)


def _ssim(reference: torch.Tensor, rendered: torch.Tensor) -> float:
    if reference.shape != rendered.shape or reference.ndim != 4 or reference.shape[1] != 3:
        raise ValueError("SSIM expects matching NCHW RGB tensors")
    height, width = int(reference.shape[-2]), int(reference.shape[-1])
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError("SSIM requires image dimensions >= 3")
    base = _gaussian_window(window_size, 1.5, dtype=reference.dtype, device=reference.device)
    window = base.expand(3, 1, window_size, window_size).contiguous()
    mu1 = F.conv2d(reference, window, groups=3)
    mu2 = F.conv2d(rendered, window, groups=3)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(reference * reference, window, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(rendered * rendered, window, groups=3) - mu2_sq
    sigma12 = F.conv2d(reference * rendered, window, groups=3) - mu12
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    value = ((2.0 * mu12 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    result = float(value.mean().item())
    if not math.isfinite(result):
        raise RuntimeError("SSIM produced a non-finite result")
    return max(0.0, min(1.0, result))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-manifest", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()

    installed_lpips = metadata.version("lpips")
    if installed_lpips != LPIPS_PACKAGE_VERSION:
        raise RuntimeError(
            f"Phase 8.7D3 requires lpips {LPIPS_PACKAGE_VERSION}; found {installed_lpips}"
        )
    import lpips  # Imported only in the isolated metrics worker.

    manifest = _load_json(Path(args.render_manifest).resolve())
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("unsupported held-out render manifest schema")
    renders = manifest.get("renders")
    validation_ids = [str(value) for value in manifest.get("validation_frame_ids", ())]
    if not isinstance(renders, list) or not renders:
        raise ValueError("held-out render manifest has no renders")
    if set(str(value.get("frame_id", "")) for value in renders) != set(validation_ids):
        raise ValueError("held-out render manifest does not match validation split")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA metric evaluation requested but CUDA is unavailable")
        device = torch.device("cuda:0")
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = lpips.LPIPS(net=LPIPS_NETWORK, version=LPIPS_METRIC_VERSION, verbose=False).to(device)
    model.eval()

    per_view: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in renders:
            frame_id = str(item["frame_id"])
            width = int(item["width"])
            height = int(item["height"])
            if min(width, height) < 64:
                raise ValueError(
                    f"LPIPS AlexNet evaluation requires held-out images >=64 px on both axes; "
                    f"{frame_id} is {width}x{height}"
                )
            reference = _load_rgb(
                Path(str(item["reference_image_path"])).resolve(),
                width=width,
                height=height,
            )
            rendered = _load_rgb(
                Path(str(item["rendered_image_path"])).resolve(),
                width=width,
                height=height,
            )
            psnr = _psnr(reference, rendered)
            ssim = _ssim(reference, rendered)
            ref_lpips = reference.to(device=device) * 2.0 - 1.0
            render_lpips = rendered.to(device=device) * 2.0 - 1.0
            distance = float(model(ref_lpips, render_lpips).reshape(-1)[0].item())
            if not math.isfinite(distance) or distance < 0.0:
                raise RuntimeError(f"LPIPS produced an invalid value for {frame_id}: {distance}")
            per_view.append(
                {
                    "frame_id": frame_id,
                    "psnr_db": psnr,
                    "ssim": ssim,
                    "lpips": distance,
                }
            )

    count = float(len(per_view))
    aggregate = {
        "measured": True,
        "psnr_db": sum(float(value["psnr_db"]) for value in per_view) / count,
        "ssim": sum(float(value["ssim"]) for value in per_view) / count,
        "lpips": sum(float(value["lpips"]) for value in per_view) / count,
    }
    result = {
        "schema_version": 1,
        "backend_kind": manifest["backend_kind"],
        "benchmark_fingerprint_sha256": manifest["benchmark_fingerprint_sha256"],
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "validation_frame_ids": validation_ids,
        "views": per_view,
        "aggregate": aggregate,
        "evaluator": {
            "name": "depth3d-heldout-quality",
            "psnr_definition": "rgb_mse_unit_range_mse_floor_1e-12",
            "ssim_definition": "gaussian_11_sigma_1.5_rgb_unit_range",
            "lpips_network": LPIPS_NETWORK,
            "lpips_version": LPIPS_METRIC_VERSION,
            "lpips_package_version": installed_lpips,
            "device": str(device),
            "torch_version": str(torch.__version__),
        },
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    target = Path(args.result).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Held-out quality: PASS views={len(per_view)} "
        f"psnr={aggregate['psnr_db']:.4f} ssim={aggregate['ssim']:.6f} lpips={aggregate['lpips']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
