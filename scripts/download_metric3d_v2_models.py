#!/usr/bin/env python3
"""Download registry-supported Metric3D v2 checkpoints from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "checkpoints" / "metric3d-v2"
REPOSITORY = "JUGGHM/Metric3D"
REVISION = "main"
VARIANTS = {
    "small": "metric_depth_vit_small_800k.pth",
    "large": "metric_depth_vit_large_800k.pth",
}
_MIN_CHECKPOINT_BYTES = 1024 * 1024


def auto_variant(torch_module=None) -> str:
    """Select ViT-S below 20 GiB and ViT-L on RTX 5090-class GPUs."""

    try:
        torch = torch_module
        if torch is None:
            import torch  # type: ignore
        if not torch.cuda.is_available():
            return "small"
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception:  # noqa: BLE001 - conservative fallback
        return "small"
    return "large" if total_gib >= 20.0 else "small"


def _checkpoint_complete(path: Path, *, minimum_bytes: int = _MIN_CHECKPOINT_BYTES) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def download_variant(
    variant: str,
    output_root: Path,
    *,
    force: bool = False,
    hf_download_fn: Callable[..., str] | None = None,
    minimum_bytes: int = _MIN_CHECKPOINT_BYTES,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Metric3D v2 variant: {variant!r}")
    filename = VARIANTS[variant]
    destination_root = output_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / filename
    if _checkpoint_complete(destination, minimum_bytes=minimum_bytes) and not force:
        print(f"Metric3D v2 {variant} checkpoint already exists: {destination}")
        return destination

    if hf_download_fn is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is missing. Run: python scripts/install_metric3d_v2.py"
            ) from exc
        hf_download_fn = hf_hub_download

    print(f"Downloading {REPOSITORY}/{filename} to: {destination_root}")
    downloaded = Path(
        hf_download_fn(
            repo_id=REPOSITORY,
            filename=filename,
            revision=REVISION,
            local_dir=str(destination_root),
            force_download=force,
        )
    )
    if downloaded.resolve() != destination.resolve() and downloaded.is_file():
        destination.write_bytes(downloaded.read_bytes())
    if not _checkpoint_complete(destination, minimum_bytes=minimum_bytes):
        raise RuntimeError(f"Downloaded Metric3D v2 checkpoint is incomplete: {destination}")
    return destination


def write_ready_marker(output_root: Path, downloaded: list[str]) -> Path:
    marker = output_root.expanduser().resolve() / ".ready"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "downloaded_variants": sorted(set(downloaded)),
                "files": [VARIANTS[item] for item in sorted(set(downloaded))],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("auto", "small", "large", "all"), default="auto"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    selected = args.variant
    if selected == "auto":
        selected = auto_variant()
        print(f"Automatic GPU selection: {selected}")
    targets = list(VARIANTS) if selected == "all" else [selected]
    downloaded: list[str] = []
    try:
        for target in targets:
            download_variant(target, args.output, force=args.force)
            downloaded.append(target)
        marker = write_ready_marker(args.output, downloaded)
    except Exception as exc:  # noqa: BLE001
        print(f"Metric3D v2 checkpoint download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Metric3D v2 readiness marker: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
