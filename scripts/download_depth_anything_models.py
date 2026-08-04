#!/usr/bin/env python3
"""Download registry-supported Depth Anything V2 and metric VDA checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CheckpointSpec:
    repo_id: str
    filename: str
    revision: str = "main"


CHECKPOINTS: dict[str, dict[str, CheckpointSpec]] = {
    "image": {
        "small": CheckpointSpec("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
        "base": CheckpointSpec("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth"),
        "large": CheckpointSpec("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"),
    },
    "video": {
        "small": CheckpointSpec("depth-anything/Metric-Video-Depth-Anything-Small", "metric_video_depth_anything_vits.pth"),
        "base": CheckpointSpec("depth-anything/Metric-Video-Depth-Anything-Base", "metric_video_depth_anything_vitb.pth"),
        "large": CheckpointSpec("depth-anything/Metric-Video-Depth-Anything-Large", "metric_video_depth_anything_vitl.pth"),
    },
}
OUTPUTS = {
    "image": ROOT / "checkpoints" / "depth-anything-v2",
    "video": ROOT / "checkpoints" / "video-depth-anything",
}
_MIN_CHECKPOINT_BYTES = 1024 * 1024


def auto_variant(torch_module=None) -> str:
    """Select Small below 10 GiB, Base below 20 GiB, otherwise Large."""

    try:
        torch = torch_module
        if torch is None:
            import torch  # type: ignore
        if not torch.cuda.is_available():
            return "small"
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception:  # noqa: BLE001
        return "small"
    if total_gib >= 20.0:
        return "large"
    if total_gib >= 10.0:
        return "base"
    return "small"


def _complete(path: Path, minimum_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def download_checkpoint(
    family: str,
    variant: str,
    output_root: Path,
    *,
    force: bool = False,
    hf_download_fn: Callable[..., str] | None = None,
    minimum_bytes: int = _MIN_CHECKPOINT_BYTES,
) -> Path:
    try:
        spec = CHECKPOINTS[family][variant]
    except KeyError:
        raise ValueError(f"Unknown Depth Anything target: {family}/{variant}") from None
    destination_root = output_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / spec.filename
    if _complete(destination, minimum_bytes) and not force:
        print(f"Checkpoint already exists: {destination}")
        return destination
    if hf_download_fn is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is missing. Run: python scripts/install_depth_anything_models.py"
            ) from exc
        hf_download_fn = hf_hub_download
    print(f"Downloading {spec.repo_id}/{spec.filename} to: {destination_root}")
    downloaded = Path(
        hf_download_fn(
            repo_id=spec.repo_id,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(destination_root),
            force_download=force,
        )
    )
    if downloaded.resolve() != destination.resolve() and downloaded.is_file():
        destination.write_bytes(downloaded.read_bytes())
    if not _complete(destination, minimum_bytes):
        raise RuntimeError(f"Downloaded checkpoint is incomplete: {destination}")
    return destination


def write_ready_marker(family: str, output_root: Path, variants: list[str]) -> Path:
    marker = output_root.expanduser().resolve() / ".ready"
    marker.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(variants))
    marker.write_text(
        json.dumps(
            {
                "family": family,
                "variants": unique,
                "checkpoints": [
                    {
                        "repo_id": CHECKPOINTS[family][variant].repo_id,
                        "filename": CHECKPOINTS[family][variant].filename,
                        "revision": CHECKPOINTS[family][variant].revision,
                    }
                    for variant in unique
                ],
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
    parser.add_argument("--family", choices=("image", "video", "all"), default="all")
    parser.add_argument("--variant", choices=("auto", "small", "base", "large", "all"), default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    families = list(CHECKPOINTS) if args.family == "all" else [args.family]
    selected = args.variant
    if selected == "auto":
        selected = auto_variant()
        print(f"Automatic GPU selection: {selected}")
    variants = list(CHECKPOINTS["image"]) if selected == "all" else [selected]
    try:
        for family in families:
            downloaded: list[str] = []
            for variant in variants:
                download_checkpoint(family, variant, OUTPUTS[family], force=args.force)
                downloaded.append(variant)
            marker = write_ready_marker(family, OUTPUTS[family], downloaded)
            print(f"{family.title()} readiness marker: {marker}")
    except Exception as exc:  # noqa: BLE001
        print(f"Depth Anything checkpoint download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
