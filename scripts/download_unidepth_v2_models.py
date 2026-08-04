#!/usr/bin/env python3
"""Download registry-supported UniDepthV2 model snapshots from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "checkpoints" / "unidepth-v2"
VARIANTS = {
    "small": ("vits14", "lpiccinelli/unidepth-v2-vits14", "038c238f06c87b6c2f5b3749fd51fbf442b1f218"),
    "base": ("vitb14", "lpiccinelli/unidepth-v2-vitb14", "6830c15e415da7f94babbe0e83b46e1a04bc28ea"),
    "large": ("vitl14", "lpiccinelli/unidepth-v2-vitl14", "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"),
}


def auto_variant(torch_module=None) -> str:
    try:
        torch = torch_module
        if torch is None:
            import torch  # type: ignore
        if not torch.cuda.is_available():
            return "small"
        total_gib = float(torch.cuda.get_device_properties(0).total_memory) / 1024**3
    except Exception:  # noqa: BLE001 - conservative downloader fallback
        return "small"
    if total_gib >= 20.0:
        return "large"
    if total_gib >= 8.0:
        return "base"
    return "small"


def _snapshot_complete(path: Path) -> bool:
    return (path / "config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def download_variant(
    variant: str,
    output_root: Path,
    *,
    force: bool = False,
    snapshot_download_fn: Callable[..., str] | None = None,
) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown UniDepthV2 variant: {variant!r}")
    folder, repository, revision = VARIANTS[variant]
    destination = output_root.expanduser().resolve() / folder
    if _snapshot_complete(destination) and not force:
        print(f"UniDepthV2 {variant} snapshot already exists: {destination}")
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    if snapshot_download_fn is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is missing. Run: python scripts/install_unidepth_v2.py"
            ) from exc
        snapshot_download_fn = snapshot_download
    print(f"Downloading {repository} to: {destination}")
    snapshot_download_fn(
        repo_id=repository,
        revision=revision,
        local_dir=str(destination),
        allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
        force_download=force,
    )
    if not _snapshot_complete(destination):
        raise RuntimeError(f"Downloaded UniDepthV2 snapshot is incomplete: {destination}")
    return destination


def write_ready_marker(output_root: Path, downloaded: list[str]) -> Path:
    marker = output_root.expanduser().resolve() / ".ready"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"downloaded_variants": sorted(set(downloaded))}, indent=2) + "\n",
        encoding="utf-8",
    )
    return marker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("auto", "small", "base", "large", "all"),
        default="auto",
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
        print(f"UniDepthV2 model download failed: {exc}", file=sys.stderr)
        return 1
    print(f"UniDepthV2 readiness marker: {marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
