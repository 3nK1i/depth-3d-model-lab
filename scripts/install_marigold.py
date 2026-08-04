#!/usr/bin/env python3
"""Install Marigold dependencies without replacing the validated PyTorch build."""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT_ROOT / "requirements" / "models-marigold.txt"


def _torch_version() -> str:
    try:
        import torch
    except ImportError:
        return ""
    return str(torch.__version__)


def install(*, dry_run: bool = False) -> None:
    if not REQUIREMENTS.is_file():
        raise FileNotFoundError(f"Missing requirements file: {REQUIREMENTS}")
    before = _torch_version()
    if not before:
        raise RuntimeError(
            "PyTorch is not installed. Install the validated project environment first."
        )
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade-strategy",
        "only-if-needed",
        "-r",
        str(REQUIREMENTS),
    ]
    print("Command:", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)
    importlib.invalidate_caches()
    after = _torch_version()
    if after != before:
        raise RuntimeError(
            "The Marigold installation changed PyTorch from "
            f"{before} to {after}. Restore the validated environment before continuing."
        )
    from diffusers import MarigoldDepthPipeline  # noqa: F401

    print(f"Marigold runtime: OK (PyTorch preserved at {after})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        install(dry_run=args.dry_run)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
