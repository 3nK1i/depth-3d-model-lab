#!/usr/bin/env python3
"""Download the official Apple Depth Pro checkpoint atomically."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "checkpoints" / "depth_pro.pt"
OFFICIAL_URL = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
_MINIMUM_EXPECTED_BYTES = 100 * 1024 * 1024


def download_checkpoint(output: Path, *, force: bool = False) -> Path:
    output = output.expanduser().resolve()
    if output.is_file() and output.stat().st_size > 0 and not force:
        print(f"Depth Pro checkpoint already exists: {output}")
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading Depth Pro checkpoint to: {output}")
    try:
        with urllib.request.urlopen(OFFICIAL_URL, timeout=60) as response:
            with partial.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
        size = partial.stat().st_size
        if size < _MINIMUM_EXPECTED_BYTES:
            raise RuntimeError(
                f"Downloaded file is unexpectedly small ({size:,} bytes)."
            )
        partial.replace(output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    print(f"Downloaded {output.stat().st_size:,} bytes.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Checkpoint destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing file.")
    args = parser.parse_args()
    try:
        download_checkpoint(args.output, force=args.force)
    except Exception as exc:  # noqa: BLE001 - CLI should provide a concise failure
        print(f"Depth Pro checkpoint download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
