#!/usr/bin/env python3
"""Download a pinned local snapshot of the official Marigold Depth v1.1 model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "prs-eth/marigold-depth-v1-1"
DEFAULT_DESTINATION = PROJECT_ROOT / "checkpoints" / "marigold" / "depth-v1-1"


def download_snapshot(
    *,
    repository: str = DEFAULT_REPOSITORY,
    destination: Path = DEFAULT_DESTINATION,
    revision: str = "main",
    force: bool = False,
    api_factory: Callable[[], Any] | None = None,
    snapshot_downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    ready = destination / ".ready"
    model_index = destination / "model_index.json"
    if ready.is_file() and model_index.is_file() and not force:
        payload = json.loads(ready.read_text(encoding="utf-8"))
        print(f"Marigold snapshot already ready: {destination}")
        return payload
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is required. Run: python scripts/install_marigold.py"
        ) from exc
    api = api_factory() if api_factory is not None else HfApi()
    downloader = snapshot_downloader or snapshot_download
    info = api.model_info(repository, revision=revision)
    resolved_revision = str(getattr(info, "sha", "") or revision)
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    downloader(
        repo_id=repository,
        revision=resolved_revision,
        local_dir=str(destination),
    )
    if not model_index.is_file():
        raise RuntimeError(
            f"Downloaded snapshot is incomplete; missing {model_index}."
        )
    payload = {
        "repository": repository,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "destination": str(destination),
        "downloaded_at_unix": time.time(),
    }
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(ready)
    print(f"Marigold snapshot ready: {destination}")
    print(f"Pinned revision: {resolved_revision}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        download_snapshot(
            repository=args.repository,
            destination=args.destination,
            revision=args.revision,
            force=args.force,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
