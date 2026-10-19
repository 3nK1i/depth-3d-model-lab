#!/usr/bin/env python3
"""Install a pinned official Metric3D v2 source checkout safely."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party" / "Metric3D"
OFFICIAL_REPOSITORY = "https://github.com/YvanYin/Metric3D.git"
OFFICIAL_COMMIT = "eb5b6fac0dc155e4e52f576e304fbf11655ff339"
REQUIREMENTS = ROOT / "requirements" / "models-metric3d-v2.txt"
READY_MARKER = ".depth3d_metric3d_v2_ready.json"
_REQUIRED_SOURCE_FILES = (
    "hubconf.py",
    "LICENSE",
    "mono/configs/HourglassDecoder/vit.raft5.small.py",
    "mono/configs/HourglassDecoder/vit.raft5.large.py",
)
_REQUIRED_IMPORTS = ("torch", "torchvision", "cv2", "mmengine", "mmcv", "timm")


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _torch_version() -> str:
    output = subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        text=True,
    )
    return output.strip()


def prepare_source(
    source_dir: Path,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    ref: str = OFFICIAL_COMMIT,
    force: bool = False,
) -> Path:
    """Clone the exact official revision and validate its inference layout."""

    source_dir = source_dir.expanduser().resolve()
    if source_dir.exists() and force:
        shutil.rmtree(source_dir)
    if not source_dir.exists():
        source_dir.parent.mkdir(parents=True, exist_ok=True)
        source_dir.mkdir(parents=True)
        _run(["git", "init", "--quiet"], cwd=source_dir)
        _run(["git", "remote", "add", "origin", repository], cwd=source_dir)
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=source_dir)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source_dir)
    if not (source_dir / ".git").is_dir():
        raise RuntimeError(f"Metric3D source directory is not a Git clone: {source_dir}")

    missing = [relative for relative in _REQUIRED_SOURCE_FILES if not (source_dir / relative).is_file()]
    if missing:
        raise RuntimeError(
            "The official Metric3D source layout is incomplete; missing: "
            + ", ".join(missing)
        )

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
    ).strip()
    marker = {
        "repository": repository,
        "ref": ref,
        "commit": commit,
        "models": ["metric3d_vit_small", "metric3d_vit_large"],
        "integration": "local torch hub source; pretrained downloads disabled",
    }
    (source_dir / READY_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source_dir


def install_runtime(*, install_dependencies: bool = True) -> None:
    """Install optional packages without replacing the project's PyTorch build."""

    before = _torch_version()
    if install_dependencies:
        _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    after = _torch_version()
    if after != before:
        raise RuntimeError(
            "PyTorch changed during Metric3D installation "
            f"({before} -> {after}). Restore the project's CUDA PyTorch build."
        )
    import_check = "; ".join(f"import {name}" for name in _REQUIRED_IMPORTS)
    _run(
        [
            sys.executable,
            "-c",
            import_check + "; print('Metric3D v2 dependencies: OK')",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--repository", default=OFFICIAL_REPOSITORY)
    parser.add_argument("--ref", default=OFFICIAL_COMMIT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = prepare_source(
            args.source_dir,
            repository=args.repository,
            ref=args.ref,
            force=args.force,
        )
        install_runtime(install_dependencies=not args.skip_dependencies)
    except Exception as exc:  # noqa: BLE001
        print(f"Metric3D v2 installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Metric3D v2 source prepared at: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
