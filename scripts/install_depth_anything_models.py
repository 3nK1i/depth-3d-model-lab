#!/usr/bin/env python3
"""Prepare official Depth Anything V2 and Video Depth Anything source trees safely."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class SourceSpec:
    family: str
    repository: str
    source_dir: Path
    ref: str
    marker: str
    required_files: tuple[str, ...]
    requirement_file: Path


SOURCES: dict[str, SourceSpec] = {
    "image": SourceSpec(
        family="image",
        repository="https://github.com/DepthAnything/Depth-Anything-V2.git",
        source_dir=ROOT / "third_party" / "Depth-Anything-V2",
        ref="main",
        marker=".eia_depth_anything_v2_ready.json",
        required_files=("depth_anything_v2/dpt.py", "depth_anything_v2/dinov2.py", "LICENSE"),
        requirement_file=ROOT / "requirements" / "models-depth-anything-v2.txt",
    ),
    "video": SourceSpec(
        family="video",
        repository="https://github.com/DepthAnything/Video-Depth-Anything.git",
        source_dir=ROOT / "third_party" / "Video-Depth-Anything",
        ref="main",
        marker=".eia_video_depth_anything_ready.json",
        required_files=(
            "video_depth_anything/video_depth_stream.py",
            "video_depth_anything/dpt_temporal.py",
            "utils/util.py",
            "LICENSE",
        ),
        requirement_file=ROOT / "requirements" / "models-video-depth-anything.txt",
    ),
}

_IMPORT_PROBES: dict[str, str] = {
    "image": (
        "from depth_anything_v2.dpt import DepthAnythingV2; "
        "print('Depth Anything V2 source import: OK')"
    ),
    "video": (
        "from video_depth_anything.video_depth_stream import VideoDepthAnything; "
        "print('Video Depth Anything source import: OK')"
    ),
}



def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _torch_version() -> str:
    return subprocess.check_output(
        [sys.executable, "-c", "import torch; print(torch.__version__)"], text=True
    ).strip()


def prepare_source(
    spec: SourceSpec,
    *,
    source_dir: Path | None = None,
    repository: str | None = None,
    ref: str | None = None,
    force: bool = False,
) -> Path:
    """Clone/update one official source and record the exact resolved commit."""

    destination = (source_dir or spec.source_dir).expanduser().resolve()
    repository = repository or spec.repository
    ref = ref or spec.ref
    if destination.exists() and force:
        shutil.rmtree(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True)
        _run(["git", "init", "--quiet"], cwd=destination)
        _run(["git", "remote", "add", "origin", repository], cwd=destination)
        _run(["git", "fetch", "--depth", "1", "origin", ref], cwd=destination)
        _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=destination)
    if not (destination / ".git").is_dir():
        raise RuntimeError(f"Source directory is not a Git checkout: {destination}")

    missing = [item for item in spec.required_files if not (destination / item).is_file()]
    if missing:
        raise RuntimeError(
            f"Official {spec.family} source layout is incomplete; missing: "
            + ", ".join(missing)
        )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=destination, text=True
    ).strip()
    payload = {
        "family": spec.family,
        "repository": repository,
        "requested_ref": ref,
        "commit": commit,
        "integration": "local official source; weights managed separately",
    }
    (destination / spec.marker).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def install_runtime(specs: list[SourceSpec]) -> None:
    """Install only compatible extras and prove that PyTorch was not replaced."""

    before = _torch_version()
    seen: set[Path] = set()
    for spec in specs:
        requirement = spec.requirement_file.resolve()
        if requirement in seen:
            continue
        seen.add(requirement)
        _run([sys.executable, "-m", "pip", "install", "-r", str(requirement)])
    after = _torch_version()
    if after != before:
        raise RuntimeError(
            "PyTorch changed during Depth Anything installation "
            f"({before} -> {after}). Restore the validated CUDA PyTorch build."
        )
    _run(
        [
            sys.executable,
            "-c",
            "import torch, torchvision, cv2, numpy; "
            "import huggingface_hub, einops; "
            "print('Depth Anything dependencies: OK')",
        ]
    )


def verify_source_import(spec: SourceSpec, source_dir: Path) -> None:
    """Import the official model class in an isolated interpreter."""

    source = str(source_dir.expanduser().resolve())
    probe = (
        "import sys; "
        f"sys.path.insert(0, {source!r}); "
        + _IMPORT_PROBES[spec.family]
    )
    _run([sys.executable, "-c", probe], cwd=ROOT)


def selected_specs(family: str) -> list[SourceSpec]:
    return list(SOURCES.values()) if family == "all" else [SOURCES[family]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("image", "video", "all"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-dependencies", action="store_true")
    args = parser.parse_args(argv)
    specs = selected_specs(args.family)
    try:
        prepared = [(spec, prepare_source(spec, force=args.force)) for spec in specs]
        if not args.skip_dependencies:
            install_runtime(specs)
        for spec, destination in prepared:
            verify_source_import(spec, destination)
    except Exception as exc:  # noqa: BLE001
        print(f"Depth Anything source installation failed: {exc}", file=sys.stderr)
        return 1
    for _spec, destination in prepared:
        print(f"Prepared official source: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
