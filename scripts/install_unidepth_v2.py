#!/usr/bin/env python3
"""Install the official UniDepth repository with an isolated V2 import path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "third_party" / "UniDepth"
OFFICIAL_REPOSITORY = "https://github.com/lpiccinelli-eth/UniDepth.git"
OFFICIAL_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
REQUIREMENTS = ROOT / "requirements" / "models-unidepth-v2.txt"
COMPAT_MARKER = ".depth3d_unidepth_v2_compat.json"

_MODELS_INIT = '''"""V2-only compatibility import for the official UniDepth source."""\n\nfrom .unidepthv2 import UniDepthV2\n\n__all__ = ["UniDepthV2"]\n'''

_UNIDEPTH_V2_INIT = '''"""V2-only package import."""\n\nfrom .unidepthv2 import UniDepthV2\n\n__all__ = ["UniDepthV2"]\n'''

_LAYERS_INIT = '''"""V2 inference layers without the legacy V1 Nystrom/xFormers import."""\n\nfrom .activation import GEGLU, SwiGLU\nfrom .attention import AttentionBlock, AttentionDecoderBlock, AttentionLayer\nfrom .convnext import CvnxtBlock\nfrom .mlp import MLP\nfrom .positional_encoding import PositionEmbeddingSine\nfrom .upsample import (\n    ConvUpsample,\n    ConvUpsampleShuffle,\n    ConvUpsampleShuffleResidual,\n    ResUpsampleBil,\n)\n\n__all__ = [\n    "SwiGLU",\n    "GEGLU",\n    "CvnxtBlock",\n    "AttentionBlock",\n    "PositionEmbeddingSine",\n    "ConvUpsample",\n    "MLP",\n    "ConvUpsampleShuffle",\n    "AttentionDecoderBlock",\n    "AttentionLayer",\n    "ConvUpsampleShuffleResidual",\n    "ResUpsampleBil",\n]\n'''


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def prepare_source(
    source_dir: Path,
    *,
    repository: str = OFFICIAL_REPOSITORY,
    ref: str = OFFICIAL_COMMIT,
    force: bool = False,
) -> Path:
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
        raise RuntimeError(f"UniDepth source directory is not a Git clone: {source_dir}")

    models_init = source_dir / "unidepth" / "models" / "__init__.py"
    unidepth_v2_init = source_dir / "unidepth" / "models" / "unidepthv2" / "__init__.py"
    layers_init = source_dir / "unidepth" / "layers" / "__init__.py"
    if not models_init.is_file() or not unidepth_v2_init.is_file() or not layers_init.is_file():
        raise RuntimeError("The official UniDepth source layout is not recognized.")

    models_init.write_text(_MODELS_INIT, encoding="utf-8")
    unidepth_v2_init.write_text(_UNIDEPTH_V2_INIT, encoding="utf-8")
    layers_init.write_text(_LAYERS_INIT, encoding="utf-8")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_dir, text=True
    ).strip()
    marker = {
        "repository": repository,
        "ref": ref,
        "commit": commit,
        "purpose": "V2-only imports; model and inference implementation unchanged",
    }
    (source_dir / COMPAT_MARKER).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return source_dir


def install_runtime(
    source_dir: Path,
    *,
    install_dependencies: bool = True,
) -> None:
    if install_dependencies:
        _run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    _run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(source_dir)]
    )
    _run(
        [
            sys.executable,
            "-c",
            "from unidepth.models import UniDepthV2; print('UniDepthV2 import: OK')",
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
        install_runtime(source, install_dependencies=not args.skip_dependencies)
    except Exception as exc:  # noqa: BLE001
        print(f"UniDepthV2 installation failed: {exc}", file=sys.stderr)
        return 1
    print(f"UniDepthV2 runtime installed from: {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
