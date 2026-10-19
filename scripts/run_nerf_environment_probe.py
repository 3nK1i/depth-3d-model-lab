#!/usr/bin/env python3
"""Print the Phase 8.7E2 isolated Nerfstudio environment probe."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import probe_nerf_environment


def _default_python() -> Path:
    return Path.home() / "anaconda3/envs/depth3d-nerf-nerfstudio/bin/python"


@contextmanager
def _isolated_runtime_paths(python_executable: Path):
    """Expose only the selected conda runtime paths to the external probe."""

    python = python_executable.expanduser().resolve()
    prefix = python.parent.parent
    if python.parent.name != "bin":
        raise RuntimeError(f"cannot infer isolated environment prefix from {python}")
    keys = ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CUDA_HOME", "CUDACXX", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONNOUSERSITE")
    previous = {key: os.environ.get(key) for key in keys}

    def prepend(key: str, values: list[Path]) -> None:
        parts = [str(value) for value in values if value.is_dir()]
        old = previous.get(key) or ""
        if old:
            parts.append(old)
        os.environ[key] = os.pathsep.join(parts)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)
    lib_dirs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", lib_dirs)
    prepend("LIBRARY_PATH", lib_dirs)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=_default_python())
    args = parser.parse_args()
    python = args.python.expanduser().resolve()
    with _isolated_runtime_paths(python):
        probe = probe_nerf_environment(python, timeout_s=60.0)
    print(json.dumps(probe.as_dict(), indent=2, sort_keys=True))
    return 0 if probe.backend_ready and probe.interpreter_is_isolated else 1


if __name__ == "__main__":
    raise SystemExit(main())
