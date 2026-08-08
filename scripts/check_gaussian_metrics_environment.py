#!/usr/bin/env python3
"""Probe the isolated Phase 8.7D3 held-out image-quality environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REQUIRED_PYTHON = "3.10"
REQUIRED_TORCH = "2.7.1"
REQUIRED_TORCHVISION = "0.22.1"
REQUIRED_LPIPS = "0.1.4"


def _base(value: str) -> str:
    return value.split("+")[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args()
    python = args.python.expanduser().resolve()
    if not python.is_file():
        print(json.dumps({"ready": False, "error": f"missing interpreter: {python}"}, indent=2))
        return 1

    code = r'''
import importlib.metadata as md
import json
import sys
payload = {
    "python_executable": sys.executable,
    "python_version": sys.version.split()[0],
}
try:
    import torch
    import torchvision
    import lpips
    payload.update({
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "lpips_package_version": md.version("lpips"),
    })
    model = lpips.LPIPS(net="alex", version="0.1", verbose=False)
    model.eval()
    payload["lpips_alex_v0_1_ready"] = True
    payload["error"] = ""
except Exception as exc:
    payload["lpips_alex_v0_1_ready"] = False
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''
    completed = subprocess.run(
        [str(python), "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180.0,
        check=False,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if completed.returncode != 0:
        print(json.dumps({"ready": False, "stderr": completed.stderr[-4000:]}, indent=2))
        return 1
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        print(json.dumps({"ready": False, "stdout": completed.stdout, "stderr": completed.stderr}, indent=2))
        return 1

    payload["interpreter_is_isolated"] = python != Path(sys.executable).resolve()
    ready = (
        str(payload.get("python_version", "")).startswith(REQUIRED_PYTHON + ".")
        and _base(str(payload.get("torch_version", ""))) == REQUIRED_TORCH
        and _base(str(payload.get("torchvision_version", ""))) == REQUIRED_TORCHVISION
        and str(payload.get("lpips_package_version", "")) == REQUIRED_LPIPS
        and bool(payload.get("lpips_alex_v0_1_ready"))
        and bool(payload["interpreter_is_isolated"])
    )
    payload["ready"] = ready
    print(json.dumps(payload, indent=2, sort_keys=True))
    print("Phase 8.7D3 metrics environment: " + ("READY" if ready else "NOT READY"))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
