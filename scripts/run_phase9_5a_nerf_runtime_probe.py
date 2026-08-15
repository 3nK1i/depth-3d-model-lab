#!/usr/bin/env python3
"""Read-only real probe of the isolated Phase 9.5 Nerfstudio/Nerfacto runtime.

This command starts only short external capability-probe subprocesses. It does
not train NeRF, author a checkpoint, or render an image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NerfInteractiveRuntimeSpec,
    default_phase9_nerf_runtime_spec,
    probe_phase9_nerf_interactive_runtime,
)


def main() -> int:
    defaults = default_phase9_nerf_runtime_spec()
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=defaults.python_executable)
    parser.add_argument("--nerfstudio-root", default=defaults.nerfstudio_root)
    parser.add_argument("--tiny-cuda-nn-root", default=defaults.tiny_cuda_nn_root)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    spec = NerfInteractiveRuntimeSpec(
        python_executable=str(Path(args.python).expanduser().resolve()),
        nerfstudio_root=str(Path(args.nerfstudio_root).expanduser().resolve()),
        tiny_cuda_nn_root=str(Path(args.tiny_cuda_nn_root).expanduser().resolve()),
        probe_timeout_s=args.timeout_seconds,
    )
    report = probe_phase9_nerf_interactive_runtime(spec)
    payload = report.as_dict()
    payload.update(
        {
            "status": "PASS" if report.ready else "NOT_READY",
            "real_capability_probe_performed": True,
            "optimization_started": False,
            "training_started": False,
            "rendering_started": False,
            "nerf_artifact_created": False,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.ready:
        print("Phase 9.5A isolated Nerfstudio/Nerfacto runtime probe: READY")
        return 0
    print(f"Phase 9.5A isolated Nerfstudio/Nerfacto runtime probe: {report.state.value.upper()}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
