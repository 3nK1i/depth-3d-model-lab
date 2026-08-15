#!/usr/bin/env python3
"""Read-only real probe of the isolated Phase 9.4 gsplat runtime.

This command starts only short external capability-probe subprocesses.  It does
not optimize/train Gaussians, author an artifact, or render an interactive view.
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
    GaussianInteractiveRuntimeSpec,
    default_phase9_gaussian_runtime_spec,
    probe_phase9_gaussian_interactive_runtime,
)


def main() -> int:
    defaults = default_phase9_gaussian_runtime_spec()
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=defaults.python_executable)
    parser.add_argument("--checkout", default=defaults.checkout_root)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    spec = GaussianInteractiveRuntimeSpec(
        python_executable=str(Path(args.python).expanduser().resolve()),
        checkout_root=str(Path(args.checkout).expanduser().resolve()),
        probe_timeout_s=args.timeout_seconds,
    )
    report = probe_phase9_gaussian_interactive_runtime(spec)
    payload = report.as_dict()
    payload.update(
        {
            "status": "PASS" if report.ready else "NOT_READY",
            "real_capability_probe_performed": True,
            "optimization_started": False,
            "rendering_started": False,
            "gaussian_artifact_created": False,
        }
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if report.ready:
        print("Phase 9.4A isolated gsplat runtime probe: READY")
        return 0
    print(f"Phase 9.4A isolated gsplat runtime probe: {report.state.value.upper()}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
