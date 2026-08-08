#!/usr/bin/env python3
"""Probe a separate interpreter for a real Gaussian backend without importing it here."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import GaussianBackendKind, probe_gaussian_environment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True, help="Absolute path to the isolated Python executable")
    parser.add_argument(
        "--backend",
        choices=("gsplat", "graphdeco-reference"),
        default="gsplat",
        help="Real backend to qualify",
    )
    parser.add_argument(
        "--graphdeco-root",
        default="",
        help="Absolute Graphdeco gaussian-splatting checkout; required for graphdeco-reference",
    )
    args = parser.parse_args()

    backend_kind = (
        GaussianBackendKind.GSPLAT_EXTERNAL
        if args.backend == "gsplat"
        else GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
    )
    graphdeco_root = Path(args.graphdeco_root) if args.graphdeco_root else None
    probe = probe_gaussian_environment(
        Path(args.python),
        backend_kind=backend_kind,
        graphdeco_root=graphdeco_root,
    )
    print(json.dumps(probe.as_dict(), indent=2, sort_keys=True))
    ready = probe.backend_ready and probe.interpreter_is_isolated
    print(
        f"Phase 8.7D {args.backend} environment probe: "
        + ("READY" if ready else "NOT READY")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
