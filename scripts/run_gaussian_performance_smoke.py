#!/usr/bin/env python3
"""Real CUDA Gaussian performance smoke for Phase 8.7D3 Step 3."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_gaussian_heldout_quality_smoke import (
    _default_env_python,
    prepare_heldout_smoke_inputs,
)

DEFAULT_OUTPUT = Path("/tmp/phase8_7d3_gaussian_performance_smoke")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gsplat", "graphdeco", "both"), default="both")
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("eia-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("eia-gaussian-graphdeco"))
    parser.add_argument("--metrics-python", type=Path, default=_default_env_python("eia-gaussian-metrics"))
    parser.add_argument(
        "--graphdeco-root", type=Path,
        default=Path.home() / "Desktop" / "eia-gaussian-backends" / "gaussian-splatting",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-steps", type=int, default=64)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument(
        "--hardware-profile",
        choices=("rtx_2060_6gb", "rtx_5090_32gb", "custom"),
        default="rtx_2060_6gb",
    )
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    manifest, seed_path = prepare_heldout_smoke_inputs(output)
    benchmark_output = output / "benchmark"
    command = [
        sys.executable, str(ROOT / "scripts" / "run_gaussian_real_scene_benchmark.py"),
        "--dataset-manifest", str(manifest),
        "--seed", str(seed_path),
        "--validation-frame-id", "heldout-smoke-4",
        "--benchmark-id", "phase8-7d3-performance-smoke",
        "--source-label", "deterministic-performance-smoke",
        "--backend", args.backend,
        "--profile", "efficient",
        "--training-steps", str(args.training_steps),
        "--measure-performance",
        "--render-repeats", str(args.render_repeats),
        "--hardware-profile", args.hardware_profile,
        "--hardware-label", args.hardware_profile,
        "--gsplat-python", str(args.gsplat_python.expanduser().resolve()),
        "--graphdeco-python", str(args.graphdeco_python.expanduser().resolve()),
        "--metrics-python", str(args.metrics_python.expanduser().resolve()),
        "--graphdeco-root", str(args.graphdeco_root.expanduser().resolve()),
        "--metrics-device", "cpu",
        "--output-dir", str(benchmark_output),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode
    print("Phase 8.7D3 Gaussian performance smoke: PASS")
    print(f"summary={benchmark_output / 'step3_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
