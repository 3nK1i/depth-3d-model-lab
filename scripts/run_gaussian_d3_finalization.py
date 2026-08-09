#!/usr/bin/env python3
"""Run fresh real evidence and close Phase 8.7D3 Gaussian validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gaussian_d3_finalization import finalize_gaussian_d3, write_finalization_outputs
from scripts.run_gaussian_heldout_quality_smoke import _default_env_python

DEFAULT_OUTPUT = Path("/tmp/phase8_7d3_finalization")


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"closure command failed with exit code {completed.returncode}: {command[1]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("eia-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("eia-gaussian-graphdeco"))
    parser.add_argument("--metrics-python", type=Path, default=_default_env_python("eia-gaussian-metrics"))
    parser.add_argument("--graphdeco-root", type=Path, default=Path.home() / "Desktop" / "eia-gaussian-backends" / "gaussian-splatting")
    parser.add_argument("--hardware-profile", choices=("rtx_2060_6gb", "rtx_5090_32gb", "custom"), default="rtx_2060_6gb")
    parser.add_argument("--quality-training-steps", type=int, default=64)
    parser.add_argument("--performance-training-steps", type=int, default=64)
    parser.add_argument("--render-repeats", type=int, default=3)
    parser.add_argument("--recovery-training-steps", type=int, default=16)
    parser.add_argument("--cancel-after-s", type=float, default=0.75)
    parser.add_argument("--fault-timeout-s", type=float, default=1.0)
    parser.add_argument("--backend-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    gsplat = str(args.gsplat_python.expanduser().resolve())
    graphdeco_python = str(args.graphdeco_python.expanduser().resolve())
    metrics = str(args.metrics_python.expanduser().resolve())
    graphdeco_root = str(args.graphdeco_root.expanduser().resolve())

    try:
        _run([
            sys.executable, str(ROOT / "scripts" / "run_gaussian_backend_qualification.py"),
            "--backend", "both", "--gsplat-python", gsplat, "--graphdeco-python", graphdeco_python,
            "--graphdeco-root", graphdeco_root, "--profile", "efficient",
            "--timeout-s", str(args.backend_timeout_s), "--output-dir", str(output / "d2_qualification"),
        ])
        _run([
            sys.executable, str(ROOT / "scripts" / "run_gaussian_heldout_quality_smoke.py"),
            "--backend", "both", "--gsplat-python", gsplat, "--graphdeco-python", graphdeco_python,
            "--metrics-python", metrics, "--graphdeco-root", graphdeco_root,
            "--training-steps", str(args.quality_training_steps), "--output-dir", str(output / "step2_quality"),
        ])
        _run([
            sys.executable, str(ROOT / "scripts" / "run_gaussian_performance_smoke.py"),
            "--backend", "both", "--gsplat-python", gsplat, "--graphdeco-python", graphdeco_python,
            "--metrics-python", metrics, "--graphdeco-root", graphdeco_root,
            "--training-steps", str(args.performance_training_steps), "--render-repeats", str(args.render_repeats),
            "--hardware-profile", args.hardware_profile, "--output-dir", str(output / "step3_performance"),
        ])
        _run([
            sys.executable, str(ROOT / "scripts" / "run_gaussian_backend_robustness_smoke.py"),
            "--backend", "both", "--gsplat-python", gsplat, "--graphdeco-python", graphdeco_python,
            "--graphdeco-root", graphdeco_root, "--recovery-training-steps", str(args.recovery_training_steps),
            "--cancel-after-s", str(args.cancel_after_s), "--timeout-s", str(args.fault_timeout_s),
            "--output-dir", str(output / "step4_robustness"),
        ])
    except RuntimeError as exc:
        print(f"Phase 8.7D3 Finalization: FAIL ({exc})")
        return 1

    result = finalize_gaussian_d3(
        d2_summary_path=output / "d2_qualification" / "summary.json",
        step2_summary_path=output / "step2_quality" / "benchmark" / "step2_summary.json",
        step3_summary_path=output / "step3_performance" / "benchmark" / "step3_summary.json",
        step4_summary_path=output / "step4_robustness" / "step4_summary.json",
        expected_hardware_profile=args.hardware_profile,
        repository_root=ROOT,
    )
    json_path, md_path = write_finalization_outputs(result, output)
    if not result.passed:
        print("Phase 8.7D3 Finalization: FAIL")
        for name, gate in result.summary["gates"].items():
            print(f"{name}={'PASS' if gate['pass'] else 'FAIL'} reasons={','.join(gate['reasons'])}")
        print(f"summary={json_path}")
        return 1

    print("Phase 8.7D3 Finalization: PASS")
    print("D3_CLOSED=True next_phase=8.7E1")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print("gaussian_scene_role=appearance_only")
    print("engine_level_recovery_deferred_to_phase8_7g=True")
    print(f"summary={json_path}")
    print(f"report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
