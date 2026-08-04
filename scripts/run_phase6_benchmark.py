#!/usr/bin/env python3
"""Run the registry-driven Phase 6 cross-model benchmark.

Each model runs in its own Python process so optional runtimes, CUDA allocators,
and temporal caches are fully isolated between measurements.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark.assets import ensure_benchmark_assets  # noqa: E402
from core.benchmark.reports import (  # noqa: E402
    write_benchmark_csv,
    write_benchmark_summary,
    write_capability_matrix,
    write_json,
)
from core.benchmark.runner import (  # noqa: E402
    BenchmarkOptions,
    benchmark_model,
    hardware_snapshot,
    production_benchmark_descriptors,
)
from core.hardware.gpu_info import detect_hardware  # noqa: E402
from core.hardware.profiles import ComputeProfile  # noqa: E402
from core.models.registry import get_model_descriptor, iter_model_descriptors  # noqa: E402
from core.version import RELEASE_NAME, __version__  # noqa: E402

_DEFAULT_OUTPUT = ROOT / "outputs" / "phase6"
_DEFAULT_ASSETS = _DEFAULT_OUTPUT / "benchmark_assets"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated registry model IDs, or 'all' for production models.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(item.value for item in ComputeProfile),
        default=ComputeProfile.AUTO.value,
    )
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--sequence-frames", type=int)
    parser.add_argument("--point-budget", type=int, default=100_000)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--include-mock", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=_DEFAULT_ASSETS)

    # Internal isolated-worker mode. These options are intentionally public
    # enough for deterministic tests, but normal users call the parent mode.
    parser.add_argument("--worker-model", default="")
    parser.add_argument("--worker-output", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.worker_model:
        return _worker_main(args)
    return _parent_main(args)


def _worker_main(args: argparse.Namespace) -> int:
    if args.worker_output is None:
        raise SystemExit("--worker-output is required with --worker-model")
    options = BenchmarkOptions(
        profile=ComputeProfile(args.profile),
        iterations=args.iterations,
        sequence_frames=args.sequence_frames,
        point_budget=max(1, int(args.point_budget)),
        smoke_test=bool(args.smoke_test),
    )
    result = benchmark_model(
        args.worker_model,
        assets_root=args.assets_dir,
        options=options,
    )
    write_json(args.worker_output, result)
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    assets_dir = args.assets_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = ensure_benchmark_assets(assets_dir)
    descriptors = _selected_descriptors(args.models, include_mock=args.include_mock)
    hardware = detect_hardware()

    print(f"EIA TECH Depth & 3D Model Lab v{__version__} — {RELEASE_NAME}")
    print(f"Hardware: {hardware.short_label}")
    print(f"Models: {', '.join(item.model_id for item in descriptors)}")
    print(f"Assets: {manifest.manifest}\n")

    results: list[dict[str, object]] = []
    for descriptor in descriptors:
        policy = descriptor.benchmark
        timeout = max(30.0, float(policy.timeout_seconds))
        with tempfile.TemporaryDirectory(prefix="depth3d-phase6-benchmark-") as directory:
            worker_output = Path(directory) / f"{descriptor.model_id}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-model",
                descriptor.model_id,
                "--worker-output",
                str(worker_output),
                "--profile",
                args.profile,
                "--point-budget",
                str(max(1, int(args.point_budget))),
                "--assets-dir",
                str(assets_dir),
            ]
            if args.iterations is not None:
                command.extend(["--iterations", str(max(1, args.iterations))])
            if args.sequence_frames is not None:
                command.extend(["--sequence-frames", str(max(1, args.sequence_frames))])
            if args.smoke_test:
                command.append("--smoke-test")

            print(f"[{descriptor.display_name}] running…", flush=True)
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                )
                elapsed = time.perf_counter() - started
                if worker_output.is_file():
                    result = json.loads(worker_output.read_text(encoding="utf-8"))
                else:
                    result = _subprocess_failure(
                        descriptor.model_id,
                        descriptor.display_name,
                        f"Worker exited with code {completed.returncode} without a report.",
                        completed.stderr or completed.stdout,
                    )
                result["isolated_process_seconds"] = elapsed
                result["worker_returncode"] = completed.returncode
                if completed.returncode != 0 and result.get("status") == "passed":
                    result["status"] = "failed"
                    result["error"] = "Benchmark worker returned a non-zero exit code."
                if completed.stderr.strip():
                    result["worker_stderr_tail"] = completed.stderr.strip()[-4000:]
            except subprocess.TimeoutExpired as exc:
                elapsed = time.perf_counter() - started
                result = _subprocess_failure(
                    descriptor.model_id,
                    descriptor.display_name,
                    f"Benchmark timed out after {timeout:.0f} seconds.",
                    _text(exc.stderr) or _text(exc.stdout),
                )
                result["isolated_process_seconds"] = elapsed
                result["worker_returncode"] = 124
            results.append(result)
            print(f"[{descriptor.display_name}] {str(result.get('status', 'unknown')).upper()}")

    passed = all(item.get("status") == "passed" for item in results)
    payload: dict[str, object] = {
        "schema_version": 1,
        "application": {
            "name": "EIA TECH - Depth & 3D Model Lab",
            "version": __version__,
            "release_name": RELEASE_NAME,
        },
        "generated_at_unix": time.time(),
        "passed": passed,
        "scope": {
            "performance": True,
            "capability_matrix": True,
            "ground_truth_accuracy": False,
            "note": (
                "The deterministic synthetic workload measures runtime and output contracts; "
                "it does not claim ground-truth depth accuracy."
            ),
        },
        "hardware": hardware_snapshot(hardware),
        "configuration": {
            "profile": args.profile,
            "iterations_override": args.iterations,
            "sequence_frames_override": args.sequence_frames,
            "point_budget": max(1, int(args.point_budget)),
            "smoke_test": bool(args.smoke_test),
            "isolated_process_per_model": True,
        },
        "asset_manifest": str(manifest.manifest),
        "asset_checksums": manifest.checksums,
        "results": results,
    }

    json_path = write_json(output_dir / "benchmark_results.json", payload)
    hardware_path = write_json(
        output_dir / "hardware_summary.json",
        {
            "schema_version": 1,
            "application_version": __version__,
            "generated_at_unix": payload["generated_at_unix"],
            "hardware": payload["hardware"],
            "configuration": payload["configuration"],
        },
    )
    csv_path = write_benchmark_csv(output_dir / "benchmark_results.csv", results)
    matrix_path = write_capability_matrix(
        output_dir / "model_capability_matrix.md",
        production_benchmark_descriptors(),
    )
    summary_path = write_benchmark_summary(output_dir / "benchmark_summary.md", payload)

    print("\nReports:")
    for path in (json_path, hardware_path, csv_path, matrix_path, summary_path):
        print(f"  {path}")
    print(f"\nBenchmark: {'PASSED' if passed else 'COMPLETED WITH NON-PASSING MODELS'}")
    return 1 if args.require_all and not passed else 0


def _selected_descriptors(value: str, *, include_mock: bool) -> tuple[object, ...]:
    normalized = str(value).strip()
    if normalized.lower() == "all":
        selected = list(production_benchmark_descriptors())
        if include_mock:
            selected.extend(
                descriptor
                for descriptor in iter_model_descriptors(selectable_only=True, enabled_only=True)
                if descriptor.development_only
            )
        return tuple(selected)

    ids = [item.strip() for item in normalized.split(",") if item.strip()]
    if not ids:
        raise SystemExit("No benchmark model IDs were selected.")
    descriptors = []
    for model_id in ids:
        descriptor = get_model_descriptor(model_id)
        if descriptor.development_only and not include_mock:
            raise SystemExit(
                f"{model_id!r} is development-only; pass --include-mock explicitly."
            )
        descriptors.append(descriptor)
    return tuple(descriptors)


def _subprocess_failure(
    model_id: str,
    display_name: str,
    error: str,
    details: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "application_version": __version__,
        "model_id": model_id,
        "display_name": display_name,
        "status": "failed",
        "error_type": "BenchmarkWorkerError",
        "error": error,
        "worker_output_tail": details.strip()[-4000:],
    }


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
