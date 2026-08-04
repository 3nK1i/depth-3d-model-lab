#!/usr/bin/env python3
"""Run the registry-driven cross-model benchmark.

Each model runs in its own Python process so optional runtimes, CUDA allocators,
and temporal caches are fully isolated between measurements. Phase 7 reuses
this CLI for the interactive benchmark page through structured progress events.
"""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ACTIVE_WORKER: subprocess.Popen[str] | None = None
_CANCEL_REQUESTED = False
_PROGRESS_JSONL = False


def _early_signal_handler(_signum: int, _frame: object) -> None:
    """Record cancellation even if requested during optional-runtime imports."""
    global _CANCEL_REQUESTED
    _CANCEL_REQUESTED = True
    process = _ACTIVE_WORKER
    if process is not None and process.poll() is None:
        process.terminate()


signal.signal(signal.SIGTERM, _early_signal_handler)
signal.signal(signal.SIGINT, _early_signal_handler)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark.assets import ensure_benchmark_assets  # noqa: E402
from core.benchmark.events import encode_event  # noqa: E402
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
    parser.add_argument(
        "--image-asset",
        choices=("registry", "landscape", "portrait"),
        default="registry",
        help="Override the registry image workload with one deterministic asset.",
    )
    parser.add_argument(
        "--sequence-asset",
        choices=("registry", "temporal"),
        default="registry",
        help="Override the registry temporal workload with one deterministic sequence.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--include-mock", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--progress-jsonl", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--assets-dir", type=Path, default=_DEFAULT_ASSETS)

    # Internal isolated-worker mode. These options remain available for tests.
    parser.add_argument("--worker-model", default="")
    parser.add_argument("--worker-output", type=Path)
    return parser


def main() -> int:
    global _PROGRESS_JSONL
    args = _parser().parse_args()
    _PROGRESS_JSONL = bool(args.progress_jsonl)
    if args.worker_model:
        return _worker_main(args)
    _install_signal_handlers()
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
        image_asset_key=None if args.image_asset == "registry" else args.image_asset,
        sequence_asset_key=None if args.sequence_asset == "registry" else args.sequence_asset,
    )
    result = benchmark_model(
        args.worker_model,
        assets_root=args.assets_dir,
        options=options,
    )
    write_json(args.worker_output, result)
    return 0


def _parent_main(args: argparse.Namespace) -> int:
    global _ACTIVE_WORKER

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
    _emit_event(
        "benchmark_started",
        total_models=len(descriptors),
        model_ids=[item.model_id for item in descriptors],
        hardware=hardware.short_label,
        output_dir=str(output_dir),
    )

    results: list[dict[str, object]] = []
    for index, descriptor in enumerate(descriptors, start=1):
        if _CANCEL_REQUESTED:
            break
        policy = descriptor.benchmark
        timeout = max(30.0, float(policy.timeout_seconds))
        with tempfile.TemporaryDirectory(prefix="depth3d-phase6-benchmark-") as directory:
            worker_output = Path(directory) / f"{descriptor.model_id}.json"
            command = _worker_command(args, descriptor.model_id, worker_output, assets_dir)

            print(f"[{descriptor.display_name}] running…", flush=True)
            _emit_event(
                "model_started",
                model_id=descriptor.model_id,
                display_name=descriptor.display_name,
                index=index,
                total_models=len(descriptors),
                classification=policy.classification,
                workload=policy.workload,
            )
            started = time.perf_counter()
            try:
                _ACTIVE_WORKER = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = _ACTIVE_WORKER.communicate(timeout=timeout)
                returncode = int(_ACTIVE_WORKER.returncode or 0)
                elapsed = time.perf_counter() - started
                if _CANCEL_REQUESTED:
                    result = _cancelled_result(descriptor.model_id, descriptor.display_name)
                    result["isolated_process_seconds"] = elapsed
                    result["worker_returncode"] = returncode
                elif worker_output.is_file():
                    result = json.loads(worker_output.read_text(encoding="utf-8"))
                    result["isolated_process_seconds"] = elapsed
                    result["worker_returncode"] = returncode
                    if returncode != 0 and result.get("status") == "passed":
                        result["status"] = "failed"
                        result["error"] = "Benchmark worker returned a non-zero exit code."
                else:
                    result = _subprocess_failure(
                        descriptor.model_id,
                        descriptor.display_name,
                        f"Worker exited with code {returncode} without a report.",
                        stderr or stdout,
                    )
                    result["isolated_process_seconds"] = elapsed
                    result["worker_returncode"] = returncode
                if stderr.strip():
                    result["worker_stderr_tail"] = stderr.strip()[-4000:]
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - started
                _terminate_process(_ACTIVE_WORKER)
                stdout, stderr = _ACTIVE_WORKER.communicate()
                result = _subprocess_failure(
                    descriptor.model_id,
                    descriptor.display_name,
                    f"Benchmark timed out after {timeout:.0f} seconds.",
                    stderr or stdout,
                )
                result["isolated_process_seconds"] = elapsed
                result["worker_returncode"] = 124
            finally:
                _ACTIVE_WORKER = None

            results.append(result)
            status = str(result.get("status", "unknown"))
            print(f"[{descriptor.display_name}] {status.upper()}")
            _emit_event(
                "model_completed",
                model_id=descriptor.model_id,
                display_name=descriptor.display_name,
                index=index,
                total_models=len(descriptors),
                status=status,
                error=str(result.get("error") or result.get("availability_reason") or ""),
            )
            if _CANCEL_REQUESTED:
                break

    cancelled = bool(_CANCEL_REQUESTED)
    passed = (
        not cancelled
        and len(results) == len(descriptors)
        and all(item.get("status") == "passed" for item in results)
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "application": {
            "name": "EIA TECH - Depth & 3D Model Lab",
            "version": __version__,
            "release_name": RELEASE_NAME,
        },
        "generated_at_unix": time.time(),
        "passed": passed,
        "cancelled": cancelled,
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
            "image_asset": args.image_asset,
            "sequence_asset": args.sequence_asset,
            "smoke_test": bool(args.smoke_test),
            "isolated_process_per_model": True,
        },
        "asset_manifest": str(manifest.manifest),
        "asset_checksums": manifest.checksums,
        "requested_models": [item.model_id for item in descriptors],
        "completed_models": len(results),
        "results": results,
    }

    report_paths = _write_reports(output_dir, payload)
    print("\nReports:")
    for path in report_paths.values():
        print(f"  {path}")
    if cancelled:
        conclusion = "CANCELLED"
    elif passed:
        conclusion = "PASSED"
    else:
        conclusion = "COMPLETED WITH NON-PASSING MODELS"
    print(f"\nBenchmark: {conclusion}")
    _emit_event(
        "benchmark_finished",
        passed=passed,
        cancelled=cancelled,
        completed_models=len(results),
        total_models=len(descriptors),
        report_paths={name: str(path) for name, path in report_paths.items()},
    )
    if cancelled:
        return 130
    return 1 if args.require_all and not passed else 0


def _worker_command(
    args: argparse.Namespace,
    model_id: str,
    worker_output: Path,
    assets_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-model",
        model_id,
        "--worker-output",
        str(worker_output),
        "--profile",
        args.profile,
        "--point-budget",
        str(max(1, int(args.point_budget))),
        "--assets-dir",
        str(assets_dir),
        "--image-asset",
        args.image_asset,
        "--sequence-asset",
        args.sequence_asset,
    ]
    if args.iterations is not None:
        command.extend(["--iterations", str(max(1, args.iterations))])
    if args.sequence_frames is not None:
        command.extend(["--sequence-frames", str(max(1, args.sequence_frames))])
    if args.smoke_test:
        command.append("--smoke-test")
    return command


def _write_reports(output_dir: Path, payload: dict[str, object]) -> dict[str, Path]:
    results = list(payload.get("results", []))
    paths = {
        "json": write_json(output_dir / "benchmark_results.json", payload),
        "hardware": write_json(
            output_dir / "hardware_summary.json",
            {
                "schema_version": 1,
                "application_version": __version__,
                "generated_at_unix": payload["generated_at_unix"],
                "hardware": payload["hardware"],
                "configuration": payload["configuration"],
            },
        ),
        "csv": write_benchmark_csv(output_dir / "benchmark_results.csv", results),
        "capability_matrix": write_capability_matrix(
            output_dir / "model_capability_matrix.md",
            production_benchmark_descriptors(),
        ),
        "summary": write_benchmark_summary(output_dir / "benchmark_summary.md", payload),
    }
    return paths


def _selected_descriptors(value: str, *, include_mock: bool) -> tuple[Any, ...]:
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


def _install_signal_handlers() -> None:
    def _handle_signal(_signum: int, _frame: object) -> None:
        global _CANCEL_REQUESTED
        if _CANCEL_REQUESTED:
            return
        _CANCEL_REQUESTED = True
        _emit_event("cancellation_requested")
        _terminate_process(_ACTIVE_WORKER)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3.0)


def _emit_event(event: str, **payload: object) -> None:
    if not _PROGRESS_JSONL:
        return
    print(encode_event(event, **payload), flush=True)


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


def _cancelled_result(model_id: str, display_name: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "application_version": __version__,
        "model_id": model_id,
        "display_name": display_name,
        "status": "cancelled",
        "error_type": "BenchmarkCancelled",
        "error": "Benchmark cancelled by the user.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
