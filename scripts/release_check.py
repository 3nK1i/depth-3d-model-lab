#!/usr/bin/env python3
"""Run the reproducible Phase 6 release gate and write a JSON report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.release import (  # noqa: E402
    CommandResult,
    ReleaseCheck,
    benchmark_report_check,
    build_release_report,
    collect_static_checks,
    release_failed,
    write_release_report,
)
from core.version import RELEASE_NAME, __version__  # noqa: E402

_DEFAULT_REPORT = ROOT / "outputs" / "release" / "phase6_release_report.json"
_DEFAULT_BENCHMARK = ROOT / "outputs" / "phase6" / "benchmark_results.json"


def _run_command(
    name: str,
    command: list[str],
    *,
    required: bool = True,
    env: dict[str, str] | None = None,
    timeout_seconds: float = 600.0,
) -> CommandResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(0.01, float(timeout_seconds)),
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr) + f"\nCommand timed out after {timeout_seconds:.0f}s."
    return CommandResult(
        name=name,
        command=tuple(command),
        returncode=returncode,
        duration_seconds=time.perf_counter() - started,
        stdout=stdout,
        stderr=stderr,
        required=required,
    )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _git_snapshot() -> tuple[dict[str, object], ReleaseCheck | None]:
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True,
            capture_output=True, check=False, timeout=10,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=False, timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            capture_output=True, check=False, timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}, None

    available = branch.returncode == commit.returncode == status.returncode == 0
    if not available:
        return {"available": False}, None
    dirty_lines = [line for line in status.stdout.splitlines() if line.strip()]
    return (
        {
            "available": True,
            "branch": branch.stdout.strip(),
            "commit": commit.stdout.strip(),
            "clean": not dirty_lines,
            "dirty_entries": dirty_lines,
        },
        ReleaseCheck(
            "git_worktree_clean",
            not dirty_lines,
            "clean" if not dirty_lines else f"{len(dirty_lines)} changed entries",
            required=False,
        ),
    )


def _print_result(name: str, passed: bool, detail: str = "", *, required: bool = True) -> None:
    suffix = f" — {detail}" if detail else ""
    status = "PASS" if passed else ("FAIL" if required else "WARN")
    print(f"  {status:<4}  {name}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    parser.add_argument("--skip-model-dependencies", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--benchmark-smoke-test", action="store_true")
    parser.add_argument("--require-benchmark", action="store_true")
    parser.add_argument("--benchmark-report", type=Path, default=_DEFAULT_BENCHMARK)
    parser.add_argument(
        "--moge-stress-cycles",
        type=int,
        default=0,
        help="Backward-compatible optional real MoGe-2 memory stress cycles.",
    )
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT)
    args = parser.parse_args()

    print(f"EIA TECH | Depth & 3D Model Lab v{__version__} — {RELEASE_NAME}")
    print(f"Release gate mode: {args.mode}\n")

    checks = collect_static_checks(
        ROOT,
        require_model_dependencies=not args.skip_model_dependencies,
    )
    git, git_check = _git_snapshot()
    if git_check is not None:
        if args.require_clean_git:
            git_check = ReleaseCheck(
                git_check.name, git_check.passed, git_check.detail, required=True
            )
        checks.append(git_check)
    elif args.require_clean_git:
        checks.append(
            ReleaseCheck("git_worktree_clean", False, "Git metadata is unavailable.", required=True)
        )

    commands: list[CommandResult] = [
        _run_command(
            "python_compile",
            [sys.executable, "-m", "compileall", "-q", "app", "core", "tests", "scripts", "main.py"],
            timeout_seconds=120,
        ),
        _run_command(
            "benchmark_asset_validation",
            [
                sys.executable,
                "-c",
                (
                    "from core.benchmark.assets import ensure_benchmark_assets; "
                    "print(ensure_benchmark_assets('outputs/phase6/benchmark_assets').manifest)"
                ),
            ],
            timeout_seconds=120,
        ),
    ]

    if not args.skip_hardware:
        commands.extend(
            [
                _run_command(
                    "physical_cuda_probe",
                    [sys.executable, "scripts/check_hardware.py", "--assert-ready", "--require-cuda"],
                    timeout_seconds=60,
                ),
                _run_command(
                    "rtx2060_profile_simulation",
                    [sys.executable, "scripts/check_hardware.py", "--simulate", "rtx2060"],
                    timeout_seconds=30,
                ),
                _run_command(
                    "rtx5090_profile_simulation",
                    [sys.executable, "scripts/check_hardware.py", "--simulate", "rtx5090"],
                    timeout_seconds=30,
                ),
                _run_command(
                    "rtx5090_pressure_simulation",
                    [sys.executable, "scripts/check_hardware.py", "--simulate", "rtx5090-pressure"],
                    timeout_seconds=30,
                ),
            ]
        )

    if not args.skip_tests:
        test_env = os.environ.copy()
        test_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        test_env.setdefault("QT_OPENGL", "software")
        test_env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        test_env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
        command = ["bash", "scripts/run_phase6_step7_tests.sh", "quick" if args.mode == "quick" else "full"]
        commands.append(
            _run_command(
                "focused_test_suite" if args.mode == "quick" else "complete_test_suite",
                command,
                env=test_env,
                timeout_seconds=1200,
            )
        )

    if args.run_benchmark:
        command = [
            sys.executable,
            "scripts/run_phase6_benchmark.py",
            "--models",
            "all",
            "--require-all",
            "--output-dir",
            str(args.benchmark_report.parent),
        ]
        if args.benchmark_smoke_test:
            command.append("--smoke-test")
        commands.append(
            _run_command(
                "phase6_cross_model_benchmark",
                command,
                timeout_seconds=7200,
            )
        )

    benchmark_payload: dict[str, object] | None = None
    if args.require_benchmark or args.run_benchmark or args.benchmark_report.is_file():
        benchmark_check, benchmark_payload = benchmark_report_check(
            args.benchmark_report,
            require_all_models=True,
        )
        benchmark_check = ReleaseCheck(
            benchmark_check.name,
            benchmark_check.passed,
            benchmark_check.detail,
            required=args.require_benchmark or args.run_benchmark,
        )
        checks.append(benchmark_check)

    if args.moge_stress_cycles > 0:
        commands.append(
            _run_command(
                "moge2_memory_stress",
                [
                    sys.executable,
                    "scripts/run_memory_stress.py",
                    "--model", "moge2",
                    "--profile", "auto",
                    "--cycles", str(max(1, args.moge_stress_cycles)),
                    "--settle-ms", "1000",
                    "--strict-device-vram",
                    "--output", "outputs/release/moge2_release_stress.json",
                ],
                timeout_seconds=1800,
            )
        )

    payload = build_release_report(
        checks=checks,
        commands=commands,
        mode=args.mode,
        git=git,
        benchmark=benchmark_payload,
    )
    report_path = write_release_report(args.output, payload)

    print("Static checks:")
    for check in checks:
        _print_result(check.name, check.passed, check.detail, required=check.required)

    print("\nCommand checks:")
    for command in commands:
        detail = f"{command.duration_seconds:.2f}s"
        if command.passed:
            lines = [line.strip() for line in command.stdout.splitlines() if line.strip()]
            if lines:
                detail += f" · {lines[-1][:160]}"
        _print_result(command.name, command.passed, detail, required=command.required)
        if not command.passed:
            output = (command.stderr or command.stdout).strip()
            if output:
                print("    " + output.replace("\n", "\n    ")[-4000:])

    failed = release_failed(checks, commands)
    print(f"\nRelease report: {report_path}")
    print(f"Release gate: {'PASSED' if not failed else 'FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
