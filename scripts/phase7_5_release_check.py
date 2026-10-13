#!/usr/bin/env python3
"""Run the Phase 7.5 pre-release gate after known-size validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.release import CommandResult, ReleaseCheck, collect_static_checks, release_failed  # noqa: E402
from core.validation import forbidden_release_artifacts_check, known_size_report_check, phase75_layout_check  # noqa: E402
from core.version import RELEASE_NAME, __version__  # noqa: E402


def _run(name: str, command: list[str], *, timeout: float, required: bool = True) -> CommandResult:
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        return CommandResult(
            name=name,
            command=tuple(command),
            returncode=completed.returncode,
            duration_seconds=time.perf_counter() - started,
            stdout=completed.stdout,
            stderr=completed.stderr,
            required=required,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            name=name,
            command=tuple(command),
            returncode=124,
            duration_seconds=time.perf_counter() - started,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"Command timed out after {timeout:.0f}s.",
            required=required,
        )


def _git_check(require_clean: bool) -> tuple[ReleaseCheck, dict[str, object]]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=10
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True, check=False, timeout=10
        )
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck("git_worktree_clean", not require_clean, str(exc), required=require_clean), {"available": False}
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    passed = status.returncode == 0 and (not dirty or not require_clean)
    detail = "clean" if not dirty else f"{len(dirty)} changed entries"
    return ReleaseCheck("git_worktree_clean", passed, detail, required=require_clean), {
        "available": commit.returncode == 0 and status.returncode == 0,
        "commit": commit.stdout.strip(),
        "clean": not dirty,
        "dirty_entries": dirty,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="full")
    parser.add_argument(
        "--known-size-report",
        type=Path,
        default=ROOT / "outputs" / "phase7_5" / "known_size_validation",
    )
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-static-dependencies", action="store_true")
    parser.add_argument(
        "--dependency-light",
        action="store_true",
        help="Package-development probe only: omit Python/dependency checks. Never use for final release approval.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase7_5" / "phase7_5_release_gate.json",
    )
    args = parser.parse_args()

    print(f"Depth & 3D Model Lab v{__version__} — {RELEASE_NAME}")
    print("Phase 7.5 release-candidate gate\n")

    checks = collect_static_checks(
        ROOT,
        require_model_dependencies=not args.skip_static_dependencies,
    )
    if args.dependency_light:
        checks = [
            check
            for check in checks
            if check.name in {
                "project_layout",
                "delivery_manifest_removed",
                "version_consistency",
            }
            or check.name.startswith("writable_")
        ]
    checks.append(phase75_layout_check(ROOT))
    checks.append(forbidden_release_artifacts_check(ROOT))
    known_size_check, known_size_payload = known_size_report_check(args.known_size_report)
    checks.append(known_size_check)
    git_check, git_payload = _git_check(args.require_clean_git)
    checks.append(git_check)

    commands = [
        _run(
            "python_compile",
            [sys.executable, "-m", "compileall", "-q", "app", "core", "scripts", "tests", "main.py"],
            timeout=180,
        ),
    ]
    if not args.skip_tests:
        commands.append(
            _run(
                "phase7_5_step10_tests",
                ["bash", "scripts/run_phase7_5_step10_tests.sh", args.mode],
                timeout=2400 if args.mode == "full" else 300,
            )
        )

    failed = release_failed(checks, commands)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "application": {"version": __version__, "release_name": RELEASE_NAME},
        "release_target": "Phase 7.5 Stable",
        "release_metadata_finalized": False,
        "passed": not failed,
        "checks": [{**asdict(item)} for item in checks],
        "commands": [{**asdict(item), "passed": item.passed} for item in commands],
        "git": git_payload,
        "known_size_validation": known_size_payload,
        "next_action": (
            "Finalize v0.7.5 metadata and annotated tag after this gate passes on target hardware."
            if not failed
            else "Resolve required failures before release metadata or tag creation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(args.output)

    for check in checks:
        status = "PASS" if check.passed else ("FAIL" if check.required else "WARN")
        print(f"{status:4} {check.name}: {check.detail}")
    for command in commands:
        status = "PASS" if command.passed else "FAIL"
        print(f"{status:4} {command.name}: {command.duration_seconds:.2f}s")
        if not command.passed:
            output = (command.stderr or command.stdout).strip()
            if output:
                print(output[-4000:])
    print(f"\nRelease gate: {'PASSED' if not failed else 'FAILED'}")
    print(f"Report: {args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
