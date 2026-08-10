#!/usr/bin/env python3
"""Run and record the final Phase 8.7 full regression gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE = "8.7G4"
_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|xfailed|xpassed|error|errors)\b",
    re.IGNORECASE,
)


def _pytest_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _SUMMARY_RE.finditer(text):
        kind = match.group("kind").lower()
        if kind == "error":
            kind = "errors"
        counts[kind] = max(counts.get(kind, 0), int(match.group("count")))
    return counts


def _run(name: str, command: list[str], env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path = output_dir / f"{name}.log"
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    counts = _pytest_counts(proc.stdout or "") if "pytest" in " ".join(command) else {}
    return {
        "name": name,
        "command": command,
        "return_code": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_s": elapsed,
        "counts": counts,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_7g4_full_regression"))
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")

    commands = [
        ("compileall", [sys.executable, "-m", "compileall", "-q", "app", "core", "tests", "scripts", "main.py"]),
        ("unit", [sys.executable, "-m", "pytest", "-q", "tests/unit"]),
        ("ui", [sys.executable, "-m", "pytest", "-q", "tests/ui"]),
        ("integration", [sys.executable, "-m", "pytest", "-q", "tests/integration"]),
        ("stress", [sys.executable, "-m", "pytest", "-q", "tests/stress"]),
    ]

    results: list[dict[str, Any]] = []
    for name, command in commands:
        result = _run(name, command, env, output_dir)
        results.append(result)
        print(
            f"[{name}] {'PASS' if result['passed'] else 'FAIL'} "
            f"elapsed={result['elapsed_s']:.2f}s counts={result['counts']}"
        )
        if not result["passed"]:
            break

    by_name = {item["name"]: item for item in results}
    required = ("compileall", "unit", "ui", "integration", "stress")
    passed = all(name in by_name and by_name[name]["passed"] for name in required)
    payload = {
        "phase": PHASE,
        "pass": passed,
        "full_app_regression_performed": True,
        "full_app_regression_passed": passed,
        "groups": results,
        "required_groups": list(required),
        "ui_readiness_gate_passed": bool(passed and by_name.get("ui", {}).get("passed", False)),
        "reconstruction_ui_enabled": False,
        "phase8_8_ui_handoff_ready": passed,
        "new_reconstruction_algorithms_added": False,
    }
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if passed:
        print("Phase 8.7G4 full regression: PASS")
        print("unit=PASS ui=PASS integration=PASS stress=PASS")
        print("reconstruction_ui_enabled=False phase8_8_ui_handoff_ready=True")
        print(f"summary={summary}")
        return 0

    print("Phase 8.7G4 full regression: FAIL", file=sys.stderr)
    print(f"summary={summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
