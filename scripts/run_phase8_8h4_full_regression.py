#!/usr/bin/env python3
"""Run and record the Phase 8.8H4 full regression gate."""
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
PHASE = "8.8H4"
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
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


def _run(name: str, command: list[str], env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    log_path = output_dir / f"{name}.log"
    lines: list[str] = []
    print(f"\n=== H4 {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        return_code = proc.wait()

    elapsed = time.monotonic() - started
    text = "".join(lines)
    result = {
        "name": name,
        "command": command,
        "return_code": return_code,
        "passed": return_code == 0,
        "elapsed_s": round(elapsed, 3),
        "counts": _pytest_counts(text) if "pytest" in text or "pytest" in " ".join(command) else {},
        "log": str(log_path),
    }
    print(
        f"[H4:{name}] {'PASS' if result['passed'] else 'FAIL'} "
        f"elapsed={elapsed:.2f}s counts={result['counts']}",
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_8h4_full_regression"),
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")

    groups: list[tuple[str, list[str]]] = [
        (
            "compileall",
            [sys.executable, "-m", "compileall", "-q", "app", "core", "tests", "scripts", "main.py"],
        ),
        ("unit", [sys.executable, "-m", "pytest", "-q", "tests/unit"]),
        ("ui", [sys.executable, "-m", "pytest", "-q", "tests/ui"]),
        ("integration", [sys.executable, "-m", "pytest", "-q", "tests/integration"]),
        ("stress", [sys.executable, "-m", "pytest", "-q", "tests/stress"]),
        ("phase8_8_protocol", ["bash", "scripts/run_phase8_8h3_tests.sh", "protocol"]),
        ("phase8_8_backward", ["bash", "scripts/run_phase8_8h3_tests.sh", "backward"]),
    ]

    results: list[dict[str, Any]] = []
    for name, command in groups:
        result = _run(name, command, env, output_dir)
        results.append(result)
        if not result["passed"]:
            break

    by_name = {item["name"]: item for item in results}
    required = [name for name, _ in groups]
    passed = all(name in by_name and by_name[name]["passed"] for name in required)

    payload = {
        "phase": PHASE,
        "pass": passed,
        "full_phase8_8_regression_performed": True,
        "required_groups": required,
        "groups": results,
        "unit_passed": bool(by_name.get("unit", {}).get("passed", False)),
        "ui_passed": bool(by_name.get("ui", {}).get("passed", False)),
        "integration_passed": bool(by_name.get("integration", {}).get("passed", False)),
        "stress_passed": bool(by_name.get("stress", {}).get("passed", False)),
        "phase8_8_protocol_chain_passed": bool(by_name.get("phase8_8_protocol", {}).get("passed", False)),
        "phase8_8_backward_chain_passed": bool(by_name.get("phase8_8_backward", {}).get("passed", False)),
        "new_feature_work_added_by_h4": False,
        "production_behavior_changed_by_h4": False,
        "heavy_backend_executed_by_h4": False,
        "hardware_profile_claimed_by_h4": False,
        "release_tag_created_by_h4": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if passed:
        print("\nPhase 8.8H4 full regression: PASS")
        print("unit=PASS ui=PASS integration=PASS stress=PASS protocol=PASS backward=PASS")
        print(f"summary={summary_path}")
        return 0

    failed = next((item["name"] for item in results if not item["passed"]), "incomplete")
    print(f"\nPhase 8.8H4 full regression: FAIL group={failed}", file=sys.stderr)
    print(f"summary={summary_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
