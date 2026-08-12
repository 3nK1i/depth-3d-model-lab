#!/usr/bin/env python3
"""Phase 8.9G3 backward-compatibility and final regression gate.

G3 executes the targeted compatibility matrix frozen by G1 after G2 has
completed the broad current-tree regression.  It deliberately avoids repeating
all four full pytest suites and does not start heavy real reconstruction
campaigns.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE = "8.9G3"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "a7cae9f1477a814b7561739a07d4d2cfd1b3f593"
G3_OVERLAY = {
    "docs/phase_history/PHASE8_9G3_VALIDATION.md",
    "scripts/run_phase8_9g3_backward_compatibility.py",
    "scripts/run_phase8_9g3_tests.sh",
    "tests/integration/test_phase8_9g3_backward_compatibility_contract.py",
    "tests/unit/test_phase8_9g3_backward_compatibility.py",
}
G2_EVIDENCE = "docs/phase_history/PHASE8_9G2_VALIDATION.md"
HEAVY_MODULES = {
    "depth_anything_3",
    "diff_gaussian_rasterization",
    "gsplat",
    "nerfstudio",
    "pycolmap",
    "simple_knn",
    "tinycudann",
}
GROUP_NAMES = (
    "phase5_smoke",
    "phase6_release_quick",
    "phase7_release_quick",
    "phase7_5_release_quick",
    "phase8_8_backward",
    "main_window_startup_shutdown",
)
_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|xfailed|xpassed|error|errors)\b",
    re.IGNORECASE,
)


def build_matrix() -> list[dict[str, Any]]:
    return [
        {
            "name": "phase5_smoke",
            "command": ["bash", "scripts/run_phase5_tests.sh", "smoke"],
            "scope": "pre-reconstruction core/runtime smoke compatibility",
        },
        {
            "name": "phase6_release_quick",
            "command": ["bash", "scripts/run_phase6_step7_tests.sh", "quick"],
            "scope": "Phase 6 model/benchmark release behavior",
        },
        {
            "name": "phase7_release_quick",
            "command": ["bash", "scripts/run_phase7_step4_tests.sh", "quick"],
            "scope": "Phase 7 operational modes and release behavior",
        },
        {
            "name": "phase7_5_release_quick",
            "command": ["bash", "scripts/run_phase7_5_step10_tests.sh", "quick"],
            "scope": "Phase 7.5 measurement/inspection release behavior",
        },
        {
            "name": "phase8_8_backward",
            "command": ["bash", "scripts/run_phase8_8h3_tests.sh", "backward"],
            "scope": "inherited Phase 8.8 backward-compatibility chain",
        },
        {
            "name": "main_window_startup_shutdown",
            "command": [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/integration/test_phase8_9g3_backward_compatibility_contract.py",
            ],
            "scope": "bounded actual MainWindow legacy-mode startup/switch/shutdown sanity",
        },
    ]


def _git(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _status_paths(raw: str) -> list[str]:
    paths: list[str] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip())
    return sorted(paths)


def _version_from_python(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in targets):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
    return None


def _config_version(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^\s*version\s*:\s*[\"']?([^\"'\s#]+)", text, re.MULTILINE)
    return match.group(1) if match else None


def _direct_heavy_imports() -> list[str]:
    findings: list[str] = []
    root = ROOT / "core" / "reconstruction"
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name in HEAVY_MODULES:
                    findings.append(f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', '?')}:{name}")
    return findings


def _repository_snapshot(*, strict_git: bool) -> dict[str, Any]:
    direct_heavy = _direct_heavy_imports()
    core_version = _version_from_python(ROOT / "core" / "version.py")
    config_version = _config_version(ROOT / "config" / "app.yaml")

    if strict_git:
        head_rc, head_out, _ = _git("rev-parse", "HEAD")
        origin_rc, origin_out, _ = _git("rev-parse", f"origin/{EXPECTED_BRANCH}")
        branch_rc, branch_out, _ = _git("branch", "--show-current")
        status_rc, status_out, _ = _git("status", "--short", "--untracked-files=all")
        diff_rc, _, diff_err = _git("diff", "--check")
        tag_rc, tag_out, _ = _git("tag", "--list", "v0.8.0")
        status_paths = _status_paths(status_out) if status_rc == 0 else []
        checks = {
            "branch": branch_rc == 0 and branch_out.strip() == EXPECTED_BRANCH,
            "head": head_rc == 0 and head_out.strip() == EXPECTED_HEAD,
            "origin": origin_rc == 0 and origin_out.strip() == EXPECTED_HEAD,
            "overlay_exact": set(status_paths) == G3_OVERLAY,
            "diff_check_clean": diff_rc == 0,
            "version_0_7_5": core_version == "0.7.5" and config_version == "0.7.5",
            "v0_8_0_absent": tag_rc == 0 and not tag_out.strip(),
            "no_direct_heavy_imports": not direct_heavy,
            "g2_evidence_present": (ROOT / G2_EVIDENCE).is_file(),
        }
        branch = branch_out.strip()
        head = head_out.strip()
        origin = origin_out.strip()
        tag_present: bool | None = bool(tag_out.strip()) if tag_rc == 0 else None
    else:
        branch = "<non-strict>"
        head = "<non-strict>"
        origin = "<non-strict>"
        status_paths = []
        diff_err = ""
        tag_present = False
        checks = {
            "branch": True,
            "head": True,
            "origin": True,
            "overlay_exact": True,
            "diff_check_clean": True,
            "version_0_7_5": core_version == "0.7.5" and config_version == "0.7.5",
            "v0_8_0_absent": True,
            "no_direct_heavy_imports": not direct_heavy,
            "g2_evidence_present": (ROOT / G2_EVIDENCE).is_file(),
        }

    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": EXPECTED_HEAD,
        "status_paths": status_paths,
        "expected_overlay_paths": sorted(G3_OVERLAY),
        "direct_heavy_imports": direct_heavy,
        "release": {
            "core_version": core_version,
            "config_version": config_version,
            "v0_8_0_tag_present": tag_present,
        },
        "checks": checks,
        "diff_check_error": diff_err.strip(),
        "pass": all(checks.values()),
    }


def _pytest_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _SUMMARY_RE.finditer(text):
        kind = match.group("kind").lower()
        if kind == "error":
            kind = "errors"
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


def _run_group(entry: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    name = str(entry["name"])
    command = [str(part) for part in entry["command"]]
    log = output_dir / f"{name}.log"
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n=== G3 {name} ===")
    print("$ " + " ".join(command))
    started = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    output = proc.stdout or ""
    print(output, end="" if output.endswith("\n") or not output else "\n")
    log.write_text(output, encoding="utf-8")
    result = {
        "name": name,
        "scope": entry.get("scope", ""),
        "command": command,
        "return_code": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_s": round(elapsed, 3),
        "counts": _pytest_counts(output),
        "log": str(log),
    }
    print(
        f"[G3:{name}] {'PASS' if result['passed'] else 'FAIL'} "
        f"elapsed={result['elapsed_s']}s counts={result['counts']}"
    )
    return result


def run_gate(output_dir: Path, *, strict_git: bool = True, plan_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix = build_matrix()
    pre = _repository_snapshot(strict_git=strict_git)
    blockers: list[str] = []
    if not pre["pass"]:
        blockers.append("repository_preflight")

    groups: list[dict[str, Any]] = []
    execution_started = False
    if not blockers and not plan_only:
        execution_started = True
        for entry in matrix:
            result = _run_group(entry, output_dir)
            groups.append(result)
            if not result["passed"]:
                blockers.append(str(result["name"]))
                break

    post = _repository_snapshot(strict_git=strict_git)
    if not post["pass"]:
        blockers.append("repository_postflight")

    group_by_name = {item["name"]: item for item in groups}
    all_six = len(groups) == len(matrix) and all(item["passed"] for item in groups)
    payload: dict[str, Any] = {
        "phase": PHASE,
        "pass": not blockers and (plan_only or all_six),
        "blockers": blockers,
        "failure_policy": "stop_and_diagnose; never weaken tests to obtain green",
        "plan_only": plan_only,
        "execution_started": execution_started,
        "matrix": matrix,
        "groups": groups,
        "backward_compatibility_final_gate_performed": all_six,
        "legacy_release_gates": {
            name: bool(group_by_name.get(name, {}).get("passed"))
            for name in GROUP_NAMES[:-1]
        },
        "main_window_startup_shutdown_passed": bool(
            group_by_name.get("main_window_startup_shutdown", {}).get("passed")
        ),
        "repository_pre": pre,
        "repository_post": post,
        "production_code_changed_by_8_9g3": False,
        "heavy_real_reconstruction_campaign_started_by_8_9g3": False,
        "release_tag_created_by_8_9g3": False,
        "truthful_scope": {
            "g2_full_suite_duplication": False,
            "phase5_smoke_executed": bool(group_by_name.get("phase5_smoke", {}).get("passed")),
            "phase6_release_quick_executed": bool(group_by_name.get("phase6_release_quick", {}).get("passed")),
            "phase7_release_quick_executed": bool(group_by_name.get("phase7_release_quick", {}).get("passed")),
            "phase7_5_release_quick_executed": bool(group_by_name.get("phase7_5_release_quick", {}).get("passed")),
            "phase8_8_backward_executed": bool(group_by_name.get("phase8_8_backward", {}).get("passed")),
            "actual_main_window_bounded_sanity_executed": bool(
                group_by_name.get("main_window_startup_shutdown", {}).get("passed")
            ),
            "production_code_changed_by_8_9g3": False,
            "next_step_if_pass": "8.9H_documentation_release_preparation",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_9g3_backward_compatibility"),
    )
    parser.add_argument("--non-strict-git", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    payload = run_gate(
        args.output_dir,
        strict_git=not args.non_strict_git,
        plan_only=args.plan_only,
    )
    print(
        f"\nPhase 8.9G3 backward compatibility + final regression gate: "
        f"{'PASS' if payload['pass'] else 'FAIL'}"
    )
    print(f"blockers={payload['blockers']}")
    if payload["groups"]:
        states = " ".join(
            f"{item['name']}={'PASS' if item['passed'] else 'FAIL'}"
            for item in payload["groups"]
        )
        print(states)
    print(
        "production_code_changed=False "
        "heavy_real_reconstruction_campaign_started=False "
        "release_tag_created=False"
    )
    if payload["pass"] and not payload["plan_only"]:
        print("next_step_if_pass=8.9H_documentation_release_preparation")
    print(f"summary={args.output_dir / 'summary.json'}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
