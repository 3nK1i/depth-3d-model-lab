#!/usr/bin/env python3
"""Phase 8.9G2 complete current-tree regression runner.

G2 executes the frozen G1 regression matrix without changing production code or
starting expensive real reconstruction campaigns.  The four primary pytest
suites run in separate subprocesses, followed by the inherited Phase 8.8
protocol chain and every Phase 8.9 A1-F6 quick gate.
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
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
PHASE = "8.9G2"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "c8ef0fcbc9ab1cb25f4f7606b8aca71596bba85f"
G2_OVERLAY = {
    "docs/phase_history/PHASE8_9G2_VALIDATION.md",
    "scripts/run_phase8_9g2_full_regression.py",
    "scripts/run_phase8_9g2_tests.sh",
    "tests/integration/test_phase8_9g2_full_regression_contract.py",
    "tests/unit/test_phase8_9g2_full_regression.py",
}
HEAVY_MODULES = (
    "depth_anything_3",
    "diff_gaussian_rasterization",
    "gsplat",
    "nerfstudio",
    "pycolmap",
    "simple_knn",
    "tinycudann",
)
PHASE89_QUICK_RUNNERS = (
    "scripts/run_phase8_9a1_tests.sh",
    "scripts/run_phase8_9a2_tests.sh",
    "scripts/run_phase8_9b_tests.sh",
    "scripts/run_phase8_9c_tests.sh",
    "scripts/run_phase8_9d_tests.sh",
    "scripts/run_phase8_9e_tests.sh",
    "scripts/run_phase8_9f1_tests.sh",
    "scripts/run_phase8_9f2_tests.sh",
    "scripts/run_phase8_9f3_tests.sh",
    "scripts/run_phase8_9f4_tests.sh",
    "scripts/run_phase8_9f5_tests.sh",
    "scripts/run_phase8_9f6_tests.sh",
)
GROUP_NAMES = (
    "compileall",
    "unit",
    "ui",
    "integration",
    "stress",
    "phase8_8_protocol",
    "phase8_9_runner_chain",
)
_SUMMARY_RE = re.compile(
    r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|xfailed|xpassed|error|errors)\b",
    re.IGNORECASE,
)


def _git(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_text(*args: str) -> str:
    code, out, _ = _git(*args)
    return out if code == 0 else ""


def _status_paths() -> list[str]:
    paths: list[str] = []
    for line in _git_text("status", "--short").splitlines():
        if not line:
            continue
        raw = line[3:] if len(line) >= 4 else line
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip())
    return sorted(paths)


def _release_version() -> dict[str, Any]:
    version_file = ROOT / "core" / "version.py"
    app_yaml = ROOT / "config" / "app.yaml"
    version_text = version_file.read_text(encoding="utf-8") if version_file.is_file() else ""
    yaml_text = app_yaml.read_text(encoding="utf-8") if app_yaml.is_file() else ""
    version_match = re.search(r"__version__\s*=\s*[\"']([^\"']+)", version_text)
    yaml_match = re.search(r"^\s*version\s*:\s*[\"']?([^\s\"']+)", yaml_text, flags=re.MULTILINE)
    return {
        "core_version": version_match.group(1) if version_match else None,
        "config_version": yaml_match.group(1) if yaml_match else None,
        "v0_8_0_tag_present": bool(_git_text("tag", "--list", "v0.8.0")),
    }


def _direct_heavy_imports() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    base = ROOT / "core" / "reconstruction"
    if not base.is_dir():
        return [{"path": str(base), "module": "<missing reconstruction root>"}]
    for path in sorted(base.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            findings.append({"path": str(path.relative_to(ROOT)), "module": f"<parse-error:{exc}>"})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".", 1)[0] in HEAVY_MODULES:
                        findings.append({"path": str(path.relative_to(ROOT)), "module": alias.name})
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".", 1)[0] in HEAVY_MODULES:
                    findings.append({"path": str(path.relative_to(ROOT)), "module": node.module})
    return findings


def _pytest_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in _SUMMARY_RE.finditer(text):
        kind = match.group("kind").lower()
        if kind == "error":
            kind = "errors"
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
    return env


def build_matrix() -> list[dict[str, Any]]:
    return [
        {
            "name": "compileall",
            "command": [sys.executable, "-m", "compileall", "-q", "app", "core", "tests", "scripts", "main.py"],
        },
        {"name": "unit", "command": [sys.executable, "-m", "pytest", "-q", "tests/unit"]},
        {"name": "ui", "command": [sys.executable, "-m", "pytest", "-q", "tests/ui"]},
        {"name": "integration", "command": [sys.executable, "-m", "pytest", "-q", "tests/integration"]},
        {"name": "stress", "command": [sys.executable, "-m", "pytest", "-q", "tests/stress"]},
        {"name": "phase8_8_protocol", "command": ["bash", "scripts/run_phase8_8h3_tests.sh", "protocol"]},
        {
            "name": "phase8_9_runner_chain",
            "command": ["<internal-chain>", "12 Phase 8.9 quick runners A1..F6"],
            "runners": list(PHASE89_QUICK_RUNNERS),
        },
    ]


def _repository_snapshot(*, strict_git: bool) -> dict[str, Any]:
    release = _release_version()
    heavy_imports = _direct_heavy_imports()
    if strict_git:
        branch = _git_text("branch", "--show-current")
        head = _git_text("rev-parse", "HEAD")
        origin = _git_text("rev-parse", "origin/phase8-reconstruction")
        status_paths = _status_paths()
        diff_check_code, _, diff_check_err = _git("diff", "--check")
        checks = {
            "branch": branch == EXPECTED_BRANCH,
            "head": head == EXPECTED_HEAD,
            "origin": origin == EXPECTED_HEAD,
            "overlay_exact": set(status_paths) == G2_OVERLAY,
            "diff_check_clean": diff_check_code == 0,
            "version_0_7_5": release["core_version"] == "0.7.5" and release["config_version"] == "0.7.5",
            "v0_8_0_absent": not release["v0_8_0_tag_present"],
            "no_direct_heavy_imports": not heavy_imports,
        }
    else:
        branch = "<non-strict>"
        head = "<non-strict>"
        origin = "<non-strict>"
        status_paths = []
        diff_check_err = ""
        checks = {
            "branch": True,
            "head": True,
            "origin": True,
            "overlay_exact": True,
            "diff_check_clean": True,
            "version_0_7_5": release["core_version"] == "0.7.5" and release["config_version"] == "0.7.5",
            "v0_8_0_absent": True,
            "no_direct_heavy_imports": not heavy_imports,
        }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": EXPECTED_HEAD,
        "status_paths": status_paths,
        "expected_overlay_paths": sorted(G2_OVERLAY),
        "release": release,
        "direct_heavy_imports": heavy_imports,
        "diff_check_error": diff_check_err,
    }


def _run_streamed(name: str, command: list[str], env: dict[str, str], log_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    lines: list[str] = []
    print(f"\n=== G2 {name} ===", flush=True)
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
        "counts": _pytest_counts(text),
        "log": str(log_path),
    }
    print(
        f"[G2:{name}] {'PASS' if result['passed'] else 'FAIL'} "
        f"elapsed={elapsed:.2f}s counts={result['counts']}",
        flush=True,
    )
    return result


def _sum_counts(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for result in results:
        for kind, value in result.get("counts", {}).items():
            total[kind] = total.get(kind, 0) + int(value)
    return total


def _run_phase89_chain(env: dict[str, str], output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    chain_dir = output_dir / "phase8_9_quick"
    chain_dir.mkdir(parents=True, exist_ok=True)
    children: list[dict[str, Any]] = []
    for index, runner in enumerate(PHASE89_QUICK_RUNNERS, start=1):
        stem = Path(runner).stem
        child = _run_streamed(
            f"phase8_9_{index:02d}_{stem}",
            ["bash", runner, "quick"],
            env,
            chain_dir / f"{index:02d}_{stem}.log",
        )
        child["runner"] = runner
        children.append(child)
        if not child["passed"]:
            break
    elapsed = time.monotonic() - started
    passed = len(children) == len(PHASE89_QUICK_RUNNERS) and all(item["passed"] for item in children)
    result = {
        "name": "phase8_9_runner_chain",
        "command": ["<internal-chain>", "12 Phase 8.9 quick runners A1..F6"],
        "return_code": 0 if passed else 1,
        "passed": passed,
        "elapsed_s": round(elapsed, 3),
        "counts": _sum_counts(children),
        "runner_count_required": len(PHASE89_QUICK_RUNNERS),
        "runner_count_executed": len(children),
        "children": children,
        "log": str(chain_dir),
    }
    print(
        f"[G2:phase8_9_runner_chain] {'PASS' if passed else 'FAIL'} "
        f"executed={len(children)}/{len(PHASE89_QUICK_RUNNERS)} "
        f"elapsed={elapsed:.2f}s counts={result['counts']}",
        flush=True,
    )
    return result


def _empty_payload(pre: dict[str, Any], *, plan_only: bool) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "pass": False,
        "blockers": [],
        "plan_only": plan_only,
        "execution_started": False,
        "matrix": build_matrix(),
        "groups": [],
        "repository_pre": pre,
        "repository_post": None,
        "full_current_tree_regression_performed": False,
        "phase8_9_quick_gate_chain_performed": False,
        "heavy_real_training_or_inference_started": False,
        "production_code_changed_by_8_9g2": False,
        "release_tag_created_by_8_9g2": False,
        "failure_policy": "stop_and_diagnose; never weaken tests to obtain green",
    }


def run_regression(output_dir: Path, *, strict_git: bool = True, plan_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre = _repository_snapshot(strict_git=strict_git)
    payload = _empty_payload(pre, plan_only=plan_only)
    if not pre["pass"]:
        payload["blockers"] = [f"repository_preflight:{name}" for name, ok in pre["checks"].items() if not ok]
        return payload

    if plan_only:
        payload["pass"] = True
        payload["blockers"] = []
        return payload

    env = _base_env()
    payload["execution_started"] = True
    results: list[dict[str, Any]] = []
    for entry in build_matrix():
        name = entry["name"]
        if name == "phase8_9_runner_chain":
            result = _run_phase89_chain(env, output_dir)
        else:
            result = _run_streamed(name, list(entry["command"]), env, output_dir / f"{name}.log")
        results.append(result)
        if not result["passed"]:
            break

    post = _repository_snapshot(strict_git=strict_git)
    by_name = {item["name"]: item for item in results}
    all_groups = all(name in by_name and by_name[name]["passed"] for name in GROUP_NAMES)
    post_ok = post["pass"]
    blockers: list[str] = []
    if not all_groups:
        failed = next((item["name"] for item in results if not item["passed"]), "incomplete_group_matrix")
        blockers.append(f"regression_group:{failed}")
    if not post_ok:
        blockers.extend(f"repository_postflight:{name}" for name, ok in post["checks"].items() if not ok)

    payload.update(
        {
            "pass": not blockers,
            "blockers": blockers,
            "groups": results,
            "repository_post": post,
            "full_current_tree_regression_performed": all_groups,
            "phase8_9_quick_gate_chain_performed": bool(by_name.get("phase8_9_runner_chain", {}).get("passed", False)),
            "suite_results": {
                name: {
                    "passed": bool(by_name.get(name, {}).get("passed", False)),
                    "counts": dict(by_name.get(name, {}).get("counts", {})),
                    "elapsed_s": by_name.get(name, {}).get("elapsed_s"),
                }
                for name in ("unit", "ui", "integration", "stress")
            },
            "truthful_scope": {
                "compileall_and_all_four_primary_pytest_suites_executed": all(
                    bool(by_name.get(name, {}).get("passed", False))
                    for name in ("compileall", "unit", "ui", "integration", "stress")
                ),
                "phase8_8_protocol_executed": bool(by_name.get("phase8_8_protocol", {}).get("passed", False)),
                "phase8_9_a1_through_f6_quick_chain_executed": bool(by_name.get("phase8_9_runner_chain", {}).get("passed", False)),
                "heavy_real_campaigns_reexecuted": False,
                "backward_compatibility_final_gate_deferred_to": "8.9G3",
                "production_code_changed_by_8_9g2": False,
            },
        }
    )
    return payload


def _write_summary(output_dir: Path, payload: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_9g2_full_regression"))
    parser.add_argument(
        "--non-strict-git",
        action="store_true",
        help="Disable workstation branch/HEAD/worktree enforcement for package tests only.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate the frozen matrix and repository preflight without executing regression groups.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    payload = run_regression(
        output_dir,
        strict_git=not args.non_strict_git,
        plan_only=args.plan_only,
    )
    summary = _write_summary(output_dir, payload)

    if payload["pass"]:
        if args.plan_only:
            print("Phase 8.9G2 regression plan preflight: PASS")
            print("groups=compileall,unit,ui,integration,stress,phase8_8_protocol,phase8_9_runner_chain")
            print("phase8_9_quick_runners=12 heavy_real_training_or_inference=False")
        else:
            suites = payload.get("suite_results", {})
            print("\nPhase 8.9G2 complete current-tree regression: PASS")
            print(
                "unit=PASS ui=PASS integration=PASS stress=PASS "
                "phase8_8_protocol=PASS phase8_9_A1_F6_quick_chain=PASS"
            )
            print(
                "measured_counts="
                + " ".join(f"{name}:{suites.get(name, {}).get('counts', {})}" for name in ("unit", "ui", "integration", "stress"))
            )
            print("heavy_real_training_or_inference_started=False production_code_changed=False")
            print("next_step_if_pass=8.9G3_backward_compatibility_final_gate")
        print(f"summary={summary}")
        return 0

    print("Phase 8.9G2 complete current-tree regression: FAIL", file=sys.stderr)
    print(f"blockers={payload.get('blockers', [])}", file=sys.stderr)
    print(f"summary={summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
