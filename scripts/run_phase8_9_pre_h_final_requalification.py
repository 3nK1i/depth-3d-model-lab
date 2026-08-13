#!/usr/bin/env python3
"""Final post-fix whole-application requalification before Phase 8.9H.

This qualification-only gate replays the frozen G2 current-tree regression and
G3 backward-compatibility matrices against the post-RTSP/post-UI head, then
adds bounded RTSP and combined UI qualification.  Historical G2/G3 runners are
invoked with their package-only non-strict Git mode because their embedded
EXPECTED_HEAD values intentionally describe their original historical gates;
this wrapper supplies the strict current-head/worktree/tag/version enforcement.
No real SfM, Gaussian, NeRF, or DA3 training/inference campaign is started.
"""
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
PHASE = "PRE-H FINAL POST-FIX REQUALIFICATION"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "d8882391df8f3d976223a08893f66fd6d2ba14f9"
OVERLAY = {
    "docs/phase_history/PHASE8_9_PRE_H_FINAL_POSTFIX_REQUALIFICATION.md",
    "scripts/run_phase8_9_pre_h_final_requalification.py",
    "scripts/run_phase8_9_pre_h_final_requalification_tests.sh",
    "tests/unit/test_phase8_9_pre_h_final_requalification_contract.py",
}
GROUPS = (
    (
        "g2_complete_current_tree",
        [
            sys.executable,
            "scripts/run_phase8_9g2_full_regression.py",
            "--non-strict-git",
            "--output-dir",
            "/tmp/eia_pre_h_final_requal/g2",
        ],
    ),
    (
        "g3_backward_compatibility",
        [
            sys.executable,
            "scripts/run_phase8_9g3_backward_compatibility.py",
            "--non-strict-git",
            "--output-dir",
            "/tmp/eia_pre_h_final_requal/g3",
        ],
    ),
    (
        "rtsp_postfix_affected",
        ["bash", "scripts/run_phase8_9_pre_h_rtsp_fps_fix_tests.sh", "affected"],
    ),
    (
        "ui4_combined_affected",
        ["bash", "scripts/run_phase8_9_pre_h_ui4_tests.sh", "affected"],
    ),
)


def _git(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _git_text(*args: str) -> str:
    code, out, _ = _git(*args)
    return out if code == 0 else ""


def _status_paths() -> list[str]:
    paths: list[str] = []
    for line in _git_text("status", "--short", "--untracked-files=all").splitlines():
        if not line:
            continue
        path = line[3:] if len(line) >= 4 else line
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip())
    return sorted(paths)


def _version(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = re.search(r"__version__\s*=\s*[\"']([^\"']+)", path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _config_version(path: Path) -> str | None:
    if not path.is_file():
        return None
    match = re.search(
        r"^\s*version\s*:\s*[\"']?([^\s\"']+)",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def _snapshot() -> dict[str, Any]:
    branch = _git_text("branch", "--show-current")
    head = _git_text("rev-parse", "HEAD")
    origin = _git_text("rev-parse", "origin/phase8-reconstruction")
    status = _status_paths()
    diff_code, _, diff_error = _git("diff", "--check")
    tag = _git_text("tag", "--list", "v0.8.0")
    core_version = _version(ROOT / "core/version.py")
    config_version = _config_version(ROOT / "config/app.yaml")
    required = [
        "scripts/run_phase8_9g2_full_regression.py",
        "scripts/run_phase8_9g3_backward_compatibility.py",
        "scripts/run_phase8_9_pre_h_rtsp_fps_fix_tests.sh",
        "scripts/run_phase8_9_pre_h_ui4_tests.sh",
    ]
    checks = {
        "branch": branch == EXPECTED_BRANCH,
        "head": head == EXPECTED_HEAD,
        "origin": origin == EXPECTED_HEAD,
        "overlay_exact": set(status) == OVERLAY,
        "diff_check_clean": diff_code == 0,
        "version_0_7_5": core_version == "0.7.5" and config_version == "0.7.5",
        "v0_8_0_absent": not bool(tag),
        "required_runners_present": all((ROOT / path).is_file() for path in required),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": EXPECTED_HEAD,
        "status_paths": status,
        "expected_overlay_paths": sorted(OVERLAY),
        "release": {
            "core_version": core_version,
            "config_version": config_version,
            "v0_8_0_tag_present": bool(tag),
        },
        "diff_check_error": diff_error,
    }


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_OPENGL", "software")
    env.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _run_group(name: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / f"{name}.log"
    print(f"\n=== {PHASE}: {name} ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        return_code = proc.wait()
    elapsed = round(time.monotonic() - started, 3)
    result = {
        "name": name,
        "command": command,
        "return_code": return_code,
        "passed": return_code == 0,
        "elapsed_s": elapsed,
        "log": str(log_path),
    }
    print(f"[{name}] {'PASS' if result['passed'] else 'FAIL'} elapsed={elapsed}s", flush=True)
    return result


def run(output_dir: Path, *, plan_only: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre = _snapshot()
    groups: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not pre["pass"]:
        blockers.append("repository_preflight")

    if not blockers and not plan_only:
        for name, command in GROUPS:
            result = _run_group(name, list(command), output_dir)
            groups.append(result)
            if not result["passed"]:
                blockers.append(name)
                break

    post = _snapshot()
    if not post["pass"]:
        blockers.append("repository_postflight")

    all_groups = len(groups) == len(GROUPS) and all(item["passed"] for item in groups)
    payload = {
        "phase": PHASE,
        "pass": not blockers and (plan_only or all_groups),
        "plan_only": plan_only,
        "blockers": blockers,
        "failure_policy": "stop_and_diagnose; do not weaken tests or mutate production to force green",
        "groups": groups,
        "matrix": [{"name": name, "command": command} for name, command in GROUPS],
        "repository_pre": pre,
        "repository_post": post,
        "production_code_changed_by_gate": False,
        "release_tag_created_by_gate": False,
        "heavy_real_reconstruction_campaign_started": False,
        "heavy_campaigns_explicitly_not_replayed": [
            "F2 real SfM + DA3 canonical campaign",
            "F3 real Gaussian campaign",
            "F4 real Nerfacto campaign",
            "F5 persistent multi-artifact real-scene campaign",
        ],
        "qualification_scope": {
            "g2_equivalent_current_tree": True,
            "g3_equivalent_backward_compatibility": True,
            "rtsp_postfix_affected": True,
            "combined_ui_postfix": True,
            "next_step_if_pass": "8.9H_documentation_release_preparation",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/eia_pre_h_final_requal"),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    payload = run(args.output_dir.expanduser().resolve(), plan_only=args.plan_only)
    print(f"\n{PHASE}: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"blockers={payload['blockers']}")
    if payload["groups"]:
        print(" ".join(
            f"{item['name']}={'PASS' if item['passed'] else 'FAIL'}"
            for item in payload["groups"]
        ))
    print("production_code_changed=False release_tag_created=False heavy_real_campaign_started=False")
    if payload["pass"] and not payload["plan_only"]:
        print("next_step_if_pass=8.9H_documentation_release_preparation")
    print(f"summary={args.output_dir / 'summary.json'}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
