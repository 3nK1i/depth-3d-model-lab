#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_OUT = Path("/tmp/phase8_8h6_closure")
EXPECTED_BRANCH = "phase8-reconstruction"
ALLOWED_H6_STATUS = {
    "?? docs/phase_history/PHASE8_8H6_VALIDATION.md",
    "?? scripts/run_phase8_8h6_closure.py",
    "?? scripts/run_phase8_8h6_tests.sh",
    "?? tests/unit/test_phase8_8h6_closure_contract.py",
}
HEAVY_IMPORT_ROOTS = {
    "pycolmap",
    "gsplat",
    "nerfstudio",
    "stella_vslam",
    "depth_anything_3",
}


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"Required accepted summary is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Summary root must be an object: {path}")
    return data


def _direct_heavy_imports() -> list[str]:
    findings: list[str] = []
    root = ROOT / "core/reconstruction"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module.split(".", 1)[0] in HEAVY_IMPORT_ROOTS:
                        findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if module.split(".", 1)[0] in HEAVY_IMPORT_ROOTS:
                    findings.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}")
    return findings


def _required_history_present() -> bool:
    return all(
        (ROOT / f"docs/phase_history/PHASE8_8H{index}_VALIDATION.md").is_file()
        for index in range(1, 7)
    )


def _required_runners_present() -> bool:
    return all((ROOT / f"scripts/run_phase8_8h{index}_tests.sh").is_file() for index in range(1, 7))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8.8H6 closure verification")
    parser.add_argument(
        "--h4-summary",
        type=Path,
        default=Path("/tmp/phase8_8h4_full_regression/summary.json"),
    )
    parser.add_argument(
        "--h5-summary",
        type=Path,
        default=Path("/tmp/phase8_8h5_real_ui_acceptance/summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    errors: list[str] = []

    try:
        h4 = _load_json(args.h4_summary.resolve())
        h5 = _load_json(args.h5_summary.resolve())
    except Exception as exc:
        h4 = {}
        h5 = {}
        errors.append(str(exc))

    try:
        branch = _git("branch", "--show-current")
        head = _git("rev-parse", "HEAD")
        origin = _git("rev-parse", "origin/phase8-reconstruction")
        porcelain = {line for line in _git("status", "--porcelain").splitlines() if line}
        v080_tags = [line for line in _git("tag", "--list", "v0.8.0").splitlines() if line]
    except Exception as exc:
        branch = ""
        head = ""
        origin = ""
        porcelain = set()
        v080_tags = []
        errors.append(str(exc))

    unexpected_status = sorted(porcelain - ALLOWED_H6_STATUS)
    missing_h6_status = sorted(ALLOWED_H6_STATUS - porcelain)
    heavy_imports = _direct_heavy_imports()

    h4_ok = bool(
        h4.get("pass") is True
        and h4.get("full_phase8_8_regression_performed") is True
        and h4.get("unit_passed") is True
        and h4.get("ui_passed") is True
        and h4.get("integration_passed") is True
        and h4.get("stress_passed") is True
        and h4.get("phase8_8_protocol_chain_passed") is True
        and h4.get("phase8_8_backward_chain_passed") is True
    )
    h5_ok = bool(
        h5.get("pass") is True
        and h5.get("real_application_executed") is True
        and h5.get("real_external_pycolmap_executed") is True
        and h5.get("offline_visualization_loaded") is True
        and int(h5.get("registered_keyframes", 0) or 0) > 1
        and int(h5.get("landmarks", 0) or 0) > 0
        and h5.get("selected_camera_visible") is True
        and h5.get("resource_release_after_completion") is True
        and h5.get("responsive_1366x768") is True
        and h5.get("production_depth_imports_pycolmap") is False
        and h5.get("pycolmap_in_application_process") is False
        and h5.get("errors") == []
    )

    try:
        from core.version import RELEASE_NAME, RELEASE_PHASE, __version__

        release_metadata_preserved = bool(
            __version__ == "0.7.5"
            and RELEASE_NAME == "Phase 7.5 Stable"
            and float(RELEASE_PHASE) == 7.5
        )
    except Exception as exc:
        release_metadata_preserved = False
        errors.append(f"release metadata check failed: {exc}")

    branch_ok = branch == EXPECTED_BRANCH
    remote_sync = bool(head and head == origin)
    scope_ok = not unexpected_status and not missing_h6_status
    history_ok = _required_history_present()
    runners_ok = _required_runners_present()
    heavy_isolation_ok = not heavy_imports
    no_release_tag = not v080_tags

    passed = all(
        (
            h4_ok,
            h5_ok,
            branch_ok,
            remote_sync,
            scope_ok,
            history_ok,
            runners_ok,
            heavy_isolation_ok,
            release_metadata_preserved,
            no_release_tag,
            not errors,
        )
    )

    summary = {
        "phase": "8.8H6",
        "pass": passed,
        "phase8_8_complete": passed,
        "handoff_to_phase8_9_ready": passed,
        "h4_full_regression_accepted": h4_ok,
        "h5_real_ui_acceptance_accepted": h5_ok,
        "phase8_8_history_complete": history_ok,
        "phase8_8_runner_chain_present": runners_ok,
        "branch": branch,
        "expected_branch": EXPECTED_BRANCH,
        "head": head,
        "origin_phase8_reconstruction": origin,
        "head_matches_origin": remote_sync,
        "h6_git_scope_exact": scope_ok,
        "unexpected_git_status": unexpected_status,
        "missing_h6_git_status": missing_h6_status,
        "direct_heavy_imports_in_core_reconstruction": heavy_imports,
        "heavy_backend_isolation_preserved": heavy_isolation_ok,
        "release_metadata_preserved_at_0_7_5": release_metadata_preserved,
        "v0_8_0_tag_present": bool(v080_tags),
        "production_behavior_changed_by_h6": False,
        "new_feature_work_added_by_h6": False,
        "release_tag_created_by_h6": False,
        "hardware_profile_claimed_by_h6": False,
        "heavy_backend_executed_by_h6": False,
        "phase8_release_created_by_h6": False,
        "errors": errors,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.8H6 closure: {'PASS' if passed else 'FAIL'}")
    print(
        f"h4={h4_ok} h5={h5_ok} branch={branch_ok} remote_sync={remote_sync} "
        f"scope={scope_ok} history={history_ok} isolation={heavy_isolation_ok}"
    )
    print(
        f"phase8_8_complete={passed} handoff_to_phase8_9_ready={passed} "
        f"release_created=False tag_created=False"
    )
    print(f"summary={summary_path}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
    if unexpected_status:
        print(f"ERROR: unexpected Git scope: {unexpected_status}", file=sys.stderr)
    if missing_h6_status:
        print(f"ERROR: missing H6 Git scope: {missing_h6_status}", file=sys.stderr)
    if heavy_imports:
        print(f"ERROR: direct heavy imports: {heavy_imports}", file=sys.stderr)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
