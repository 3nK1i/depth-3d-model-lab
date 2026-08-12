#!/usr/bin/env python3
"""Phase 8.9G1 regression inventory and release-gate preflight.

This is intentionally qualification-only.  It inventories the current automated
regression surface and records the exact G2/G3 execution plan without starting
heavy reconstruction backends or changing production code.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE = "8.9G1"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "0a987bc392bd03a8b56bdc62884a8e624ebd764c"
G1_OVERLAY = {
    "docs/phase_history/PHASE8_9G1_VALIDATION.md",
    "scripts/run_phase8_9g1_regression_inventory.py",
    "scripts/run_phase8_9g1_tests.sh",
    "tests/integration/test_phase8_9g1_regression_contract.py",
    "tests/unit/test_phase8_9g1_regression_inventory.py",
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
PHASE89_RUNNERS = (
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

# G2 is the broad current-tree regression.  The four pytest suites are run in
# separate processes to preserve the established Qt isolation discipline.
G2_MATRIX = (
    {"name": "compileall", "command": [sys.executable, "-m", "compileall", "-q", "app", "core", "tests", "scripts", "main.py"]},
    {"name": "unit", "command": [sys.executable, "-m", "pytest", "-q", "tests/unit"]},
    {"name": "ui", "command": [sys.executable, "-m", "pytest", "-q", "tests/ui"]},
    {"name": "integration", "command": [sys.executable, "-m", "pytest", "-q", "tests/integration"]},
    {"name": "stress", "command": [sys.executable, "-m", "pytest", "-q", "tests/stress"]},
    {"name": "phase8_8_protocol", "command": ["bash", "scripts/run_phase8_8h3_tests.sh", "protocol"]},
    {"name": "phase8_9_runner_chain", "command": ["<G2 runner>", "all Phase 8.9 quick gates A1..F6"]},
)

# G3 is deliberately explicit rather than another copy of the whole suite.  It
# proves stable pre-Phase-8 behaviors and the inherited 8.8 backward chain.
G3_MATRIX = (
    {"name": "phase5_smoke", "command": ["bash", "scripts/run_phase5_tests.sh", "smoke"]},
    {"name": "phase6_release_quick", "command": ["bash", "scripts/run_phase6_step7_tests.sh", "quick"]},
    {"name": "phase7_release_quick", "command": ["bash", "scripts/run_phase7_step4_tests.sh", "quick"]},
    {"name": "phase7_5_release_quick", "command": ["bash", "scripts/run_phase7_5_step10_tests.sh", "quick"]},
    {"name": "phase8_8_backward", "command": ["bash", "scripts/run_phase8_8h3_tests.sh", "backward"]},
    {"name": "main_window_startup_shutdown", "command": ["<G3 bounded application sanity>"]},
)


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _status_paths() -> list[str]:
    paths: list[str] = []
    for line in _git("status", "--short").splitlines():
        if not line:
            continue
        raw = line[3:] if len(line) >= 4 else line
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.append(raw.strip())
    return sorted(paths)


def _test_file_counts(root: Path = ROOT) -> dict[str, int]:
    result: dict[str, int] = {}
    for group in ("unit", "ui", "integration", "stress"):
        result[group] = len(list((root / "tests" / group).glob("test_*.py")))
    return result


def _phase89_test_file_counts(root: Path = ROOT) -> dict[str, int]:
    result: dict[str, int] = {}
    for group in ("unit", "ui", "integration", "stress"):
        result[group] = len(list((root / "tests" / group).glob("test_phase8_9*.py")))
    return result


def _runner_modes(path: Path) -> list[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    return sorted(set(re.findall(r"^\s*([a-zA-Z0-9_-]+)\)\s*", text, flags=re.MULTILINE)))


def _direct_heavy_imports(root: Path = ROOT) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    base = root / "core" / "reconstruction"
    if not base.is_dir():
        return [{"path": str(base), "module": "<missing reconstruction root>"}]
    for path in sorted(base.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            findings.append({"path": str(path.relative_to(root)), "module": f"<parse-error:{exc}>"})
            continue
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".", 1)[0]
                    if top in HEAVY_MODULES:
                        findings.append({"path": str(path.relative_to(root)), "module": alias.name})
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".", 1)[0]
                if top in HEAVY_MODULES:
                    findings.append({"path": str(path.relative_to(root)), "module": node.module})
    return findings


def _release_version(root: Path = ROOT) -> dict[str, Any]:
    version_file = root / "core" / "version.py"
    app_yaml = root / "config" / "app.yaml"
    version_text = version_file.read_text(encoding="utf-8") if version_file.is_file() else ""
    yaml_text = app_yaml.read_text(encoding="utf-8") if app_yaml.is_file() else ""
    version_match = re.search(r"__version__\s*=\s*[\"']([^\"']+)", version_text)
    yaml_match = re.search(r"^\s*version\s*:\s*[\"']?([^\s\"']+)", yaml_text, flags=re.MULTILINE)
    return {
        "core_version": version_match.group(1) if version_match else None,
        "config_version": yaml_match.group(1) if yaml_match else None,
        "v0_8_0_tag_present": bool(_git("tag", "--list", "v0.8.0")),
    }


def build_inventory(root: Path = ROOT, *, strict_git: bool = True) -> dict[str, Any]:
    branch = _git("branch", "--show-current") if root == ROOT else ""
    head = _git("rev-parse", "HEAD") if root == ROOT else ""
    origin = _git("rev-parse", "origin/phase8-reconstruction") if root == ROOT else ""
    status_paths = _status_paths() if root == ROOT else []

    runner_modes = {
        runner: _runner_modes(root / runner)
        for runner in PHASE89_RUNNERS
    }
    missing_runners = [runner for runner, modes in runner_modes.items() if not modes]
    bad_phase89_modes = {
        runner: modes for runner, modes in runner_modes.items()
        if modes and not {"quick", "affected"}.issubset(set(modes))
    }

    h4 = root / "scripts" / "run_phase8_8h4_full_regression.py"
    h4_text = h4.read_text(encoding="utf-8") if h4.is_file() else ""
    inherited_h4_groups = {
        name: token in h4_text
        for name, token in {
            "compileall": '"compileall"',
            "unit": '"unit"',
            "ui": '"ui"',
            "integration": '"integration"',
            "stress": '"stress"',
            "phase8_8_protocol": '"phase8_8_protocol"',
            "phase8_8_backward": '"phase8_8_backward"',
        }.items()
    }

    legacy_runners = {
        "phase5": _runner_modes(root / "scripts/run_phase5_tests.sh"),
        "phase6_release": _runner_modes(root / "scripts/run_phase6_step7_tests.sh"),
        "phase7_release": _runner_modes(root / "scripts/run_phase7_step4_tests.sh"),
        "phase7_5_release": _runner_modes(root / "scripts/run_phase7_5_step10_tests.sh"),
        "phase8_8_h3": _runner_modes(root / "scripts/run_phase8_8h3_tests.sh"),
    }

    required_legacy_modes = {
        "phase5": "smoke",
        "phase6_release": "quick",
        "phase7_release": "quick",
        "phase7_5_release": "quick",
        "phase8_8_h3": "backward",
    }
    legacy_ready = all(
        required in legacy_runners[name]
        for name, required in required_legacy_modes.items()
    )

    release = _release_version(root)
    heavy_imports = _direct_heavy_imports(root)
    overlay_exact = set(status_paths) == G1_OVERLAY if strict_git else True
    git_ok = (
        branch == EXPECTED_BRANCH
        and head == EXPECTED_HEAD
        and origin == EXPECTED_HEAD
        and overlay_exact
    ) if strict_git else True

    checks = {
        "git_baseline_and_overlay_exact": git_ok,
        "pytest_config_present": (root / "pytest.ini").is_file(),
        "four_primary_test_suites_present": all((root / "tests" / g).is_dir() for g in ("unit", "ui", "integration", "stress")),
        "phase8_9_runner_chain_complete": not missing_runners and not bad_phase89_modes,
        "inherited_h4_full_regression_groups_present": all(inherited_h4_groups.values()),
        "legacy_backward_gate_inputs_present": legacy_ready,
        "no_direct_heavy_imports_in_core_reconstruction": not heavy_imports,
        "release_version_still_0_7_5": release["core_version"] == "0.7.5" and release["config_version"] == "0.7.5",
        "v0_8_0_tag_absent": not release["v0_8_0_tag_present"],
    }

    blockers = sorted(name for name, passed in checks.items() if not passed)
    return {
        "phase": PHASE,
        "pass": not blockers,
        "blockers": blockers,
        "checks": checks,
        "repository": {
            "expected_branch": EXPECTED_BRANCH,
            "expected_head": EXPECTED_HEAD,
            "branch": branch,
            "head": head,
            "origin_head": origin,
            "overlay_paths": status_paths,
            "expected_overlay_paths": sorted(G1_OVERLAY),
        },
        "inventory": {
            "test_files": _test_file_counts(root),
            "phase8_9_test_files": _phase89_test_file_counts(root),
            "phase_test_runner_count": len(list((root / "scripts").glob("run_phase*_tests.sh"))),
            "phase8_test_runner_count": len(list((root / "scripts").glob("run_phase8*_tests.sh"))),
            "phase8_9_runner_count": len(PHASE89_RUNNERS),
            "phase8_9_runners": runner_modes,
            "inherited_h4_groups": inherited_h4_groups,
            "legacy_runners": legacy_runners,
            "direct_heavy_imports_in_core_reconstruction": heavy_imports,
            "release": release,
        },
        "planned_g2": {
            "scope": "complete_current_tree_regression",
            "groups": list(G2_MATRIX),
            "heavy_real_training_or_inference": False,
            "suite_process_isolation": True,
            "failure_policy": "stop_and_diagnose; never weaken tests to obtain green",
        },
        "planned_g3": {
            "scope": "explicit_backward_compatibility_and_final_regression_gate",
            "groups": list(G3_MATRIX),
            "full_suite_duplication": False,
            "legacy_product_behaviors_targeted": [
                "pre-reconstruction core/runtime",
                "Phase 6 model/benchmark release behavior",
                "Phase 7 operational modes",
                "Phase 7.5 measurement/inspection release behavior",
                "Phase 8.8 inherited backward chain",
                "bounded application startup/shutdown",
            ],
        },
        "truthful_scope": {
            "production_code_changed_by_8_9g1": False,
            "heavy_backend_started_by_8_9g1": False,
            "new_regression_framework_created": False,
            "release_tag_created": False,
            "g1_is_inventory_and_preflight_only": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_9g1_regression_inventory"))
    parser.add_argument(
        "--non-strict-git",
        action="store_true",
        help="Developer/test-only: inventory source without enforcing the production Git baseline.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_inventory(strict_git=not args.non_strict_git)
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    status = "PASS" if payload["pass"] else "FAIL"
    counts = payload["inventory"]["test_files"]
    print(f"Phase 8.9G1 regression inventory + preflight: {status}")
    print(
        "test_files="
        f"unit:{counts['unit']} ui:{counts['ui']} integration:{counts['integration']} stress:{counts['stress']}"
    )
    print(
        f"phase8_9_runners={payload['inventory']['phase8_9_runner_count']} "
        f"legacy_backward_ready={payload['checks']['legacy_backward_gate_inputs_present']}"
    )
    print(
        "release_version="
        f"{payload['inventory']['release']['core_version']} "
        f"v0.8.0_tag_present={payload['inventory']['release']['v0_8_0_tag_present']}"
    )
    print(
        "direct_heavy_imports="
        f"{len(payload['inventory']['direct_heavy_imports_in_core_reconstruction'])} "
        "heavy_backend_started=False"
    )
    print(f"G2_groups={len(payload['planned_g2']['groups'])} G3_groups={len(payload['planned_g3']['groups'])}")
    print(f"blockers={payload['blockers']}")
    print(f"summary={summary}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
