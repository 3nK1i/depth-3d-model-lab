#!/usr/bin/env python3
"""Strict repository qualification for the Phase 8.9I2 v0.8.0 source transition."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "f959c369f8aa27347cab20cc90b469c6674d0840"
FINAL_VERSION = "0.8.0"
FINAL_RELEASE_NAME = "Phase 8 Reconstruction Stable"
FINAL_RELEASE_PHASE = 8.0

EXPECTED_OVERLAY = {
    "README.md",
    "RELEASE_NOTES_v0.8.0.md",
    "config/app.yaml",
    "core/version.py",
    "docs/phase_history/PHASE8_9I1_FINAL_RELEASE_PREFLIGHT.md",
    "docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md",
    "pyproject.toml",
    "scripts/run_phase8_9i2_release_transition.py",
    "scripts/run_phase8_9i2_tests.sh",
    "tests/ui/test_phase7_release_modes.py",
    "tests/integration/test_phase8_9g1_regression_contract.py",
    "tests/integration/test_phase8_9g2_full_regression_contract.py",
    "tests/unit/test_phase6_release_metadata.py",
    "tests/unit/test_phase75_step10_backward_compatibility.py",
    "tests/unit/test_phase75_step1_backward_compatibility.py",
    "tests/unit/test_phase75_step2_backward_compatibility.py",
    "tests/unit/test_phase75_step3_backward_compatibility.py",
    "tests/unit/test_phase75_step4_backward_compatibility.py",
    "tests/unit/test_phase75_step5_backward_compatibility.py",
    "tests/unit/test_phase75_step6_backward_compatibility.py",
    "tests/unit/test_phase75_step7_backward_compatibility.py",
    "tests/unit/test_phase75_step8_backward_compatibility.py",
    "tests/unit/test_phase7_release_metadata.py",
    "tests/unit/test_phase8_1_backward_compatibility.py",
    "tests/unit/test_phase8_2_backward_compatibility.py",
    "tests/unit/test_phase8_3b_backward_compatibility.py",
    "tests/unit/test_phase8_3c_backward_compatibility.py",
    "tests/unit/test_phase8_3d_backward_compatibility.py",
    "tests/unit/test_phase8_4a_backward_compatibility.py",
    "tests/unit/test_phase8_7g4_readiness.py",
    "tests/unit/test_phase8_8d5_contract.py",
    "tests/unit/test_phase8_9g1_regression_inventory.py",
    "tests/unit/test_phase8_9g2_full_regression.py",
    "tests/unit/test_phase8_9g3_backward_compatibility.py",
    "tests/unit/test_phase8_9h2_release_prep_contract.py",
    "tests/unit/test_phase8_9h3_release_candidate_contract.py",
    "tests/unit/test_phase8_9i2_release_transition_contract.py",
    "tests/unit/test_release_metadata.py",
}

PROTECTED_HISTORICAL_HASHES = {
    "main.py": "1b8a6f67c9f58e4fdd0d7c35c8a58cd7b133080d32d4449ea0c9047a9c4785f0",
    "core/runtime/release.py": "222b9f47200caf4a192d1db5ca741150d1ab5dfb270f1681cdcd42c3af31abbf",
    "scripts/release_check.py": "b70eef840660ff895ced038b9211ca1cdc28020130c24e6bd16d50845c50a72a",
    "scripts/run_phase8_9g1_regression_inventory.py": "0f190bbdb57bd56feeb04bf63dd037760ec18eaaf6ca63ebd5931ded8069e76a",
    "scripts/run_phase8_9g2_full_regression.py": "33c453ef3dbd84560a296692a2cba9066284249a5a0a87dd1bacc1ddf0a9eb82",
    "scripts/run_phase8_9g3_backward_compatibility.py": "a1db5661fef02d3bafa432be005c1bb75c31afdd6eb7e65f92b013f775127620",
    "scripts/run_phase8_9h2_release_prep_tests.sh": "32493ee5092a1ed5f3e0453fbcb56929f510164967631c42271aac5126e66c7d",
    "scripts/run_phase8_9h3_release_candidate_qualification.py": "f030f00c3bace5cbaf78844c2d03e352620e4bc5705003aa68d1ada581ae3960",
    "scripts/run_phase8_9h3_release_candidate_tests.sh": "8e41a6ddb2cd2a248dc8a599bfaf56eb26fce2d67aa05ebd98845b4974dd5c66",
    "docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md": "8adc45bafcb202ac2ca7aa83e4b78c86f267c4fb244533b3d204c76375af62e7",
    "docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md": "76640c79409e5b82d142bc10be5756300d8f4aa00bfd4cdd09dee680d58113e4",
    "docs/phase_history/PHASE8_9H3_RELEASE_CANDIDATE_QUALIFICATION.md": "699a3394a5910762cc46667ea8a21621444f9dc64eea430026dae0e2dee22a9a",
}

CURRENT_IDENTITY_TESTS = {
    "tests/ui/test_phase7_release_modes.py",
    "tests/unit/test_release_metadata.py",
    "tests/unit/test_phase6_release_metadata.py",
    "tests/unit/test_phase7_release_metadata.py",
    "tests/unit/test_phase75_step1_backward_compatibility.py",
    "tests/unit/test_phase75_step2_backward_compatibility.py",
    "tests/unit/test_phase75_step3_backward_compatibility.py",
    "tests/unit/test_phase75_step4_backward_compatibility.py",
    "tests/unit/test_phase75_step5_backward_compatibility.py",
    "tests/unit/test_phase75_step6_backward_compatibility.py",
    "tests/unit/test_phase75_step7_backward_compatibility.py",
    "tests/unit/test_phase75_step8_backward_compatibility.py",
    "tests/unit/test_phase75_step10_backward_compatibility.py",
    "tests/unit/test_phase8_1_backward_compatibility.py",
    "tests/unit/test_phase8_2_backward_compatibility.py",
    "tests/unit/test_phase8_3b_backward_compatibility.py",
    "tests/unit/test_phase8_3c_backward_compatibility.py",
    "tests/unit/test_phase8_3d_backward_compatibility.py",
    "tests/unit/test_phase8_4a_backward_compatibility.py",
    "tests/unit/test_phase8_7g4_readiness.py",
    "tests/unit/test_phase8_8d5_contract.py",
}

POST_TRANSITION_HISTORICAL_CLI_CONTRACTS = {
    "tests/integration/test_phase8_9g1_regression_contract.py",
    "tests/integration/test_phase8_9g2_full_regression_contract.py",
}


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def _git(*args: str, check: bool = True) -> str:
    return _run("git", *args, check=check).stdout.strip()


def _status_paths() -> list[str]:
    paths: list[str] = []
    # Preserve the two-column porcelain prefix exactly. _git() intentionally
    # strips command output for scalar Git queries; using it here would remove
    # the leading space from the first " M <path>" record and corrupt a path
    # such as README.md into EADME.md.
    output = _run("git", "status", "--short", "--untracked-files=all").stdout
    for line in output.splitlines():
        if len(line) < 4:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        paths.append(value.strip())
    return sorted(paths)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_metadata() -> dict[str, Any]:
    version_text = (ROOT / "core/version.py").read_text(encoding="utf-8")
    tree = ast.parse(version_text)
    values: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {"__version__", "RELEASE_NAME", "RELEASE_PHASE"}:
            if isinstance(node.value, ast.Constant):
                values[target.id] = node.value.value

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    app_text = (ROOT / "config/app.yaml").read_text(encoding="utf-8")
    config_version = re.search(r'^\s*version:\s*["\']?([^"\'\s#]+)', app_text, re.MULTILINE)
    config_release = re.search(r'^\s*release:\s*["\']?([^\n#"\']+)', app_text, re.MULTILINE)
    return {
        **values,
        "pyproject_version": str(pyproject["project"]["version"]),
        "pyproject_description": str(pyproject["project"].get("description", "")),
        "config_version": config_version.group(1).strip() if config_version else None,
        "config_release": config_release.group(1).strip() if config_release else None,
    }


def _old_runtime_assertions() -> list[str]:
    patterns = (
        re.compile(r"assert\s+__version__\s*==\s*['\"]0\.7\.5['\"]"),
        re.compile(r"assert\s+RELEASE_NAME\s*==\s*['\"]Phase 7\.5 Stable['\"]"),
        re.compile(r"assert\s+RELEASE_PHASE\s*==\s*7\.5"),
        re.compile(r"assert\s+APP_VERSION[^\n]*['\"]v0\.7\.5['\"]"),
    )
    findings: list[str] = []
    for relative in sorted(CURRENT_IDENTITY_TESTS):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                findings.append(f"{relative}:{line_no}:{line.strip()}")
    return findings


def qualify(*, strict_git: bool = True) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    blockers: list[str] = []

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = bool(passed)
        if detail is not None:
            details[name] = detail
        if not passed:
            blockers.append(name)

    metadata = _version_metadata()
    record(
        "metadata_final_0_8_0",
        metadata.get("__version__") == FINAL_VERSION
        and metadata.get("RELEASE_NAME") == FINAL_RELEASE_NAME
        and float(metadata.get("RELEASE_PHASE", -1)) == FINAL_RELEASE_PHASE
        and metadata.get("pyproject_version") == FINAL_VERSION
        and metadata.get("config_version") == FINAL_VERSION
        and metadata.get("config_release") == FINAL_RELEASE_NAME,
        metadata,
    )

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    notes = (ROOT / "RELEASE_NOTES_v0.8.0.md").read_text(encoding="utf-8")
    record(
        "release_docs_finalized",
        "Version `0.8.0` is the **Phase 8 Reconstruction Stable** release" in readme
        and "The `v0.8.0` release includes:" in readme
        and "runtime/package metadata intentionally remains `0.7.5`" not in readme
        and "Release status" in notes
        and "Release-preparation status" not in notes
        and "0.8.0` / `Phase 8 Reconstruction Stable" in notes
        and "8.9I2" in notes
        and "8.9I3" in notes,
    )
    record(
        "authority_boundaries_preserved",
        all(
            phrase in readme
            for phrase in (
                "canonical geometry is the authoritative scene geometry",
                "semantic geometry supplies semantic/object context",
                "Gaussian and NeRF/Nerfacto outputs are **appearance only**",
                "`certified_metrology=false`",
                "camera intrinsics alone do not establish metric scale",
                "heavy reconstruction, Gaussian and NeRF runtimes remain outside the production `depth` environment",
            )
        )
        and all(
            phrase in notes
            for phrase in (
                "**Canonical geometry** is the authoritative scene geometry",
                "**Gaussian and NeRF/Nerfacto representations are appearance only.**",
                "`certified_metrology=false`",
                "camera intrinsics alone do not establish metric scale",
            )
        ),
    )

    protected_bad: dict[str, str] = {}
    for relative, expected in PROTECTED_HISTORICAL_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else "<missing>"
        if actual != expected:
            protected_bad[relative] = actual
    record("historical_release_evidence_immutable", not protected_bad, protected_bad)

    stale_assertions = _old_runtime_assertions()
    record("current_identity_tests_transitioned", not stale_assertions, stale_assertions)

    historical_cli_sources = {
        relative: (ROOT / relative).read_text(encoding="utf-8")
        for relative in sorted(POST_TRANSITION_HISTORICAL_CLI_CONTRACTS)
    }
    historical_cli_ok = (
        'assert "release_version_still_0_7_5" in blockers' in historical_cli_sources[
            "tests/integration/test_phase8_9g1_regression_contract.py"
        ]
        and 'assert release["core_version"] == "0.8.0"' in historical_cli_sources[
            "tests/integration/test_phase8_9g1_regression_contract.py"
        ]
        and 'assert "repository_preflight:version_0_7_5" in blockers' in historical_cli_sources[
            "tests/integration/test_phase8_9g2_full_regression_contract.py"
        ]
        and 'assert release["core_version"] == "0.8.0"' in historical_cli_sources[
            "tests/integration/test_phase8_9g2_full_regression_contract.py"
        ]
    )
    record(
        "historical_g1_g2_cli_contracts_transitioned",
        historical_cli_ok,
        sorted(POST_TRANSITION_HISTORICAL_CLI_CONTRACTS),
    )

    i1 = ROOT / "docs/phase_history/PHASE8_9I1_FINAL_RELEASE_PREFLIGHT.md"
    i2 = ROOT / "docs/phase_history/PHASE8_9I2_RELEASE_TRANSITION.md"
    record(
        "i1_i2_records_present",
        i1.is_file()
        and i2.is_file()
        and EXPECTED_HEAD in i1.read_text(encoding="utf-8")
        and "Final identity" in i2.read_text(encoding="utf-8"),
    )

    if strict_git:
        branch = _git("branch", "--show-current")
        head = _git("rev-parse", "HEAD")
        origin = _git("rev-parse", f"origin/{EXPECTED_BRANCH}")
        status_paths = _status_paths()
        diff_check = _run("git", "diff", "--check", check=False)
        local_tag = _run("git", "rev-parse", "-q", "--verify", "refs/tags/v0.8.0", check=False)
        remote_tag = _git("ls-remote", "--tags", "origin", "refs/tags/v0.8.0", "refs/tags/v0.8.0^{}", check=False)
        record("branch", branch == EXPECTED_BRANCH, branch)
        record("head", head == EXPECTED_HEAD, head)
        record("origin", origin == EXPECTED_HEAD, origin)
        record("overlay_exact", set(status_paths) == EXPECTED_OVERLAY, status_paths)
        record("diff_check_clean", diff_check.returncode == 0, diff_check.stderr.strip())
        record("v0_8_0_local_absent", local_tag.returncode != 0)
        record("v0_8_0_remote_absent", not remote_tag, remote_tag)
    else:
        details["strict_git"] = False

    return {
        "phase": "8.9I2",
        "pass": not blockers,
        "blockers": sorted(set(blockers)),
        "checks": checks,
        "details": details,
        "expected_overlay": sorted(EXPECTED_OVERLAY),
        "release_tag_created_by_i2": False,
        "production_reconstruction_implementation_changed_by_i2": False,
        "heavy_real_backend_campaign_started_by_i2": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--non-strict-git", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/depth3d_phase8_9i2_release_transition/summary.json"),
    )
    args = parser.parse_args()

    payload = qualify(strict_git=not args.non_strict_git)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.9I2 release-transition repository qualification: {'PASS' if payload['pass'] else 'FAIL'}")
    print(f"blockers={payload['blockers']}")
    print(f"summary={output}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
