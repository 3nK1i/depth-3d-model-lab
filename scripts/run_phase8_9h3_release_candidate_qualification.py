#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "dff62357874e28de8d0ceadb9a222dd0cd17d999"

H2_HASHES = {
    "README.md": "ee9b5936b0502a5c9bbe09cc62b0782a5e674d620d210e495d317e26d14c7539",
    "RELEASE_NOTES_v0.8.0.md": "d2b1f0e19a0f86ea7cd80ed306f87d33d7e5b230273ed7a58048823e98696ff5",
    "docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md": "8adc45bafcb202ac2ca7aa83e4b78c86f267c4fb244533b3d204c76375af62e7",
    "docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md": "76640c79409e5b82d142bc10be5756300d8f4aa00bfd4cdd09dee680d58113e4",
    "scripts/run_phase8_9h2_release_prep_tests.sh": "32493ee5092a1ed5f3e0453fbcb56929f510164967631c42271aac5126e66c7d",
    "tests/unit/test_phase8_9h2_release_prep_contract.py": "46fd581e30f6a856230d309d550469f5b7f998d5bd86a5c1bf42ace6a521628a",
}

PROTECTED_HASHES = {
    "core/version.py": "d3519af6d79762497f05de8499d6ea641a235fbc7e3d19b936ddf9e8654af4bb",
    "pyproject.toml": "88d1fa6b87e8c72720c1f453bde72b595abd4c337a42c705612f2a813c48fd9b",
    "config/app.yaml": "dabddbf14c00a8692993b00a2e6a5d1fadd130c4079814c92db94ef6cf431bc9",
    "main.py": "1b8a6f67c9f58e4fdd0d7c35c8a58cd7b133080d32d4449ea0c9047a9c4785f0",
}

EXPECTED_SCOPE = sorted(
    [
        *H2_HASHES,
        "docs/phase_history/PHASE8_9H3_RELEASE_CANDIDATE_QUALIFICATION.md",
        "scripts/run_phase8_9h3_release_candidate_qualification.py",
        "scripts/run_phase8_9h3_release_candidate_tests.sh",
        "tests/unit/test_phase8_9h3_release_candidate_contract.py",
    ]
)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def check(condition: bool, name: str, blockers: list[str]) -> bool:
    if not condition:
        blockers.append(name)
    return condition


def inspect_repository() -> dict[str, Any]:
    blockers: list[str] = []
    checks: dict[str, bool] = {}

    branch = run("git", "branch", "--show-current").stdout.strip()
    head = run("git", "rev-parse", "HEAD").stdout.strip()
    remote = run("git", "ls-remote", "origin", "refs/heads/phase8-reconstruction").stdout.split()
    remote_head = remote[0] if remote else ""

    checks["branch"] = check(branch == EXPECTED_BRANCH, "wrong_branch", blockers)
    checks["head"] = check(head == EXPECTED_HEAD, "unexpected_head", blockers)
    checks["origin"] = check(remote_head == EXPECTED_HEAD, "origin_moved", blockers)

    diff_check = run("git", "diff", "--check", check=False)
    checks["diff_check_clean"] = check(diff_check.returncode == 0, "git_diff_check_failed", blockers)

    status = run("git", "status", "--porcelain=v1").stdout.splitlines()
    actual_scope = sorted(line[3:] for line in status if len(line) >= 4)
    checks["overlay_exact"] = check(actual_scope == EXPECTED_SCOPE, "overlay_scope_mismatch", blockers)

    local_tag = run("git", "rev-parse", "-q", "--verify", "refs/tags/v0.8.0", check=False)
    checks["v0_8_0_local_absent"] = check(local_tag.returncode != 0, "local_v0.8.0_exists", blockers)
    remote_tag = run(
        "git",
        "ls-remote",
        "--tags",
        "origin",
        "refs/tags/v0.8.0",
        "refs/tags/v0.8.0^{}",
    ).stdout.strip()
    checks["v0_8_0_remote_absent"] = check(not remote_tag, "remote_v0.8.0_exists", blockers)

    h2_hash_results: dict[str, bool] = {}
    for rel, expected in H2_HASHES.items():
        ok = (ROOT / rel).is_file() and sha256(ROOT / rel) == expected
        h2_hash_results[rel] = ok
        check(ok, f"h2_hash_mismatch:{rel}", blockers)
    checks["h2_overlay_immutable"] = all(h2_hash_results.values())

    protected_hash_results: dict[str, bool] = {}
    for rel, expected in PROTECTED_HASHES.items():
        ok = (ROOT / rel).is_file() and sha256(ROOT / rel) == expected
        protected_hash_results[rel] = ok
        check(ok, f"protected_hash_mismatch:{rel}", blockers)
    checks["protected_release_files_unchanged"] = all(protected_hash_results.values())

    version_py = read("core/version.py")
    pyproject = read("pyproject.toml")
    app_yaml = read("config/app.yaml")
    metadata_ok = all(
        [
            '__version__ = "0.7.5"' in version_py,
            'RELEASE_NAME = "Phase 7.5 Stable"' in version_py,
            "RELEASE_PHASE = 7.5" in version_py,
            'version = "0.7.5"' in pyproject,
            'version: "0.7.5"' in app_yaml,
            'release: "Phase 7.5 Stable"' in app_yaml,
            '__version__ = "0.8.0"' not in version_py,
        ]
    )
    checks["metadata_still_0_7_5"] = check(metadata_ok, "metadata_transitioned_too_early", blockers)

    readme = read("README.md")
    notes = read("RELEASE_NOTES_v0.8.0.md")
    docs_ok = all(
        [
            "target `v0.8.0` (Phase 8 Reconstruction)" in readme,
            "runtime/package metadata intentionally remains `0.7.5`" in readme,
            "## Phase 8 Reconstruction" in readme,
            "canonical geometry is the authoritative scene geometry" in readme,
            "Gaussian and NeRF/Nerfacto outputs are **appearance only**" in readme,
            "camera intrinsics alone do not establish metric scale" in readme,
            "post-`v0.8.0`" in readme,
            "planned after Phase 8" not in readme,
            "Release-preparation status" in notes,
            "the `v0.8.0` tag must remain absent" in notes,
            "Gaussian and NeRF/Nerfacto representations are appearance only" in notes,
            "Known limitations and deliberate exclusions" in notes,
            "8.9I" in notes,
        ]
    )
    checks["release_docs_consistent"] = check(docs_ok, "release_documentation_inconsistent", blockers)

    required_paths = [
        "KNOWN_SIZE_BENCHMARK_GUIDE.md",
        "docs/test_strategy/TEST_STRATEGY_PHASE7_5_RELEASE.md",
        "docs/phase_history/PHASE8_9G2_VALIDATION.md",
        "docs/phase_history/PHASE8_9G3_VALIDATION.md",
        "docs/phase_history/PHASE8_9_PRE_H_FINAL_POSTFIX_REQUALIFICATION.md",
        "scripts/run_phase8_9_pre_h_final_requalification_tests.sh",
    ]
    refs_ok = all((ROOT / path).exists() for path in required_paths)
    checks["release_evidence_paths_present"] = check(refs_ok, "release_evidence_path_missing", blockers)

    return {
        "pass": not blockers,
        "blockers": blockers,
        "checks": checks,
        "branch": branch,
        "head": head,
        "origin": remote_head,
        "actual_scope": actual_scope,
        "expected_scope": EXPECTED_SCOPE,
        "h2_hashes": h2_hash_results,
        "protected_hashes": protected_hash_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/eia_phase8_9h3_release_candidate/summary.json")
    args = parser.parse_args()

    result = inspect_repository()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Phase 8.9H3 release-candidate repository qualification: {'PASS' if result['pass'] else 'FAIL'}")
    print(f"blockers={result['blockers']}")
    print(f"summary={output}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
