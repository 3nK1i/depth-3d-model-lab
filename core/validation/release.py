"""Phase 7.5 release-specific static and artifact checks."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from core.runtime.release import ReleaseCheck

_FORBIDDEN_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".plan"}
_FORBIDDEN_DIR_NAMES = {"__pycache__", ".pytest_cache", "logs", "outputs"}

_PHASE75_REQUIRED_FILES = (
    "core/validation/__init__.py",
    "core/validation/known_size.py",
    "core/validation/release.py",
    "scripts/create_known_size_benchmark_template.py",
    "scripts/run_known_size_validation.py",
    "scripts/run_known_size_validation_smoke.py",
    "scripts/phase7_5_release_check.py",
    "scripts/run_phase7_5_release.sh",
    "scripts/run_phase7_5_step10_tests.sh",
    "KNOWN_SIZE_BENCHMARK_GUIDE.md",
    "TEST_STRATEGY_PHASE7_5_RELEASE.md",
    "PHASE7_5_STEP10_VALIDATION.md",
)


def phase75_layout_check(root: str | Path) -> ReleaseCheck:
    """Verify that the final Phase 7.5 validation surface is complete."""

    root_path = Path(root)
    missing = [relative for relative in _PHASE75_REQUIRED_FILES if not (root_path / relative).is_file()]
    detail = "all Phase 7.5 release files present" if not missing else "missing: " + ", ".join(missing)
    return ReleaseCheck("phase7_5_project_layout", not missing, detail)


def forbidden_release_artifacts_check(root: str | Path) -> ReleaseCheck:
    """Reject tracked/package-local checkpoints and generated runtime artifacts.

    In a Git checkout only tracked files are inspected, so harmless local logs or
    cached model weights do not block the release gate.  In an archive without
    Git metadata, the archive tree itself is inspected.
    """

    root_path = Path(root)
    candidates: list[Path] = []
    using_git_index = False
    if (root_path / ".git").exists():
        try:
            completed = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=root_path,
                capture_output=True,
                check=False,
                timeout=15,
            )
            if completed.returncode == 0:
                using_git_index = True
                candidates = [
                    root_path / item.decode("utf-8", errors="replace")
                    for item in completed.stdout.split(b"\0")
                    if item
                ]
        except (OSError, subprocess.TimeoutExpired):
            candidates = []
    if not candidates:
        candidates = [path for path in root_path.rglob("*") if path.is_file()]

    offenders: list[str] = []
    for path in candidates:
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            continue
        if any(part in _FORBIDDEN_DIR_NAMES for part in relative.parts):
            # Imports/tests can create cache directories in an extracted archive.
            # They are release blockers only when actually tracked by Git.
            cache_only = any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts)
            if using_git_index or not cache_only:
                offenders.append(str(relative))
            continue
        if path.suffix.lower() in _FORBIDDEN_SUFFIXES:
            offenders.append(str(relative))
    detail = "clean" if not offenders else "forbidden artifacts: " + ", ".join(sorted(offenders)[:12])
    return ReleaseCheck("forbidden_release_artifacts", not offenders, detail)


def known_size_report_check(path: str | Path) -> tuple[ReleaseCheck, dict[str, object] | None]:
    """Validate a finalized known-size benchmark report."""

    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / "known_size_validation.json"
    if not report_path.is_file():
        return ReleaseCheck("known_size_validation", False, f"missing: {report_path}"), None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("unsupported report schema")
        if not payload.get("thresholds_confirmed"):
            raise ValueError("acceptance thresholds are not confirmed")
        if not payload.get("passed"):
            raise ValueError("known-size benchmark did not pass")
        case_count = int(summary.get("case_count", 0))
        dimension_count = int(summary.get("available_dimension_count", 0))
        if case_count <= 0 or dimension_count <= 0:
            raise ValueError("report contains no usable measurements")
        detail = (
            f"{case_count} cases · {dimension_count} dimensions · "
            f"MAE={float(summary.get('mean_absolute_error_mm', 0.0)):.2f} mm · "
            f"coverage={float(summary.get('uncertainty_coverage_percent', 0.0)):.1f}%"
        )
        return ReleaseCheck("known_size_validation", True, detail), payload
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck("known_size_validation", False, f"invalid report: {exc}"), None
