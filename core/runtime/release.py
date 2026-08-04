"""Phase 6 release-gate diagnostics and atomic report helpers."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import tempfile
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from core.version import RELEASE_NAME, RELEASE_PHASE, __version__

_REQUIRED_PROJECT_FILES = (
    "main.py",
    "pyproject.toml",
    "README.md",
    "app/main_window.py",
    "app/inference_controller.py",
    "app/widgets/performance_bar.py",
    "core/benchmark/assets.py",
    "core/benchmark/reports.py",
    "core/benchmark/runner.py",
    "core/inference/worker.py",
    "core/models/registry.py",
    "core/runtime/logging_setup.py",
    "core/runtime/release.py",
    "core/version.py",
    "scripts/check_hardware.py",
    "scripts/release_check.py",
    "scripts/run_phase6_benchmark.py",
    "scripts/run_phase6_release.sh",
    "scripts/run_phase6_step7_tests.sh",
    "scripts/run_phase5_tests.sh",
    "PHASE6_STEP7_VALIDATION.md",
    "TEST_STRATEGY_PHASE6_RELEASE.md",
)

_BASE_DEPENDENCIES = (
    ("PyQt6", "PyQt6"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("vispy", "vispy"),
)


@dataclass(frozen=True, slots=True)
class ReleaseCheck:
    """One deterministic release-gate result."""

    name: str
    passed: bool
    detail: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Captured subprocess result included in the release report."""

    name: str
    command: tuple[str, ...]
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str
    required: bool = True

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def collect_static_checks(
    root: str | os.PathLike[str] | None = None,
    *,
    require_model_dependencies: bool = True,
) -> list[ReleaseCheck]:
    """Collect dependency-light checks that never initialize CUDA."""

    root_path = Path(root) if root is not None else project_root()
    checks: list[ReleaseCheck] = []
    checks.append(_python_version_check())
    checks.extend(_layout_checks(root_path))
    checks.append(version_consistency_check(root_path))
    checks.extend(_dependency_checks(_BASE_DEPENDENCIES, required=True))
    checks.extend(_model_registry_checks(required=require_model_dependencies))
    checks.extend(_writable_directory_checks(root_path))
    return checks


def version_consistency_check(root: str | os.PathLike[str] | Path) -> ReleaseCheck:
    """Verify pyproject, config and runtime version metadata are identical."""

    root_path = Path(root)
    discovered: dict[str, str] = {"runtime": __version__}

    try:
        pyproject = tomllib.loads((root_path / "pyproject.toml").read_text(encoding="utf-8"))
        discovered["pyproject"] = str(pyproject["project"]["version"])
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck("version_consistency", False, f"Cannot read pyproject version: {exc}")

    config_path = root_path / "config" / "app.yaml"
    try:
        config_text = config_path.read_text(encoding="utf-8")
        match = re.search(r"^\s*version:\s*[\"']?([^\"'\s#]+)", config_text, re.MULTILINE)
        if match is None:
            raise ValueError("app.version is missing")
        discovered["config"] = match.group(1)
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck("version_consistency", False, f"Cannot read config version: {exc}")

    unique = set(discovered.values())
    detail = ", ".join(f"{key}={value}" for key, value in sorted(discovered.items()))
    return ReleaseCheck("version_consistency", len(unique) == 1, detail)


def benchmark_report_check(
    path: str | os.PathLike[str],
    *,
    require_all_models: bool = True,
) -> tuple[ReleaseCheck, dict[str, object] | None]:
    """Validate a Phase 6 benchmark report without rerunning inference."""

    report_path = Path(path)
    if not report_path.is_file():
        return ReleaseCheck(
            "phase6_benchmark_report",
            False,
            f"missing: {report_path}",
        ), None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        application = payload.get("application", {})
        results = payload.get("results", [])
        if not isinstance(results, list) or not results:
            raise ValueError("report contains no model results")
        if str(application.get("version", "")) != __version__:
            raise ValueError(
                f"report version {application.get('version')!r} does not match {__version__!r}"
            )
        from core.benchmark.runner import production_benchmark_descriptors

        required_ids = {item.model_id for item in production_benchmark_descriptors()}
        result_by_id = {
            str(item.get("model_id", "")): item
            for item in results
            if isinstance(item, dict)
        }
        missing = sorted(required_ids - set(result_by_id))
        nonpassing = sorted(
            model_id
            for model_id in required_ids
            if model_id in result_by_id
            and result_by_id[model_id].get("status") != "passed"
        )
        passed = not missing and (not require_all_models or not nonpassing)
        details = [f"{len(required_ids) - len(missing)} / {len(required_ids)} models reported"]
        if missing:
            details.append("missing: " + ", ".join(missing))
        if nonpassing:
            details.append("non-passing: " + ", ".join(nonpassing))
        return ReleaseCheck(
            "phase6_benchmark_report",
            passed,
            "; ".join(details),
        ), payload
    except Exception as exc:  # noqa: BLE001
        return ReleaseCheck(
            "phase6_benchmark_report",
            False,
            f"invalid report: {exc}",
        ), None


def build_release_report(
    *,
    checks: Iterable[ReleaseCheck],
    commands: Iterable[CommandResult],
    mode: str,
    git: dict[str, object] | None = None,
    benchmark: dict[str, object] | None = None,
) -> dict[str, object]:
    check_list = list(checks)
    command_list = list(commands)
    required_passed = all(check.passed or not check.required for check in check_list) and all(
        command.passed or not command.required for command in command_list
    )
    return {
        "schema_version": 2,
        "application": {
            "name": "EIA TECH | Depth & 3D Model Lab",
            "version": __version__,
            "release_name": RELEASE_NAME,
            "release_phase": RELEASE_PHASE,
        },
        "generated_at_unix": time.time(),
        "mode": mode,
        "passed": required_passed,
        "environment": environment_snapshot(),
        "checks": [asdict(item) for item in check_list],
        "commands": [
            {
                **asdict(item),
                "command": list(item.command),
                "passed": item.passed,
            }
            for item in command_list
        ],
        "benchmark": benchmark or {},
        "git": git or {},
    }


def write_release_report(
    path: str | os.PathLike[str],
    payload: dict[str, object],
) -> Path:
    """Write a UTF-8 JSON report atomically."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def environment_snapshot() -> dict[str, object]:
    dependencies = {distribution for _, distribution in _BASE_DEPENDENCIES}
    try:
        from core.models.registry import iter_model_descriptors

        for descriptor in iter_model_descriptors(enabled_only=True):
            dependencies.update(item.distribution for item in descriptor.dependencies)
    except Exception:
        pass
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "process_id": os.getpid(),
        "dependencies": {
            distribution: _distribution_version(distribution)
            for distribution in sorted(dependencies)
        },
    }


def release_failed(
    checks: Iterable[ReleaseCheck],
    commands: Iterable[CommandResult],
) -> bool:
    return any(not check.passed and check.required for check in checks) or any(
        not command.passed and command.required for command in commands
    )


def _python_version_check() -> ReleaseCheck:
    supported = (3, 11) <= sys.version_info[:2] < (3, 13)
    return ReleaseCheck(
        "python_version",
        supported,
        f"Python {sys.version.split()[0]} via {sys.executable}",
    )


def _layout_checks(root: Path) -> list[ReleaseCheck]:
    missing = [relative for relative in _REQUIRED_PROJECT_FILES if not (root / relative).is_file()]
    checks = [
        ReleaseCheck(
            "project_layout",
            not missing,
            "all required files present" if not missing else f"missing: {', '.join(missing)}",
        )
    ]
    manifest = root / "DELETE_THESE_FILES.txt"
    checks.append(
        ReleaseCheck(
            "delivery_manifest_removed",
            not manifest.exists(),
            "delivery manifest absent" if not manifest.exists() else str(manifest),
        )
    )
    return checks


def _dependency_checks(
    dependencies: Iterable[tuple[str, str]],
    *,
    required: bool,
) -> list[ReleaseCheck]:
    checks: list[ReleaseCheck] = []
    for module_name, distribution_name in dependencies:
        available = _module_available(module_name)
        version = _distribution_version(distribution_name)
        detail = (
            f"{distribution_name} {version or '(version unavailable)'}"
            if available
            else f"module {module_name!r} not found"
        )
        checks.append(
            ReleaseCheck(
                f"dependency_{module_name.replace('.', '_')}",
                available,
                detail,
                required=required,
            )
        )
    return checks


def _model_registry_checks(*, required: bool) -> list[ReleaseCheck]:
    try:
        from core.benchmark.runner import production_benchmark_descriptors
        from core.models.registry import check_model_availability
    except Exception as exc:  # noqa: BLE001
        return [ReleaseCheck("model_registry", False, str(exc), required=required)]

    checks: list[ReleaseCheck] = []
    seen_dependencies: set[str] = {module for module, _ in _BASE_DEPENDENCIES}
    for descriptor in production_benchmark_descriptors():
        for dependency in descriptor.dependencies:
            if dependency.import_name in seen_dependencies:
                continue
            seen_dependencies.add(dependency.import_name)
            checks.extend(
                _dependency_checks(
                    ((dependency.import_name, dependency.distribution),),
                    required=required,
                )
            )
        availability = check_model_availability(descriptor.model_id)
        asset_detail = (
            "available"
            if not availability.missing_assets
            else "missing: " + ", ".join(availability.missing_assets)
        )
        checks.append(
            ReleaseCheck(
                f"model_{descriptor.model_id.replace('-', '_')}_assets",
                not availability.missing_assets,
                asset_detail,
                required=required,
            )
        )
    return checks


def _writable_directory_checks(root: Path) -> list[ReleaseCheck]:
    results: list[ReleaseCheck] = []
    for relative in ("logs", "outputs/release", "outputs/stress", "outputs/phase6"):
        directory = root / relative
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".depth3d-write-test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            passed = True
            detail = f"writable: {directory}"
        except Exception as exc:  # noqa: BLE001
            passed = False
            detail = f"not writable: {directory}: {exc}"
        results.append(ReleaseCheck(f"writable_{relative.replace('/', '_')}", passed, detail))
    return results


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None
