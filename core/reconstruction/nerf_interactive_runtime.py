"""Phase 9.5A interactive NeRF runtime-isolation foundation.

This module promotes the already-qualified Phase 8 Nerfstudio/Nerfacto backend
contracts into one explicit Phase 9 interactive-runtime boundary.  It performs
read-only capability qualification only: no NeRF training, optimization,
checkpoint authoring, or rendering is started here.

NeRF remains an appearance-only representation.  Canonical reconstruction
geometry retains geometry, picking, scale, measurement, and QC authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import importlib.util
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterable

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.nerf import (
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_PYTHON,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfEnvironmentProbe,
    NerfSceneRole,
    nerf_backend_provenance,
    probe_nerf_environment,
)

PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION = 1
PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
PHASE9_5A_TCNN_QUALIFIED_REVISION = TINY_CUDA_NN_E2_REVISION
PHASE9_5A_HEAVY_NERF_MODULES = ("nerfstudio", "tinycudann")


class NerfInteractiveRuntimeState(str, Enum):
    """Readiness state for the future interactive NeRF runtime."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class NerfInteractiveProcessPolicy:
    """Process-lifecycle policy that later Phase 9.5 workers must obey."""

    timeout_s: float = 3600.0
    terminate_grace_s: float = 2.0
    start_new_session: bool = True
    shell: bool = False
    terminate_signal: str = "SIGTERM"
    kill_signal: str = "SIGKILL"

    def __post_init__(self) -> None:
        timeout = float(self.timeout_s)
        grace = float(self.terminate_grace_s)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ReconstructionValidationError("NeRF process timeout_s must be finite and positive.")
        if not isfinite(grace) or grace <= 0.0:
            raise ReconstructionValidationError(
                "NeRF process terminate_grace_s must be finite and positive."
            )
        if not bool(self.start_new_session):
            raise ReconstructionValidationError(
                "Interactive NeRF workers must run in their own process group/session."
            )
        if bool(self.shell):
            raise ReconstructionValidationError("Interactive NeRF workers must never use shell=True.")
        if str(self.terminate_signal).strip() != "SIGTERM" or str(self.kill_signal).strip() != "SIGKILL":
            raise ReconstructionValidationError(
                "Interactive NeRF cleanup must use SIGTERM followed by SIGKILL escalation."
            )
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "terminate_grace_s", grace)
        object.__setattr__(self, "start_new_session", True)
        object.__setattr__(self, "shell", False)
        object.__setattr__(self, "terminate_signal", "SIGTERM")
        object.__setattr__(self, "kill_signal", "SIGKILL")

    def as_dict(self) -> dict[str, Any]:
        return {
            "timeout_s": self.timeout_s,
            "terminate_grace_s": self.terminate_grace_s,
            "start_new_session": self.start_new_session,
            "shell": self.shell,
            "termination_order": [self.terminate_signal, self.kill_signal],
        }


@dataclass(frozen=True, slots=True)
class NerfInteractiveRuntimeSpec:
    """Qualified external-runtime locations for Phase 9.5 interactive NeRF work."""

    python_executable: str
    nerfstudio_root: str
    tiny_cuda_nn_root: str
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    expected_nerfstudio_revision: str = PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION
    expected_tiny_cuda_nn_revision: str = PHASE9_5A_TCNN_QUALIFIED_REVISION
    probe_timeout_s: float = 60.0
    process_policy: NerfInteractiveProcessPolicy = NerfInteractiveProcessPolicy()
    schema_version: int = PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        python = Path(str(self.python_executable))
        nerfstudio = Path(str(self.nerfstudio_root))
        tcnn = Path(str(self.tiny_cuda_nn_root))
        if not python.is_absolute() or not nerfstudio.is_absolute() or not tcnn.is_absolute():
            raise ReconstructionValidationError(
                "Interactive NeRF Python and checkout paths must be absolute."
            )
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError(
                "Phase 9.5 interactive runtime specs must reference real isolated Nerfstudio/Nerfacto."
            )
        ns_revision = str(self.expected_nerfstudio_revision).strip().lower()
        tcnn_revision = str(self.expected_tiny_cuda_nn_revision).strip().lower()
        for value, label in ((ns_revision, "expected_nerfstudio_revision"), (tcnn_revision, "expected_tiny_cuda_nn_revision")):
            if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
                raise ReconstructionValidationError(f"{label} must be an exact 40-character Git revision.")
        timeout = float(self.probe_timeout_s)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ReconstructionValidationError("probe_timeout_s must be finite and positive.")
        if int(self.schema_version) != PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Phase 9.5A NeRF runtime schema version.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "nerfstudio_root", str(nerfstudio))
        object.__setattr__(self, "tiny_cuda_nn_root", str(tcnn))
        object.__setattr__(self, "expected_nerfstudio_revision", ns_revision)
        object.__setattr__(self, "expected_tiny_cuda_nn_revision", tcnn_revision)
        object.__setattr__(self, "probe_timeout_s", timeout)
        object.__setattr__(self, "schema_version", PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION)

    @property
    def interpreter_is_external(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "python_executable": self.python_executable,
            "nerfstudio_root": self.nerfstudio_root,
            "tiny_cuda_nn_root": self.tiny_cuda_nn_root,
            "expected_nerfstudio_revision": self.expected_nerfstudio_revision,
            "expected_tiny_cuda_nn_revision": self.expected_tiny_cuda_nn_revision,
            "interpreter_is_external": self.interpreter_is_external,
            "probe_timeout_s": self.probe_timeout_s,
            "process_policy": self.process_policy.as_dict(),
        }




@contextmanager
def _isolated_runtime_paths(python_executable: str | Path):
    """Temporarily expose only the selected conda runtime paths to child probes."""

    python = Path(python_executable).expanduser().resolve()
    prefix = python.parent.parent
    if python.parent.name != "bin":
        raise ReconstructionValidationError(
            f"Cannot infer isolated NeRF environment prefix from {python}."
        )
    keys = (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CUDA_HOME",
        "CUDACXX",
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PYTHONNOUSERSITE",
    )
    previous = {key: os.environ.get(key) for key in keys}

    def prepend(key: str, values: list[Path]) -> None:
        parts = [str(value) for value in values if value.is_dir()]
        old = previous.get(key) or ""
        if old:
            parts.append(old)
        os.environ[key] = os.pathsep.join(parts)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)
    prepend(
        "LD_LIBRARY_PATH",
        [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"],
    )
    prepend(
        "LIBRARY_PATH",
        [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"],
    )
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def default_phase9_nerf_runtime_spec(home: str | Path | None = None) -> NerfInteractiveRuntimeSpec:
    root = Path.home() if home is None else Path(home)
    backend_root = root / "Desktop" / "eia-nerf-backends"
    return NerfInteractiveRuntimeSpec(
        python_executable=str(root / "anaconda3" / "envs" / "eia-nerf-nerfstudio" / "bin" / "python"),
        nerfstudio_root=str(backend_root / "nerfstudio"),
        tiny_cuda_nn_root=str(backend_root / "tiny-cuda-nn"),
    )


@dataclass(frozen=True, slots=True)
class NerfCheckoutEvidence:
    root: str
    exists: bool
    revision: str
    clean: bool
    status_lines: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        root = Path(str(self.root))
        if not root.is_absolute():
            raise ReconstructionValidationError("NeRF checkout evidence root must be absolute.")
        object.__setattr__(self, "root", str(root))
        object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(self, "revision", str(self.revision).strip().lower())
        object.__setattr__(self, "clean", bool(self.clean))
        object.__setattr__(self, "status_lines", tuple(str(line).rstrip() for line in self.status_lines if str(line).strip()))
        object.__setattr__(self, "error", str(self.error).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "exists": self.exists,
            "revision": self.revision,
            "clean": self.clean,
            "status_lines": list(self.status_lines),
            "error": self.error,
        }


def probe_nerf_checkout(root: str | Path, *, timeout_s: float = 10.0) -> NerfCheckoutEvidence:
    path = Path(root)
    if not path.is_absolute():
        raise ReconstructionValidationError("NeRF checkout root must be absolute.")
    if not path.is_dir() or not (path / ".git").exists():
        return NerfCheckoutEvidence(root=str(path), exists=False, revision="", clean=False, error="checkout_not_found")
    try:
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--short", "--untracked-files=all"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NerfCheckoutEvidence(
            root=str(path), exists=True, revision="", clean=False, error=f"{type(exc).__name__}: {exc}"
        )
    if revision.returncode != 0 or status.returncode != 0:
        detail = revision.stderr.strip() or status.stderr.strip() or "git_probe_failed"
        return NerfCheckoutEvidence(root=str(path), exists=True, revision="", clean=False, error=detail)
    lines = tuple(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    return NerfCheckoutEvidence(
        root=str(path),
        exists=True,
        revision=revision.stdout.strip().splitlines()[-1] if revision.stdout.strip() else "",
        clean=not lines,
        status_lines=lines,
    )


@dataclass(frozen=True, slots=True)
class NerfPythonRuntimeEvidence:
    python_executable: str
    python_version: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        python = Path(str(self.python_executable))
        if not python.is_absolute():
            raise ReconstructionValidationError("NeRF runtime python_executable must be absolute.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "python_version", str(self.python_version).strip())
        object.__setattr__(self, "error", str(self.error).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "error": self.error,
        }


def probe_nerf_python_runtime(python_executable: str | Path, *, timeout_s: float = 15.0) -> NerfPythonRuntimeEvidence:
    python = Path(python_executable)
    if not python.is_absolute():
        raise ReconstructionValidationError("NeRF runtime python_executable must be absolute.")
    if not python.is_file():
        return NerfPythonRuntimeEvidence(str(python), error="python_executable_missing")
    code = "import json,platform; print(json.dumps({'python_version': platform.python_version()}))"
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NerfPythonRuntimeEvidence(str(python), error=f"probe_failed:{type(exc).__name__}:{exc}")
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    if completed.returncode != 0 or not line:
        return NerfPythonRuntimeEvidence(str(python), error=completed.stderr.strip() or "probe_process_failed")
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return NerfPythonRuntimeEvidence(str(python), error=f"invalid_probe_json:{exc}")
    return NerfPythonRuntimeEvidence(str(python), python_version=str(payload.get("python_version", "")))


def current_process_nerf_heavy_modules(
    names: Iterable[str] = PHASE9_5A_HEAVY_NERF_MODULES,
    *,
    finder: Callable[[str], object | None] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Report discoverable NeRF-heavy modules without importing them."""

    found: list[str] = []
    for name in names:
        module = str(name).strip()
        if module and finder(module) is not None:
            found.append(module)
    return tuple(found)


def _base_version(value: str) -> str:
    return str(value).strip().split("+", 1)[0]


def _version_prefix_matches(actual: str, expected_major_minor: str) -> bool:
    actual_parts = _base_version(actual).split(".")
    expected_parts = _base_version(expected_major_minor).split(".")
    return actual_parts[: len(expected_parts)] == expected_parts


@dataclass(frozen=True, slots=True)
class NerfInteractiveRuntimeReport:
    state: NerfInteractiveRuntimeState
    reasons: tuple[str, ...]
    spec: NerfInteractiveRuntimeSpec
    environment_probe: NerfEnvironmentProbe
    python_runtime: NerfPythonRuntimeEvidence
    nerfstudio_checkout: NerfCheckoutEvidence
    tiny_cuda_nn_checkout: NerfCheckoutEvidence
    production_heavy_modules: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.state is NerfInteractiveRuntimeState.READY

    def as_dict(self) -> dict[str, Any]:
        provenance = nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)
        return {
            "schema_version": PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION,
            "state": self.state.value,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "spec": self.spec.as_dict(),
            "environment_probe": self.environment_probe.as_dict(),
            "python_runtime": self.python_runtime.as_dict(),
            "nerfstudio_checkout": self.nerfstudio_checkout.as_dict(),
            "tiny_cuda_nn_checkout": self.tiny_cuda_nn_checkout.as_dict(),
            "backend_provenance": provenance.as_dict(),
            "production_heavy_modules": list(self.production_heavy_modules),
            "capability_probe_only": True,
            "optimization_started": False,
            "training_performed": False,
            "rendering_performed": False,
            "nerf_artifact_created": False,
            "nerf_scene_role": NerfSceneRole.APPEARANCE_ONLY.value,
            "canonical_geometry_authoritative": False,
            "metric_scale_authoritative": False,
            "measurement_authoritative": False,
            "qc_authoritative": False,
            "certified_metrology": False,
            "cross_representation_fallback": False,
        }


def assess_nerf_interactive_runtime(
    spec: NerfInteractiveRuntimeSpec,
    *,
    environment_probe: NerfEnvironmentProbe,
    python_runtime: NerfPythonRuntimeEvidence,
    nerfstudio_checkout: NerfCheckoutEvidence,
    tiny_cuda_nn_checkout: NerfCheckoutEvidence,
    production_heavy_modules: Iterable[str] = (),
) -> NerfInteractiveRuntimeReport:
    """Assess probe evidence without launching training or rendering."""

    unavailable: list[str] = []
    rejected: list[str] = []
    heavy = tuple(str(name).strip() for name in production_heavy_modules if str(name).strip())

    if spec.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
        rejected.append("interactive_backend_must_be_nerfstudio_nerfacto")
    if not spec.interpreter_is_external or not environment_probe.interpreter_is_isolated:
        rejected.append("external_interpreter_required")
    if heavy:
        rejected.append("heavy_nerf_modules_present_in_production_runtime")

    if not Path(spec.python_executable).is_file():
        unavailable.append("external_python_missing")

    for label, evidence, expected in (
        ("nerfstudio", nerfstudio_checkout, spec.expected_nerfstudio_revision),
        ("tiny_cuda_nn", tiny_cuda_nn_checkout, spec.expected_tiny_cuda_nn_revision),
    ):
        if not evidence.exists:
            unavailable.append(f"{label}_checkout_missing")
        elif evidence.error:
            unavailable.append(f"{label}_checkout_probe_failed")
        else:
            if evidence.revision != expected:
                rejected.append(f"{label}_checkout_revision_mismatch")
            if not evidence.clean:
                rejected.append(f"{label}_checkout_not_clean")

    if not environment_probe.backend_ready:
        unavailable.append("nerfstudio_environment_not_ready")
    else:
        if _base_version(environment_probe.nerfstudio_version) != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
            rejected.append("nerfstudio_version_mismatch")
        if _base_version(environment_probe.torch_version) != NERFSTUDIO_E2_TORCH:
            rejected.append("torch_version_mismatch")
        if not _base_version(environment_probe.torch_cuda_version).startswith(NERFSTUDIO_E2_CUDA):
            rejected.append("torch_cuda_version_mismatch")
        if not environment_probe.cuda_available or not environment_probe.cuda_device_names:
            unavailable.append("cuda_device_unavailable")
        elif any(
            capability < NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY
            for capability in environment_probe.cuda_compute_capabilities
        ):
            rejected.append("cuda_compute_capability_below_qualified_minimum")

    if python_runtime.error:
        unavailable.append("python_runtime_probe_failed")
    elif not _version_prefix_matches(python_runtime.python_version, NERFSTUDIO_E2_PYTHON):
        rejected.append("python_version_mismatch")

    if rejected:
        state = NerfInteractiveRuntimeState.REJECTED
        reasons = tuple(dict.fromkeys(rejected + unavailable))
    elif unavailable:
        state = NerfInteractiveRuntimeState.UNAVAILABLE
        reasons = tuple(dict.fromkeys(unavailable))
    else:
        state = NerfInteractiveRuntimeState.READY
        reasons = ()

    return NerfInteractiveRuntimeReport(
        state=state,
        reasons=reasons,
        spec=spec,
        environment_probe=environment_probe,
        python_runtime=python_runtime,
        nerfstudio_checkout=nerfstudio_checkout,
        tiny_cuda_nn_checkout=tiny_cuda_nn_checkout,
        production_heavy_modules=heavy,
    )


def probe_phase9_nerf_interactive_runtime(
    spec: NerfInteractiveRuntimeSpec | None = None,
) -> NerfInteractiveRuntimeReport:
    """Read-only real probe of the isolated Phase 9.5 Nerfstudio/Nerfacto runtime."""

    resolved = spec or default_phase9_nerf_runtime_spec()
    with _isolated_runtime_paths(resolved.python_executable):
        environment = probe_nerf_environment(
            resolved.python_executable,
            timeout_s=resolved.probe_timeout_s,
        )
        python_runtime = probe_nerf_python_runtime(
            resolved.python_executable,
            timeout_s=min(15.0, resolved.probe_timeout_s),
        )
    nerfstudio_checkout = probe_nerf_checkout(
        resolved.nerfstudio_root,
        timeout_s=min(10.0, resolved.probe_timeout_s),
    )
    tcnn_checkout = probe_nerf_checkout(
        resolved.tiny_cuda_nn_root,
        timeout_s=min(10.0, resolved.probe_timeout_s),
    )
    heavy = current_process_nerf_heavy_modules()
    return assess_nerf_interactive_runtime(
        resolved,
        environment_probe=environment,
        python_runtime=python_runtime,
        nerfstudio_checkout=nerfstudio_checkout,
        tiny_cuda_nn_checkout=tcnn_checkout,
        production_heavy_modules=heavy,
    )


__all__ = [
    "PHASE9_5A_NERF_RUNTIME_SCHEMA_VERSION",
    "PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION",
    "PHASE9_5A_TCNN_QUALIFIED_REVISION",
    "PHASE9_5A_HEAVY_NERF_MODULES",
    "NerfInteractiveRuntimeState",
    "NerfInteractiveProcessPolicy",
    "NerfInteractiveRuntimeSpec",
    "NerfCheckoutEvidence",
    "NerfPythonRuntimeEvidence",
    "NerfInteractiveRuntimeReport",
    "default_phase9_nerf_runtime_spec",
    "probe_nerf_checkout",
    "probe_nerf_python_runtime",
    "current_process_nerf_heavy_modules",
    "assess_nerf_interactive_runtime",
    "probe_phase9_nerf_interactive_runtime",
]
