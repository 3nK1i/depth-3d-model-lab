"""Phase 9.4A interactive Gaussian runtime-isolation foundation.

This module promotes the already-qualified Phase 8 Gaussian backend contracts into
one explicit Phase 9 interactive-runtime boundary.  It performs read-only
capability qualification only: no Gaussian training, optimization, or rendering
is started here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
from math import isfinite
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Callable, Iterable

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_backends import (
    GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY,
    GAUSSIAN_QUALIFIED_CUDA,
    GAUSSIAN_QUALIFIED_PYTHON,
    GAUSSIAN_QUALIFIED_TORCH,
    GSPLAT_QUALIFIED_VERSION,
    GaussianRuntimeEvidence,
    probe_gaussian_runtime_evidence,
)
from core.reconstruction.gaussian_splatting import (
    GaussianBackendKind,
    GaussianEnvironmentProbe,
    GaussianSceneRole,
    gaussian_backend_provenance,
    probe_gaussian_environment,
)

PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION = 1
PHASE9_4A_GSPLAT_QUALIFIED_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
PHASE9_4A_HEAVY_GAUSSIAN_MODULES = (
    "gsplat",
    "diff_gaussian_rasterization",
    "simple_knn",
)


class GaussianInteractiveRuntimeState(str, Enum):
    """Readiness state for the future interactive Gaussian runtime."""

    READY = "ready"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GaussianInteractiveProcessPolicy:
    """Process-lifecycle policy that later 9.4 workers must obey."""

    timeout_s: float = 1800.0
    terminate_grace_s: float = 2.0
    start_new_session: bool = True
    shell: bool = False
    terminate_signal: str = "SIGTERM"
    kill_signal: str = "SIGKILL"

    def __post_init__(self) -> None:
        timeout = float(self.timeout_s)
        grace = float(self.terminate_grace_s)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ReconstructionValidationError("Gaussian process timeout_s must be finite and positive.")
        if not isfinite(grace) or grace <= 0.0:
            raise ReconstructionValidationError(
                "Gaussian process terminate_grace_s must be finite and positive."
            )
        if not bool(self.start_new_session):
            raise ReconstructionValidationError(
                "Interactive Gaussian workers must run in their own process group/session."
            )
        if bool(self.shell):
            raise ReconstructionValidationError("Interactive Gaussian workers must never use shell=True.")
        if str(self.terminate_signal).strip() != "SIGTERM" or str(self.kill_signal).strip() != "SIGKILL":
            raise ReconstructionValidationError(
                "Interactive Gaussian cleanup must use SIGTERM followed by SIGKILL escalation."
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
class GaussianInteractiveRuntimeSpec:
    """Qualified external-runtime location for Phase 9.4 interactive Gaussian work."""

    python_executable: str
    checkout_root: str
    backend_kind: GaussianBackendKind = GaussianBackendKind.GSPLAT_EXTERNAL
    expected_checkout_revision: str = PHASE9_4A_GSPLAT_QUALIFIED_REVISION
    probe_timeout_s: float = 30.0
    process_policy: GaussianInteractiveProcessPolicy = GaussianInteractiveProcessPolicy()
    schema_version: int = PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        python = Path(str(self.python_executable))
        checkout = Path(str(self.checkout_root))
        if not python.is_absolute() or not checkout.is_absolute():
            raise ReconstructionValidationError(
                "Interactive Gaussian Python and checkout paths must be absolute."
            )
        if not isinstance(self.backend_kind, GaussianBackendKind):
            raise TypeError("backend_kind must be GaussianBackendKind.")
        if self.backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK:
            raise ReconstructionValidationError(
                "Phase 9.4 interactive runtime specs must reference a real isolated backend."
            )
        revision = str(self.expected_checkout_revision).strip().lower()
        if not revision:
            raise ReconstructionValidationError("expected_checkout_revision must be non-empty.")
        timeout = float(self.probe_timeout_s)
        if not isfinite(timeout) or timeout <= 0.0:
            raise ReconstructionValidationError("probe_timeout_s must be finite and positive.")
        if int(self.schema_version) != PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Phase 9.4A Gaussian runtime schema version.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "checkout_root", str(checkout))
        object.__setattr__(self, "expected_checkout_revision", revision)
        object.__setattr__(self, "probe_timeout_s", timeout)
        object.__setattr__(self, "schema_version", PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION)

    @property
    def interpreter_is_external(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "python_executable": self.python_executable,
            "checkout_root": self.checkout_root,
            "expected_checkout_revision": self.expected_checkout_revision,
            "interpreter_is_external": self.interpreter_is_external,
            "probe_timeout_s": self.probe_timeout_s,
            "process_policy": self.process_policy.as_dict(),
        }


def default_phase9_gaussian_runtime_spec(home: str | Path | None = None) -> GaussianInteractiveRuntimeSpec:
    root = Path.home() if home is None else Path(home)
    return GaussianInteractiveRuntimeSpec(
        python_executable=str(root / "anaconda3" / "envs" / "eia-gaussian-gsplat" / "bin" / "python"),
        checkout_root=str(root / "Desktop" / "eia-gaussian-backends" / "gsplat"),
    )


@dataclass(frozen=True, slots=True)
class GaussianCheckoutEvidence:
    root: str
    exists: bool
    revision: str
    clean: bool
    status_lines: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        root = Path(str(self.root))
        if not root.is_absolute():
            raise ReconstructionValidationError("Gaussian checkout evidence root must be absolute.")
        object.__setattr__(self, "root", str(root))
        object.__setattr__(self, "exists", bool(self.exists))
        object.__setattr__(self, "revision", str(self.revision).strip().lower())
        object.__setattr__(self, "clean", bool(self.clean))
        object.__setattr__(
            self,
            "status_lines",
            tuple(str(line).rstrip() for line in self.status_lines if str(line).strip()),
        )
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


def probe_gaussian_checkout(root: str | Path, *, timeout_s: float = 10.0) -> GaussianCheckoutEvidence:
    path = Path(root)
    if not path.is_absolute():
        raise ReconstructionValidationError("Gaussian checkout root must be absolute.")
    if not path.is_dir() or not (path / ".git").exists():
        return GaussianCheckoutEvidence(
            root=str(path), exists=False, revision="", clean=False, error="checkout_not_found"
        )
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
        return GaussianCheckoutEvidence(
            root=str(path), exists=True, revision="", clean=False, error=f"{type(exc).__name__}: {exc}"
        )
    if revision.returncode != 0 or status.returncode != 0:
        detail = revision.stderr.strip() or status.stderr.strip() or "git_probe_failed"
        return GaussianCheckoutEvidence(
            root=str(path), exists=True, revision="", clean=False, error=detail
        )
    lines = tuple(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    return GaussianCheckoutEvidence(
        root=str(path),
        exists=True,
        revision=revision.stdout.strip().splitlines()[-1] if revision.stdout.strip() else "",
        clean=not lines,
        status_lines=lines,
    )


def current_process_gaussian_heavy_modules(
    modules: Iterable[str] = PHASE9_4A_HEAVY_GAUSSIAN_MODULES,
    *,
    finder: Callable[[str], Any] = importlib.util.find_spec,
) -> tuple[str, ...]:
    """Return heavy Gaussian modules importable by the production interpreter without importing them."""

    found: list[str] = []
    for name in tuple(str(value).strip() for value in modules if str(value).strip()):
        try:
            if finder(name) is not None:
                found.append(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return tuple(found)


@dataclass(frozen=True, slots=True)
class GaussianInteractiveRuntimeReport:
    state: GaussianInteractiveRuntimeState
    reasons: tuple[str, ...]
    spec: GaussianInteractiveRuntimeSpec
    environment_probe: GaussianEnvironmentProbe
    runtime_evidence: GaussianRuntimeEvidence
    checkout_evidence: GaussianCheckoutEvidence
    production_heavy_modules: tuple[str, ...] = ()
    capability_probe_only: bool = True
    training_performed: bool = False
    rendering_performed: bool = False
    gaussian_scene_role: GaussianSceneRole = GaussianSceneRole.APPEARANCE_ONLY
    canonical_geometry_authoritative: bool = False
    metric_scale_authoritative: bool = False
    measurement_authoritative: bool = False
    qc_authoritative: bool = False
    certified_metrology: bool = False
    cross_representation_fallback: bool = False
    schema_version: int = PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.state, GaussianInteractiveRuntimeState):
            raise TypeError("state must be GaussianInteractiveRuntimeState.")
        if self.environment_probe.backend_kind is not self.spec.backend_kind:
            raise ReconstructionValidationError("Gaussian runtime report backend mismatch.")
        if os.path.realpath(self.runtime_evidence.python_executable) != os.path.realpath(
            self.spec.python_executable
        ):
            raise ReconstructionValidationError("Gaussian runtime report interpreter mismatch.")
        if Path(self.checkout_evidence.root) != Path(self.spec.checkout_root):
            raise ReconstructionValidationError("Gaussian runtime report checkout mismatch.")
        reasons = tuple(str(reason).strip() for reason in self.reasons if str(reason).strip())
        heavy = tuple(str(name).strip() for name in self.production_heavy_modules if str(name).strip())
        if self.state is GaussianInteractiveRuntimeState.READY and reasons:
            raise ReconstructionValidationError("READY Gaussian runtime reports cannot contain rejection reasons.")
        if self.state is not GaussianInteractiveRuntimeState.READY and not reasons:
            raise ReconstructionValidationError("Non-ready Gaussian runtime reports require explicit reasons.")
        if not bool(self.capability_probe_only) or bool(self.training_performed) or bool(self.rendering_performed):
            raise ReconstructionValidationError("Phase 9.4A runtime qualification must remain probe-only.")
        if self.gaussian_scene_role is not GaussianSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("Interactive Gaussian runtime must remain appearance-only.")
        if any(
            bool(value)
            for value in (
                self.canonical_geometry_authoritative,
                self.metric_scale_authoritative,
                self.measurement_authoritative,
                self.qc_authoritative,
                self.certified_metrology,
                self.cross_representation_fallback,
            )
        ):
            raise ReconstructionValidationError(
                "Gaussian runtime qualification cannot claim geometry/metric/measurement/QC/metrology authority or cross-fallback."
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "production_heavy_modules", heavy)
        object.__setattr__(self, "capability_probe_only", True)
        object.__setattr__(self, "training_performed", False)
        object.__setattr__(self, "rendering_performed", False)
        object.__setattr__(self, "gaussian_scene_role", GaussianSceneRole.APPEARANCE_ONLY)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "metric_scale_authoritative", False)
        object.__setattr__(self, "measurement_authoritative", False)
        object.__setattr__(self, "qc_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "cross_representation_fallback", False)
        object.__setattr__(self, "schema_version", PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION)

    @property
    def ready(self) -> bool:
        return self.state is GaussianInteractiveRuntimeState.READY

    @property
    def backend_provenance(self):
        return gaussian_backend_provenance(self.spec.backend_kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "spec": self.spec.as_dict(),
            "backend_provenance": self.backend_provenance.as_dict(),
            "environment_probe": self.environment_probe.as_dict(),
            "runtime_evidence": self.runtime_evidence.as_dict(),
            "checkout_evidence": self.checkout_evidence.as_dict(),
            "production_heavy_modules": list(self.production_heavy_modules),
            "capability_probe_only": self.capability_probe_only,
            "training_performed": self.training_performed,
            "rendering_performed": self.rendering_performed,
            "gaussian_scene_role": self.gaussian_scene_role.value,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "metric_scale_authoritative": self.metric_scale_authoritative,
            "measurement_authoritative": self.measurement_authoritative,
            "qc_authoritative": self.qc_authoritative,
            "certified_metrology": self.certified_metrology,
            "cross_representation_fallback": self.cross_representation_fallback,
        }


def _base_version(value: str) -> str:
    return str(value).strip().split("+", 1)[0]


def _version_prefix_matches(value: str, required: str) -> bool:
    actual = _base_version(value)
    expected = _base_version(required)
    return actual == expected or actual.startswith(expected + ".")


def assess_gaussian_interactive_runtime(
    spec: GaussianInteractiveRuntimeSpec,
    *,
    environment_probe: GaussianEnvironmentProbe,
    runtime_evidence: GaussianRuntimeEvidence,
    checkout_evidence: GaussianCheckoutEvidence,
    production_heavy_modules: Iterable[str] = (),
) -> GaussianInteractiveRuntimeReport:
    """Assess probe evidence without launching training or rendering."""

    unavailable: list[str] = []
    rejected: list[str] = []
    heavy = tuple(str(name).strip() for name in production_heavy_modules if str(name).strip())

    if spec.backend_kind is not GaussianBackendKind.GSPLAT_EXTERNAL:
        rejected.append("interactive_backend_must_be_permissive_gsplat")
    if not spec.interpreter_is_external or not environment_probe.interpreter_is_isolated:
        rejected.append("external_interpreter_required")
    if heavy:
        rejected.append("heavy_gaussian_modules_present_in_production_runtime")

    if not Path(spec.python_executable).is_file():
        unavailable.append("external_python_missing")
    if not checkout_evidence.exists:
        unavailable.append("qualified_checkout_missing")
    elif checkout_evidence.error:
        unavailable.append("checkout_probe_failed")
    else:
        if checkout_evidence.revision != spec.expected_checkout_revision:
            rejected.append("checkout_revision_mismatch")
        if not checkout_evidence.clean:
            rejected.append("checkout_not_clean")

    if not environment_probe.backend_ready:
        unavailable.append("gsplat_environment_not_ready")
    else:
        if _base_version(environment_probe.gsplat_version) != GSPLAT_QUALIFIED_VERSION:
            rejected.append("gsplat_version_mismatch")

    if runtime_evidence.error:
        unavailable.append("runtime_probe_failed")
    else:
        if not _version_prefix_matches(runtime_evidence.python_version, GAUSSIAN_QUALIFIED_PYTHON):
            rejected.append("python_version_mismatch")
        if _base_version(runtime_evidence.torch_version) != GAUSSIAN_QUALIFIED_TORCH:
            rejected.append("torch_version_mismatch")
        if _base_version(runtime_evidence.torch_cuda_version) != GAUSSIAN_QUALIFIED_CUDA:
            rejected.append("torch_cuda_version_mismatch")
        if not runtime_evidence.cuda_available or not runtime_evidence.device_names:
            unavailable.append("cuda_device_unavailable")
        elif any(
            capability < GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY
            for capability in runtime_evidence.device_capabilities
        ):
            rejected.append("cuda_compute_capability_below_qualified_minimum")

    if rejected:
        state = GaussianInteractiveRuntimeState.REJECTED
        reasons = tuple(dict.fromkeys(rejected + unavailable))
    elif unavailable:
        state = GaussianInteractiveRuntimeState.UNAVAILABLE
        reasons = tuple(dict.fromkeys(unavailable))
    else:
        state = GaussianInteractiveRuntimeState.READY
        reasons = ()

    return GaussianInteractiveRuntimeReport(
        state=state,
        reasons=reasons,
        spec=spec,
        environment_probe=environment_probe,
        runtime_evidence=runtime_evidence,
        checkout_evidence=checkout_evidence,
        production_heavy_modules=heavy,
    )


def probe_phase9_gaussian_interactive_runtime(
    spec: GaussianInteractiveRuntimeSpec | None = None,
) -> GaussianInteractiveRuntimeReport:
    """Read-only real runtime probe for the isolated Phase 9.4 gsplat backend."""

    resolved = spec or default_phase9_gaussian_runtime_spec()
    environment = probe_gaussian_environment(
        resolved.python_executable,
        backend_kind=resolved.backend_kind,
        timeout_s=resolved.probe_timeout_s,
    )
    runtime = probe_gaussian_runtime_evidence(
        resolved.python_executable,
        timeout_s=resolved.probe_timeout_s,
    )
    checkout = probe_gaussian_checkout(
        resolved.checkout_root,
        timeout_s=min(10.0, resolved.probe_timeout_s),
    )
    heavy = current_process_gaussian_heavy_modules()
    return assess_gaussian_interactive_runtime(
        resolved,
        environment_probe=environment,
        runtime_evidence=runtime,
        checkout_evidence=checkout,
        production_heavy_modules=heavy,
    )


__all__ = [
    "PHASE9_4A_GAUSSIAN_RUNTIME_SCHEMA_VERSION",
    "PHASE9_4A_GSPLAT_QUALIFIED_REVISION",
    "PHASE9_4A_HEAVY_GAUSSIAN_MODULES",
    "GaussianInteractiveRuntimeState",
    "GaussianInteractiveProcessPolicy",
    "GaussianInteractiveRuntimeSpec",
    "GaussianCheckoutEvidence",
    "GaussianInteractiveRuntimeReport",
    "default_phase9_gaussian_runtime_spec",
    "probe_gaussian_checkout",
    "current_process_gaussian_heavy_modules",
    "assess_gaussian_interactive_runtime",
    "probe_phase9_gaussian_interactive_runtime",
]
