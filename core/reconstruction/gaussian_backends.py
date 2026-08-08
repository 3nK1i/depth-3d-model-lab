"""Phase 8.7D2 real Gaussian backend qualification contracts.

This module deliberately does not import Graphdeco, gsplat, CUDA extensions, or
backend-specific Python packages.  Heavy backends remain external processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
import subprocess
from typing import Any, Mapping

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import (
    GaussianBackendKind,
    GaussianEnvironmentProbe,
    GaussianOptimizationState,
    GaussianSceneArtifact,
)

GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION = 1

# Reproducible qualification pins.  These are intentionally conservative and
# must only be changed in an explicit future qualification step.
GSPLAT_QUALIFIED_VERSION = "1.5.3"
GSPLAT_QUALIFIED_TAG = "v1.5.3"
GSPLAT_QUALIFIED_COMMIT_SHORT = "937e299"
GRAPHDECO_QUALIFIED_REVISION = "54c035f7834b564019656c3e3fcc3646292f727d"
GRAPHDECO_QUALIFIED_COMPATIBILITY_PATCH = "diff-gaussian-rasterization-pr49-cstdint"
GAUSSIAN_QUALIFIED_PYTHON = "3.10"
GAUSSIAN_QUALIFIED_TORCH = "2.7.1"
GAUSSIAN_QUALIFIED_TORCHVISION = "0.22.1"
GAUSSIAN_QUALIFIED_CUDA = "12.8"
GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY = (7, 0)


class GaussianBackendQualificationStatus(str, Enum):
    QUALIFIED = "qualified"
    NOT_READY = "not_ready"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class GaussianBackendRuntimePin:
    backend_kind: GaussianBackendKind
    backend_version: str
    upstream_revision: str
    python_version: str = GAUSSIAN_QUALIFIED_PYTHON
    torch_version: str = GAUSSIAN_QUALIFIED_TORCH
    cuda_version: str = GAUSSIAN_QUALIFIED_CUDA
    compatibility_patches: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.backend_kind not in {
            GaussianBackendKind.GSPLAT_EXTERNAL,
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        }:
            raise ReconstructionValidationError("Runtime pins are only defined for real Gaussian backends.")
        for field_name in (
            "backend_version",
            "upstream_revision",
            "python_version",
            "torch_version",
            "cuda_version",
        ):
            object.__setattr__(
                self,
                field_name,
                require_nonempty(getattr(self, field_name), field_name=field_name),
            )
        patches = tuple(
            require_nonempty(value, field_name="compatibility_patch")
            for value in self.compatibility_patches
        )
        object.__setattr__(self, "compatibility_patches", patches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "backend_version": self.backend_version,
            "upstream_revision": self.upstream_revision,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "compatibility_patches": list(self.compatibility_patches),
        }


def gaussian_backend_runtime_pin(backend_kind: GaussianBackendKind) -> GaussianBackendRuntimePin:
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        return GaussianBackendRuntimePin(
            backend_kind=backend_kind,
            backend_version=GSPLAT_QUALIFIED_VERSION,
            upstream_revision=GSPLAT_QUALIFIED_TAG,
        )
    if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
        return GaussianBackendRuntimePin(
            backend_kind=backend_kind,
            backend_version=GRAPHDECO_QUALIFIED_REVISION,
            upstream_revision=GRAPHDECO_QUALIFIED_REVISION,
            compatibility_patches=(GRAPHDECO_QUALIFIED_COMPATIBILITY_PATCH,),
        )
    raise ReconstructionValidationError("No real-backend runtime pin exists for this backend kind.")


@dataclass(frozen=True, slots=True)
class GaussianRuntimeEvidence:
    python_executable: str
    python_version: str
    torch_version: str
    torch_cuda_version: str
    cuda_available: bool
    device_names: tuple[str, ...]
    device_capabilities: tuple[tuple[int, int], ...]
    error: str = ""

    def __post_init__(self) -> None:
        python = Path(require_nonempty(self.python_executable, field_name="python_executable"))
        if not python.is_absolute():
            raise ReconstructionValidationError("Gaussian runtime evidence requires an absolute Python path.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "python_version", str(self.python_version).strip())
        object.__setattr__(self, "torch_version", str(self.torch_version).strip())
        object.__setattr__(self, "torch_cuda_version", str(self.torch_cuda_version).strip())
        object.__setattr__(self, "cuda_available", bool(self.cuda_available))
        names = tuple(str(name).strip() for name in self.device_names if str(name).strip())
        caps = tuple((int(major), int(minor)) for major, minor in self.device_capabilities)
        if len(names) != len(caps):
            raise ReconstructionValidationError("Gaussian runtime device names/capabilities must have equal length.")
        if names and not self.cuda_available:
            raise ReconstructionValidationError("CUDA devices cannot be reported when CUDA is unavailable.")
        object.__setattr__(self, "device_names", names)
        object.__setattr__(self, "device_capabilities", caps)
        object.__setattr__(self, "error", str(self.error).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
            "cuda_available": self.cuda_available,
            "device_names": list(self.device_names),
            "device_capabilities": [list(value) for value in self.device_capabilities],
            "error": self.error,
        }


_RUNTIME_EVIDENCE_CODE = r'''
import importlib
import json
import platform
payload = {
    "python_version": platform.python_version(),
    "torch_version": "",
    "torch_cuda_version": "",
    "cuda_available": False,
    "device_names": [],
    "device_capabilities": [],
    "error": "",
}
try:
    torch = importlib.import_module("torch")
    payload["torch_version"] = str(getattr(torch, "__version__", ""))
    payload["torch_cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if payload["cuda_available"]:
        count = int(torch.cuda.device_count())
        payload["device_names"] = [str(torch.cuda.get_device_name(i)) for i in range(count)]
        payload["device_capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(count)]
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''


def probe_gaussian_runtime_evidence(
    python_executable: str | Path,
    *,
    timeout_s: float = 30.0,
) -> GaussianRuntimeEvidence:
    python = Path(python_executable)
    if not python.is_absolute():
        raise ReconstructionValidationError("Gaussian runtime-evidence Python must be absolute.")
    timeout = float(timeout_s)
    if not isfinite(timeout) or timeout <= 0:
        raise ReconstructionValidationError("timeout_s must be finite and positive.")
    if not python.is_file():
        return GaussianRuntimeEvidence(
            python_executable=str(python),
            python_version="",
            torch_version="",
            torch_cuda_version="",
            cuda_available=False,
            device_names=(),
            device_capabilities=(),
            error="python_executable_not_found",
        )
    try:
        completed = subprocess.run(
            [str(python), "-c", _RUNTIME_EVIDENCE_CODE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GaussianRuntimeEvidence(
            python_executable=str(python),
            python_version="",
            torch_version="",
            torch_cuda_version="",
            cuda_available=False,
            device_names=(),
            device_capabilities=(),
            error=f"{type(exc).__name__}: {exc}",
        )
    if completed.returncode != 0:
        return GaussianRuntimeEvidence(
            python_executable=str(python),
            python_version="",
            torch_version="",
            torch_cuda_version="",
            cuda_available=False,
            device_names=(),
            device_capabilities=(),
            error=completed.stderr.strip() or f"probe_exit_{completed.returncode}",
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return GaussianRuntimeEvidence(
            python_executable=str(python),
            python_version="",
            torch_version="",
            torch_cuda_version="",
            cuda_available=False,
            device_names=(),
            device_capabilities=(),
            error=f"invalid_runtime_probe_output: {exc}",
        )
    return GaussianRuntimeEvidence(
        python_executable=str(python),
        python_version=str(payload.get("python_version", "")),
        torch_version=str(payload.get("torch_version", "")),
        torch_cuda_version=str(payload.get("torch_cuda_version", "")),
        cuda_available=bool(payload.get("cuda_available", False)),
        device_names=tuple(payload.get("device_names", ())),
        device_capabilities=tuple(tuple(value) for value in payload.get("device_capabilities", ())),
        error=str(payload.get("error", "")),
    )


def graphdeco_checkout_revision(root: str | Path, *, timeout_s: float = 10.0) -> str:
    path = Path(root)
    if not path.is_absolute():
        raise ReconstructionValidationError("Graphdeco checkout root must be absolute.")
    if not (path / ".git").exists():
        return ""
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip().lower() if completed.returncode == 0 else ""


@dataclass(frozen=True, slots=True)
class GaussianWorkerQualificationEvidence:
    backend_kind: GaussianBackendKind
    real_cuda_execution: bool
    optimization_completed: bool
    render_written: bool
    training_steps: int
    source_revision: str
    preview_path: str
    metrics_path: str

    def __post_init__(self) -> None:
        if self.backend_kind not in {
            GaussianBackendKind.GSPLAT_EXTERNAL,
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        }:
            raise ReconstructionValidationError("Worker qualification evidence requires a real backend.")
        object.__setattr__(self, "real_cuda_execution", bool(self.real_cuda_execution))
        object.__setattr__(self, "optimization_completed", bool(self.optimization_completed))
        object.__setattr__(self, "render_written", bool(self.render_written))
        steps = int(self.training_steps)
        if steps < 0:
            raise ReconstructionValidationError("training_steps must be non-negative.")
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "source_revision", require_nonempty(self.source_revision, field_name="source_revision"))
        for field_name in ("preview_path", "metrics_path"):
            path = Path(require_nonempty(getattr(self, field_name), field_name=field_name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{field_name} must be absolute.")
            object.__setattr__(self, field_name, str(path))

    @classmethod
    def from_file(cls, path: str | Path) -> "GaussianWorkerQualificationEvidence":
        metrics_path = Path(path)
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("Gaussian worker qualification metrics must be a JSON object.")
        return cls(
            backend_kind=GaussianBackendKind(str(payload["backend_kind"])),
            real_cuda_execution=bool(payload["real_cuda_execution"]),
            optimization_completed=bool(payload["optimization_completed"]),
            render_written=bool(payload["render_written"]),
            training_steps=int(payload["training_steps"]),
            source_revision=str(payload["source_revision"]),
            preview_path=str(payload["preview_path"]),
            metrics_path=str(metrics_path.resolve()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "real_cuda_execution": self.real_cuda_execution,
            "optimization_completed": self.optimization_completed,
            "render_written": self.render_written,
            "training_steps": self.training_steps,
            "source_revision": self.source_revision,
            "preview_path": self.preview_path,
            "metrics_path": self.metrics_path,
        }


@dataclass(frozen=True, slots=True)
class GaussianBackendQualification:
    backend_kind: GaussianBackendKind
    status: GaussianBackendQualificationStatus
    runtime_pin: GaussianBackendRuntimePin
    environment_probe: GaussianEnvironmentProbe
    runtime_evidence: GaussianRuntimeEvidence
    worker_evidence: GaussianWorkerQualificationEvidence | None
    artifact: GaussianSceneArtifact | None
    reasons: tuple[str, ...]
    schema_version: int = GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, GaussianBackendQualificationStatus):
            raise TypeError("status must be GaussianBackendQualificationStatus.")
        if self.backend_kind is not self.runtime_pin.backend_kind:
            raise ReconstructionValidationError("Qualification backend does not match runtime pin.")
        if self.environment_probe.backend_kind is not self.backend_kind:
            raise ReconstructionValidationError("Qualification backend does not match environment probe.")
        if self.worker_evidence is not None and self.worker_evidence.backend_kind is not self.backend_kind:
            raise ReconstructionValidationError("Qualification backend does not match worker evidence.")
        object.__setattr__(self, "reasons", tuple(str(value) for value in self.reasons))
        if int(self.schema_version) != GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Gaussian backend qualification schema version.")
        object.__setattr__(self, "schema_version", GAUSSIAN_BACKEND_QUALIFICATION_SCHEMA_VERSION)

    @property
    def qualified(self) -> bool:
        return self.status is GaussianBackendQualificationStatus.QUALIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "status": self.status.value,
            "qualified": self.qualified,
            "runtime_pin": self.runtime_pin.as_dict(),
            "environment_probe": self.environment_probe.as_dict(),
            "runtime_evidence": self.runtime_evidence.as_dict(),
            "worker_evidence": None if self.worker_evidence is None else self.worker_evidence.as_dict(),
            "artifact": None if self.artifact is None else self.artifact.as_dict(),
            "reasons": list(self.reasons),
        }


def _base_version(value: str) -> str:
    return str(value).strip().split("+")[0]


def _version_prefix_matches(value: str, required: str) -> bool:
    actual = _base_version(value)
    return actual == required or actual.startswith(required + ".")


def assess_gaussian_backend_qualification(
    *,
    backend_kind: GaussianBackendKind,
    environment_probe: GaussianEnvironmentProbe,
    runtime_evidence: GaussianRuntimeEvidence,
    artifact: GaussianSceneArtifact | None,
    worker_evidence: GaussianWorkerQualificationEvidence | None,
    expected_dataset_fingerprint_sha256: str,
    graphdeco_revision: str = "",
) -> GaussianBackendQualification:
    pin = gaussian_backend_runtime_pin(backend_kind)
    reasons: list[str] = []

    if not environment_probe.interpreter_is_isolated:
        reasons.append("interpreter_not_isolated")
    if not environment_probe.backend_ready:
        reasons.append("backend_environment_not_ready")
    if runtime_evidence.error:
        reasons.append("runtime_probe_failed")
    if not _version_prefix_matches(runtime_evidence.python_version, pin.python_version):
        reasons.append("python_version_mismatch")
    if not _version_prefix_matches(runtime_evidence.torch_version, pin.torch_version):
        reasons.append("torch_version_mismatch")
    if not str(runtime_evidence.torch_cuda_version).startswith(pin.cuda_version):
        reasons.append("torch_cuda_version_mismatch")
    if not runtime_evidence.cuda_available or not runtime_evidence.device_capabilities:
        reasons.append("cuda_device_unavailable")
    elif any(capability < GAUSSIAN_MINIMUM_COMPUTE_CAPABILITY for capability in runtime_evidence.device_capabilities):
        reasons.append("compute_capability_below_minimum")

    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        if _base_version(environment_probe.gsplat_version) != GSPLAT_QUALIFIED_VERSION:
            reasons.append("gsplat_version_mismatch")
    else:
        if str(graphdeco_revision).lower() != GRAPHDECO_QUALIFIED_REVISION:
            reasons.append("graphdeco_revision_mismatch")

    digest = str(expected_dataset_fingerprint_sha256).strip().lower()
    if len(digest) != 64:
        raise ReconstructionValidationError("expected_dataset_fingerprint_sha256 must be a SHA-256 digest.")

    if artifact is None:
        reasons.append("artifact_missing")
    else:
        if artifact.backend_provenance.backend_kind is not backend_kind:
            reasons.append("artifact_backend_mismatch")
        if artifact.optimization_state is not GaussianOptimizationState.OPTIMIZED:
            reasons.append("artifact_not_optimized")
        if not artifact.render_capable:
            reasons.append("artifact_not_render_capable")
        if artifact.training_steps < 1:
            reasons.append("artifact_has_no_training_steps")
        if artifact.dataset_fingerprint_sha256 != digest:
            reasons.append("dataset_fingerprint_mismatch")
        if artifact.canonical_geometry_authoritative:
            reasons.append("gaussian_claims_canonical_geometry")
        if artifact.certified_metrology:
            reasons.append("gaussian_claims_certified_metrology")
        expected_backend_version = pin.backend_version
        if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
            if _base_version(artifact.backend_version) != expected_backend_version:
                reasons.append("artifact_backend_version_mismatch")
        elif artifact.backend_version.lower() != expected_backend_version:
            reasons.append("artifact_backend_version_mismatch")
        if not Path(artifact.scene_path).is_file():
            reasons.append("scene_artifact_missing_on_disk")

    if worker_evidence is None:
        reasons.append("worker_evidence_missing")
    else:
        if not worker_evidence.real_cuda_execution:
            reasons.append("worker_did_not_execute_cuda")
        if not worker_evidence.optimization_completed:
            reasons.append("worker_optimization_incomplete")
        if not worker_evidence.render_written:
            reasons.append("worker_render_missing")
        if worker_evidence.training_steps < 1:
            reasons.append("worker_has_no_training_steps")
        expected_revision = pin.upstream_revision
        if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
            if worker_evidence.source_revision not in {GSPLAT_QUALIFIED_TAG, GSPLAT_QUALIFIED_VERSION, GSPLAT_QUALIFIED_COMMIT_SHORT}:
                reasons.append("worker_source_revision_mismatch")
        elif worker_evidence.source_revision.lower() != expected_revision.lower():
            reasons.append("worker_source_revision_mismatch")
        if not Path(worker_evidence.preview_path).is_file():
            reasons.append("preview_missing_on_disk")

    status = (
        GaussianBackendQualificationStatus.QUALIFIED
        if not reasons
        else GaussianBackendQualificationStatus.REJECTED
    )
    return GaussianBackendQualification(
        backend_kind=backend_kind,
        status=status,
        runtime_pin=pin,
        environment_probe=environment_probe,
        runtime_evidence=runtime_evidence,
        worker_evidence=worker_evidence,
        artifact=artifact,
        reasons=tuple(reasons),
    )


def not_ready_gaussian_backend_qualification(
    *,
    backend_kind: GaussianBackendKind,
    environment_probe: GaussianEnvironmentProbe,
    runtime_evidence: GaussianRuntimeEvidence,
    reason: str,
) -> GaussianBackendQualification:
    return GaussianBackendQualification(
        backend_kind=backend_kind,
        status=GaussianBackendQualificationStatus.NOT_READY,
        runtime_pin=gaussian_backend_runtime_pin(backend_kind),
        environment_probe=environment_probe,
        runtime_evidence=runtime_evidence,
        worker_evidence=None,
        artifact=None,
        reasons=(require_nonempty(reason, field_name="reason"),),
    )


def write_gaussian_backend_qualification_json(
    qualification: GaussianBackendQualification,
    path: str | Path,
) -> Path:
    if not isinstance(qualification, GaussianBackendQualification):
        raise TypeError("qualification must be GaussianBackendQualification.")
    target = Path(path)
    if not target.is_absolute():
        raise ReconstructionValidationError("Gaussian qualification path must be absolute.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(qualification.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
