"""Phase 8.7E1 NeRF contracts and external-process isolation.

NeRF is an appearance representation.  It must never silently become the
canonical inspection geometry or a certified metrology source.  Heavy NeRF
runtimes (Nerfstudio, tiny-cuda-nn, CUDA extensions) are intentionally probed
and executed through another Python interpreter instead of being imported by
this production module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionError, ReconstructionValidationError
from core.reconstruction.neural_scene import (
    NEURAL_SCENE_POSE_CONVENTION,
    NeuralSceneDataset,
    neural_scene_dataset_fingerprint,
)
from core.reconstruction.scale_reference import MetricGeometryStatus

NERF_SCHEMA_VERSION = 1
NERF_POSE_CONVENTION = NEURAL_SCENE_POSE_CONVENTION
NERFSTUDIO_REPOSITORY = "https://github.com/nerfstudio-project/nerfstudio"
NERFSTUDIO_LICENSE_URL = NERFSTUDIO_REPOSITORY + "/blob/main/LICENSE"
NERFSTUDIO_RELEASE_TAG = "v1.1.5"
NERFSTUDIO_RELEASE_REVISION_PREFIX = "6b60855"
NERFSTUDIO_METHOD = "nerfacto"
NERFSTUDIO_DOCUMENTED_MEMORY_MIB = 6144
TINY_CUDA_NN_REPOSITORY = "https://github.com/NVlabs/tiny-cuda-nn"
TINY_CUDA_NN_LICENSE = "BSD-3-Clause"
TINY_CUDA_NN_E2_REVISION = "b3473c81396fe927293bdfd5a6be32df8769927c"
NERFSTUDIO_E2_PYTHON = "3.10"
NERFSTUDIO_E2_TORCH = "2.1.2"
NERFSTUDIO_E2_TORCHVISION = "0.16.2"
NERFSTUDIO_E2_CUDA = "11.8"
NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY = (7, 5)


class NerfWorkerExecutionError(ReconstructionError):
    """Raised when an isolated NeRF worker cannot complete safely."""


class NerfBackendKind(str, Enum):
    DEVELOPMENT_MOCK = "development_mock"
    NERFSTUDIO_NERFACTO_EXTERNAL = "nerfstudio_nerfacto_external"


class NerfOptimizationProfile(str, Enum):
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class NerfUsageScope(str, Enum):
    DEVELOPMENT_ONLY = "development_only"
    PERMISSIVE_GENERAL_USE = "permissive_general_use"


class NerfTrainingState(str, Enum):
    CONTRACT_ONLY = "contract_only"
    TRAINED = "trained"


class NerfSceneRole(str, Enum):
    APPEARANCE_ONLY = "appearance_only"


@dataclass(frozen=True, slots=True)
class NerfBackendProvenance:
    backend_kind: NerfBackendKind
    backend_name: str
    method_name: str
    upstream_repository: str
    license_name: str
    license_url: str
    usage_scope: NerfUsageScope
    commercial_use_permitted_by_backend_license: bool
    redistribution_license_notice_required: bool
    runtime_dependency_name: str = ""
    runtime_dependency_license: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.backend_kind, NerfBackendKind):
            raise TypeError("backend_kind must be NerfBackendKind.")
        if not isinstance(self.usage_scope, NerfUsageScope):
            raise TypeError("usage_scope must be NerfUsageScope.")
        for name in ("backend_name", "method_name", "upstream_repository", "license_name", "license_url"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        object.__setattr__(self, "commercial_use_permitted_by_backend_license", bool(self.commercial_use_permitted_by_backend_license))
        object.__setattr__(self, "redistribution_license_notice_required", bool(self.redistribution_license_notice_required))
        object.__setattr__(self, "runtime_dependency_name", str(self.runtime_dependency_name).strip())
        object.__setattr__(self, "runtime_dependency_license", str(self.runtime_dependency_license).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "backend_name": self.backend_name,
            "method_name": self.method_name,
            "upstream_repository": self.upstream_repository,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "usage_scope": self.usage_scope.value,
            "commercial_use_permitted_by_backend_license": self.commercial_use_permitted_by_backend_license,
            "redistribution_license_notice_required": self.redistribution_license_notice_required,
            "runtime_dependency_name": self.runtime_dependency_name,
            "runtime_dependency_license": self.runtime_dependency_license,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NerfBackendProvenance":
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("NeRF backend provenance must be a JSON object.")
        try:
            return cls(
                backend_kind=NerfBackendKind(str(payload["backend_kind"])),
                backend_name=str(payload["backend_name"]),
                method_name=str(payload["method_name"]),
                upstream_repository=str(payload["upstream_repository"]),
                license_name=str(payload["license_name"]),
                license_url=str(payload["license_url"]),
                usage_scope=NerfUsageScope(str(payload["usage_scope"])),
                commercial_use_permitted_by_backend_license=bool(payload["commercial_use_permitted_by_backend_license"]),
                redistribution_license_notice_required=bool(payload["redistribution_license_notice_required"]),
                runtime_dependency_name=str(payload.get("runtime_dependency_name", "")),
                runtime_dependency_license=str(payload.get("runtime_dependency_license", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid NeRF backend provenance: {exc}") from exc


def nerf_backend_provenance(backend_kind: NerfBackendKind) -> NerfBackendProvenance:
    if not isinstance(backend_kind, NerfBackendKind):
        raise TypeError("backend_kind must be NerfBackendKind.")
    if backend_kind is NerfBackendKind.DEVELOPMENT_MOCK:
        return NerfBackendProvenance(
            backend_kind=backend_kind,
            backend_name="Development NeRF mock",
            method_name=NERFSTUDIO_METHOD,
            upstream_repository="internal://depth3d/development-nerf-mock",
            license_name="Development-only",
            license_url="internal://depth3d/development-nerf-mock/license",
            usage_scope=NerfUsageScope.DEVELOPMENT_ONLY,
            commercial_use_permitted_by_backend_license=False,
            redistribution_license_notice_required=False,
        )
    if backend_kind is NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
        return NerfBackendProvenance(
            backend_kind=backend_kind,
            backend_name="Nerfstudio Nerfacto",
            method_name=NERFSTUDIO_METHOD,
            upstream_repository=NERFSTUDIO_REPOSITORY,
            license_name="Apache-2.0",
            license_url=NERFSTUDIO_LICENSE_URL,
            usage_scope=NerfUsageScope.PERMISSIVE_GENERAL_USE,
            commercial_use_permitted_by_backend_license=True,
            redistribution_license_notice_required=True,
            runtime_dependency_name="tiny-cuda-nn",
            runtime_dependency_license=TINY_CUDA_NN_LICENSE,
        )
    raise ReconstructionValidationError(f"Unsupported NeRF backend: {backend_kind}")


def _require_canonical_provenance(value: NerfBackendProvenance) -> NerfBackendProvenance:
    expected = nerf_backend_provenance(value.backend_kind)
    if value != expected:
        raise ReconstructionValidationError("NeRF backend provenance does not match canonical policy.")
    return value


def _positive(value: object, field_name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and positive.")
    return parsed


def _nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ReconstructionValidationError(f"{field_name} must be a non-negative integer.")
    parsed = int(value)
    if parsed < 0:
        raise ReconstructionValidationError(f"{field_name} must be a non-negative integer.")
    return parsed


def _sha256(value: object, field_name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ReconstructionValidationError(f"{field_name} must be a SHA-256 hex digest.")
    return digest


@dataclass(frozen=True, slots=True)
class NerfBackendRuntimePin:
    backend_kind: NerfBackendKind
    release_tag: str
    revision_prefix: str
    method_name: str
    documented_memory_mib: int

    def __post_init__(self) -> None:
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("Runtime pin is only defined for real Nerfstudio backend.")
        object.__setattr__(self, "release_tag", require_nonempty(self.release_tag, field_name="release_tag"))
        revision = require_nonempty(self.revision_prefix, field_name="revision_prefix").lower()
        if len(revision) < 7 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ReconstructionValidationError("revision_prefix must be a Git hexadecimal prefix.")
        object.__setattr__(self, "revision_prefix", revision)
        object.__setattr__(self, "method_name", require_nonempty(self.method_name, field_name="method_name"))
        memory = _nonnegative_int(self.documented_memory_mib, "documented_memory_mib")
        if memory < 1:
            raise ReconstructionValidationError("documented_memory_mib must be positive.")
        object.__setattr__(self, "documented_memory_mib", memory)

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "release_tag": self.release_tag,
            "revision_prefix": self.revision_prefix,
            "method_name": self.method_name,
            "documented_memory_mib": self.documented_memory_mib,
        }


def nerf_backend_runtime_pin() -> NerfBackendRuntimePin:
    return NerfBackendRuntimePin(
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        release_tag=NERFSTUDIO_RELEASE_TAG,
        revision_prefix=NERFSTUDIO_RELEASE_REVISION_PREFIX,
        method_name=NERFSTUDIO_METHOD,
        documented_memory_mib=NERFSTUDIO_DOCUMENTED_MEMORY_MIB,
    )


def nerfstudio_camera_to_world_from_eia_pose(value: object) -> list[list[float]]:
    """Convert canonical OpenCV-style camera-to-world pose to Nerfstudio/OpenGL c2w.

    Canonical reconstruction cameras use X-right, Y-down, Z-forward.  Nerfstudio
    camera-to-world matrices use X-right, Y-up, Z-back.  Therefore the world
    transform is preserved and only camera-axis columns Y/Z are negated.
    """

    try:
        import numpy as np

        matrix = np.asarray(value, dtype=np.float64)
    except Exception as exc:
        raise ReconstructionValidationError("NeRF pose must be convertible to a 4x4 matrix.") from exc
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError("NeRF pose must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ReconstructionValidationError("NeRF pose must be homogeneous.")
    result = matrix.copy()
    result[:3, 1:3] *= -1.0
    return result.tolist()


def eia_pose_from_nerfstudio_camera_to_world(value: object) -> list[list[float]]:
    """Convert Nerfstudio/OpenGL camera-to-world to canonical OpenCV-style c2w.

    This is the exact inverse of :func:`nerfstudio_camera_to_world_from_eia_pose`.
    It is used by E3 dataset preparation so public real-scene camera poses enter
    the same canonical ``T_reconstruction_from_camera`` convention as the rest
    of the reconstruction stack.
    """

    # The convention change is its own inverse: negate camera-axis columns Y/Z.
    return nerfstudio_camera_to_world_from_eia_pose(value)


@dataclass(frozen=True, slots=True)
class NerfWorkerRequest:
    request_id: str
    backend_kind: NerfBackendKind
    profile: NerfOptimizationProfile
    dataset_manifest_path: str
    dataset_fingerprint_sha256: str
    output_dir: str
    scene_profile: DenseSceneProfile
    maximum_input_frames: int
    qualification_training_steps: int = 0
    method_name: str = NERFSTUDIO_METHOD
    schema_version: int = NERF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        if not isinstance(self.backend_kind, NerfBackendKind):
            raise TypeError("backend_kind must be NerfBackendKind.")
        if not isinstance(self.profile, NerfOptimizationProfile):
            raise TypeError("profile must be NerfOptimizationProfile.")
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        for name in ("dataset_manifest_path", "output_dir"):
            path = Path(require_nonempty(getattr(self, name), field_name=name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{name} must be absolute.")
            object.__setattr__(self, name, str(path))
        object.__setattr__(self, "dataset_fingerprint_sha256", _sha256(self.dataset_fingerprint_sha256, "dataset_fingerprint_sha256"))
        maximum = _nonnegative_int(self.maximum_input_frames, "maximum_input_frames")
        if maximum < 1:
            raise ReconstructionValidationError("maximum_input_frames must be >= 1.")
        object.__setattr__(self, "maximum_input_frames", maximum)
        qualification_steps = _nonnegative_int(self.qualification_training_steps, "qualification_training_steps")
        if qualification_steps > 256:
            raise ReconstructionValidationError("qualification_training_steps must be <= 256.")
        object.__setattr__(self, "qualification_training_steps", qualification_steps)
        method = require_nonempty(self.method_name, field_name="method_name")
        if method != NERFSTUDIO_METHOD:
            raise ReconstructionValidationError("Phase 8.7E1 supports the Nerfacto contract only.")
        object.__setattr__(self, "method_name", method)
        if int(self.schema_version) != NERF_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported NeRF worker request schema version.")
        object.__setattr__(self, "schema_version", NERF_SCHEMA_VERSION)

    @classmethod
    def from_dataset(
        cls,
        *,
        request_id: str,
        backend_kind: NerfBackendKind,
        profile: NerfOptimizationProfile,
        dataset: NeuralSceneDataset,
        dataset_manifest_path: str | Path,
        output_dir: str | Path,
        maximum_input_frames: int,
        qualification_training_steps: int = 0,
    ) -> "NerfWorkerRequest":
        if not isinstance(dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        return cls(
            request_id=request_id,
            backend_kind=backend_kind,
            profile=profile,
            dataset_manifest_path=str(Path(dataset_manifest_path)),
            dataset_fingerprint_sha256=neural_scene_dataset_fingerprint(dataset),
            output_dir=str(Path(output_dir)),
            scene_profile=dataset.scene_profile,
            maximum_input_frames=maximum_input_frames,
            qualification_training_steps=qualification_training_steps,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "backend_kind": self.backend_kind.value,
            "backend_provenance": nerf_backend_provenance(self.backend_kind).as_dict(),
            "profile": self.profile.value,
            "method_name": self.method_name,
            "dataset_manifest_path": self.dataset_manifest_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "output_dir": self.output_dir,
            "scene_profile": self.scene_profile.value,
            "maximum_input_frames": self.maximum_input_frames,
            "qualification_training_steps": self.qualification_training_steps,
        }


@dataclass(frozen=True, slots=True)
class NerfSceneArtifact:
    request_id: str
    backend_id: str
    backend_version: str
    backend_revision: str
    backend_provenance: NerfBackendProvenance
    method_name: str
    training_state: NerfTrainingState
    scene_role: NerfSceneRole
    artifact_dir: str
    config_path: str
    checkpoint_path: str
    dataset_fingerprint_sha256: str
    input_frame_count: int
    training_steps: int
    training_duration_s: float
    render_capable: bool
    device: str
    pose_convention: str
    reconstruction_frame_id: str
    metric_geometry_status: MetricGeometryStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = NERF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("request_id", "backend_id", "backend_version", "backend_revision", "device", "reconstruction_frame_id"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        if not isinstance(self.backend_provenance, NerfBackendProvenance):
            raise TypeError("backend_provenance must be NerfBackendProvenance.")
        object.__setattr__(self, "backend_provenance", _require_canonical_provenance(self.backend_provenance))
        if not isinstance(self.training_state, NerfTrainingState):
            raise TypeError("training_state must be NerfTrainingState.")
        if not isinstance(self.scene_role, NerfSceneRole) or self.scene_role is not NerfSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("Phase 8.7E NeRF scenes are appearance-only.")
        method = require_nonempty(self.method_name, field_name="method_name")
        if method != self.backend_provenance.method_name:
            raise ReconstructionValidationError("NeRF artifact method does not match backend provenance.")
        object.__setattr__(self, "method_name", method)
        for name in ("artifact_dir", "config_path", "checkpoint_path"):
            path = Path(require_nonempty(getattr(self, name), field_name=name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{name} must be absolute.")
            object.__setattr__(self, name, str(path))
        object.__setattr__(self, "dataset_fingerprint_sha256", _sha256(self.dataset_fingerprint_sha256, "dataset_fingerprint_sha256"))
        input_frames = _nonnegative_int(self.input_frame_count, "input_frame_count")
        if input_frames < 1:
            raise ReconstructionValidationError("NeRF artifacts require at least one input frame.")
        steps = _nonnegative_int(self.training_steps, "training_steps")
        duration = float(self.training_duration_s)
        if not isfinite(duration) or duration < 0.0:
            raise ReconstructionValidationError("training_duration_s must be finite and non-negative.")
        if self.training_state is NerfTrainingState.CONTRACT_ONLY:
            if steps != 0 or bool(self.render_capable):
                raise ReconstructionValidationError("Contract-only NeRF artifacts cannot claim training or rendering.")
        elif steps < 1 or not bool(self.render_capable):
            raise ReconstructionValidationError("Trained NeRF artifacts require steps and render capability.")
        if str(self.pose_convention) != NERF_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported NeRF pose convention.")
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        unit = self.unit_to_metre
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None:
                raise ReconstructionValidationError("Arbitrary NeRF artifacts must not expose unit_to_metre.")
        else:
            if unit is None:
                raise ReconstructionValidationError("Metric-aware NeRF artifacts require unit_to_metre.")
            unit = _positive(unit, "unit_to_metre")
        uncertainty = self.relative_scale_uncertainty
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty < 0.0:
                raise ReconstructionValidationError("relative_scale_uncertainty must be finite and non-negative.")
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY and uncertainty is not None:
            raise ReconstructionValidationError("Arbitrary NeRF artifacts must not expose metric uncertainty.")
        if bool(self.canonical_geometry_authoritative):
            raise ReconstructionValidationError("NeRF appearance artifacts must never be canonical inspection geometry.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("NeRF appearance artifacts must not claim certified metrology.")
        if int(self.schema_version) != NERF_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported NeRF artifact schema version.")
        object.__setattr__(self, "input_frame_count", input_frames)
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "training_duration_s", duration)
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", NERF_SCHEMA_VERSION)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "NerfSceneArtifact":
        if not isinstance(payload, Mapping):
            raise NerfWorkerExecutionError("NeRF worker result must be a JSON object.")
        try:
            return cls(
                request_id=str(payload["request_id"]),
                backend_id=str(payload["backend_id"]),
                backend_version=str(payload["backend_version"]),
                backend_revision=str(payload["backend_revision"]),
                backend_provenance=NerfBackendProvenance.from_payload(payload["backend_provenance"]),
                method_name=str(payload["method_name"]),
                training_state=NerfTrainingState(str(payload["training_state"])),
                scene_role=NerfSceneRole(str(payload.get("scene_role", "appearance_only"))),
                artifact_dir=str(payload["artifact_dir"]),
                config_path=str(payload["config_path"]),
                checkpoint_path=str(payload["checkpoint_path"]),
                dataset_fingerprint_sha256=str(payload["dataset_fingerprint_sha256"]),
                input_frame_count=int(payload["input_frame_count"]),
                training_steps=int(payload["training_steps"]),
                training_duration_s=float(payload["training_duration_s"]),
                render_capable=bool(payload["render_capable"]),
                device=str(payload["device"]),
                pose_convention=str(payload["pose_convention"]),
                reconstruction_frame_id=str(payload["reconstruction_frame_id"]),
                metric_geometry_status=MetricGeometryStatus(str(payload["metric_geometry_status"])),
                unit_to_metre=payload.get("unit_to_metre"),
                relative_scale_uncertainty=payload.get("relative_scale_uncertainty"),
                canonical_geometry_authoritative=bool(payload.get("canonical_geometry_authoritative", False)),
                certified_metrology=bool(payload.get("certified_metrology", False)),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise NerfWorkerExecutionError(f"Invalid NeRF worker result: {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "backend_revision": self.backend_revision,
            "backend_provenance": self.backend_provenance.as_dict(),
            "method_name": self.method_name,
            "training_state": self.training_state.value,
            "scene_role": self.scene_role.value,
            "artifact_dir": self.artifact_dir,
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "input_frame_count": self.input_frame_count,
            "training_steps": self.training_steps,
            "training_duration_s": self.training_duration_s,
            "render_capable": self.render_capable,
            "device": self.device,
            "pose_convention": self.pose_convention,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": self.certified_metrology,
        }


@dataclass(frozen=True, slots=True)
class NerfWorkerProcessSpec:
    python_executable: str
    worker_script: str
    timeout_s: float = 900.0
    require_isolated_interpreter: bool = True

    def __post_init__(self) -> None:
        python = Path(require_nonempty(self.python_executable, field_name="python_executable"))
        worker = Path(require_nonempty(self.worker_script, field_name="worker_script"))
        if not python.is_absolute() or not worker.is_absolute():
            raise ReconstructionValidationError("NeRF worker Python executable and script must be absolute.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "worker_script", str(worker))
        object.__setattr__(self, "timeout_s", _positive(self.timeout_s, "timeout_s"))
        object.__setattr__(self, "require_isolated_interpreter", bool(self.require_isolated_interpreter))

    @property
    def interpreter_is_isolated(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)


def run_nerf_external_worker(spec: NerfWorkerProcessSpec, request: NerfWorkerRequest) -> NerfSceneArtifact:
    if not isinstance(spec, NerfWorkerProcessSpec):
        raise TypeError("spec must be NerfWorkerProcessSpec.")
    if not isinstance(request, NerfWorkerRequest):
        raise TypeError("request must be NerfWorkerRequest.")
    if spec.require_isolated_interpreter and not spec.interpreter_is_isolated:
        raise NerfWorkerExecutionError("NeRF must run in an isolated interpreter, not production Python.")
    python = Path(spec.python_executable)
    worker = Path(spec.worker_script)
    if not python.is_file():
        raise NerfWorkerExecutionError(f"NeRF worker Python does not exist: {python}")
    if not worker.is_file():
        raise NerfWorkerExecutionError(f"NeRF worker script does not exist: {worker}")
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eia-nerf-request-", dir=str(output_dir)) as tmp:
        tmp_dir = Path(tmp)
        request_path = tmp_dir / "request.json"
        result_path = tmp_dir / "result.json"
        request_path.write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [str(python), str(worker), "--request", str(request_path), "--result", str(result_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(worker.parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=spec.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NerfWorkerExecutionError("Isolated NeRF worker timed out.") from exc
        except OSError as exc:
            raise NerfWorkerExecutionError("Could not start isolated NeRF worker.") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown worker failure"
            raise NerfWorkerExecutionError(f"Isolated NeRF worker exited with code {completed.returncode}: {detail}")
        if not result_path.is_file():
            raise NerfWorkerExecutionError("Isolated NeRF worker did not produce result.json.")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NerfWorkerExecutionError("Could not parse NeRF worker result.json.") from exc
        artifact = NerfSceneArtifact.from_payload(payload)
        if artifact.request_id != request.request_id:
            raise NerfWorkerExecutionError("NeRF worker result request_id does not match request.")
        if artifact.dataset_fingerprint_sha256 != request.dataset_fingerprint_sha256:
            raise NerfWorkerExecutionError("NeRF worker result dataset fingerprint does not match request.")
        if artifact.backend_provenance != nerf_backend_provenance(request.backend_kind):
            raise NerfWorkerExecutionError("NeRF worker result provenance does not match requested backend.")
        if artifact.method_name != request.method_name:
            raise NerfWorkerExecutionError("NeRF worker result method does not match request.")
        return artifact


_NERFSTUDIO_PROBE_CODE = r'''
import importlib
import importlib.metadata as md
import json
import shutil
from pathlib import Path
import sys

payload = {
    "available": False,
    "nerfstudio_version": "",
    "torch_version": "",
    "tinycudann_available": False,
    "ns_train_available": False,
    "ns_render_available": False,
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_names": [],
    "cuda_compute_capabilities": [],
    "cuda_total_vram_mib": [],
    "torch_cuda_version": "",
    "error": "",
}
try:
    torch = importlib.import_module("torch")
    importlib.import_module("nerfstudio")
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
else:
    payload["available"] = True
    payload["torch_version"] = str(getattr(torch, "__version__", ""))
    try:
        payload["nerfstudio_version"] = md.version("nerfstudio")
    except Exception:
        payload["nerfstudio_version"] = ""
    try:
        importlib.import_module("tinycudann")
        payload["tinycudann_available"] = True
    except Exception as exc:
        payload["error"] = f"tinycudann probe failed: {type(exc).__name__}: {exc}"
    bin_dir = Path(sys.executable).resolve().parent
    payload["ns_train_available"] = ((bin_dir / "ns-train").is_file() or shutil.which("ns-train") is not None)
    payload["ns_render_available"] = ((bin_dir / "ns-render").is_file() or shutil.which("ns-render") is not None)
    payload["torch_cuda_version"] = str(getattr(getattr(torch, "version", None), "cuda", "") or "")
    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        if payload["cuda_available"]:
            payload["cuda_device_names"] = [str(torch.cuda.get_device_name(i)) for i in range(payload["cuda_device_count"])]
            payload["cuda_compute_capabilities"] = [list(torch.cuda.get_device_capability(i)) for i in range(payload["cuda_device_count"])]
            payload["cuda_total_vram_mib"] = [float(torch.cuda.get_device_properties(i).total_memory / (1024 ** 2)) for i in range(payload["cuda_device_count"])]
    except Exception as exc:
        payload["error"] = f"CUDA probe failed: {type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''


@dataclass(frozen=True, slots=True)
class NerfEnvironmentProbe:
    python_executable: str
    available: bool
    nerfstudio_version: str = ""
    torch_version: str = ""
    tinycudann_available: bool = False
    ns_train_available: bool = False
    ns_render_available: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_names: tuple[str, ...] = ()
    cuda_compute_capabilities: tuple[tuple[int, int], ...] = ()
    cuda_total_vram_mib: tuple[float, ...] = ()
    torch_cuda_version: str = ""
    error: str = ""

    def __post_init__(self) -> None:
        python = Path(require_nonempty(self.python_executable, field_name="python_executable"))
        if not python.is_absolute():
            raise ReconstructionValidationError("NeRF probe python_executable must be absolute.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "nerfstudio_version", str(self.nerfstudio_version).strip())
        object.__setattr__(self, "torch_version", str(self.torch_version).strip())
        object.__setattr__(self, "tinycudann_available", bool(self.tinycudann_available))
        object.__setattr__(self, "ns_train_available", bool(self.ns_train_available))
        object.__setattr__(self, "ns_render_available", bool(self.ns_render_available))
        object.__setattr__(self, "cuda_available", bool(self.cuda_available))
        count = _nonnegative_int(self.cuda_device_count, "cuda_device_count")
        names = tuple(str(value).strip() for value in self.cuda_device_names if str(value).strip())
        capabilities = tuple(tuple(int(part) for part in value) for value in self.cuda_compute_capabilities)
        total_vram = tuple(float(value) for value in self.cuda_total_vram_mib)
        if count != len(names) or count != len(capabilities) or count != len(total_vram):
            raise ReconstructionValidationError("NeRF CUDA device evidence lengths must match cuda_device_count.")
        if any(len(value) != 2 or value[0] < 1 or value[1] < 0 for value in capabilities):
            raise ReconstructionValidationError("NeRF compute capabilities must contain positive major/minor pairs.")
        if any(not isfinite(value) or value <= 0.0 for value in total_vram):
            raise ReconstructionValidationError("NeRF CUDA VRAM values must be finite and positive.")
        if count > 0 and not self.cuda_available:
            raise ReconstructionValidationError("NeRF CUDA device count cannot be positive when CUDA is unavailable.")
        if self.available and (not self.nerfstudio_version or not self.torch_version):
            raise ReconstructionValidationError("Available NeRF environments must report Nerfstudio and torch versions.")
        object.__setattr__(self, "cuda_device_count", count)
        object.__setattr__(self, "cuda_device_names", names)
        object.__setattr__(self, "cuda_compute_capabilities", capabilities)
        object.__setattr__(self, "cuda_total_vram_mib", total_vram)
        object.__setattr__(self, "torch_cuda_version", str(self.torch_cuda_version).strip())
        object.__setattr__(self, "error", str(self.error).strip())

    @property
    def interpreter_is_isolated(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)

    @property
    def backend_ready(self) -> bool:
        return (
            self.available
            and self.tinycudann_available
            and self.ns_train_available
            and self.ns_render_available
            and self.cuda_available
            and self.cuda_device_count > 0
        )

    @property
    def backend_provenance(self) -> NerfBackendProvenance:
        return nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "interpreter_is_isolated": self.interpreter_is_isolated,
            "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
            "backend_provenance": self.backend_provenance.as_dict(),
            "runtime_pin": nerf_backend_runtime_pin().as_dict(),
            "available": self.available,
            "nerfstudio_version": self.nerfstudio_version,
            "torch_version": self.torch_version,
            "tinycudann_available": self.tinycudann_available,
            "ns_train_available": self.ns_train_available,
            "ns_render_available": self.ns_render_available,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": list(self.cuda_device_names),
            "cuda_compute_capabilities": [list(value) for value in self.cuda_compute_capabilities],
            "cuda_total_vram_mib": list(self.cuda_total_vram_mib),
            "torch_cuda_version": self.torch_cuda_version,
            "backend_ready": self.backend_ready,
            "error": self.error,
        }


def probe_nerf_environment(python_executable: str | Path, *, timeout_s: float = 30.0) -> NerfEnvironmentProbe:
    python = Path(python_executable)
    if not python.is_absolute():
        raise ReconstructionValidationError("NeRF probe python_executable must be absolute.")
    if not python.is_file():
        return NerfEnvironmentProbe(python_executable=str(python), available=False, error="python_executable_missing")
    try:
        completed = subprocess.run(
            [str(python), "-c", _NERFSTUDIO_PROBE_CODE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_positive(timeout_s, "timeout_s"),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return NerfEnvironmentProbe(python_executable=str(python), available=False, error=f"probe_failed:{type(exc).__name__}:{exc}")
    line = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    if completed.returncode != 0 or not line:
        detail = completed.stderr.strip() or "probe_process_failed"
        return NerfEnvironmentProbe(python_executable=str(python), available=False, error=detail)
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        return NerfEnvironmentProbe(python_executable=str(python), available=False, error=f"invalid_probe_json:{exc}")
    return NerfEnvironmentProbe(
        python_executable=str(python),
        available=bool(payload.get("available", False)),
        nerfstudio_version=str(payload.get("nerfstudio_version", "")),
        torch_version=str(payload.get("torch_version", "")),
        tinycudann_available=bool(payload.get("tinycudann_available", False)),
        ns_train_available=bool(payload.get("ns_train_available", False)),
        ns_render_available=bool(payload.get("ns_render_available", False)),
        cuda_available=bool(payload.get("cuda_available", False)),
        cuda_device_count=int(payload.get("cuda_device_count", 0)),
        cuda_device_names=tuple(payload.get("cuda_device_names", ())),
        cuda_compute_capabilities=tuple(tuple(value) for value in payload.get("cuda_compute_capabilities", ())),
        cuda_total_vram_mib=tuple(payload.get("cuda_total_vram_mib", ())),
        torch_cuda_version=str(payload.get("torch_cuda_version", "")),
        error=str(payload.get("error", "")),
    )
