"""Phase 8.7E4 NeRF performance and backend-hardening contracts.

E4 is intentionally limited to the isolated Nerfstudio/Nerfacto backend
boundary.  Whole reconstruction-engine cancellation, checkpoint/resume, and
endurance remain Phase 8.7G responsibilities.  Whole-application RTX profiling
remains Phase 8.9D.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.nerf import NerfBackendKind, NerfOptimizationProfile

NERF_E4_SCHEMA_VERSION = 1
NERF_E4_RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB = 5500.0
NERF_E4_RTX5090_NOMINAL_VRAM_MIB = 32768
NERF_E4_RTX5090_COMPUTE_CAPABILITY = (12, 0)
NERF_E4_BLACKWELL_MIN_CUDA_TOOLKIT = "12.8"
NERF_E4_DEFAULT_RENDER_REPEATS = 8
NERF_E4_MIN_RENDER_REPEATS = 3
NERF_E4_MAX_RENDER_REPEATS = 100


class NerfHardwareProfile(str, Enum):
    RTX_2060_6GB = "rtx_2060_6gb"
    RTX_5090_32GB = "rtx_5090_32gb"
    CUSTOM = "custom"


class NerfBackendRunState(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"
    RECOVERED = "recovered"
    FAILED = "failed"


class NerfBackendRobustnessScenario(str, Enum):
    CANCELLATION = "cancellation"
    TIMEOUT = "timeout"
    CRASH = "crash"
    PARTIAL_ARTIFACT = "partial_artifact"


def _finite_nonnegative(value: object, field_name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
    return parsed


def _finite_positive(value: object, field_name: str) -> float:
    parsed = _finite_nonnegative(value, field_name)
    if parsed <= 0.0:
        raise ReconstructionValidationError(f"{field_name} must be positive.")
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


def nerf_rtx5090_readiness_profile() -> dict[str, Any]:
    """Return the E4 Blackwell readiness contract without inventing measurements.

    The E2-qualified runtime is CUDA 11.8 and tiny-cuda-nn was compiled only for
    SM75.  NVIDIA first added compiler support for GeForce RTX 5090 / SM120 in
    CUDA 12.8, so the current Turing-qualified runtime must not be presented as
    RTX 5090-qualified.  A separate Blackwell-capable isolated runtime is a
    prerequisite for the real whole-application profiling gate in Phase 8.9D.
    """

    return {
        "profile": NerfHardwareProfile.RTX_5090_32GB.value,
        "target_gpu": "NVIDIA GeForce RTX 5090",
        "nominal_vram_mib": NERF_E4_RTX5090_NOMINAL_VRAM_MIB,
        "compute_capability": list(NERF_E4_RTX5090_COMPUTE_CAPABILITY),
        "minimum_cuda_toolkit_for_native_sm120": NERF_E4_BLACKWELL_MIN_CUDA_TOOLKIT,
        "current_e2_qualified_cuda_toolkit": "11.8",
        "current_e2_tcnn_architectures": ["75"],
        "current_runtime_blackwell_qualified": False,
        "profile_defined": True,
        "real_performance_measurements": None,
        "required_before_real_5090_execution": [
            "separate_isolated_Blackwell_runtime",
            "CUDA_toolkit_12_8_or_newer",
            "tiny_cuda_nn_SM120_build",
            "real_RTX5090_backend_qualification",
        ],
        "whole_application_validation_gate": "8.9D",
        "no_fake_numbers": True,
    }


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"artifact file is missing: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NerfHardwareEvidence:
    profile: NerfHardwareProfile
    gpu_name: str
    gpu_uuid: str
    compute_capability: tuple[int, int]
    total_vram_mib: float
    driver_version: str
    torch_cuda_version: str
    measured: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.profile, NerfHardwareProfile):
            raise TypeError("profile must be NerfHardwareProfile.")
        object.__setattr__(self, "gpu_name", require_nonempty(self.gpu_name, field_name="gpu_name"))
        object.__setattr__(self, "gpu_uuid", require_nonempty(self.gpu_uuid, field_name="gpu_uuid"))
        capability = tuple(int(v) for v in self.compute_capability)
        if len(capability) != 2 or min(capability) < 0:
            raise ReconstructionValidationError("compute_capability must contain two non-negative integers.")
        object.__setattr__(self, "compute_capability", capability)
        object.__setattr__(self, "total_vram_mib", _finite_positive(self.total_vram_mib, "total_vram_mib"))
        object.__setattr__(self, "driver_version", require_nonempty(self.driver_version, field_name="driver_version"))
        object.__setattr__(self, "torch_cuda_version", require_nonempty(self.torch_cuda_version, field_name="torch_cuda_version"))
        object.__setattr__(self, "measured", bool(self.measured))

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "compute_capability": list(self.compute_capability),
            "total_vram_mib": self.total_vram_mib,
            "driver_version": self.driver_version,
            "torch_cuda_version": self.torch_cuda_version,
            "measured": self.measured,
        }


@dataclass(frozen=True, slots=True)
class NerfPerformanceMetrics:
    training_steps: int
    training_duration_s: float
    training_peak_vram_mib: float
    checkpoint_reload_s: float
    render_latency_ms: tuple[float, ...]
    render_peak_vram_mib: float
    render_width: int
    render_height: int
    rendering_definition: str = "native_model_full_image_cuda_latency_excludes_file_io"

    def __post_init__(self) -> None:
        steps = _nonnegative_int(self.training_steps, "training_steps")
        if steps < 1:
            raise ReconstructionValidationError("training_steps must be positive.")
        object.__setattr__(self, "training_steps", steps)
        object.__setattr__(self, "training_duration_s", _finite_positive(self.training_duration_s, "training_duration_s"))
        object.__setattr__(self, "training_peak_vram_mib", _finite_nonnegative(self.training_peak_vram_mib, "training_peak_vram_mib"))
        object.__setattr__(self, "checkpoint_reload_s", _finite_positive(self.checkpoint_reload_s, "checkpoint_reload_s"))
        samples = tuple(_finite_positive(value, "render_latency_ms") for value in self.render_latency_ms)
        if len(samples) < NERF_E4_MIN_RENDER_REPEATS:
            raise ReconstructionValidationError(
                f"render_latency_ms requires at least {NERF_E4_MIN_RENDER_REPEATS} samples."
            )
        if len(samples) > NERF_E4_MAX_RENDER_REPEATS:
            raise ReconstructionValidationError("too many render latency samples.")
        object.__setattr__(self, "render_latency_ms", samples)
        object.__setattr__(self, "render_peak_vram_mib", _finite_nonnegative(self.render_peak_vram_mib, "render_peak_vram_mib"))
        width = _nonnegative_int(self.render_width, "render_width")
        height = _nonnegative_int(self.render_height, "render_height")
        if width < 1 or height < 1:
            raise ReconstructionValidationError("render dimensions must be positive.")
        object.__setattr__(self, "render_width", width)
        object.__setattr__(self, "render_height", height)
        object.__setattr__(self, "rendering_definition", require_nonempty(self.rendering_definition, field_name="rendering_definition"))

    @property
    def mean_render_latency_ms(self) -> float:
        return sum(self.render_latency_ms) / len(self.render_latency_ms)

    @property
    def p50_render_latency_ms(self) -> float:
        return float(median(self.render_latency_ms))

    @property
    def p95_render_latency_ms(self) -> float:
        ordered = sorted(self.render_latency_ms)
        index = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
        return float(ordered[index])

    @property
    def render_fps(self) -> float:
        return 1000.0 / self.mean_render_latency_ms

    def as_dict(self) -> dict[str, Any]:
        return {
            "training_steps": self.training_steps,
            "training_duration_s": self.training_duration_s,
            "training_peak_vram_mib": self.training_peak_vram_mib,
            "checkpoint_reload_s": self.checkpoint_reload_s,
            "render_latency_ms": list(self.render_latency_ms),
            "mean_render_latency_ms": self.mean_render_latency_ms,
            "p50_render_latency_ms": self.p50_render_latency_ms,
            "p95_render_latency_ms": self.p95_render_latency_ms,
            "render_fps": self.render_fps,
            "render_peak_vram_mib": self.render_peak_vram_mib,
            "render_width": self.render_width,
            "render_height": self.render_height,
            "rendering_definition": self.rendering_definition,
        }


@dataclass(frozen=True, slots=True)
class NerfArtifactIntegrityEvidence:
    config_path: str
    checkpoint_path: str
    config_size_bytes: int
    checkpoint_size_bytes: int
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_reload_s: float
    render_probe_passed: bool
    missing_artifact_rejected: bool
    malformed_checkpoint_rejected: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", require_nonempty(self.config_path, field_name="config_path"))
        object.__setattr__(self, "checkpoint_path", require_nonempty(self.checkpoint_path, field_name="checkpoint_path"))
        for field_name in ("config_size_bytes", "checkpoint_size_bytes"):
            parsed = _nonnegative_int(getattr(self, field_name), field_name)
            if parsed < 1:
                raise ReconstructionValidationError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, parsed)
        object.__setattr__(self, "config_sha256", _sha256(self.config_sha256, "config_sha256"))
        object.__setattr__(self, "checkpoint_sha256", _sha256(self.checkpoint_sha256, "checkpoint_sha256"))
        object.__setattr__(self, "checkpoint_reload_s", _finite_positive(self.checkpoint_reload_s, "checkpoint_reload_s"))
        for field_name in ("render_probe_passed", "missing_artifact_rejected", "malformed_checkpoint_rejected"):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    @property
    def passed(self) -> bool:
        return bool(
            self.render_probe_passed
            and self.missing_artifact_rejected
            and self.malformed_checkpoint_rejected
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "config_size_bytes": self.config_size_bytes,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "config_sha256": self.config_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_reload_s": self.checkpoint_reload_s,
            "render_probe_passed": self.render_probe_passed,
            "missing_artifact_rejected": self.missing_artifact_rejected,
            "malformed_checkpoint_rejected": self.malformed_checkpoint_rejected,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class NerfPerformanceHardeningReport:
    hardware: NerfHardwareEvidence
    profile: NerfOptimizationProfile
    performance: NerfPerformanceMetrics | None
    artifact_integrity: NerfArtifactIntegrityEvidence | None
    real_measurements_complete: bool
    rtx5090_readiness_only: bool = False
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
    schema_version: int = NERF_E4_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.hardware, NerfHardwareEvidence):
            raise TypeError("hardware must be NerfHardwareEvidence.")
        if not isinstance(self.profile, NerfOptimizationProfile):
            raise TypeError("profile must be NerfOptimizationProfile.")
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("E4 only supports the E2-qualified Nerfstudio backend.")
        if int(self.schema_version) != NERF_E4_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported E4 schema version.")
        real = bool(self.real_measurements_complete)
        readiness_only = bool(self.rtx5090_readiness_only)
        if readiness_only and real:
            raise ReconstructionValidationError("RTX 5090 readiness-only reports cannot claim real measurements.")
        if real and (self.performance is None or self.artifact_integrity is None):
            raise ReconstructionValidationError("real E4 measurements require performance and artifact evidence.")
        if readiness_only and self.hardware.profile is not NerfHardwareProfile.RTX_5090_32GB:
            raise ReconstructionValidationError("rtx5090_readiness_only requires the RTX 5090 hardware profile.")
        object.__setattr__(self, "real_measurements_complete", real)
        object.__setattr__(self, "rtx5090_readiness_only", readiness_only)
        object.__setattr__(self, "schema_version", NERF_E4_SCHEMA_VERSION)

    @property
    def passed(self) -> bool:
        if self.rtx5090_readiness_only:
            return not self.real_measurements_complete and not self.hardware.measured
        return bool(
            self.real_measurements_complete
            and self.performance is not None
            and self.artifact_integrity is not None
            and self.artifact_integrity.passed
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend_kind": self.backend_kind.value,
            "optimization_profile": self.profile.value,
            "hardware": self.hardware.as_dict(),
            "performance": None if self.performance is None else self.performance.as_dict(),
            "artifact_integrity": None if self.artifact_integrity is None else self.artifact_integrity.as_dict(),
            "real_measurements_complete": self.real_measurements_complete,
            "rtx5090_readiness_only": self.rtx5090_readiness_only,
            "passed": self.passed,
            "scope": "nerf_backend_only",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "whole_engine_hardening_deferred_to": "8.7G",
            "whole_application_gpu_profiling_deferred_to": "8.9D",
        }


@dataclass(frozen=True, slots=True)
class NerfLifecycleEvidence:
    state: NerfBackendRunState
    worker_exit_code: int | None
    cancellation_requested: bool = False
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, NerfBackendRunState):
            raise TypeError("state must be NerfBackendRunState.")
        if self.worker_exit_code is not None:
            object.__setattr__(self, "worker_exit_code", int(self.worker_exit_code))
        object.__setattr__(self, "cancellation_requested", bool(self.cancellation_requested))
        object.__setattr__(self, "recovery_attempted", bool(self.recovery_attempted))
        object.__setattr__(self, "recovery_succeeded", bool(self.recovery_succeeded))
        object.__setattr__(self, "error", str(self.error).strip())

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "worker_exit_code": self.worker_exit_code,
            "cancellation_requested": self.cancellation_requested,
            "recovery_attempted": self.recovery_attempted,
            "recovery_succeeded": self.recovery_succeeded,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class NerfBackendRobustnessEvidence:
    scenario: NerfBackendRobustnessScenario
    fault_lifecycle: NerfLifecycleEvidence
    process_group_clean: bool
    partial_artifacts_detected: bool
    partial_artifacts_cleaned: bool
    recovery_lifecycle: NerfLifecycleEvidence
    recovery_result_valid: bool
    backend_kind: NerfBackendKind = NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, NerfBackendRobustnessScenario):
            raise TypeError("scenario must be NerfBackendRobustnessScenario.")
        if self.backend_kind is not NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL:
            raise ReconstructionValidationError("E4 robustness only supports Nerfstudio/Nerfacto.")
        if not isinstance(self.fault_lifecycle, NerfLifecycleEvidence):
            raise TypeError("fault_lifecycle must be NerfLifecycleEvidence.")
        if not isinstance(self.recovery_lifecycle, NerfLifecycleEvidence):
            raise TypeError("recovery_lifecycle must be NerfLifecycleEvidence.")
        for field_name in (
            "process_group_clean", "partial_artifacts_detected", "partial_artifacts_cleaned", "recovery_result_valid"
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    @property
    def passed(self) -> bool:
        expected = {
            NerfBackendRobustnessScenario.CANCELLATION: NerfBackendRunState.CANCELLED,
            NerfBackendRobustnessScenario.TIMEOUT: NerfBackendRunState.TIMED_OUT,
            NerfBackendRobustnessScenario.CRASH: NerfBackendRunState.CRASHED,
            NerfBackendRobustnessScenario.PARTIAL_ARTIFACT: NerfBackendRunState.CRASHED,
        }[self.scenario]
        partial_ok = True
        if self.scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT:
            partial_ok = self.partial_artifacts_detected and self.partial_artifacts_cleaned
        return bool(
            self.fault_lifecycle.state is expected
            and self.process_group_clean
            and partial_ok
            and self.recovery_lifecycle.state is NerfBackendRunState.RECOVERED
            and self.recovery_lifecycle.recovery_succeeded
            and self.recovery_result_valid
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "scenario": self.scenario.value,
            "fault_lifecycle": self.fault_lifecycle.as_dict(),
            "process_group_clean": self.process_group_clean,
            "partial_artifacts_detected": self.partial_artifacts_detected,
            "partial_artifacts_cleaned": self.partial_artifacts_cleaned,
            "recovery_lifecycle": self.recovery_lifecycle.as_dict(),
            "recovery_result_valid": self.recovery_result_valid,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class NerfBackendRobustnessReport:
    evidence: tuple[NerfBackendRobustnessEvidence, ...]
    schema_version: int = NERF_E4_SCHEMA_VERSION

    def __post_init__(self) -> None:
        values = tuple(self.evidence)
        if len(values) != len(NerfBackendRobustnessScenario):
            raise ReconstructionValidationError("E4 robustness report must contain all four backend scenarios.")
        scenarios = tuple(item.scenario for item in values)
        if set(scenarios) != set(NerfBackendRobustnessScenario):
            raise ReconstructionValidationError("E4 robustness report scenarios are incomplete or duplicated.")
        if int(self.schema_version) != NERF_E4_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported E4 robustness schema version.")
        object.__setattr__(self, "evidence", values)
        object.__setattr__(self, "schema_version", NERF_E4_SCHEMA_VERSION)

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.evidence)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": "8.7E4",
            "scope": "nerf_backend_process_only",
            "evidence": [item.as_dict() for item in self.evidence],
            "passed": self.passed,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "whole_engine_hardening_deferred_to": "8.7G",
        }


def validate_nerf_artifact_files(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    artifact_root: str | Path | None = None,
) -> tuple[Path, Path]:
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not config.is_file() or config.stat().st_size <= 0:
        raise ReconstructionValidationError(f"NeRF config artifact is missing or empty: {config}")
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise ReconstructionValidationError(f"NeRF checkpoint artifact is missing or empty: {checkpoint}")
    if config.suffix.lower() not in {".yml", ".yaml"}:
        raise ReconstructionValidationError("NeRF config artifact must be YAML.")
    if checkpoint.suffix.lower() != ".ckpt":
        raise ReconstructionValidationError("NeRF checkpoint artifact must use .ckpt.")
    if artifact_root is not None:
        root = Path(artifact_root).expanduser().resolve()
        for candidate in (config, checkpoint):
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ReconstructionValidationError("NeRF artifact escapes the declared artifact root.") from exc
    return config, checkpoint


def write_nerf_e4_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
