"""Phase 8.7D3 real-scene Gaussian benchmark contracts.

This module is intentionally dependency-light. It defines reproducible benchmark
splits, target-hardware identities, result/evidence contracts, and JSON writers.
Actual PSNR/SSIM/LPIPS computation and real backend execution are added in later
D3 steps so Step 1 cannot fabricate quality or performance measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from core.contracts.validation import require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.gaussian_splatting import (
    GaussianBackendKind,
    GaussianOptimizationProfile,
)
from core.reconstruction.neural_scene import (
    NeuralSceneDataset,
    neural_scene_dataset_fingerprint,
)

GAUSSIAN_BENCHMARK_SCHEMA_VERSION = 1


class GaussianBenchmarkHardwareProfile(str, Enum):
    """Named D3 target-hardware profiles without embedding tuning policy yet."""

    RTX_2060_6GB = "rtx_2060_6gb"
    RTX_5090_32GB = "rtx_5090_32gb"
    CUSTOM = "custom"


class GaussianBenchmarkRunState(str, Enum):
    """Lifecycle outcome for one backend benchmark attempt."""

    PLANNED = "planned"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"
    RECOVERED = "recovered"
    FAILED = "failed"


def _finite_nonnegative(value: object, field_name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ReconstructionValidationError(f"{field_name} must be finite and non-negative.")
    return parsed


def _optional_finite_nonnegative(value: object | None, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_nonnegative(value, field_name)


def _optional_int(value: object | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ReconstructionValidationError(f"{field_name} must be an integer when present.")
    parsed = int(value)
    return parsed


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class GaussianImageQualityMetrics:
    """Aggregate held-out-view quality metrics.

    Step 1 permits an unmeasured object only. Later D3 metric execution will
    populate all three values together from held-out views.
    """

    psnr_db: float | None = None
    ssim: float | None = None
    lpips: float | None = None

    def __post_init__(self) -> None:
        values = (self.psnr_db, self.ssim, self.lpips)
        present = tuple(value is not None for value in values)
        if any(present) and not all(present):
            raise ReconstructionValidationError(
                "PSNR, SSIM, and LPIPS must be measured together for a benchmark result."
            )
        if not all(present):
            return
        psnr = _finite_nonnegative(self.psnr_db, "psnr_db")
        ssim = float(self.ssim)
        lpips = _finite_nonnegative(self.lpips, "lpips")
        if not isfinite(ssim) or not 0.0 <= ssim <= 1.0:
            raise ReconstructionValidationError("ssim must be finite and lie in [0, 1].")
        object.__setattr__(self, "psnr_db", psnr)
        object.__setattr__(self, "ssim", ssim)
        object.__setattr__(self, "lpips", lpips)

    @property
    def measured(self) -> bool:
        return self.psnr_db is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured,
            "psnr_db": self.psnr_db,
            "ssim": self.ssim,
            "lpips": self.lpips,
        }


@dataclass(frozen=True, slots=True)
class GaussianPerformanceMetrics:
    """Comparable performance measurements for one backend run."""

    peak_vram_mib: float | None = None
    training_duration_s: float | None = None
    render_fps: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("peak_vram_mib", "training_duration_s", "render_fps"):
            object.__setattr__(
                self,
                field_name,
                _optional_finite_nonnegative(getattr(self, field_name), field_name),
            )

    @property
    def measured(self) -> bool:
        return all(
            value is not None
            for value in (self.peak_vram_mib, self.training_duration_s, self.render_fps)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "measured": self.measured,
            "peak_vram_mib": self.peak_vram_mib,
            "training_duration_s": self.training_duration_s,
            "render_fps": self.render_fps,
        }


@dataclass(frozen=True, slots=True)
class GaussianLifecycleEvidence:
    """Crash/cancel/timeout/recovery evidence for one benchmark attempt."""

    state: GaussianBenchmarkRunState = GaussianBenchmarkRunState.PLANNED
    worker_exit_code: int | None = None
    cancellation_requested: bool = False
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, GaussianBenchmarkRunState):
            raise TypeError("state must be GaussianBenchmarkRunState.")
        object.__setattr__(self, "worker_exit_code", _optional_int(self.worker_exit_code, "worker_exit_code"))
        object.__setattr__(self, "cancellation_requested", bool(self.cancellation_requested))
        object.__setattr__(self, "recovery_attempted", bool(self.recovery_attempted))
        object.__setattr__(self, "recovery_succeeded", bool(self.recovery_succeeded))
        object.__setattr__(self, "error", str(self.error).strip())
        if self.recovery_succeeded and not self.recovery_attempted:
            raise ReconstructionValidationError("Successful recovery requires recovery_attempted=true.")
        if self.state is GaussianBenchmarkRunState.CANCELLED and not self.cancellation_requested:
            raise ReconstructionValidationError("Cancelled result requires cancellation_requested=true.")
        if self.state is GaussianBenchmarkRunState.RECOVERED and not self.recovery_succeeded:
            raise ReconstructionValidationError("Recovered state requires recovery_succeeded=true.")
        if self.state in {
            GaussianBenchmarkRunState.CRASHED,
            GaussianBenchmarkRunState.TIMED_OUT,
            GaussianBenchmarkRunState.FAILED,
        } and not self.error:
            raise ReconstructionValidationError(f"{self.state.value} lifecycle evidence requires an error.")

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
class RealSceneGaussianBenchmarkSpec:
    """Reproducible real-scene split and backend/hardware benchmark identity."""

    benchmark_id: str
    source_label: str
    dataset: NeuralSceneDataset
    training_frame_ids: tuple[str, ...]
    validation_frame_ids: tuple[str, ...]
    backends: tuple[GaussianBackendKind, ...]
    optimization_profile: GaussianOptimizationProfile
    hardware_profile: GaussianBenchmarkHardwareProfile
    hardware_label: str = ""
    schema_version: int = GAUSSIAN_BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "benchmark_id", require_nonempty(self.benchmark_id, field_name="benchmark_id"))
        object.__setattr__(self, "source_label", require_nonempty(self.source_label, field_name="source_label"))
        if not isinstance(self.dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        train = tuple(require_nonempty(value, field_name="training_frame_id") for value in self.training_frame_ids)
        validation = tuple(
            require_nonempty(value, field_name="validation_frame_id") for value in self.validation_frame_ids
        )
        if len(train) < 2:
            raise ReconstructionValidationError("Real-scene benchmark requires at least two training views.")
        if not validation:
            raise ReconstructionValidationError("Real-scene benchmark requires held-out validation views.")
        if len(set(train)) != len(train) or len(set(validation)) != len(validation):
            raise ReconstructionValidationError("Benchmark frame IDs must be unique within each split.")
        if set(train).intersection(validation):
            raise ReconstructionValidationError("Training and validation views must be disjoint.")
        dataset_ids = {frame.frame_id for frame in self.dataset.frames}
        split_ids = set(train).union(validation)
        if split_ids != dataset_ids:
            missing = sorted(dataset_ids - split_ids)
            unknown = sorted(split_ids - dataset_ids)
            raise ReconstructionValidationError(
                f"Benchmark split must partition every dataset frame exactly once; missing={missing}, unknown={unknown}."
            )
        backends = tuple(self.backends)
        if not backends:
            raise ReconstructionValidationError("At least one real Gaussian backend is required.")
        if len(set(backends)) != len(backends):
            raise ReconstructionValidationError("Benchmark backends must be unique.")
        supported = {
            GaussianBackendKind.GSPLAT_EXTERNAL,
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        }
        if any(not isinstance(value, GaussianBackendKind) for value in backends):
            raise TypeError("backends must contain GaussianBackendKind values.")
        if any(value not in supported for value in backends):
            raise ReconstructionValidationError("D3 benchmarking supports only real external Gaussian backends.")
        if not isinstance(self.optimization_profile, GaussianOptimizationProfile):
            raise TypeError("optimization_profile must be GaussianOptimizationProfile.")
        if not isinstance(self.hardware_profile, GaussianBenchmarkHardwareProfile):
            raise TypeError("hardware_profile must be GaussianBenchmarkHardwareProfile.")
        label = str(self.hardware_label).strip()
        if self.hardware_profile is GaussianBenchmarkHardwareProfile.CUSTOM and not label:
            raise ReconstructionValidationError("Custom hardware profile requires hardware_label.")
        if int(self.schema_version) != GAUSSIAN_BENCHMARK_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Gaussian benchmark schema version.")
        object.__setattr__(self, "training_frame_ids", train)
        object.__setattr__(self, "validation_frame_ids", validation)
        object.__setattr__(self, "backends", backends)
        object.__setattr__(self, "hardware_label", label)
        object.__setattr__(self, "schema_version", GAUSSIAN_BENCHMARK_SCHEMA_VERSION)

    @property
    def dataset_fingerprint_sha256(self) -> str:
        return neural_scene_dataset_fingerprint(self.dataset)

    @property
    def benchmark_fingerprint_sha256(self) -> str:
        return _sha256_json(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "source_label": self.source_label,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "training_frame_ids": list(self.training_frame_ids),
            "validation_frame_ids": list(self.validation_frame_ids),
            "backends": [value.value for value in self.backends],
            "optimization_profile": self.optimization_profile.value,
            "hardware_profile": self.hardware_profile.value,
            "hardware_label": self.hardware_label,
            "pose_convention": self.dataset.pose_convention,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }


@dataclass(frozen=True, slots=True)
class GaussianBenchmarkBackendResult:
    """One backend's benchmark evidence without changing canonical geometry authority."""

    backend_kind: GaussianBackendKind
    benchmark_fingerprint_sha256: str
    dataset_fingerprint_sha256: str
    optimization_profile: GaussianOptimizationProfile
    hardware_profile: GaussianBenchmarkHardwareProfile
    training_view_count: int
    validation_view_count: int
    quality: GaussianImageQualityMetrics = field(default_factory=GaussianImageQualityMetrics)
    performance: GaussianPerformanceMetrics = field(default_factory=GaussianPerformanceMetrics)
    lifecycle: GaussianLifecycleEvidence = field(default_factory=GaussianLifecycleEvidence)
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        if self.backend_kind not in {
            GaussianBackendKind.GSPLAT_EXTERNAL,
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        }:
            raise ReconstructionValidationError("Benchmark results require a real external Gaussian backend.")
        for field_name in ("benchmark_fingerprint_sha256", "dataset_fingerprint_sha256"):
            digest = str(getattr(self, field_name)).lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ReconstructionValidationError(f"{field_name} must be a SHA-256 hex digest.")
            object.__setattr__(self, field_name, digest)
        if not isinstance(self.optimization_profile, GaussianOptimizationProfile):
            raise TypeError("optimization_profile must be GaussianOptimizationProfile.")
        if not isinstance(self.hardware_profile, GaussianBenchmarkHardwareProfile):
            raise TypeError("hardware_profile must be GaussianBenchmarkHardwareProfile.")
        for field_name in ("training_view_count", "validation_view_count"):
            if isinstance(getattr(self, field_name), bool):
                raise ReconstructionValidationError(f"{field_name} must be a positive integer.")
            value = int(getattr(self, field_name))
            if value < 1:
                raise ReconstructionValidationError(f"{field_name} must be >= 1.")
            object.__setattr__(self, field_name, value)
        if not isinstance(self.quality, GaussianImageQualityMetrics):
            raise TypeError("quality must be GaussianImageQualityMetrics.")
        if not isinstance(self.performance, GaussianPerformanceMetrics):
            raise TypeError("performance must be GaussianPerformanceMetrics.")
        if not isinstance(self.lifecycle, GaussianLifecycleEvidence):
            raise TypeError("lifecycle must be GaussianLifecycleEvidence.")
        if bool(self.canonical_geometry_authoritative):
            raise ReconstructionValidationError("Gaussian benchmark output cannot become canonical geometry.")
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Gaussian benchmark output cannot claim certified metrology.")
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def comparison_ready(self) -> bool:
        return (
            self.lifecycle.state in {GaussianBenchmarkRunState.COMPLETED, GaussianBenchmarkRunState.RECOVERED}
            and self.quality.measured
            and self.performance.measured
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "benchmark_fingerprint_sha256": self.benchmark_fingerprint_sha256,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "optimization_profile": self.optimization_profile.value,
            "hardware_profile": self.hardware_profile.value,
            "training_view_count": self.training_view_count,
            "validation_view_count": self.validation_view_count,
            "quality": self.quality.as_dict(),
            "performance": self.performance.as_dict(),
            "lifecycle": self.lifecycle.as_dict(),
            "comparison_ready": self.comparison_ready,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }


@dataclass(frozen=True, slots=True)
class RealSceneGaussianBenchmarkReport:
    """Backend-neutral D3 comparison report for one exact benchmark spec."""

    spec: RealSceneGaussianBenchmarkSpec
    results: tuple[GaussianBenchmarkBackendResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RealSceneGaussianBenchmarkSpec):
            raise TypeError("spec must be RealSceneGaussianBenchmarkSpec.")
        results = tuple(self.results)
        if any(not isinstance(result, GaussianBenchmarkBackendResult) for result in results):
            raise TypeError("results must contain GaussianBenchmarkBackendResult values.")
        if len({result.backend_kind for result in results}) != len(results):
            raise ReconstructionValidationError("Benchmark report contains duplicate backend results.")
        expected_fingerprint = self.spec.benchmark_fingerprint_sha256
        expected_dataset = self.spec.dataset_fingerprint_sha256
        allowed = set(self.spec.backends)
        for result in results:
            if result.backend_kind not in allowed:
                raise ReconstructionValidationError("Benchmark result backend is not selected by the spec.")
            if result.benchmark_fingerprint_sha256 != expected_fingerprint:
                raise ReconstructionValidationError("Benchmark result/spec fingerprint mismatch.")
            if result.dataset_fingerprint_sha256 != expected_dataset:
                raise ReconstructionValidationError("Benchmark result/dataset fingerprint mismatch.")
            if result.optimization_profile is not self.spec.optimization_profile:
                raise ReconstructionValidationError("Benchmark result optimization profile mismatch.")
            if result.hardware_profile is not self.spec.hardware_profile:
                raise ReconstructionValidationError("Benchmark result hardware profile mismatch.")
            if result.training_view_count != len(self.spec.training_frame_ids):
                raise ReconstructionValidationError("Benchmark result training-view count mismatch.")
            if result.validation_view_count != len(self.spec.validation_frame_ids):
                raise ReconstructionValidationError("Benchmark result validation-view count mismatch.")
        object.__setattr__(self, "results", results)

    @property
    def comparison_ready(self) -> bool:
        return (
            len(self.results) == len(self.spec.backends)
            and all(result.comparison_ready for result in self.results)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GAUSSIAN_BENCHMARK_SCHEMA_VERSION,
            "benchmark_fingerprint_sha256": self.spec.benchmark_fingerprint_sha256,
            "spec": self.spec.as_dict(),
            "results": [result.as_dict() for result in self.results],
            "comparison_ready": self.comparison_ready,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }


def write_real_scene_gaussian_benchmark_spec(
    spec: RealSceneGaussianBenchmarkSpec,
    path: str | Path,
) -> Path:
    if not isinstance(spec, RealSceneGaussianBenchmarkSpec):
        raise TypeError("spec must be RealSceneGaussianBenchmarkSpec.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_real_scene_gaussian_benchmark_report(
    report: RealSceneGaussianBenchmarkReport,
    path: str | Path,
) -> Path:
    if not isinstance(report, RealSceneGaussianBenchmarkReport):
        raise TypeError("report must be RealSceneGaussianBenchmarkReport.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
