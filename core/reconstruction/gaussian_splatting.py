"""Phase 8.7D1 Gaussian-splatting foundation and external-worker isolation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite, log
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionError, ReconstructionValidationError
from core.reconstruction.neural_scene import (
    NEURAL_SCENE_POSE_CONVENTION,
    NeuralSceneDataset,
    neural_scene_dataset_fingerprint,
)
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.surface_mesh import SurfaceMeshResult

GAUSSIAN_SPLATTING_SCHEMA_VERSION = 1
GAUSSIAN_SPLATTING_POSE_CONVENTION = NEURAL_SCENE_POSE_CONVENTION

# The permissive backend remains the default future production path.  The
# Graphdeco reference implementation is also supported as an isolated research
# backend, but its upstream license limits use to non-commercial research and
# evaluation unless the licensors grant explicit additional permission.
GAUSSIAN_PREFERRED_PACKAGE = "gsplat"
GAUSSIAN_PREFERRED_LICENSE = "Apache-2.0"
GAUSSIAN_GSPLAT_REPOSITORY = "https://github.com/nerfstudio-project/gsplat"
GAUSSIAN_GSPLAT_LICENSE_URL = GAUSSIAN_GSPLAT_REPOSITORY + "/blob/main/LICENSE"
GAUSSIAN_GRAPHDECO_REPOSITORY = "https://github.com/graphdeco-inria/gaussian-splatting"
GAUSSIAN_GRAPHDECO_LICENSE_URL = GAUSSIAN_GRAPHDECO_REPOSITORY + "/blob/main/LICENSE.md"


class GaussianWorkerExecutionError(ReconstructionError):
    """Raised when the isolated Gaussian worker cannot complete safely."""


class GaussianOptimizationProfile(str, Enum):
    EFFICIENT = "efficient"
    BALANCED = "balanced"
    HIGH_QUALITY = "high_quality"


class GaussianBackendKind(str, Enum):
    DEVELOPMENT_MOCK = "development_mock"
    GSPLAT_EXTERNAL = "gsplat_external"
    GRAPHDECO_REFERENCE_EXTERNAL = "graphdeco_reference_external"


class GaussianUsageScope(str, Enum):
    DEVELOPMENT_ONLY = "development_only"
    PERMISSIVE_GENERAL_USE = "permissive_general_use"
    RESEARCH_EVALUATION_ONLY = "research_evaluation_only"


class GaussianOptimizationState(str, Enum):
    SEED_ONLY = "seed_only"
    OPTIMIZED = "optimized"


class GaussianSceneRole(str, Enum):
    APPEARANCE_ONLY = "appearance_only"


@dataclass(frozen=True, slots=True)
class GaussianBackendProvenance:
    """Canonical upstream/license provenance for a Gaussian backend.

    This is an engineering policy contract, not a substitute for legal review.
    It prevents a research-only Graphdeco artifact from being silently relabeled
    as a permissive/commercially deployable backend later in the pipeline.
    """

    backend_kind: GaussianBackendKind
    backend_name: str
    upstream_repository: str
    license_name: str
    license_url: str
    usage_scope: GaussianUsageScope
    commercial_use_permitted_by_backend_license: bool
    explicit_commercial_consent_required: bool
    redistribution_license_notice_required: bool
    research_publication_supported: bool

    def __post_init__(self) -> None:
        if not isinstance(self.backend_kind, GaussianBackendKind):
            raise TypeError("backend_kind must be GaussianBackendKind.")
        if not isinstance(self.usage_scope, GaussianUsageScope):
            raise TypeError("usage_scope must be GaussianUsageScope.")
        for field_name in (
            "backend_name",
            "upstream_repository",
            "license_name",
            "license_url",
        ):
            object.__setattr__(
                self, field_name, require_nonempty(getattr(self, field_name), field_name=field_name)
            )
        for field_name in (
            "commercial_use_permitted_by_backend_license",
            "explicit_commercial_consent_required",
            "redistribution_license_notice_required",
            "research_publication_supported",
        ):
            object.__setattr__(self, field_name, bool(getattr(self, field_name)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind.value,
            "backend_name": self.backend_name,
            "upstream_repository": self.upstream_repository,
            "license_name": self.license_name,
            "license_url": self.license_url,
            "usage_scope": self.usage_scope.value,
            "commercial_use_permitted_by_backend_license": (
                self.commercial_use_permitted_by_backend_license
            ),
            "explicit_commercial_consent_required": self.explicit_commercial_consent_required,
            "redistribution_license_notice_required": self.redistribution_license_notice_required,
            "research_publication_supported": self.research_publication_supported,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GaussianBackendProvenance":
        if not isinstance(payload, Mapping):
            raise ReconstructionValidationError("backend_provenance must be a JSON object.")
        try:
            return cls(
                backend_kind=GaussianBackendKind(str(payload["backend_kind"])),
                backend_name=str(payload["backend_name"]),
                upstream_repository=str(payload["upstream_repository"]),
                license_name=str(payload["license_name"]),
                license_url=str(payload["license_url"]),
                usage_scope=GaussianUsageScope(str(payload["usage_scope"])),
                commercial_use_permitted_by_backend_license=bool(
                    payload["commercial_use_permitted_by_backend_license"]
                ),
                explicit_commercial_consent_required=bool(
                    payload["explicit_commercial_consent_required"]
                ),
                redistribution_license_notice_required=bool(
                    payload["redistribution_license_notice_required"]
                ),
                research_publication_supported=bool(payload["research_publication_supported"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ReconstructionValidationError(f"Invalid Gaussian backend provenance: {exc}") from exc


def gaussian_backend_provenance(backend_kind: GaussianBackendKind) -> GaussianBackendProvenance:
    """Return the immutable policy descriptor for a supported backend."""

    if not isinstance(backend_kind, GaussianBackendKind):
        raise TypeError("backend_kind must be GaussianBackendKind.")
    if backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK:
        return GaussianBackendProvenance(
            backend_kind=backend_kind,
            backend_name="EIA development Gaussian mock",
            upstream_repository="internal://eia/development-gaussian-mock",
            license_name="EIA-development-only",
            license_url="internal://eia/development-gaussian-mock/license",
            usage_scope=GaussianUsageScope.DEVELOPMENT_ONLY,
            commercial_use_permitted_by_backend_license=False,
            explicit_commercial_consent_required=False,
            redistribution_license_notice_required=False,
            research_publication_supported=False,
        )
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        return GaussianBackendProvenance(
            backend_kind=backend_kind,
            backend_name="gsplat",
            upstream_repository=GAUSSIAN_GSPLAT_REPOSITORY,
            license_name="Apache-2.0",
            license_url=GAUSSIAN_GSPLAT_LICENSE_URL,
            usage_scope=GaussianUsageScope.PERMISSIVE_GENERAL_USE,
            commercial_use_permitted_by_backend_license=True,
            explicit_commercial_consent_required=False,
            redistribution_license_notice_required=True,
            research_publication_supported=True,
        )
    if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
        return GaussianBackendProvenance(
            backend_kind=backend_kind,
            backend_name="Graphdeco gaussian-splatting reference",
            upstream_repository=GAUSSIAN_GRAPHDECO_REPOSITORY,
            license_name="Gaussian-Splatting License",
            license_url=GAUSSIAN_GRAPHDECO_LICENSE_URL,
            usage_scope=GaussianUsageScope.RESEARCH_EVALUATION_ONLY,
            commercial_use_permitted_by_backend_license=False,
            explicit_commercial_consent_required=True,
            redistribution_license_notice_required=True,
            research_publication_supported=True,
        )
    raise ReconstructionValidationError(f"Unsupported Gaussian backend: {backend_kind}")


def _require_canonical_backend_provenance(
    provenance: GaussianBackendProvenance,
) -> GaussianBackendProvenance:
    expected = gaussian_backend_provenance(provenance.backend_kind)
    if provenance != expected:
        raise ReconstructionValidationError(
            "Gaussian backend provenance does not match the canonical license/usage policy."
        )
    return provenance


def _readonly(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


def _probability(value: object, field_name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ReconstructionValidationError(f"{field_name} must be finite in [0, 1].")
    return parsed


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


@dataclass(frozen=True, slots=True)
class GaussianSeed:
    """Quality-aware appearance seed sampled from canonical mesh vertices.

    This is not an optimized 3DGS scene.  Positions remain exact canonical mesh
    vertices and exist only to initialize a future isolated optimizer.
    """

    positions_reconstruction: np.ndarray
    colors_rgb: np.ndarray
    log_scales_reconstruction: np.ndarray
    opacities: np.ndarray
    quality_scores: np.ndarray
    source_vertex_indices: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_reconstruction, dtype=np.float32)
        colors = np.asarray(self.colors_rgb, dtype=np.uint8)
        log_scales = np.asarray(self.log_scales_reconstruction, dtype=np.float32)
        opacities = np.asarray(self.opacities, dtype=np.float32)
        quality = np.asarray(self.quality_scores, dtype=np.float32)
        source = np.asarray(self.source_vertex_indices, dtype=np.uint32)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 1:
            raise ReconstructionValidationError("Gaussian seed positions must be non-empty Nx3.")
        if not np.isfinite(positions).all():
            raise ReconstructionValidationError("Gaussian seed positions must be finite.")
        count = len(positions)
        if colors.shape != (count, 3):
            raise ReconstructionValidationError("Gaussian seed colors must have shape Nx3.")
        if log_scales.shape != (count, 3) or not np.isfinite(log_scales).all():
            raise ReconstructionValidationError("Gaussian log scales must be finite Nx3.")
        if opacities.shape != (count,) or not np.isfinite(opacities).all():
            raise ReconstructionValidationError("Gaussian opacities must be finite N values.")
        if np.any(opacities <= 0.0) or np.any(opacities >= 1.0):
            raise ReconstructionValidationError("Gaussian opacities must lie strictly in (0, 1).")
        if quality.shape != (count,) or not np.isfinite(quality).all():
            raise ReconstructionValidationError("Gaussian quality scores must be finite N values.")
        if np.any(quality < 0.0) or np.any(quality > 1.0):
            raise ReconstructionValidationError("Gaussian quality scores must lie in [0, 1].")
        if source.shape != (count,) or len(np.unique(source)) != count:
            raise ReconstructionValidationError(
                "Gaussian seed source_vertex_indices must be one unique index per Gaussian."
            )
        object.__setattr__(self, "positions_reconstruction", _readonly(positions, np.float32))
        object.__setattr__(self, "colors_rgb", _readonly(colors, np.uint8))
        object.__setattr__(self, "log_scales_reconstruction", _readonly(log_scales, np.float32))
        object.__setattr__(self, "opacities", _readonly(opacities, np.float32))
        object.__setattr__(self, "quality_scores", _readonly(quality, np.float32))
        object.__setattr__(self, "source_vertex_indices", _readonly(source, np.uint32))

    @property
    def gaussian_count(self) -> int:
        return len(self.positions_reconstruction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gaussian_count": self.gaussian_count,
            "mean_quality": float(np.mean(self.quality_scores)),
            "minimum_quality": float(np.min(self.quality_scores)),
            "positions_are_canonical_mesh_vertices": True,
        }


def _median_mesh_edge_length(surface: SurfaceMeshResult) -> float:
    mesh = surface.mesh
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices_reconstruction, dtype=np.float64)
    if len(faces) == 0:
        raise ReconstructionValidationError("Cannot seed Gaussians from an empty mesh.")
    sampled = faces
    if len(sampled) > 20_000:
        indices = np.linspace(0, len(sampled) - 1, 20_000, dtype=np.int64)
        sampled = sampled[indices]
    edge_pairs = np.concatenate(
        [sampled[:, [0, 1]], sampled[:, [1, 2]], sampled[:, [2, 0]]], axis=0
    )
    lengths = np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    if len(lengths) == 0:
        raise ReconstructionValidationError("Mesh does not contain a usable positive edge length.")
    return float(np.median(lengths))


def build_gaussian_seed(
    surface: SurfaceMeshResult,
    *,
    max_gaussians: int = 20_000,
    minimum_vertex_quality: float = 0.0,
) -> GaussianSeed:
    """Create a bounded seed without altering canonical mesh coordinates."""

    if not isinstance(surface, SurfaceMeshResult):
        raise TypeError("surface must be SurfaceMeshResult.")
    if isinstance(max_gaussians, bool) or int(max_gaussians) < 1:
        raise ReconstructionValidationError("max_gaussians must be >= 1.")
    max_gaussians = int(max_gaussians)
    minimum_vertex_quality = _probability(minimum_vertex_quality, "minimum_vertex_quality")
    mesh = surface.mesh
    accepted = np.flatnonzero(mesh.vertex_quality_scores >= minimum_vertex_quality)
    if len(accepted) == 0:
        raise ReconstructionValidationError("No mesh vertices satisfy the Gaussian seed quality gate.")
    if len(accepted) > max_gaussians:
        # Deterministic, spatial-order-preserving bounded sampling.  We do not
        # reorder by confidence because that would cluster seeds in easy areas.
        pick = np.linspace(0, len(accepted) - 1, max_gaussians, dtype=np.int64)
        accepted = accepted[pick]
    edge_length = _median_mesh_edge_length(surface)
    base_scale = max(edge_length * 0.35, np.finfo(np.float32).tiny)
    log_scale = float(log(base_scale))
    quality = np.asarray(mesh.vertex_quality_scores[accepted], dtype=np.float32)
    opacities = np.clip(0.15 + 0.75 * quality, 0.05, 0.95).astype(np.float32)
    return GaussianSeed(
        positions_reconstruction=np.asarray(mesh.vertices_reconstruction[accepted], dtype=np.float32),
        colors_rgb=np.asarray(mesh.colors_rgb[accepted], dtype=np.uint8),
        log_scales_reconstruction=np.full((len(accepted), 3), log_scale, dtype=np.float32),
        opacities=opacities,
        quality_scores=quality,
        source_vertex_indices=np.asarray(accepted, dtype=np.uint32),
    )


def write_gaussian_seed_npz(seed: GaussianSeed, path: str | Path) -> Path:
    if not isinstance(seed, GaussianSeed):
        raise TypeError("seed must be GaussianSeed.")
    target = Path(path)
    if not target.is_absolute():
        raise ReconstructionValidationError("Gaussian seed path must be absolute.")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        schema_version=np.asarray([GAUSSIAN_SPLATTING_SCHEMA_VERSION], dtype=np.int32),
        positions_reconstruction=seed.positions_reconstruction,
        colors_rgb=seed.colors_rgb,
        log_scales_reconstruction=seed.log_scales_reconstruction,
        opacities=seed.opacities,
        quality_scores=seed.quality_scores,
        source_vertex_indices=seed.source_vertex_indices,
    )
    return target


@dataclass(frozen=True, slots=True)
class GaussianWorkerRequest:
    request_id: str
    backend_kind: GaussianBackendKind
    profile: GaussianOptimizationProfile
    dataset_manifest_path: str
    dataset_fingerprint_sha256: str
    seed_path: str
    output_dir: str
    scene_profile: DenseSceneProfile
    maximum_input_frames: int
    schema_version: int = GAUSSIAN_SPLATTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", require_nonempty(self.request_id, field_name="request_id"))
        if not isinstance(self.backend_kind, GaussianBackendKind):
            raise TypeError("backend_kind must be GaussianBackendKind.")
        if not isinstance(self.profile, GaussianOptimizationProfile):
            raise TypeError("profile must be GaussianOptimizationProfile.")
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        for field_name in ("dataset_manifest_path", "seed_path", "output_dir"):
            path = Path(require_nonempty(getattr(self, field_name), field_name=field_name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{field_name} must be absolute.")
            object.__setattr__(self, field_name, str(path))
        digest = str(self.dataset_fingerprint_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("dataset_fingerprint_sha256 must be a SHA-256 hex digest.")
        object.__setattr__(self, "dataset_fingerprint_sha256", digest)
        if isinstance(self.maximum_input_frames, bool) or int(self.maximum_input_frames) < 1:
            raise ReconstructionValidationError("maximum_input_frames must be >= 1.")
        object.__setattr__(self, "maximum_input_frames", int(self.maximum_input_frames))
        if int(self.schema_version) != GAUSSIAN_SPLATTING_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Gaussian worker request schema version.")
        object.__setattr__(self, "schema_version", GAUSSIAN_SPLATTING_SCHEMA_VERSION)

    @classmethod
    def from_dataset(
        cls,
        *,
        request_id: str,
        backend_kind: GaussianBackendKind,
        profile: GaussianOptimizationProfile,
        dataset: NeuralSceneDataset,
        dataset_manifest_path: str | Path,
        seed_path: str | Path,
        output_dir: str | Path,
        maximum_input_frames: int,
    ) -> "GaussianWorkerRequest":
        if not isinstance(dataset, NeuralSceneDataset):
            raise TypeError("dataset must be NeuralSceneDataset.")
        return cls(
            request_id=request_id,
            backend_kind=backend_kind,
            profile=profile,
            dataset_manifest_path=str(Path(dataset_manifest_path)),
            dataset_fingerprint_sha256=neural_scene_dataset_fingerprint(dataset),
            seed_path=str(Path(seed_path)),
            output_dir=str(Path(output_dir)),
            scene_profile=dataset.scene_profile,
            maximum_input_frames=maximum_input_frames,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "backend_kind": self.backend_kind.value,
            "backend_provenance": gaussian_backend_provenance(self.backend_kind).as_dict(),
            "profile": self.profile.value,
            "dataset_manifest_path": self.dataset_manifest_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "seed_path": self.seed_path,
            "output_dir": self.output_dir,
            "scene_profile": self.scene_profile.value,
            "maximum_input_frames": self.maximum_input_frames,
        }


@dataclass(frozen=True, slots=True)
class GaussianSceneArtifact:
    request_id: str
    backend_id: str
    backend_version: str
    backend_provenance: GaussianBackendProvenance
    optimization_state: GaussianOptimizationState
    scene_role: GaussianSceneRole
    artifact_dir: str
    scene_path: str
    dataset_fingerprint_sha256: str
    input_frame_count: int
    gaussian_count: int
    training_steps: int
    training_duration_s: float
    render_capable: bool
    device: str
    pose_convention: str
    reconstruction_frame_id: str
    metric_geometry_status: MetricGeometryStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    peak_vram_mib: float | None = None
    canonical_geometry_authoritative: bool = False
    certified_metrology: bool = False
    schema_version: int = GAUSSIAN_SPLATTING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("request_id", "backend_id", "backend_version", "device", "reconstruction_frame_id"):
            object.__setattr__(
                self, field_name, require_nonempty(getattr(self, field_name), field_name=field_name)
            )
        if not isinstance(self.backend_provenance, GaussianBackendProvenance):
            raise TypeError("backend_provenance must be GaussianBackendProvenance.")
        object.__setattr__(
            self,
            "backend_provenance",
            _require_canonical_backend_provenance(self.backend_provenance),
        )
        if not isinstance(self.optimization_state, GaussianOptimizationState):
            raise TypeError("optimization_state must be GaussianOptimizationState.")
        if not isinstance(self.scene_role, GaussianSceneRole):
            raise TypeError("scene_role must be GaussianSceneRole.")
        if self.scene_role is not GaussianSceneRole.APPEARANCE_ONLY:
            raise ReconstructionValidationError("Phase 8.7D Gaussian scenes are appearance-only.")
        for field_name in ("artifact_dir", "scene_path"):
            path = Path(require_nonempty(getattr(self, field_name), field_name=field_name))
            if not path.is_absolute():
                raise ReconstructionValidationError(f"{field_name} must be absolute.")
            object.__setattr__(self, field_name, str(path))
        digest = str(self.dataset_fingerprint_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError("dataset_fingerprint_sha256 must be a SHA-256 hex digest.")
        object.__setattr__(self, "dataset_fingerprint_sha256", digest)
        for field_name in ("input_frame_count", "gaussian_count", "training_steps"):
            value = _nonnegative_int(getattr(self, field_name), field_name)
            object.__setattr__(self, field_name, value)
        if self.input_frame_count < 1 or self.gaussian_count < 1:
            raise ReconstructionValidationError("Gaussian artifacts require frames and Gaussians.")
        duration = float(self.training_duration_s)
        if not isfinite(duration) or duration < 0.0:
            raise ReconstructionValidationError("training_duration_s must be finite and non-negative.")
        object.__setattr__(self, "training_duration_s", duration)
        if self.optimization_state is GaussianOptimizationState.SEED_ONLY:
            if self.training_steps != 0 or bool(self.render_capable):
                raise ReconstructionValidationError(
                    "Seed-only Gaussian artifacts must not claim training or render capability."
                )
        else:
            if self.training_steps < 1:
                raise ReconstructionValidationError("Optimized Gaussian artifacts require training_steps > 0.")
        if str(self.pose_convention) != GAUSSIAN_SPLATTING_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported Gaussian pose convention.")
        object.__setattr__(self, "pose_convention", GAUSSIAN_SPLATTING_POSE_CONVENTION)
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        unit = self.unit_to_metre
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None:
                raise ReconstructionValidationError("Arbitrary Gaussian artifacts must not expose unit_to_metre.")
        else:
            if unit is None:
                raise ReconstructionValidationError("Metric Gaussian artifacts require unit_to_metre.")
            unit = _positive(unit, "unit_to_metre")
        uncertainty = self.relative_scale_uncertainty
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY and uncertainty is not None:
            raise ReconstructionValidationError(
                "Arbitrary Gaussian artifacts must not expose metric scale uncertainty."
            )
        peak = self.peak_vram_mib
        if peak is not None:
            peak = float(peak)
            if not isfinite(peak) or peak < 0.0:
                raise ReconstructionValidationError("peak_vram_mib must be finite and non-negative.")
        if bool(self.canonical_geometry_authoritative):
            raise ReconstructionValidationError(
                "Gaussian appearance artifacts must never be canonical inspection geometry."
            )
        if bool(self.certified_metrology):
            raise ReconstructionValidationError("Gaussian appearance artifacts must not claim certified metrology.")
        if int(self.schema_version) != GAUSSIAN_SPLATTING_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported Gaussian artifact schema version.")
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)
        object.__setattr__(self, "peak_vram_mib", peak)
        object.__setattr__(self, "canonical_geometry_authoritative", False)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", GAUSSIAN_SPLATTING_SCHEMA_VERSION)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GaussianSceneArtifact":
        try:
            return cls(
                request_id=str(payload["request_id"]),
                backend_id=str(payload["backend_id"]),
                backend_version=str(payload["backend_version"]),
                backend_provenance=GaussianBackendProvenance.from_payload(payload["backend_provenance"]),
                optimization_state=GaussianOptimizationState(str(payload["optimization_state"])),
                scene_role=GaussianSceneRole(str(payload.get("scene_role", "appearance_only"))),
                artifact_dir=str(payload["artifact_dir"]),
                scene_path=str(payload["scene_path"]),
                dataset_fingerprint_sha256=str(payload["dataset_fingerprint_sha256"]),
                input_frame_count=int(payload["input_frame_count"]),
                gaussian_count=int(payload["gaussian_count"]),
                training_steps=int(payload["training_steps"]),
                training_duration_s=float(payload["training_duration_s"]),
                render_capable=bool(payload["render_capable"]),
                device=str(payload["device"]),
                pose_convention=str(payload["pose_convention"]),
                reconstruction_frame_id=str(payload["reconstruction_frame_id"]),
                metric_geometry_status=MetricGeometryStatus(str(payload["metric_geometry_status"])),
                unit_to_metre=payload.get("unit_to_metre"),
                relative_scale_uncertainty=payload.get("relative_scale_uncertainty"),
                peak_vram_mib=payload.get("peak_vram_mib"),
                canonical_geometry_authoritative=bool(payload.get("canonical_geometry_authoritative", False)),
                certified_metrology=bool(payload.get("certified_metrology", False)),
                schema_version=int(payload.get("schema_version", 0)),
            )
        except (KeyError, TypeError, ValueError, ReconstructionValidationError) as exc:
            raise GaussianWorkerExecutionError(f"Invalid Gaussian worker result: {exc}") from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "backend_provenance": self.backend_provenance.as_dict(),
            "optimization_state": self.optimization_state.value,
            "scene_role": self.scene_role.value,
            "artifact_dir": self.artifact_dir,
            "scene_path": self.scene_path,
            "dataset_fingerprint_sha256": self.dataset_fingerprint_sha256,
            "input_frame_count": self.input_frame_count,
            "gaussian_count": self.gaussian_count,
            "training_steps": self.training_steps,
            "training_duration_s": self.training_duration_s,
            "render_capable": self.render_capable,
            "device": self.device,
            "peak_vram_mib": self.peak_vram_mib,
            "pose_convention": self.pose_convention,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "certified_metrology": self.certified_metrology,
        }


@dataclass(frozen=True, slots=True)
class GaussianWorkerProcessSpec:
    python_executable: str
    worker_script: str
    timeout_s: float = 600.0
    require_isolated_interpreter: bool = True

    def __post_init__(self) -> None:
        python = Path(require_nonempty(self.python_executable, field_name="python_executable"))
        worker = Path(require_nonempty(self.worker_script, field_name="worker_script"))
        if not python.is_absolute() or not worker.is_absolute():
            raise ReconstructionValidationError(
                "Gaussian worker Python executable and worker script paths must be absolute."
            )
        timeout = _positive(self.timeout_s, "timeout_s")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "worker_script", str(worker))
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "require_isolated_interpreter", bool(self.require_isolated_interpreter))

    @property
    def interpreter_is_isolated(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)


def run_gaussian_external_worker(
    spec: GaussianWorkerProcessSpec,
    request: GaussianWorkerRequest,
) -> GaussianSceneArtifact:
    if not isinstance(spec, GaussianWorkerProcessSpec):
        raise TypeError("spec must be GaussianWorkerProcessSpec.")
    if not isinstance(request, GaussianWorkerRequest):
        raise TypeError("request must be GaussianWorkerRequest.")
    if spec.require_isolated_interpreter and not spec.interpreter_is_isolated:
        raise GaussianWorkerExecutionError(
            "Gaussian optimization must run in an isolated interpreter, not the production Python environment."
        )
    python = Path(spec.python_executable)
    worker = Path(spec.worker_script)
    if not python.is_file():
        raise GaussianWorkerExecutionError(f"Gaussian worker Python does not exist: {python}")
    if not worker.is_file():
        raise GaussianWorkerExecutionError(f"Gaussian worker script does not exist: {worker}")
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="eia-gaussian-request-", dir=str(output_dir)) as tmp:
        tmp_dir = Path(tmp)
        request_path = tmp_dir / "request.json"
        result_path = tmp_dir / "result.json"
        request_path.write_text(json.dumps(request.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            str(python),
            str(worker),
            "--request",
            str(request_path),
            "--result",
            str(result_path),
        ]
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
            raise GaussianWorkerExecutionError("Isolated Gaussian worker timed out.") from exc
        except OSError as exc:
            raise GaussianWorkerExecutionError("Could not start isolated Gaussian worker.") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "unknown worker failure"
            raise GaussianWorkerExecutionError(
                f"Isolated Gaussian worker exited with code {completed.returncode}: {detail}"
            )
        if not result_path.is_file():
            raise GaussianWorkerExecutionError("Isolated Gaussian worker did not produce result.json.")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GaussianWorkerExecutionError("Could not parse Gaussian worker result.json.") from exc
        if not isinstance(payload, Mapping):
            raise GaussianWorkerExecutionError("Gaussian worker result must be a JSON object.")
        artifact = GaussianSceneArtifact.from_payload(payload)
        if artifact.request_id != request.request_id:
            raise GaussianWorkerExecutionError("Gaussian worker result request_id does not match request.")
        if artifact.dataset_fingerprint_sha256 != request.dataset_fingerprint_sha256:
            raise GaussianWorkerExecutionError("Gaussian worker result dataset fingerprint does not match request.")
        expected_provenance = gaussian_backend_provenance(request.backend_kind)
        if artifact.backend_provenance != expected_provenance:
            raise GaussianWorkerExecutionError(
                "Gaussian worker result backend provenance does not match the requested backend."
            )
        return artifact


_GSPLAT_PROBE_CODE = r'''
import importlib
import importlib.metadata as md
import json

payload = {
    "available": False,
    "torch_version": "",
    "gsplat_version": "",
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_names": [],
    "error": "",
}
try:
    torch = importlib.import_module("torch")
    gsplat = importlib.import_module("gsplat")
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
else:
    payload["available"] = True
    payload["torch_version"] = str(getattr(torch, "__version__", ""))
    try:
        payload["gsplat_version"] = md.version("gsplat")
    except Exception:
        payload["gsplat_version"] = str(getattr(gsplat, "__version__", ""))
    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        if payload["cuda_available"]:
            payload["cuda_device_names"] = [
                str(torch.cuda.get_device_name(i)) for i in range(payload["cuda_device_count"])
            ]
    except Exception as exc:
        payload["error"] = f"CUDA probe failed: {type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''

_GRAPHDECO_PROBE_CODE = r'''
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
payload = {
    "available": False,
    "torch_version": "",
    "gsplat_version": "",
    "cuda_available": False,
    "cuda_device_count": 0,
    "cuda_device_names": [],
    "graphdeco_runtime_ready": False,
    "error": "",
}
try:
    torch = importlib.import_module("torch")
    importlib.import_module("diff_gaussian_rasterization")
    importlib.import_module("simple_knn._C")
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
else:
    payload["available"] = True
    payload["graphdeco_runtime_ready"] = True
    payload["torch_version"] = str(getattr(torch, "__version__", ""))
    try:
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
        if payload["cuda_available"]:
            payload["cuda_device_names"] = [
                str(torch.cuda.get_device_name(i)) for i in range(payload["cuda_device_count"])
            ]
    except Exception as exc:
        payload["error"] = f"CUDA probe failed: {type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
'''


@dataclass(frozen=True, slots=True)
class GaussianEnvironmentProbe:
    python_executable: str
    available: bool
    backend_kind: GaussianBackendKind = GaussianBackendKind.GSPLAT_EXTERNAL
    torch_version: str = ""
    gsplat_version: str = ""
    graphdeco_root: str = ""
    graphdeco_license_present: bool = False
    graphdeco_runtime_ready: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_names: tuple[str, ...] = ()
    error: str = ""

    def __post_init__(self) -> None:
        python = Path(require_nonempty(self.python_executable, field_name="python_executable"))
        if not python.is_absolute():
            raise ReconstructionValidationError("Gaussian probe python_executable must be absolute.")
        object.__setattr__(self, "python_executable", str(python))
        object.__setattr__(self, "available", bool(self.available))
        if not isinstance(self.backend_kind, GaussianBackendKind):
            raise TypeError("backend_kind must be GaussianBackendKind.")
        if self.backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK:
            raise ReconstructionValidationError("Environment probes are only for real Gaussian backends.")
        object.__setattr__(self, "torch_version", str(self.torch_version).strip())
        object.__setattr__(self, "gsplat_version", str(self.gsplat_version).strip())
        root = str(self.graphdeco_root).strip()
        if root:
            root_path = Path(root)
            if not root_path.is_absolute():
                raise ReconstructionValidationError("graphdeco_root must be absolute when present.")
            root = str(root_path)
        object.__setattr__(self, "graphdeco_root", root)
        object.__setattr__(self, "graphdeco_license_present", bool(self.graphdeco_license_present))
        object.__setattr__(self, "graphdeco_runtime_ready", bool(self.graphdeco_runtime_ready))
        object.__setattr__(self, "cuda_available", bool(self.cuda_available))
        object.__setattr__(
            self, "cuda_device_count", _nonnegative_int(self.cuda_device_count, "cuda_device_count")
        )
        names = tuple(str(value).strip() for value in self.cuda_device_names if str(value).strip())
        object.__setattr__(self, "cuda_device_names", names)
        object.__setattr__(self, "error", str(self.error).strip())
        if self.backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
            if self.available and (not self.torch_version or not self.gsplat_version):
                raise ReconstructionValidationError(
                    "Available gsplat environments must report torch and gsplat versions."
                )
            if self.graphdeco_root or self.graphdeco_runtime_ready or self.graphdeco_license_present:
                raise ReconstructionValidationError(
                    "gsplat environment probes must not claim Graphdeco checkout state."
                )
        elif self.backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
            if self.available and (
                not self.torch_version
                or not self.graphdeco_root
                or not self.graphdeco_runtime_ready
                or not self.graphdeco_license_present
            ):
                raise ReconstructionValidationError(
                    "Available Graphdeco environments require torch, checkout, runtime extensions, and LICENSE.md."
                )
            if self.gsplat_version:
                raise ReconstructionValidationError(
                    "Graphdeco environment probes must not report a gsplat version."
                )
        if self.cuda_device_count != len(names):
            raise ReconstructionValidationError("Gaussian CUDA device count must match device names.")
        if self.cuda_device_count > 0 and not self.cuda_available:
            raise ReconstructionValidationError("CUDA device count cannot be positive when CUDA is unavailable.")

    @property
    def interpreter_is_isolated(self) -> bool:
        return os.path.realpath(self.python_executable) != os.path.realpath(sys.executable)

    @property
    def backend_ready(self) -> bool:
        return self.available and self.cuda_available and self.cuda_device_count > 0

    @property
    def backend_provenance(self) -> GaussianBackendProvenance:
        return gaussian_backend_provenance(self.backend_kind)

    def as_dict(self) -> dict[str, Any]:
        return {
            "python_executable": self.python_executable,
            "interpreter_is_isolated": self.interpreter_is_isolated,
            "backend_kind": self.backend_kind.value,
            "backend_provenance": self.backend_provenance.as_dict(),
            "available": self.available,
            "torch_version": self.torch_version,
            "gsplat_version": self.gsplat_version,
            "graphdeco_root": self.graphdeco_root,
            "graphdeco_license_present": self.graphdeco_license_present,
            "graphdeco_runtime_ready": self.graphdeco_runtime_ready,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_names": list(self.cuda_device_names),
            "backend_ready": self.backend_ready,
            "error": self.error,
        }


def _graphdeco_checkout_status(root: Path) -> tuple[bool, str]:
    required = (
        root / "train.py",
        root / "gaussian_renderer" / "__init__.py",
        root / "scene" / "gaussian_model.py",
        root / "LICENSE.md",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return False, "missing_graphdeco_files:" + ",".join(missing)
    try:
        license_text = (root / "LICENSE.md").read_text(encoding="utf-8", errors="replace").lower()
    except OSError as exc:
        return False, f"graphdeco_license_read_failed:{type(exc).__name__}:{exc}"
    required_terms = ("research users", "non-commercial", "explicit consent")
    if not all(term in license_text for term in required_terms):
        return False, "graphdeco_license_marker_mismatch"
    return True, ""


def probe_gaussian_environment(
    python_executable: str | Path,
    *,
    backend_kind: GaussianBackendKind = GaussianBackendKind.GSPLAT_EXTERNAL,
    graphdeco_root: str | Path | None = None,
    timeout_s: float = 30.0,
) -> GaussianEnvironmentProbe:
    python = Path(python_executable)
    if not python.is_absolute():
        raise ReconstructionValidationError("Gaussian probe python_executable must be absolute.")
    if not isinstance(backend_kind, GaussianBackendKind):
        raise TypeError("backend_kind must be GaussianBackendKind.")
    if backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK:
        raise ReconstructionValidationError("Environment probes are only for real Gaussian backends.")
    timeout = _positive(timeout_s, "timeout_s")
    common: dict[str, Any] = {
        "python_executable": str(python),
        "backend_kind": backend_kind,
    }
    if not python.is_file():
        return GaussianEnvironmentProbe(**common, available=False, error="python_executable_not_found")

    command = [str(python), "-c", _GSPLAT_PROBE_CODE]
    if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
        if graphdeco_root is None:
            raise ReconstructionValidationError("graphdeco_root is required for the Graphdeco backend probe.")
        root = Path(graphdeco_root)
        if not root.is_absolute():
            raise ReconstructionValidationError("graphdeco_root must be absolute.")
        checkout_ok, checkout_error = _graphdeco_checkout_status(root)
        common.update(
            graphdeco_root=str(root),
            graphdeco_license_present=checkout_ok,
        )
        if not checkout_ok:
            return GaussianEnvironmentProbe(**common, available=False, error=checkout_error)
        command = [str(python), "-c", _GRAPHDECO_PROBE_CODE, str(root)]

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return GaussianEnvironmentProbe(**common, available=False, error=f"{type(exc).__name__}: {exc}")
    if completed.returncode != 0:
        return GaussianEnvironmentProbe(
            **common,
            available=False,
            error=(completed.stderr.strip() or f"probe_exit_{completed.returncode}"),
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return GaussianEnvironmentProbe(
            **common,
            available=False,
            error=f"invalid_probe_output: {exc}",
        )
    names = payload.get("cuda_device_names", [])
    if not isinstance(names, list):
        names = []
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        return GaussianEnvironmentProbe(
            **common,
            available=bool(payload.get("available", False)),
            torch_version=str(payload.get("torch_version", "")),
            gsplat_version=str(payload.get("gsplat_version", "")),
            cuda_available=bool(payload.get("cuda_available", False)),
            cuda_device_count=int(payload.get("cuda_device_count", 0)),
            cuda_device_names=tuple(str(v) for v in names),
            error=str(payload.get("error", "")),
        )
    return GaussianEnvironmentProbe(
        **common,
        available=bool(payload.get("available", False)),
        torch_version=str(payload.get("torch_version", "")),
        graphdeco_runtime_ready=bool(payload.get("graphdeco_runtime_ready", False)),
        cuda_available=bool(payload.get("cuda_available", False)),
        cuda_device_count=int(payload.get("cuda_device_count", 0)),
        cuda_device_names=tuple(str(v) for v in names),
        error=str(payload.get("error", "")),
    )
