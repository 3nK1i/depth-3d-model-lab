"""Gauge-aware contracts for Phase 8.3 offline sparse reconstruction.

Raw monocular SfM coordinates deliberately use ``reconstruction`` naming rather
than metric ``world_m`` naming. Absolute metres are only available after an
explicit scale constraint has been applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts import ImageSize
from core.contracts.validation import (
    JsonValue,
    metadata_as_dict,
    normalize_metadata,
    require_finite,
    require_nonempty,
)
from core.reconstruction.contracts import InputSourceMetadata
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.profiles import ReconstructionQualityProfile
from core.reconstruction.validation import (
    readonly_vector3,
    require_nonnegative_int,
    require_uuid,
)

SFM_SCHEMA_VERSION = 1
SFM_WORKSPACE_SCHEMA_VERSION = 1


class SfmScaleStatus(str, Enum):
    ARBITRARY = "arbitrary"
    ESTIMATED_METRIC = "estimated_metric"
    CALIBRATED_METRIC = "calibrated_metric"


class SfmWorldAlignmentStatus(str, Enum):
    ARBITRARY = "arbitrary"
    GRAVITY_ALIGNED = "gravity_aligned"
    WORLD_ALIGNED = "world_aligned"


class SfmMatchingStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    EXHAUSTIVE = "exhaustive"


class SfmFeatureStrategy(str, Enum):
    COLMAP_SIFT = "colmap_sift"


@dataclass(frozen=True, slots=True)
class ReconstructionGauge:
    """Describe the scale and alignment gauge of one reconstruction.

    ``unit_to_metre`` is intentionally ``None`` for raw monocular SfM. A
    positive conversion factor is required before any result may claim metric
    scale.
    """

    scale_status: SfmScaleStatus = SfmScaleStatus.ARBITRARY
    alignment_status: SfmWorldAlignmentStatus = SfmWorldAlignmentStatus.ARBITRARY
    unit_to_metre: float | None = None
    scale_source: str = ""
    relative_scale_uncertainty: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scale_status, SfmScaleStatus):
            raise TypeError("scale_status must be an SfmScaleStatus.")
        if not isinstance(self.alignment_status, SfmWorldAlignmentStatus):
            raise TypeError("alignment_status must be an SfmWorldAlignmentStatus.")
        source = str(self.scale_source).strip()
        uncertainty = self.relative_scale_uncertainty

        if self.scale_status is SfmScaleStatus.ARBITRARY:
            if self.unit_to_metre is not None:
                raise ReconstructionValidationError(
                    "Arbitrary-scale SfM must not declare unit_to_metre."
                )
            if source:
                raise ReconstructionValidationError(
                    "Arbitrary-scale SfM must not declare a metric scale source."
                )
            if uncertainty is not None:
                raise ReconstructionValidationError(
                    "Arbitrary-scale SfM cannot declare metric scale uncertainty."
                )
        else:
            if self.unit_to_metre is None:
                raise ReconstructionValidationError(
                    "Metric SfM gauge requires a positive unit_to_metre conversion."
                )
            factor = require_finite(self.unit_to_metre, field_name="unit_to_metre")
            if factor <= 0.0:
                raise ReconstructionValidationError("unit_to_metre must be positive.")
            if not source:
                raise ReconstructionValidationError(
                    "Metric SfM gauge requires an explicit scale_source."
                )
            object.__setattr__(self, "unit_to_metre", factor)
            if uncertainty is not None:
                uncertainty = require_finite(
                    uncertainty, field_name="relative_scale_uncertainty"
                )
                if uncertainty < 0.0:
                    raise ReconstructionValidationError(
                        "relative_scale_uncertainty must not be negative."
                    )
                object.__setattr__(self, "relative_scale_uncertainty", uncertainty)

        object.__setattr__(self, "scale_source", source)

    @property
    def is_metric(self) -> bool:
        return self.scale_status is not SfmScaleStatus.ARBITRARY

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale_status": self.scale_status.value,
            "alignment_status": self.alignment_status.value,
            "unit_to_metre": self.unit_to_metre,
            "scale_source": self.scale_source,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
        }


def _readonly_rigid_matrix(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError(
            f"{field_name} must be a finite homogeneous 4x4 matrix."
        )
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-7):
        raise ReconstructionValidationError(
            f"{field_name} must have homogeneous final row [0, 0, 0, 1]."
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ReconstructionValidationError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-6):
        raise ReconstructionValidationError(
            f"{field_name} rotation must have positive unit determinant."
        )
    matrix = np.ascontiguousarray(matrix).copy()
    matrix.setflags(write=False)
    return matrix


def _safe_relative_path(value: str, *, field_name: str) -> str:
    text = require_nonempty(value, field_name=field_name).replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ReconstructionValidationError(
            f"{field_name} must be a safe workspace-relative path."
        )
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class SfmWorkspaceImage:
    order: int
    keyframe_id: str
    frame_id: str
    source_sequence_index: int
    timestamp_ns: int
    image_size: ImageSize
    source_locator: str
    staged_image_path: str
    staged_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "order", require_nonnegative_int(self.order, field_name="order"))
        object.__setattr__(
            self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id")
        )
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        object.__setattr__(
            self,
            "source_sequence_index",
            require_nonnegative_int(
                self.source_sequence_index, field_name="source_sequence_index"
            ),
        )
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) < 0:
            raise ReconstructionValidationError("timestamp_ns must be non-negative.")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        if not isinstance(self.image_size, ImageSize):
            raise TypeError("image_size must be an ImageSize.")
        object.__setattr__(
            self,
            "source_locator",
            require_nonempty(self.source_locator, field_name="source_locator"),
        )
        object.__setattr__(
            self,
            "staged_image_path",
            _safe_relative_path(self.staged_image_path, field_name="staged_image_path"),
        )
        digest = str(self.staged_sha256).strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ReconstructionValidationError(
                "staged_sha256 must be a 64-character hexadecimal digest."
            )
        object.__setattr__(self, "staged_sha256", digest)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "source_sequence_index": self.source_sequence_index,
            "timestamp_ns": self.timestamp_ns,
            "image_size": self.image_size.as_dict(),
            "source_locator": self.source_locator,
            "staged_image_path": self.staged_image_path,
            "staged_sha256": self.staged_sha256,
        }


@dataclass(frozen=True, slots=True)
class SfmWorkspaceManifest:
    session_id: str
    source: InputSourceMetadata
    workspace_root: str
    images_directory: str
    database_path: str
    sparse_directory: str
    checkpoints_directory: str
    reports_directory: str
    logs_directory: str
    images: tuple[SfmWorkspaceImage, ...]
    schema_version: int = SFM_WORKSPACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.source, InputSourceMetadata):
            raise TypeError("source must be InputSourceMetadata.")
        root = require_nonempty(self.workspace_root, field_name="workspace_root")
        object.__setattr__(self, "workspace_root", root)
        for field_name in (
            "images_directory",
            "database_path",
            "sparse_directory",
            "checkpoints_directory",
            "reports_directory",
            "logs_directory",
        ):
            object.__setattr__(
                self,
                field_name,
                _safe_relative_path(getattr(self, field_name), field_name=field_name),
            )
        images = tuple(self.images)
        if len(images) < 2:
            raise ReconstructionValidationError(
                "Offline SfM workspace requires at least two materialized keyframes."
            )
        if any(not isinstance(item, SfmWorkspaceImage) for item in images):
            raise TypeError("images must contain SfmWorkspaceImage values.")
        orders = [item.order for item in images]
        if orders != list(range(len(images))):
            raise ReconstructionValidationError(
                "Workspace image order must be contiguous and start at zero."
            )
        for attribute in ("keyframe_id", "frame_id", "staged_image_path"):
            values = [getattr(item, attribute) for item in images]
            if len(values) != len(set(values)):
                raise ReconstructionValidationError(
                    f"Workspace image {attribute} values must be unique."
                )
        if isinstance(self.schema_version, bool) or int(self.schema_version) != SFM_WORKSPACE_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported SfM workspace schema version: {self.schema_version}."
            )
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "source": self.source.as_dict(),
            "workspace_root": self.workspace_root,
            "paths": {
                "images_directory": self.images_directory,
                "database_path": self.database_path,
                "sparse_directory": self.sparse_directory,
                "checkpoints_directory": self.checkpoints_directory,
                "reports_directory": self.reports_directory,
                "logs_directory": self.logs_directory,
            },
            "images": [item.as_dict() for item in self.images],
        }


@dataclass(frozen=True, slots=True)
class OfflineSfmConfiguration:
    feature_strategy: SfmFeatureStrategy = SfmFeatureStrategy.COLMAP_SIFT
    matching_strategy: SfmMatchingStrategy = SfmMatchingStrategy.SEQUENTIAL
    use_gpu: bool = True
    maximum_image_size: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.feature_strategy, SfmFeatureStrategy):
            raise TypeError("feature_strategy must be an SfmFeatureStrategy.")
        if not isinstance(self.matching_strategy, SfmMatchingStrategy):
            raise TypeError("matching_strategy must be an SfmMatchingStrategy.")
        if not isinstance(self.use_gpu, bool):
            raise TypeError("use_gpu must be a bool.")
        if self.maximum_image_size is not None:
            value = require_nonnegative_int(
                self.maximum_image_size, field_name="maximum_image_size"
            )
            if value <= 0:
                raise ReconstructionValidationError("maximum_image_size must be positive.")
            object.__setattr__(self, "maximum_image_size", value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_strategy": self.feature_strategy.value,
            "matching_strategy": self.matching_strategy.value,
            "use_gpu": self.use_gpu,
            "maximum_image_size": self.maximum_image_size,
        }


@dataclass(frozen=True, slots=True)
class OfflineSfmRequest:
    session_id: str
    profile: ReconstructionQualityProfile
    workspace: SfmWorkspaceManifest
    configuration: OfflineSfmConfiguration = field(default_factory=OfflineSfmConfiguration)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        if not isinstance(self.profile, ReconstructionQualityProfile):
            raise TypeError("profile must be a ReconstructionQualityProfile.")
        if not isinstance(self.workspace, SfmWorkspaceManifest):
            raise TypeError("workspace must be an SfmWorkspaceManifest.")
        if self.workspace.session_id != self.session_id:
            raise ReconstructionValidationError(
                "SfM request session_id must match the workspace session_id."
            )
        if not isinstance(self.configuration, OfflineSfmConfiguration):
            raise TypeError("configuration must be an OfflineSfmConfiguration.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SFM_SCHEMA_VERSION,
            "session_id": self.session_id,
            "profile": self.profile.value,
            "workspace": self.workspace.as_dict(),
            "configuration": self.configuration.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class SfmObservation:
    observation_id: str
    landmark_id: str
    keyframe_id: str
    feature_xy: tuple[float, float]
    feature_index: int | None = None
    backend_point2d_index: int | None = None

    def __post_init__(self) -> None:
        for name in ("observation_id", "landmark_id", "keyframe_id"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        if len(self.feature_xy) != 2:
            raise ReconstructionValidationError("feature_xy must contain two coordinates.")
        xy = tuple(require_finite(value, field_name="feature_xy") for value in self.feature_xy)
        object.__setattr__(self, "feature_xy", xy)
        for name in ("feature_index", "backend_point2d_index"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_nonnegative_int(value, field_name=name))

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "landmark_id": self.landmark_id,
            "keyframe_id": self.keyframe_id,
            "feature_xy": list(self.feature_xy),
            "feature_index": self.feature_index,
            "backend_point2d_index": self.backend_point2d_index,
        }


@dataclass(frozen=True, slots=True)
class SfmCameraEstimate:
    estimate_id: str
    camera_id: str
    keyframe_id: str
    frame_id: str
    backend_image_id: int | None
    T_reconstruction_from_camera: np.ndarray

    def __post_init__(self) -> None:
        for name in ("estimate_id", "camera_id", "keyframe_id", "frame_id"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), field_name=name))
        if self.backend_image_id is not None:
            object.__setattr__(
                self,
                "backend_image_id",
                require_nonnegative_int(self.backend_image_id, field_name="backend_image_id"),
            )
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            _readonly_rigid_matrix(
                self.T_reconstruction_from_camera,
                field_name="T_reconstruction_from_camera",
            ),
        )

    @property
    def camera_center_reconstruction(self) -> np.ndarray:
        center = self.T_reconstruction_from_camera[:3, 3].copy()
        center.setflags(write=False)
        return center

    def as_dict(self) -> dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "camera_id": self.camera_id,
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "backend_image_id": self.backend_image_id,
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
        }


@dataclass(frozen=True, slots=True)
class SfmLandmarkEstimate:
    landmark_id: str
    track_id: str
    position_reconstruction: np.ndarray
    observations: tuple[SfmObservation, ...]
    color_rgb: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "landmark_id", require_nonempty(self.landmark_id, field_name="landmark_id")
        )
        object.__setattr__(self, "track_id", require_nonempty(self.track_id, field_name="track_id"))
        object.__setattr__(
            self,
            "position_reconstruction",
            readonly_vector3(
                self.position_reconstruction, field_name="position_reconstruction"
            ),
        )
        observations = tuple(self.observations)
        if len(observations) < 2:
            raise ReconstructionValidationError(
                "Sparse SfM landmarks require observations from at least two keyframes."
            )
        if any(not isinstance(item, SfmObservation) for item in observations):
            raise TypeError("observations must contain SfmObservation values.")
        if any(item.landmark_id != self.landmark_id for item in observations):
            raise ReconstructionValidationError(
                "Every SfM observation must reference its parent landmark."
            )
        observation_ids = [item.observation_id for item in observations]
        keyframe_ids = [item.keyframe_id for item in observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ReconstructionValidationError("SfM observation IDs must be unique.")
        if len(keyframe_ids) != len(set(keyframe_ids)):
            raise ReconstructionValidationError(
                "One landmark cannot have duplicate observations from one keyframe."
            )
        color = self.color_rgb
        if color is not None:
            if len(color) != 3 or any(
                isinstance(value, bool) or int(value) != value or not 0 <= int(value) <= 255
                for value in color
            ):
                raise ReconstructionValidationError(
                    "color_rgb must contain three integer values within [0, 255]."
                )
            color = tuple(int(value) for value in color)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "color_rgb", color)

    def as_dict(self) -> dict[str, Any]:
        return {
            "landmark_id": self.landmark_id,
            "track_id": self.track_id,
            "position_reconstruction": self.position_reconstruction.tolist(),
            "observations": [item.as_dict() for item in self.observations],
            "color_rgb": list(self.color_rgb) if self.color_rgb is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SfmStageStatistics:
    input_keyframes: int
    registered_keyframes: int
    landmarks: int
    observations: int
    elapsed_ns: int
    mean_reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_keyframes",
            "registered_keyframes",
            "landmarks",
            "observations",
            "elapsed_ns",
        ):
            object.__setattr__(self, name, require_nonnegative_int(getattr(self, name), field_name=name))
        if self.registered_keyframes > self.input_keyframes:
            raise ReconstructionValidationError(
                "registered_keyframes cannot exceed input_keyframes."
            )
        if self.mean_reprojection_error_px is not None:
            value = require_finite(
                self.mean_reprojection_error_px,
                field_name="mean_reprojection_error_px",
            )
            if value < 0.0:
                raise ReconstructionValidationError(
                    "mean_reprojection_error_px must not be negative."
                )
            object.__setattr__(self, "mean_reprojection_error_px", value)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_keyframes": self.input_keyframes,
            "registered_keyframes": self.registered_keyframes,
            "landmarks": self.landmarks,
            "observations": self.observations,
            "elapsed_ns": self.elapsed_ns,
            "mean_reprojection_error_px": self.mean_reprojection_error_px,
        }


@dataclass(frozen=True, slots=True)
class OfflineSfmResult:
    session_id: str
    backend_id: str
    backend_version: str
    gauge: ReconstructionGauge
    cameras: tuple[SfmCameraEstimate, ...]
    landmarks: tuple[SfmLandmarkEstimate, ...]
    statistics: SfmStageStatistics
    workspace: SfmWorkspaceManifest
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(
        default_factory=tuple
    )
    schema_version: int = SFM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_uuid(self.session_id, field_name="session_id"))
        object.__setattr__(self, "backend_id", require_nonempty(self.backend_id, field_name="backend_id"))
        object.__setattr__(
            self,
            "backend_version",
            require_nonempty(self.backend_version, field_name="backend_version"),
        )
        if not isinstance(self.gauge, ReconstructionGauge):
            raise TypeError("gauge must be a ReconstructionGauge.")
        if not isinstance(self.workspace, SfmWorkspaceManifest):
            raise TypeError("workspace must be an SfmWorkspaceManifest.")
        if self.workspace.session_id != self.session_id:
            raise ReconstructionValidationError(
                "SfM result session_id must match the workspace session_id."
            )
        cameras = tuple(self.cameras)
        landmarks = tuple(self.landmarks)
        if any(not isinstance(item, SfmCameraEstimate) for item in cameras):
            raise TypeError("cameras must contain SfmCameraEstimate values.")
        if any(not isinstance(item, SfmLandmarkEstimate) for item in landmarks):
            raise TypeError("landmarks must contain SfmLandmarkEstimate values.")
        if not isinstance(self.statistics, SfmStageStatistics):
            raise TypeError("statistics must be SfmStageStatistics.")
        workspace_keyframes = {item.keyframe_id for item in self.workspace.images}
        camera_keyframes = [item.keyframe_id for item in cameras]
        if len(camera_keyframes) != len(set(camera_keyframes)):
            raise ReconstructionValidationError("SfM camera keyframe IDs must be unique.")
        if any(item not in workspace_keyframes for item in camera_keyframes):
            raise ReconstructionValidationError(
                "SfM cameras must reference materialized workspace keyframes."
            )
        if any(
            observation.keyframe_id not in workspace_keyframes
            for landmark in landmarks
            for observation in landmark.observations
        ):
            raise ReconstructionValidationError(
                "SfM observations must reference materialized workspace keyframes."
            )
        if self.statistics.registered_keyframes != len(cameras):
            raise ReconstructionValidationError(
                "registered_keyframes statistic must equal the number of camera estimates."
            )
        if self.statistics.landmarks != len(landmarks):
            raise ReconstructionValidationError(
                "landmarks statistic must equal the number of sparse landmarks."
            )
        observation_count = sum(len(item.observations) for item in landmarks)
        if self.statistics.observations != observation_count:
            raise ReconstructionValidationError(
                "observations statistic must equal the sparse observation count."
            )
        warnings = tuple(require_nonempty(item, field_name="warning") for item in self.warnings)
        object.__setattr__(self, "cameras", cameras)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))
        if isinstance(self.schema_version, bool) or int(self.schema_version) != SFM_SCHEMA_VERSION:
            raise ReconstructionValidationError(
                f"Unsupported SfM schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "backend_id": self.backend_id,
            "backend_version": self.backend_version,
            "gauge": self.gauge.as_dict(),
            "cameras": [item.as_dict() for item in self.cameras],
            "landmarks": [item.as_dict() for item in self.landmarks],
            "statistics": self.statistics.as_dict(),
            "workspace": self.workspace.as_dict(),
            "warnings": list(self.warnings),
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }
