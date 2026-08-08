"""Backend-neutral neural-scene dataset contracts for Phase 8.7D/8.7E."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from core.contracts.validation import require_nonempty
from core.reconstruction.dense_fusion import DenseSceneProfile
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.scale_reference import MetricGeometryStatus
from core.reconstruction.sfm_contracts import SfmScaleStatus, SfmWorldAlignmentStatus

NEURAL_SCENE_DATASET_SCHEMA_VERSION = 1
NEURAL_SCENE_POSE_CONVENTION = "T_reconstruction_from_camera"


def _readonly_matrix4(value: object, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError(f"{field_name} must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ReconstructionValidationError(f"{field_name} must be a rigid homogeneous transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4):
        raise ReconstructionValidationError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4):
        raise ReconstructionValidationError(f"{field_name} rotation determinant must be +1.")
    result = matrix.copy()
    result.setflags(write=False)
    return result


def _positive_finite(value: object, field_name: str) -> float:
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
class NeuralSceneFrame:
    """One RGB view plus camera-to-reconstruction pose for appearance training."""

    frame_id: str
    image_path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    T_reconstruction_from_camera: np.ndarray
    pose_layer: str
    camera_profile_id: str
    timestamp_ns: int = 0
    mask_path: str = ""
    depth_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        image = Path(require_nonempty(self.image_path, field_name="image_path"))
        if not image.is_absolute():
            raise ReconstructionValidationError("Neural-scene image_path must be absolute.")
        object.__setattr__(self, "image_path", str(image))
        for field_name in ("width", "height"):
            value = _nonnegative_int(getattr(self, field_name), field_name)
            if value < 2:
                raise ReconstructionValidationError(f"{field_name} must be >= 2.")
            object.__setattr__(self, field_name, value)
        for field_name in ("fx", "fy"):
            object.__setattr__(self, field_name, _positive_finite(getattr(self, field_name), field_name))
        for field_name in ("cx", "cy"):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ReconstructionValidationError(f"{field_name} must be finite.")
            object.__setattr__(self, field_name, value)
        if not 0.0 <= self.cx <= float(self.width):
            raise ReconstructionValidationError("cx must lie inside the image extent.")
        if not 0.0 <= self.cy <= float(self.height):
            raise ReconstructionValidationError("cy must lie inside the image extent.")
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            _readonly_matrix4(self.T_reconstruction_from_camera, "T_reconstruction_from_camera"),
        )
        object.__setattr__(self, "pose_layer", require_nonempty(self.pose_layer, field_name="pose_layer"))
        object.__setattr__(
            self,
            "camera_profile_id",
            require_nonempty(self.camera_profile_id, field_name="camera_profile_id"),
        )
        object.__setattr__(self, "timestamp_ns", _nonnegative_int(self.timestamp_ns, "timestamp_ns"))
        for field_name in ("mask_path", "depth_path"):
            raw = str(getattr(self, field_name)).strip()
            if raw and not Path(raw).is_absolute():
                raise ReconstructionValidationError(f"{field_name} must be absolute when present.")
            object.__setattr__(self, field_name, raw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
            "pose_layer": self.pose_layer,
            "camera_profile_id": self.camera_profile_id,
            "timestamp_ns": self.timestamp_ns,
            "mask_path": self.mask_path,
            "depth_path": self.depth_path,
        }


@dataclass(frozen=True, slots=True)
class NeuralSceneDataset:
    """Versioned appearance-training dataset with reconstruction provenance."""

    session_id: str
    scene_profile: DenseSceneProfile
    reconstruction_frame_id: str
    camera_profile_id: str
    scale_profile_id: str
    metric_geometry_status: MetricGeometryStatus
    scale_status: SfmScaleStatus
    world_alignment_status: SfmWorldAlignmentStatus
    unit_to_metre: float | None
    relative_scale_uncertainty: float | None
    frames: tuple[NeuralSceneFrame, ...]
    certified_metrology: bool = False
    pose_convention: str = NEURAL_SCENE_POSE_CONVENTION
    schema_version: int = NEURAL_SCENE_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", require_nonempty(self.session_id, field_name="session_id"))
        if not isinstance(self.scene_profile, DenseSceneProfile):
            raise TypeError("scene_profile must be DenseSceneProfile.")
        object.__setattr__(
            self,
            "reconstruction_frame_id",
            require_nonempty(self.reconstruction_frame_id, field_name="reconstruction_frame_id"),
        )
        object.__setattr__(
            self,
            "camera_profile_id",
            require_nonempty(self.camera_profile_id, field_name="camera_profile_id"),
        )
        object.__setattr__(
            self,
            "scale_profile_id",
            require_nonempty(self.scale_profile_id, field_name="scale_profile_id"),
        )
        if not isinstance(self.metric_geometry_status, MetricGeometryStatus):
            raise TypeError("metric_geometry_status must be MetricGeometryStatus.")
        if not isinstance(self.scale_status, SfmScaleStatus):
            raise TypeError("scale_status must be SfmScaleStatus.")
        if not isinstance(self.world_alignment_status, SfmWorldAlignmentStatus):
            raise TypeError("world_alignment_status must be SfmWorldAlignmentStatus.")
        if str(self.pose_convention) != NEURAL_SCENE_POSE_CONVENTION:
            raise ReconstructionValidationError("Unsupported neural-scene pose convention.")
        object.__setattr__(self, "pose_convention", NEURAL_SCENE_POSE_CONVENTION)
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Neural appearance datasets must not claim certified metrology."
            )
        object.__setattr__(self, "certified_metrology", False)
        frames = tuple(self.frames)
        if not frames:
            raise ReconstructionValidationError("Neural scene dataset requires at least one frame.")
        if any(not isinstance(frame, NeuralSceneFrame) for frame in frames):
            raise TypeError("frames must contain NeuralSceneFrame values.")
        frame_ids = [frame.frame_id for frame in frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ReconstructionValidationError("Neural scene frame IDs must be unique.")
        if any(frame.camera_profile_id != self.camera_profile_id for frame in frames):
            raise ReconstructionValidationError(
                "All neural scene frames must use the dataset camera profile."
            )
        unit = self.unit_to_metre
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY:
            if unit is not None:
                raise ReconstructionValidationError(
                    "Arbitrary neural scene datasets must not expose unit_to_metre."
                )
        else:
            if unit is None:
                raise ReconstructionValidationError(
                    "Metric neural scene datasets require unit_to_metre."
                )
            unit = _positive_finite(unit, "unit_to_metre")
        uncertainty = self.relative_scale_uncertainty
        if uncertainty is not None:
            uncertainty = float(uncertainty)
            if not isfinite(uncertainty) or uncertainty < 0.0:
                raise ReconstructionValidationError(
                    "relative_scale_uncertainty must be finite and non-negative."
                )
        if self.metric_geometry_status is MetricGeometryStatus.ARBITRARY and uncertainty is not None:
            raise ReconstructionValidationError(
                "Arbitrary neural scene datasets must not expose metric scale uncertainty."
            )
        if int(self.schema_version) != NEURAL_SCENE_DATASET_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported neural scene dataset schema version.")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "unit_to_metre", unit)
        object.__setattr__(self, "relative_scale_uncertainty", uncertainty)
        object.__setattr__(self, "schema_version", NEURAL_SCENE_DATASET_SCHEMA_VERSION)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def pose_layers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(frame.pose_layer for frame in self.frames))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "scene_profile": self.scene_profile.value,
            "reconstruction_frame_id": self.reconstruction_frame_id,
            "camera_profile_id": self.camera_profile_id,
            "scale_profile_id": self.scale_profile_id,
            "metric_geometry_status": self.metric_geometry_status.value,
            "scale_status": self.scale_status.value,
            "world_alignment_status": self.world_alignment_status.value,
            "unit_to_metre": self.unit_to_metre,
            "relative_scale_uncertainty": self.relative_scale_uncertainty,
            "pose_convention": self.pose_convention,
            "certified_metrology": self.certified_metrology,
            "frame_count": self.frame_count,
            "pose_layers": list(self.pose_layers),
            "frames": [frame.as_dict() for frame in self.frames],
        }


def neural_scene_dataset_fingerprint(dataset: NeuralSceneDataset) -> str:
    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    payload = json.dumps(dataset.as_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_neural_scene_dataset_manifest(dataset: NeuralSceneDataset, path: str | Path) -> Path:
    if not isinstance(dataset, NeuralSceneDataset):
        raise TypeError("dataset must be NeuralSceneDataset.")
    target = Path(path)
    if not target.is_absolute():
        raise ReconstructionValidationError("Neural scene manifest path must be absolute.")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset.as_dict()
    payload["dataset_fingerprint_sha256"] = neural_scene_dataset_fingerprint(dataset)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
