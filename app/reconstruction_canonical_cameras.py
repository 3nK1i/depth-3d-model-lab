"""Persistent canonical-camera presentation for the Reconstruction viewer.

Phase 9.3C consumes camera poses only when the existing persistent
``camera_trajectory`` artifact explicitly stores ``T_reconstruction_from_camera``
matrices in its generic ``payload_snapshot``.  Legacy scenes that preserve only
trajectory provenance remain valid but do not gain invented camera poses.

The arrays produced here are presentation resources.  They do not establish
metric scale, change canonical-geometry authority, certify metrology, or alter
the reconstruction coordinate convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from app.reconstruction_visualization import keyframe_centers_in_viewer
from core.contracts.validation import JsonValue, metadata_as_dict
from core.reconstruction.orchestration import (
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
)
from core.reconstruction.persistent_scene import (
    PersistentReconstructionScene,
    PersistentSceneArtifactRecord,
)

CANONICAL_CAMERA_POSE_CONVENTION = "T_reconstruction_from_camera"
MAX_PERSISTED_CANONICAL_CAMERA_POSES = 4096


@dataclass(frozen=True, slots=True, eq=False)
class CanonicalCameraTrajectoryPresentation:
    """Validated persistent camera poses expressed in canonical scene coordinates."""

    scene_id: str
    artifact_id: str
    provider_id: str
    pose_convention: str
    reconstruction_frame_id: str
    camera_ids: tuple[str, ...]
    pose_layers: tuple[str, ...]
    matrices_reconstruction: np.ndarray
    positions_viewer: np.ndarray
    total_pose_count: int
    certified_metrology: bool = False

    def __post_init__(self) -> None:
        matrices = np.asarray(self.matrices_reconstruction, dtype=np.float64)
        positions = np.asarray(self.positions_viewer, dtype=np.float32)
        ids = tuple(str(value).strip() for value in self.camera_ids)
        layers = tuple(str(value).strip() for value in self.pose_layers)

        if self.certified_metrology:
            raise ValueError("Canonical camera presentation cannot claim certified metrology.")
        if str(self.pose_convention).strip() != CANONICAL_CAMERA_POSE_CONVENTION:
            raise ValueError("Canonical camera presentation uses an unsupported pose convention.")
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < 1:
            raise ValueError("matrices_reconstruction must have shape (N, 4, 4) with N >= 1.")
        if matrices.shape[0] > MAX_PERSISTED_CANONICAL_CAMERA_POSES:
            raise ValueError("Persistent canonical camera trajectory exceeds the presentation safety bound.")
        if not np.isfinite(matrices).all():
            raise ValueError("Persistent canonical camera matrices must be finite.")
        if positions.shape != (matrices.shape[0], 3) or not np.isfinite(positions).all():
            raise ValueError("positions_viewer must be finite and match the camera pose count.")
        if len(ids) != matrices.shape[0] or any(not value for value in ids):
            raise ValueError("camera_ids must contain one non-empty ID per camera pose.")
        if len(set(ids)) != len(ids):
            raise ValueError("camera_ids must be unique.")
        if layers and (len(layers) != matrices.shape[0] or any(not value for value in layers)):
            raise ValueError("pose_layers must be empty or contain one non-empty layer per camera pose.")
        total = int(self.total_pose_count)
        if total < matrices.shape[0]:
            raise ValueError("total_pose_count cannot be smaller than the persisted pose count.")

        for index, matrix in enumerate(matrices):
            _validate_pose(matrix, field_name=f"T_reconstruction_from_camera[{index}]")

        matrices = np.ascontiguousarray(matrices)
        positions = np.ascontiguousarray(positions)
        matrices.setflags(write=False)
        positions.setflags(write=False)
        object.__setattr__(self, "scene_id", _nonempty(self.scene_id, "scene_id"))
        object.__setattr__(self, "artifact_id", _nonempty(self.artifact_id, "artifact_id"))
        object.__setattr__(self, "provider_id", _nonempty(self.provider_id, "provider_id"))
        object.__setattr__(self, "pose_convention", CANONICAL_CAMERA_POSE_CONVENTION)
        object.__setattr__(self, "reconstruction_frame_id", str(self.reconstruction_frame_id).strip())
        object.__setattr__(self, "camera_ids", ids)
        object.__setattr__(self, "pose_layers", layers)
        object.__setattr__(self, "matrices_reconstruction", matrices)
        object.__setattr__(self, "positions_viewer", positions)
        object.__setattr__(self, "total_pose_count", total)
        object.__setattr__(self, "certified_metrology", False)

    @property
    def persisted_pose_count(self) -> int:
        return int(self.matrices_reconstruction.shape[0])

    @property
    def trajectory_available(self) -> bool:
        return self.persisted_pose_count >= 2


def build_canonical_camera_trajectory_snapshot(
    *,
    matrices_reconstruction: np.ndarray,
    camera_ids: Sequence[str],
    provider_id: str,
    pose_layers: Sequence[str] = (),
    reconstruction_frame_id: str = "",
    total_pose_count: int | None = None,
) -> dict[str, JsonValue]:
    """Build the bounded JSON snapshot shape consumed by Phase 9.3C.

    This helper is intentionally separate from ``PipelineCameraTrajectoryEvidence.as_dict``:
    the core evidence serializer is provenance-focused and deliberately omits the dense pose
    matrices.  Callers that want persistent interactive cameras must opt in explicitly to
    storing this bounded presentation snapshot.
    """

    matrices = np.asarray(matrices_reconstruction, dtype=np.float64)
    ids = tuple(str(value).strip() for value in camera_ids)
    layers = tuple(str(value).strip() for value in pose_layers)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or matrices.shape[0] < 1:
        raise ValueError("matrices_reconstruction must have shape (N, 4, 4) with N >= 1.")
    if matrices.shape[0] > MAX_PERSISTED_CANONICAL_CAMERA_POSES:
        raise ValueError("Persistent camera snapshot exceeds the Phase 9.3C safety bound.")
    if len(ids) != matrices.shape[0] or any(not value for value in ids):
        raise ValueError("camera_ids must contain one non-empty ID per pose.")
    if len(set(ids)) != len(ids):
        raise ValueError("camera_ids must be unique.")
    if layers and (len(layers) != matrices.shape[0] or any(not value for value in layers)):
        raise ValueError("pose_layers must be empty or match the pose count.")
    for index, matrix in enumerate(matrices):
        _validate_pose(matrix, field_name=f"T_reconstruction_from_camera[{index}]")
    total = matrices.shape[0] if total_pose_count is None else int(total_pose_count)
    if total < matrices.shape[0]:
        raise ValueError("total_pose_count cannot be smaller than persisted poses.")

    return {
        "pose_convention": CANONICAL_CAMERA_POSE_CONVENTION,
        "frame_ids": list(ids),
        "frame_count": total,
        "T_reconstruction_from_camera": matrices.tolist(),
        "pose_layers": list(layers),
        "source_provider_id": _nonempty(provider_id, "provider_id"),
        "reconstruction_frame_id": str(reconstruction_frame_id).strip(),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "presentation_role": "camera_pose_context",
    }


def load_persistent_canonical_camera_trajectory(
    scene: PersistentReconstructionScene,
) -> CanonicalCameraTrajectoryPresentation | None:
    """Load explicitly persisted canonical-scene camera poses, if available.

    A legacy or provenance-only ``camera_trajectory`` snapshot is not an error and returns
    ``None``.  If pose matrices are present, they are validated strictly and fail closed.
    """

    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    artifact = _camera_trajectory_artifact(scene)
    if artifact is None:
        return None
    snapshot = metadata_as_dict(artifact.payload_snapshot)  # type: ignore[arg-type]
    raw_matrices = snapshot.get("T_reconstruction_from_camera")
    if raw_matrices is None:
        return None
    if bool(snapshot.get("canonical_geometry_authoritative", False)):
        raise ValueError("Persistent camera trajectory snapshot cannot claim canonical geometry authority.")
    if bool(snapshot.get("certified_metrology", False)):
        raise ValueError("Persistent camera trajectory snapshot cannot claim certified metrology.")

    pose_convention = str(snapshot.get("pose_convention", "")).strip()
    if pose_convention != CANONICAL_CAMERA_POSE_CONVENTION:
        raise ValueError(
            "Persistent camera trajectory with matrices must declare "
            f"pose_convention={CANONICAL_CAMERA_POSE_CONVENTION!r}."
        )
    raw_ids = snapshot.get("frame_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("Persistent camera trajectory frame_ids must be a JSON array.")
    camera_ids = tuple(str(value).strip() for value in raw_ids)
    matrices = np.asarray(raw_matrices, dtype=np.float64)
    if matrices.size == 0:
        return None
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise ValueError("Persistent camera trajectory matrices must have shape (N, 4, 4).")
    if matrices.shape[0] > MAX_PERSISTED_CANONICAL_CAMERA_POSES:
        raise ValueError("Persistent camera trajectory exceeds the Phase 9.3C safety bound.")
    if len(camera_ids) != matrices.shape[0]:
        raise ValueError("Persistent camera trajectory frame IDs and matrices must have equal length.")

    raw_layers = snapshot.get("pose_layers", [])
    if raw_layers is None:
        raw_layers = []
    if not isinstance(raw_layers, list):
        raise ValueError("Persistent camera trajectory pose_layers must be a JSON array when present.")
    pose_layers = tuple(str(value).strip() for value in raw_layers)

    provider = str(snapshot.get("source_provider_id", artifact.provider_id)).strip()
    if provider != artifact.provider_id:
        raise ValueError("Persistent camera trajectory provider provenance does not match its artifact.")
    frame_id = str(snapshot.get("reconstruction_frame_id", "")).strip()
    total = snapshot.get("frame_count", matrices.shape[0])
    if isinstance(total, bool):
        raise ValueError("Persistent camera trajectory frame_count must be an integer.")
    try:
        total_count = int(total)
    except (TypeError, ValueError) as exc:
        raise ValueError("Persistent camera trajectory frame_count must be an integer.") from exc

    positions = keyframe_centers_in_viewer(matrices)
    return CanonicalCameraTrajectoryPresentation(
        scene_id=scene.scene_id,
        artifact_id=artifact.artifact_id,
        provider_id=artifact.provider_id,
        pose_convention=pose_convention,
        reconstruction_frame_id=frame_id,
        camera_ids=camera_ids,
        pose_layers=pose_layers,
        matrices_reconstruction=matrices,
        positions_viewer=positions,
        total_pose_count=total_count,
        certified_metrology=False,
    )


def _camera_trajectory_artifact(
    scene: PersistentReconstructionScene,
) -> PersistentSceneArtifactRecord | None:
    candidates = [
        item for item in scene.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("Persistent scene must contain at most one camera_trajectory artifact.")
    artifact = candidates[0]
    if artifact.authority is not ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY:
        raise ValueError("Persistent camera trajectory must remain supporting geometry evidence.")
    if artifact.canonical_geometry_authoritative:
        raise ValueError("Persistent camera trajectory cannot become canonical geometry authority.")
    return artifact


def _validate_pose(matrix: np.ndarray, *, field_name: str) -> None:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{field_name} must be a finite 4x4 matrix.")
    if not np.allclose(value[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{field_name} must be homogeneous.")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{field_name} rotation must be orthonormal.")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-5):
        raise ValueError(f"{field_name} rotation must have determinant +1.")


def _nonempty(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


__all__ = [
    "CANONICAL_CAMERA_POSE_CONVENTION",
    "MAX_PERSISTED_CANONICAL_CAMERA_POSES",
    "CanonicalCameraTrajectoryPresentation",
    "build_canonical_camera_trajectory_snapshot",
    "load_persistent_canonical_camera_trajectory",
]
