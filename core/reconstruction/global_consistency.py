"""Phase 8.5A global-consistency contracts.

This module deliberately stays backend-neutral. It defines loop-closure,
relocalization, pose-graph, and optimized-pose contracts without enabling any
native backend feature. Raw monocular geometry remains in the arbitrary
``reconstruction`` gauge until a later explicit metric/world-alignment stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts.validation import JsonValue, metadata_as_dict, normalize_metadata, require_finite, require_nonempty
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import ReconstructionGauge, SfmScaleStatus, SfmWorldAlignmentStatus
from core.reconstruction.sfm_coordinates import validate_rigid_matrix
from core.reconstruction.validation import require_nonnegative_int, require_positive_int, require_probability

GLOBAL_CONSISTENCY_SCHEMA_VERSION = 1


class PoseGraphConstraintKind(str, Enum):
    ODOMETRY = "odometry"
    LOOP_CLOSURE = "loop_closure"


class LoopClosureDecision(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class RelocalizationStatus(str, Enum):
    SUCCESS = "success"
    REJECTED = "rejected"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    FAILED = "failed"


def _require_arbitrary_gauge(gauge: ReconstructionGauge) -> ReconstructionGauge:
    if not isinstance(gauge, ReconstructionGauge):
        raise TypeError("gauge must be a ReconstructionGauge.")
    if (
        gauge.scale_status is not SfmScaleStatus.ARBITRARY
        or gauge.alignment_status is not SfmWorldAlignmentStatus.ARBITRARY
        or gauge.unit_to_metre is not None
    ):
        raise ReconstructionValidationError(
            "Phase 8.5A monocular global-consistency data must remain arbitrary-scale and unaligned."
        )
    return gauge


def _readonly_six_vector(value: object, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (6,) or not np.isfinite(vector).all() or np.any(vector <= 0.0):
        raise ReconstructionValidationError(
            f"{field_name} must be a finite positive 6-vector."
        )
    vector = np.array(vector, copy=True)
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, slots=True)
class LoopClosureCandidate:
    candidate_id: str
    current_keyframe_id: str
    matched_keyframe_id: str
    similarity_score: float
    retrieval_rank: int
    temporal_separation: int
    decision: LoopClosureDecision = LoopClosureDecision.PROPOSED
    geometric_inlier_count: int = 0
    geometric_inlier_ratio: float | None = None
    metadata: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", require_nonempty(self.candidate_id, field_name="candidate_id"))
        object.__setattr__(self, "current_keyframe_id", require_nonempty(self.current_keyframe_id, field_name="current_keyframe_id"))
        object.__setattr__(self, "matched_keyframe_id", require_nonempty(self.matched_keyframe_id, field_name="matched_keyframe_id"))
        if self.current_keyframe_id == self.matched_keyframe_id:
            raise ReconstructionValidationError("A loop-closure candidate must reference two different keyframes.")
        object.__setattr__(self, "similarity_score", require_probability(self.similarity_score, field_name="similarity_score"))
        object.__setattr__(self, "retrieval_rank", require_positive_int(self.retrieval_rank, field_name="retrieval_rank"))
        object.__setattr__(self, "temporal_separation", require_positive_int(self.temporal_separation, field_name="temporal_separation"))
        if not isinstance(self.decision, LoopClosureDecision):
            raise TypeError("decision must be a LoopClosureDecision.")
        inliers = require_nonnegative_int(self.geometric_inlier_count, field_name="geometric_inlier_count")
        object.__setattr__(self, "geometric_inlier_count", inliers)
        ratio = self.geometric_inlier_ratio
        if ratio is not None:
            object.__setattr__(self, "geometric_inlier_ratio", require_probability(ratio, field_name="geometric_inlier_ratio"))
        if self.decision is LoopClosureDecision.VERIFIED and (inliers <= 0 or ratio is None):
            raise ReconstructionValidationError("Verified loop closures require geometric inlier evidence.")
        object.__setattr__(self, "metadata", normalize_metadata(self.metadata))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "current_keyframe_id": self.current_keyframe_id,
            "matched_keyframe_id": self.matched_keyframe_id,
            "similarity_score": self.similarity_score,
            "retrieval_rank": self.retrieval_rank,
            "temporal_separation": self.temporal_separation,
            "decision": self.decision.value,
            "geometric_inlier_count": self.geometric_inlier_count,
            "geometric_inlier_ratio": self.geometric_inlier_ratio,
            "metadata": metadata_as_dict(self.metadata),  # type: ignore[arg-type]
        }


@dataclass(frozen=True, slots=True)
class PoseGraphNode:
    keyframe_id: str
    frame_id: str
    timestamp_ns: int
    source_pose_id: str
    T_reconstruction_from_camera: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id"))
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) < 0:
            raise ReconstructionValidationError("timestamp_ns must be non-negative.")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "source_pose_id", require_nonempty(self.source_pose_id, field_name="source_pose_id"))
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            validate_rigid_matrix(self.T_reconstruction_from_camera, field_name="T_reconstruction_from_camera"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "source_pose_id": self.source_pose_id,
            "pose_convention": "T_reconstruction_from_camera",
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
        }


@dataclass(frozen=True, slots=True)
class PoseGraphConstraint:
    constraint_id: str
    kind: PoseGraphConstraintKind
    from_keyframe_id: str
    to_keyframe_id: str
    T_camera_to_from_camera_from: np.ndarray
    information_diagonal: np.ndarray
    confidence: float
    inlier_count: int = 0
    accepted: bool = True
    source_candidate_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_id", require_nonempty(self.constraint_id, field_name="constraint_id"))
        if not isinstance(self.kind, PoseGraphConstraintKind):
            raise TypeError("kind must be a PoseGraphConstraintKind.")
        object.__setattr__(self, "from_keyframe_id", require_nonempty(self.from_keyframe_id, field_name="from_keyframe_id"))
        object.__setattr__(self, "to_keyframe_id", require_nonempty(self.to_keyframe_id, field_name="to_keyframe_id"))
        if self.from_keyframe_id == self.to_keyframe_id:
            raise ReconstructionValidationError("Pose-graph constraints must connect different keyframes.")
        object.__setattr__(
            self,
            "T_camera_to_from_camera_from",
            validate_rigid_matrix(self.T_camera_to_from_camera_from, field_name="T_camera_to_from_camera_from"),
        )
        object.__setattr__(self, "information_diagonal", _readonly_six_vector(self.information_diagonal, field_name="information_diagonal"))
        object.__setattr__(self, "confidence", require_probability(self.confidence, field_name="confidence"))
        object.__setattr__(self, "inlier_count", require_nonnegative_int(self.inlier_count, field_name="inlier_count"))
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool.")
        source_id = str(self.source_candidate_id).strip()
        if self.kind is PoseGraphConstraintKind.LOOP_CLOSURE and self.accepted:
            if not source_id:
                raise ReconstructionValidationError("Accepted loop-closure constraints require source_candidate_id.")
            if self.inlier_count <= 0:
                raise ReconstructionValidationError("Accepted loop-closure constraints require positive inlier_count.")
        object.__setattr__(self, "source_candidate_id", source_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "kind": self.kind.value,
            "from_keyframe_id": self.from_keyframe_id,
            "to_keyframe_id": self.to_keyframe_id,
            "relative_pose_convention": "T_camera_to_from_camera_from",
            "T_camera_to_from_camera_from": self.T_camera_to_from_camera_from.tolist(),
            "information_diagonal": self.information_diagonal.tolist(),
            "confidence": self.confidence,
            "inlier_count": self.inlier_count,
            "accepted": self.accepted,
            "source_candidate_id": self.source_candidate_id,
        }


@dataclass(frozen=True, slots=True)
class PoseGraphSnapshot:
    revision: int
    anchor_keyframe_id: str
    nodes: tuple[PoseGraphNode, ...]
    constraints: tuple[PoseGraphConstraint, ...]
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        object.__setattr__(self, "revision", require_nonnegative_int(self.revision, field_name="revision"))
        object.__setattr__(self, "anchor_keyframe_id", require_nonempty(self.anchor_keyframe_id, field_name="anchor_keyframe_id"))
        nodes = tuple(self.nodes)
        constraints = tuple(self.constraints)
        if not nodes or any(not isinstance(node, PoseGraphNode) for node in nodes):
            raise ReconstructionValidationError("PoseGraphSnapshot requires at least one PoseGraphNode.")
        if any(not isinstance(item, PoseGraphConstraint) for item in constraints):
            raise TypeError("constraints must contain PoseGraphConstraint values.")
        node_ids = [node.keyframe_id for node in nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ReconstructionValidationError("Pose-graph keyframe IDs must be unique.")
        if self.anchor_keyframe_id not in set(node_ids):
            raise ReconstructionValidationError("anchor_keyframe_id must reference a graph node.")
        timestamps = [node.timestamp_ns for node in nodes]
        if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
            raise ReconstructionValidationError("Pose-graph node timestamps must be strictly increasing.")
        known = set(node_ids)
        constraint_ids = [item.constraint_id for item in constraints]
        if len(set(constraint_ids)) != len(constraint_ids):
            raise ReconstructionValidationError("Pose-graph constraint IDs must be unique.")
        for item in constraints:
            if item.from_keyframe_id not in known or item.to_keyframe_id not in known:
                raise ReconstructionValidationError("Pose-graph constraint endpoints must reference graph nodes.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "gauge", _require_arbitrary_gauge(self.gauge))

    @property
    def accepted_loop_closures(self) -> tuple[PoseGraphConstraint, ...]:
        return tuple(
            item
            for item in self.constraints
            if item.kind is PoseGraphConstraintKind.LOOP_CLOSURE and item.accepted
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_CONSISTENCY_SCHEMA_VERSION,
            "revision": self.revision,
            "anchor_keyframe_id": self.anchor_keyframe_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "constraints": [item.as_dict() for item in self.constraints],
            "gauge": self.gauge.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class OptimizedKeyframePose:
    keyframe_id: str
    frame_id: str
    timestamp_ns: int
    source_pose_id: str
    optimization_revision: int
    T_reconstruction_from_camera: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "keyframe_id", require_nonempty(self.keyframe_id, field_name="keyframe_id"))
        object.__setattr__(self, "frame_id", require_nonempty(self.frame_id, field_name="frame_id"))
        if isinstance(self.timestamp_ns, bool) or int(self.timestamp_ns) < 0:
            raise ReconstructionValidationError("timestamp_ns must be non-negative.")
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "source_pose_id", require_nonempty(self.source_pose_id, field_name="source_pose_id"))
        object.__setattr__(self, "optimization_revision", require_positive_int(self.optimization_revision, field_name="optimization_revision"))
        object.__setattr__(
            self,
            "T_reconstruction_from_camera",
            validate_rigid_matrix(self.T_reconstruction_from_camera, field_name="T_reconstruction_from_camera"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyframe_id": self.keyframe_id,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
            "source_pose_id": self.source_pose_id,
            "optimization_revision": self.optimization_revision,
            "pose_layer": "optimized",
            "pose_convention": "T_reconstruction_from_camera",
            "T_reconstruction_from_camera": self.T_reconstruction_from_camera.tolist(),
        }


@dataclass(frozen=True, slots=True)
class GlobalConsistencyResult:
    input_revision: int
    output_revision: int
    anchor_keyframe_id: str
    optimized_poses: tuple[OptimizedKeyframePose, ...]
    loop_closure_count: int
    converged: bool
    iterations: int
    initial_error: float
    final_error: float
    max_translation_correction: float
    max_rotation_correction_deg: float
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        input_revision = require_nonnegative_int(self.input_revision, field_name="input_revision")
        output_revision = require_positive_int(self.output_revision, field_name="output_revision")
        if output_revision <= input_revision:
            raise ReconstructionValidationError("output_revision must advance beyond input_revision.")
        object.__setattr__(self, "input_revision", input_revision)
        object.__setattr__(self, "output_revision", output_revision)
        object.__setattr__(self, "anchor_keyframe_id", require_nonempty(self.anchor_keyframe_id, field_name="anchor_keyframe_id"))
        poses = tuple(self.optimized_poses)
        if not poses or any(not isinstance(pose, OptimizedKeyframePose) for pose in poses):
            raise ReconstructionValidationError("GlobalConsistencyResult requires optimized poses.")
        if len({pose.keyframe_id for pose in poses}) != len(poses):
            raise ReconstructionValidationError("Optimized keyframe IDs must be unique.")
        if self.anchor_keyframe_id not in {pose.keyframe_id for pose in poses}:
            raise ReconstructionValidationError("Optimized result must retain the anchor keyframe.")
        if any(pose.optimization_revision != output_revision for pose in poses):
            raise ReconstructionValidationError("All optimized poses must use output_revision.")
        object.__setattr__(self, "optimized_poses", poses)
        object.__setattr__(self, "loop_closure_count", require_nonnegative_int(self.loop_closure_count, field_name="loop_closure_count"))
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool.")
        object.__setattr__(self, "iterations", require_nonnegative_int(self.iterations, field_name="iterations"))
        for field_name in ("initial_error", "final_error", "max_translation_correction", "max_rotation_correction_deg"):
            number = require_finite(getattr(self, field_name), field_name=field_name)
            if number < 0.0:
                raise ReconstructionValidationError(f"{field_name} must not be negative.")
            object.__setattr__(self, field_name, number)
        if self.final_error > self.initial_error + 1e-12:
            raise ReconstructionValidationError("Global optimization must not report a larger final error.")
        object.__setattr__(self, "gauge", _require_arbitrary_gauge(self.gauge))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_CONSISTENCY_SCHEMA_VERSION,
            "input_revision": self.input_revision,
            "output_revision": self.output_revision,
            "anchor_keyframe_id": self.anchor_keyframe_id,
            "optimized_poses": [pose.as_dict() for pose in self.optimized_poses],
            "loop_closure_count": self.loop_closure_count,
            "converged": self.converged,
            "iterations": self.iterations,
            "initial_error": self.initial_error,
            "final_error": self.final_error,
            "max_translation_correction": self.max_translation_correction,
            "max_rotation_correction_deg": self.max_rotation_correction_deg,
            "gauge": self.gauge.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class RelocalizationResult:
    query_frame_id: str
    map_revision: int
    status: RelocalizationStatus
    confidence: float
    inlier_count: int
    matched_keyframe_id: str = ""
    T_reconstruction_from_camera: np.ndarray | None = None
    reason: str = ""
    gauge: ReconstructionGauge = field(default_factory=ReconstructionGauge)

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_frame_id", require_nonempty(self.query_frame_id, field_name="query_frame_id"))
        object.__setattr__(self, "map_revision", require_nonnegative_int(self.map_revision, field_name="map_revision"))
        if not isinstance(self.status, RelocalizationStatus):
            raise TypeError("status must be a RelocalizationStatus.")
        object.__setattr__(self, "confidence", require_probability(self.confidence, field_name="confidence"))
        object.__setattr__(self, "inlier_count", require_nonnegative_int(self.inlier_count, field_name="inlier_count"))
        matched = str(self.matched_keyframe_id).strip()
        pose = self.T_reconstruction_from_camera
        if self.status is RelocalizationStatus.SUCCESS:
            matched = require_nonempty(matched, field_name="matched_keyframe_id")
            if pose is None:
                raise ReconstructionValidationError("Successful relocalization requires a reconstruction-frame pose.")
            pose = validate_rigid_matrix(pose, field_name="T_reconstruction_from_camera")
            if self.inlier_count <= 0:
                raise ReconstructionValidationError("Successful relocalization requires positive inlier_count.")
        else:
            if pose is not None:
                raise ReconstructionValidationError("Unsuccessful relocalization must not claim a pose.")
            if matched:
                raise ReconstructionValidationError("Unsuccessful relocalization must not claim a matched keyframe.")
        object.__setattr__(self, "matched_keyframe_id", matched)
        object.__setattr__(self, "T_reconstruction_from_camera", pose)
        object.__setattr__(self, "reason", str(self.reason).strip())
        object.__setattr__(self, "gauge", _require_arbitrary_gauge(self.gauge))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GLOBAL_CONSISTENCY_SCHEMA_VERSION,
            "query_frame_id": self.query_frame_id,
            "map_revision": self.map_revision,
            "status": self.status.value,
            "confidence": self.confidence,
            "inlier_count": self.inlier_count,
            "matched_keyframe_id": self.matched_keyframe_id,
            "pose_convention": "T_reconstruction_from_camera" if self.T_reconstruction_from_camera is not None else None,
            "T_reconstruction_from_camera": None if self.T_reconstruction_from_camera is None else self.T_reconstruction_from_camera.tolist(),
            "reason": self.reason,
            "gauge": self.gauge.as_dict(),
        }
