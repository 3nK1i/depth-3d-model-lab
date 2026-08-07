"""Dependency-light pose-graph utilities for Phase 8.5A.

The deterministic optimizer in this module is a test/development backend. It is
not the production optimizer. It corrects synthetic translation drift while
preserving the graph anchor and arbitrary monocular gauge, giving later native
integration a strict contract to satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import degrees, acos
from typing import Protocol, runtime_checkable

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.global_consistency import (
    GlobalConsistencyResult,
    OptimizedKeyframePose,
    PoseGraphConstraint,
    PoseGraphConstraintKind,
    PoseGraphSnapshot,
)


def predicted_relative_transform(
    T_reconstruction_from_camera_from: np.ndarray,
    T_reconstruction_from_camera_to: np.ndarray,
) -> np.ndarray:
    """Return ``T_camera_to_from_camera_from`` for two reconstruction poses."""

    return np.linalg.inv(T_reconstruction_from_camera_to) @ T_reconstruction_from_camera_from


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    trace_value = float(np.trace(rotation))
    cosine = max(-1.0, min(1.0, (trace_value - 1.0) * 0.5))
    return degrees(acos(cosine))


def constraint_residual(constraint: PoseGraphConstraint, nodes_by_id: dict[str, object]) -> tuple[float, float]:
    from_node = nodes_by_id[constraint.from_keyframe_id]
    to_node = nodes_by_id[constraint.to_keyframe_id]
    predicted = predicted_relative_transform(
        from_node.T_reconstruction_from_camera,
        to_node.T_reconstruction_from_camera,
    )
    residual = np.linalg.inv(constraint.T_camera_to_from_camera_from) @ predicted
    return float(np.linalg.norm(residual[:3, 3])), _rotation_angle_deg(residual[:3, :3])


@runtime_checkable
class PoseGraphOptimizer(Protocol):
    def optimize(self, snapshot: PoseGraphSnapshot) -> GlobalConsistencyResult: ...


@dataclass(slots=True)
class DeterministicMockPoseGraphOptimizer:
    """Translation-only deterministic mock used for contract/integration tests.

    The mock accepts one anchor-to-later verified loop constraint with negligible
    rotational residual, then distributes its translation correction across the
    ordered keyframe chain. Real SE(3) optimization belongs to later Phase 8.5
    native integration.
    """

    max_supported_rotation_residual_deg: float = 1e-5

    def optimize(self, snapshot: PoseGraphSnapshot) -> GlobalConsistencyResult:
        if not isinstance(snapshot, PoseGraphSnapshot):
            raise TypeError("snapshot must be a PoseGraphSnapshot.")
        nodes = snapshot.nodes
        nodes_by_id = {node.keyframe_id: node for node in nodes}
        anchor_index = next(i for i, node in enumerate(nodes) if node.keyframe_id == snapshot.anchor_keyframe_id)
        if anchor_index != 0:
            raise ReconstructionValidationError(
                "Deterministic mock optimizer requires the first graph node to be the anchor."
            )

        loops = sorted(
            snapshot.accepted_loop_closures,
            key=lambda item: (-item.confidence, item.constraint_id),
        )
        output_revision = snapshot.revision + 1
        original_matrices = [np.array(node.T_reconstruction_from_camera, copy=True) for node in nodes]
        optimized_matrices = [np.array(matrix, copy=True) for matrix in original_matrices]

        initial_error = 0.0
        final_error = 0.0
        max_translation_correction = 0.0
        max_rotation_correction_deg = 0.0
        iterations = 0

        if loops:
            loop = loops[0]
            if loop.from_keyframe_id != snapshot.anchor_keyframe_id:
                raise ReconstructionValidationError(
                    "Deterministic mock optimizer requires loop closure to originate at the anchor."
                )
            target_index = next(i for i, node in enumerate(nodes) if node.keyframe_id == loop.to_keyframe_id)
            if target_index <= 0:
                raise ReconstructionValidationError("Loop-closure target must occur after the anchor.")

            translation_error, rotation_error_deg = constraint_residual(loop, nodes_by_id)
            initial_error = translation_error + rotation_error_deg
            if rotation_error_deg > self.max_supported_rotation_residual_deg:
                raise ReconstructionValidationError(
                    "Deterministic mock optimizer supports translation-drift correction only."
                )

            anchor_pose = original_matrices[0]
            desired_target = anchor_pose @ np.linalg.inv(loop.T_camera_to_from_camera_from)
            correction = desired_target[:3, 3] - original_matrices[target_index][:3, 3]
            for index in range(1, target_index + 1):
                fraction = index / target_index
                optimized_matrices[index][:3, 3] += correction * fraction
                max_translation_correction = max(
                    max_translation_correction,
                    float(np.linalg.norm(correction * fraction)),
                )
            for index in range(target_index + 1, len(nodes)):
                optimized_matrices[index][:3, 3] += correction
                max_translation_correction = max(
                    max_translation_correction,
                    float(np.linalg.norm(correction)),
                )
            iterations = 1

            optimized_nodes_by_id = {
                node.keyframe_id: type("_NodeView", (), {"T_reconstruction_from_camera": matrix})()
                for node, matrix in zip(nodes, optimized_matrices)
            }
            final_translation, final_rotation_deg = constraint_residual(loop, optimized_nodes_by_id)
            final_error = final_translation + final_rotation_deg

        optimized = tuple(
            OptimizedKeyframePose(
                keyframe_id=node.keyframe_id,
                frame_id=node.frame_id,
                timestamp_ns=node.timestamp_ns,
                source_pose_id=node.source_pose_id,
                optimization_revision=output_revision,
                T_reconstruction_from_camera=matrix,
            )
            for node, matrix in zip(nodes, optimized_matrices)
        )

        if not np.allclose(
            optimized[0].T_reconstruction_from_camera,
            nodes[0].T_reconstruction_from_camera,
            atol=1e-12,
        ):
            raise ReconstructionValidationError("Pose-graph optimization must preserve the anchor pose.")

        return GlobalConsistencyResult(
            input_revision=snapshot.revision,
            output_revision=output_revision,
            anchor_keyframe_id=snapshot.anchor_keyframe_id,
            optimized_poses=optimized,
            loop_closure_count=len(loops),
            converged=True,
            iterations=iterations,
            initial_error=initial_error,
            final_error=final_error,
            max_translation_correction=max_translation_correction,
            max_rotation_correction_deg=max_rotation_correction_deg,
            gauge=snapshot.gauge,
        )
