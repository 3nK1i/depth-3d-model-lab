from __future__ import annotations

import numpy as np

from core.reconstruction import (
    LoopClosureCandidate,
    LoopClosureDecision,
    PoseGraphConstraint,
    PoseGraphConstraintKind,
    PoseGraphNode,
    PoseGraphSnapshot,
)


def pose_x(x: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = x
    return matrix


def relative_x(delta_from_to: float) -> np.ndarray:
    # T_camera_to_from_camera_from for identity-rotation cameras where the
    # reconstruction-frame camera centers differ by +delta_from_to.
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 3] = -delta_from_to
    return matrix


def make_drifted_graph(*, revision: int = 4, node_count: int = 4) -> PoseGraphSnapshot:
    nodes = tuple(
        PoseGraphNode(
            keyframe_id=f"kf-{index}",
            frame_id=f"frame-{index}",
            timestamp_ns=(index + 1) * 1_000_000,
            source_pose_id=f"pose-{index}",
            T_reconstruction_from_camera=pose_x(index * 1.1),
        )
        for index in range(node_count)
    )
    constraints = []
    for index in range(node_count - 1):
        constraints.append(
            PoseGraphConstraint(
                constraint_id=f"odom-{index}",
                kind=PoseGraphConstraintKind.ODOMETRY,
                from_keyframe_id=f"kf-{index}",
                to_keyframe_id=f"kf-{index + 1}",
                T_camera_to_from_camera_from=relative_x(1.1),
                information_diagonal=np.ones(6),
                confidence=0.95,
                inlier_count=80,
            )
        )
    constraints.append(
        PoseGraphConstraint(
            constraint_id="loop-0-last",
            kind=PoseGraphConstraintKind.LOOP_CLOSURE,
            from_keyframe_id="kf-0",
            to_keyframe_id=f"kf-{node_count - 1}",
            T_camera_to_from_camera_from=relative_x(float(node_count - 1)),
            information_diagonal=np.full(6, 2.0),
            confidence=0.99,
            inlier_count=120,
            source_candidate_id="candidate-0-last",
        )
    )
    return PoseGraphSnapshot(
        revision=revision,
        anchor_keyframe_id="kf-0",
        nodes=nodes,
        constraints=tuple(constraints),
    )


def make_verified_candidate() -> LoopClosureCandidate:
    return LoopClosureCandidate(
        candidate_id="candidate-0-last",
        current_keyframe_id="kf-3",
        matched_keyframe_id="kf-0",
        similarity_score=0.96,
        retrieval_rank=1,
        temporal_separation=3,
        decision=LoopClosureDecision.VERIFIED,
        geometric_inlier_count=120,
        geometric_inlier_ratio=0.8,
    )
