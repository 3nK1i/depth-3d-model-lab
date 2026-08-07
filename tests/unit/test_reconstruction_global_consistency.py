from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction import (
    LoopClosureCandidate,
    LoopClosureDecision,
    PoseGraphConstraint,
    PoseGraphConstraintKind,
    PoseGraphNode,
    PoseGraphSnapshot,
    ReconstructionGauge,
    ReconstructionValidationError,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)
from tests.reconstruction_global_consistency_helpers import make_drifted_graph, relative_x


def test_verified_loop_candidate_requires_geometric_evidence() -> None:
    with pytest.raises(ReconstructionValidationError):
        LoopClosureCandidate(
            candidate_id="candidate",
            current_keyframe_id="kf-10",
            matched_keyframe_id="kf-1",
            similarity_score=0.9,
            retrieval_rank=1,
            temporal_separation=9,
            decision=LoopClosureDecision.VERIFIED,
        )


def test_pose_graph_constraint_serializes_explicit_relative_pose_convention() -> None:
    constraint = PoseGraphConstraint(
        constraint_id="loop",
        kind=PoseGraphConstraintKind.LOOP_CLOSURE,
        from_keyframe_id="kf-a",
        to_keyframe_id="kf-b",
        T_camera_to_from_camera_from=relative_x(3.0),
        information_diagonal=np.ones(6),
        confidence=0.95,
        inlier_count=50,
        source_candidate_id="candidate",
    )
    assert constraint.as_dict()["relative_pose_convention"] == "T_camera_to_from_camera_from"


def test_pose_graph_rejects_unknown_constraint_endpoint() -> None:
    graph = make_drifted_graph()
    bad = PoseGraphConstraint(
        constraint_id="bad",
        kind=PoseGraphConstraintKind.ODOMETRY,
        from_keyframe_id="kf-0",
        to_keyframe_id="missing",
        T_camera_to_from_camera_from=np.eye(4),
        information_diagonal=np.ones(6),
        confidence=1.0,
    )
    with pytest.raises(ReconstructionValidationError):
        PoseGraphSnapshot(
            revision=graph.revision,
            anchor_keyframe_id=graph.anchor_keyframe_id,
            nodes=graph.nodes,
            constraints=graph.constraints + (bad,),
        )


def test_pose_graph_forbids_metric_gauge_claims() -> None:
    graph = make_drifted_graph()
    with pytest.raises(ReconstructionValidationError):
        PoseGraphSnapshot(
            revision=graph.revision,
            anchor_keyframe_id=graph.anchor_keyframe_id,
            nodes=graph.nodes,
            constraints=graph.constraints,
            gauge=ReconstructionGauge(
                scale_status=SfmScaleStatus.ESTIMATED_METRIC,
                alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
                unit_to_metre=0.5,
                scale_source="forbidden-in-8.5a",
            ),
        )


def test_pose_graph_node_matrix_is_read_only() -> None:
    node = PoseGraphNode(
        keyframe_id="kf",
        frame_id="frame",
        timestamp_ns=1,
        source_pose_id="pose",
        T_reconstruction_from_camera=np.eye(4),
    )
    assert node.T_reconstruction_from_camera.flags.writeable is False
