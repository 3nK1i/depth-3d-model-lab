from __future__ import annotations

from core.reconstruction import (
    DeterministicMockPoseGraphOptimizer,
    RelocalizationAcceptancePolicy,
    RelocalizationStatus,
)
from tests.reconstruction_global_consistency_helpers import make_drifted_graph


def test_loop_optimization_then_relocalization_keeps_arbitrary_reconstruction_gauge() -> None:
    graph = make_drifted_graph()
    optimized = DeterministicMockPoseGraphOptimizer().optimize(graph)
    final_pose = optimized.optimized_poses[-1]

    relocalized = RelocalizationAcceptancePolicy().evaluate(
        query_frame_id="query-after-loop",
        map_revision=optimized.output_revision,
        matched_keyframe_id=final_pose.keyframe_id,
        confidence=0.93,
        inlier_count=90,
        T_reconstruction_from_camera=final_pose.T_reconstruction_from_camera,
    )

    assert optimized.final_error < optimized.initial_error
    assert optimized.gauge.unit_to_metre is None
    assert relocalized.status is RelocalizationStatus.SUCCESS
    assert relocalized.gauge.unit_to_metre is None
    assert relocalized.as_dict()["pose_convention"] == "T_reconstruction_from_camera"
