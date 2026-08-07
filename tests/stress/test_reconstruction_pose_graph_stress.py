from __future__ import annotations

import numpy as np

from core.reconstruction import DeterministicMockPoseGraphOptimizer
from tests.reconstruction_global_consistency_helpers import make_drifted_graph


def test_repeated_pose_graph_optimization_is_deterministic() -> None:
    graph = make_drifted_graph(node_count=64)
    optimizer = DeterministicMockPoseGraphOptimizer()
    baseline = optimizer.optimize(graph)
    for _ in range(100):
        current = optimizer.optimize(graph)
        assert current.output_revision == baseline.output_revision
        assert current.final_error == baseline.final_error
        assert current.max_translation_correction == baseline.max_translation_correction
        for left, right in zip(current.optimized_poses, baseline.optimized_poses):
            assert np.array_equal(
                left.T_reconstruction_from_camera,
                right.T_reconstruction_from_camera,
            )
