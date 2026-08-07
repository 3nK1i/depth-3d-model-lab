from __future__ import annotations

import numpy as np

from core.reconstruction import DeterministicMockPoseGraphOptimizer, predicted_relative_transform
from tests.reconstruction_global_consistency_helpers import make_drifted_graph


def test_mock_optimizer_closes_synthetic_translation_drift_and_preserves_anchor() -> None:
    graph = make_drifted_graph()
    result = DeterministicMockPoseGraphOptimizer().optimize(graph)

    assert result.converged is True
    assert result.loop_closure_count == 1
    assert result.input_revision == 4
    assert result.output_revision == 5
    assert result.final_error < result.initial_error
    assert result.final_error < 1e-9
    assert np.allclose(
        result.optimized_poses[0].T_reconstruction_from_camera,
        graph.nodes[0].T_reconstruction_from_camera,
    )
    positions = [pose.T_reconstruction_from_camera[0, 3] for pose in result.optimized_poses]
    assert np.allclose(positions, [0.0, 1.0, 2.0, 3.0], atol=1e-9)
    assert result.gauge.unit_to_metre is None


def test_predicted_relative_transform_uses_explicit_camera_to_from_camera_from_convention() -> None:
    graph = make_drifted_graph()
    relative = predicted_relative_transform(
        graph.nodes[0].T_reconstruction_from_camera,
        graph.nodes[-1].T_reconstruction_from_camera,
    )
    assert np.isclose(relative[0, 3], -3.3)
