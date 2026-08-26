from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc


def test_i4_1_7a_large_selected_pose_set_stays_scale_invariant_and_diagnosis_only() -> None:
    matrices = _arc(120, 235.0)
    target = np.zeros(3)
    source = {
        "phase": "I4.1.7",
        "status": "degraded",
        "assessment": {"selected_orbit": orbit_coverage_metrics(matrices, target)},
    }
    core = np.tile(np.asarray([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.1, -0.1, 0.0]]), (20, 1))
    result = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=target,
        retained_core_points=core,
    )
    assert result["counts"]["selected_cameras"] == 120
    assert result["authority"]["threshold_relaxation"] is False
    assert result["decision"]["authorizes_i4_1_8_production_binding"] is False
