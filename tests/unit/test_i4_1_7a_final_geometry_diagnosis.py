from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import orbit_coverage_metrics
from tests.unit.test_i4_1_7_clean_static_orbit_qualification import _arc


def _summary(matrices: np.ndarray, target: np.ndarray) -> dict:
    return {
        "phase": "I4.1.7",
        "status": "degraded",
        "qualification_plan": "global_anchors_plus_local_neighbors",
        "assessment": {
            "selected_orbit": orbit_coverage_metrics(matrices, target),
        },
    }


def _core_cloud() -> np.ndarray:
    axis = np.linspace(-0.4, 0.4, 5)
    points = np.asarray([[x, y, z] for x in axis for y in axis for z in axis], dtype=np.float64)
    return points


def test_i4_1_7a_detects_provisional_target_collapse_on_broad_pose_arc() -> None:
    matrices = _arc(40, 220.0)
    bad_target = np.asarray([6.0, 0.0, 0.0], dtype=np.float64)
    result = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=_summary(matrices, bad_target),
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=bad_target,
        retained_core_points=_core_cloud(),
    )
    assert result["current_i4_1_7_metric_reproduction_exact"] is True
    assert result["diagnosis"]["selected_trajectory_centered_broad"] is True
    assert "registered_trajectory_circle_center" in result["diagnosis"]["robust_alternative_targets"]
    assert "retained_core_target" in result["diagnosis"]["robust_alternative_targets"]
    assert result["diagnosis"]["classification"] == "provisional_target_collapse_supported"
    assert result["diagnosis"]["orbit_target_correction_candidate"] is True
    assert result["decision"]["authorizes_orbit_metric_change"] is False
    assert result["authority"]["threshold_relaxation"] is False


def test_i4_1_7a_classifies_genuinely_narrow_selected_arc_without_authorizing_threshold_change() -> None:
    matrices = _arc(20, 85.0)
    target = np.zeros(3, dtype=np.float64)
    result = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=_summary(matrices, target),
        registered_camera_matrices=matrices,
        selected_camera_matrices=matrices,
        provisional_target_reconstruction=target,
        retained_core_points=_core_cloud(),
    )
    assert result["diagnosis"]["classification"] == "no_alternative_target_recovers_locked_orbit_geometry"
    assert result["diagnosis"]["orbit_target_correction_candidate"] is False
    assert result["decision"]["authorizes_orbit_threshold_relaxation"] is False


def test_i4_1_7a_fails_closed_without_completed_degraded_assessment() -> None:
    matrices = _arc(12, 220.0)
    with pytest.raises(ReconstructionValidationError):
        build_i4_1_7a_orbit_geometry_diagnosis(
            source_i4_1_7_summary={"phase": "I4.1.7", "status": "qualified_for_i4_1_8"},
            registered_camera_matrices=matrices,
            selected_camera_matrices=matrices,
            provisional_target_reconstruction=np.zeros(3),
        )
