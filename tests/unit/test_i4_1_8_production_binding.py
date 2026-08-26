from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_production_binding import (
    I418ProductionBinding,
    build_final_i4_1_assessment,
    build_i4_1_8_decision,
    build_production_binding,
    deterministic_capped_positions,
)


def _authorized_b() -> dict:
    return {
        "phase": "I4.1.7B",
        "status": "qualified_for_i4_1_8",
        "authority": {
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "dense_consistency_threshold_relaxation_authorized": False,
        },
        "decision": {
            "i4_1_8_production_binding_authorized": True,
            "i4_1_8_core_target_requalification_required": True,
        },
    }


def _pose(center: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = target - center
    z = z / np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z))) > 0.95:
        up = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    t = np.eye(4)
    t[:3, :3] = np.column_stack([x, y, z])
    t[:3, 3] = center
    return t


def _orbit(count: int = 40, coverage_deg: float = 220.0, target=(0.0, 0.0, 0.0)) -> np.ndarray:
    target_v = np.asarray(target, dtype=float)
    angles = np.deg2rad(np.linspace(-coverage_deg / 2.0, coverage_deg / 2.0, count))
    poses = []
    for a in angles:
        center = target_v + np.asarray([4.0 * np.cos(a), 4.0 * np.sin(a), 0.25 * np.sin(2.0 * a)])
        poses.append(_pose(center, target_v))
    return np.asarray(poses)


def _core(target=(0.0, 0.0, 0.0), count: int = 1000) -> np.ndarray:
    rng = np.random.default_rng(7)
    points = rng.normal(0.0, 0.1, size=(count, 3)) + np.asarray(target, dtype=float)
    # Mirror so the median is exactly target and target semantics are stable.
    return np.concatenate([points, 2.0 * np.asarray(target, dtype=float) - points], axis=0)


def test_deterministic_cap_preserves_endpoints_and_is_strict() -> None:
    positions = deterministic_capped_positions(118, 40)
    assert len(positions) == 40
    assert positions[0] == 0
    assert positions[-1] == 117
    assert all(b > a for a, b in zip(positions, positions[1:]))
    assert positions == deterministic_capped_positions(118, 40)


def test_binding_uses_qualified_plan_and_preserves_full_span() -> None:
    sources = tuple(range(0, 1180, 10))
    binding = build_production_binding(
        i4_1_7b_summary=_authorized_b(),
        plan_name="global_anchors_plus_local_neighbors",
        qualified_source_sequence_indices=sources,
        final_fusion_cap=40,
    )
    assert binding.selected_view_count == 40
    assert binding.selected_source_sequence_indices[0] == sources[0]
    assert binding.selected_source_sequence_indices[-1] == sources[-1]
    assert binding.source_span_preserved is True


def test_binding_fails_closed_without_i417b_authorization() -> None:
    payload = _authorized_b()
    payload["decision"]["i4_1_8_production_binding_authorized"] = False
    with pytest.raises(ReconstructionValidationError):
        build_production_binding(
            i4_1_7b_summary=payload,
            plan_name="global_anchors_plus_local_neighbors",
            qualified_source_sequence_indices=tuple(range(118)),
            final_fusion_cap=40,
        )


def test_final_assessment_uses_retained_core_and_passes_locked_gates() -> None:
    binding = I418ProductionBinding(
        plan_name="global_anchors_plus_local_neighbors",
        qualified_source_count=40,
        final_fusion_cap=40,
        selected_positions=tuple(range(40)),
        selected_source_sequence_indices=tuple(range(40)),
    )
    poses = _orbit(40, 220.0)
    assessment = build_final_i4_1_assessment(
        input_global_views=40,
        registered_global_views=40,
        production_alignment_accepted_views=40,
        registered_camera_matrices=poses,
        production_selected_camera_matrices=poses,
        retained_core_points=_core(),
        i2_summary={"phase": "I2", "automated_pass": True},
        binding=binding,
    )
    assert assessment["pass"] is True
    assert assessment["target"]["mode"] == "retained_core_robust_median"
    assert assessment["target"]["production_target_authority"] is True
    assert assessment["selected_orbit"]["azimuth_coverage_degrees"] >= 150.0
    assert assessment["selected_orbit"]["maximum_opposed_view_angle_degrees"] >= 120.0
    assert assessment["gates"]["ordinary_i2_bound_scene_quality"] is True


def test_final_assessment_does_not_reuse_provisional_target_when_core_target_breaks_orbit() -> None:
    binding = I418ProductionBinding(
        plan_name="global_anchors_plus_local_neighbors",
        qualified_source_count=40,
        final_fusion_cap=40,
        selected_positions=tuple(range(40)),
        selected_source_sequence_indices=tuple(range(40)),
    )
    poses = _orbit(40, 220.0, target=(0.0, 0.0, 0.0))
    # Retained Core target is deliberately far away.  Final assessment must
    # fail rather than carrying forward the I4.1.7B trajectory target.
    assessment = build_final_i4_1_assessment(
        input_global_views=40,
        registered_global_views=40,
        production_alignment_accepted_views=40,
        registered_camera_matrices=poses,
        production_selected_camera_matrices=poses,
        retained_core_points=_core(target=(25.0, 0.0, 0.0)),
        i2_summary={"phase": "I2", "automated_pass": True},
        binding=binding,
    )
    assert assessment["pass"] is False
    assert assessment["target"]["target_reconstruction"][0] == pytest.approx(25.0)
    assert assessment["failed_gates"]


def test_final_decision_requires_bound_scene_i2_and_keeps_i4_2_manual() -> None:
    binding = I418ProductionBinding(
        plan_name="global_anchors_plus_local_neighbors",
        qualified_source_count=40,
        final_fusion_cap=40,
        selected_positions=tuple(range(40)),
        selected_source_sequence_indices=tuple(range(40)),
    )
    final_assessment = {"pass": True, "gates": {"all": True}}
    ok = build_i4_1_8_decision(
        binding=binding,
        final_assessment=final_assessment,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=True,
    )
    assert ok["status"] == "ready_for_manual_orbit_review"
    assert ok["authority"]["production_sampling_integration_authorized"] is True
    assert ok["decision"]["i4_2_detail_preservation_authorized"] is False
    assert ok["manual_orbit_review_required"] is True

    bad = build_i4_1_8_decision(
        binding=binding,
        final_assessment=final_assessment,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=False,
    )
    assert bad["status"] == "degraded"
    assert bad["authority"]["production_sampling_integration_authorized"] is False
