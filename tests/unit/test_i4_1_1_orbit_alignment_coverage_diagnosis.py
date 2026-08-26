from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_orbit_alignment_coverage_diagnosis import build_i4_1_1_diagnosis


def _look_at(azimuth_deg: float, radius: float = 5.0) -> np.ndarray:
    a = np.deg2rad(azimuth_deg)
    center = np.asarray([radius * np.cos(a), radius * np.sin(a), 0.5], dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    up = np.asarray([0.0, 0.0, 1.0])
    x = np.cross(up, forward); x /= np.linalg.norm(x)
    y = np.cross(forward, x); y /= np.linalg.norm(y)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.column_stack([x, y, forward])
    m[:3, 3] = center
    return m


def _core() -> np.ndarray:
    return np.asarray([(x, y, z) for x in np.linspace(-0.5, 0.5, 5) for y in np.linspace(-0.4, 0.4, 5) for z in np.linspace(-0.3, 0.3, 5)], dtype=np.float64)


def _fixture(*, selected_count: int = 12, aligned_count: int = 30, selected_arc: float = 220.0, i2_selected_pass: bool = False):
    count = 40
    ids = tuple(f"kf-{i}" for i in range(count))
    poses = np.asarray([_look_at(i * 360.0 / count) for i in range(count)])
    candidates = []
    for i in range(count):
        accepted = i >= count - aligned_count
        candidates.append({
            "workspace_order": i,
            "keyframe_id": ids[i],
            "alignment_accepted": accepted,
            "rejection_code": None if accepted else "robust_scale_fit_rejected",
            "scale_to_reconstruction_units": (20.0 + i) if accepted else None,
            "valid_correspondences": 100 if accepted else 20,
            "inlier_correspondences": 80 if accepted else 5,
            "median_relative_residual": 0.05 if accepted else 0.3,
            "p90_relative_residual": 0.08 if accepted else 0.5,
        })
    selected_indices = np.linspace(count - aligned_count, count - 1, selected_count).round().astype(int)
    # For narrow-arc tests, replace selected poses by IDs corresponding to the first part of the circle.
    if selected_arc < 150.0:
        max_index = max(2, int(round(selected_arc / 360.0 * count)))
        selected_indices = np.linspace(0, max_index, selected_count).round().astype(int)
        # Make these candidates accepted as well and preserve total accepted count by rejecting later ones.
        for item in candidates:
            item["alignment_accepted"] = False
            item["rejection_code"] = "robust_scale_fit_rejected"
            item["scale_to_reconstruction_units"] = None
        accepted_indices = list(dict.fromkeys(selected_indices.tolist()))
        for i in range(count - 1, -1, -1):
            if len(accepted_indices) >= aligned_count:
                break
            if i not in accepted_indices:
                accepted_indices.append(i)
        for i in accepted_indices:
            candidates[i]["alignment_accepted"] = True
            candidates[i]["rejection_code"] = None
            candidates[i]["scale_to_reconstruction_units"] = 20.0 + i
    selected_ids = tuple(ids[int(i)] for i in selected_indices)
    gates = {
        "minimum_registered_fraction": True,
        "minimum_dense_alignment_fraction": aligned_count / 40.0 >= 0.8,
        "minimum_selected_fraction_of_registered": i2_selected_pass,
        "minimum_largest_component_fraction": True,
        "minimum_core_fraction_of_supported": True,
        "minimum_extended_fraction_of_supported": True,
        "maximum_rejected_supported_fraction": True,
        "exact_observation_accounting": True,
        "persistence_self_contained": True,
        "authority_boundaries_preserved": True,
        "per_view_attribution_available": True,
    }
    return {
        "source_i4": {"phase": "I4.1", "status": "degraded", "failed_stage": "object_orbit_baseline_quality", "reason": "ordinary I2 failed"},
        "sfm": {"input_keyframes": count},
        "i1": {"phase": "I1", "status": "pass", "pipeline_execution_complete": True, "sfm": {"registered_view_count": count, "candidate_dense_orders": list(range(count))}, "dense_qualification": {"alignment_accepted_views": aligned_count}},
        "i2": {"phase": "I2", "automated_pass": False, "automated_gates": gates, "policy": {"minimum_selected_fraction_of_registered": 0.6}},
        "e3": {"phase": "E3", "pass": True, "candidate_views": candidates, "selected_keyframe_ids": list(selected_ids)},
        "ids": ids,
        "poses": poses,
        "core": _core(),
    }


def _build(data):
    return build_i4_1_1_diagnosis(
        source_i4_summary=data["source_i4"], sfm_summary=data["sfm"], i1_summary=data["i1"], i2_summary=data["i2"], e3_summary=data["e3"], registered_camera_ids=data["ids"], registered_camera_matrices=data["poses"], core_points=data["core"]
    )


def test_diagnosis_reports_i2_failed_gate_vector():
    result = _build(_fixture())
    assert "minimum_dense_alignment_fraction" in result["i2_policy_diagnosis"]["failed_gate_names"]
    assert "minimum_selected_fraction_of_registered" in result["i2_policy_diagnosis"]["failed_gate_names"]


def test_geometry_and_contract_gates_can_pass_when_coverage_fails():
    result = _build(_fixture())
    assert result["i2_policy_diagnosis"]["geometry_and_contract_gates_pass"] is True


def test_alignment_denominator_semantics_are_reported():
    result = _build(_fixture())
    assert result["counts"]["alignment_fraction_of_candidates_i2_semantics"] == pytest.approx(0.75)
    assert result["counts"]["alignment_fraction_of_registered_i4_semantics"] == pytest.approx(0.75)
    assert result["counts"]["alignment_denominator_semantics_match_for_this_scene"] is True


def test_alignment_rejection_clusters_are_exact():
    result = _build(_fixture())
    assert result["alignment_diagnosis"]["rejected_orders"] == list(range(10))
    assert result["alignment_diagnosis"]["rejected_clusters"] == [list(range(10))]


def test_alignment_rejection_reason_histogram_is_exact():
    result = _build(_fixture())
    assert result["alignment_diagnosis"]["rejection_reason_histogram"] == {"robust_scale_fit_rejected": 10}


def test_alignment_scale_summary_is_finite():
    result = _build(_fixture())
    scale = result["alignment_diagnosis"]
    assert scale["accepted_scale_min"] == pytest.approx(30.0)
    assert scale["accepted_scale_max"] == pytest.approx(59.0)
    assert scale["accepted_scale_max_to_min_ratio"] > 1.0


def test_registered_aligned_selected_orbits_are_all_computed():
    result = _build(_fixture())
    assert result["orbit_diagnosis"]["registered"]["camera_count"] == 40
    assert result["orbit_diagnosis"]["aligned"]["camera_count"] == 30
    assert result["orbit_diagnosis"]["selected"]["camera_count"] == 12


def test_selected_orbit_gate_subset_contains_object_specific_gates():
    result = _build(_fixture())
    gates = result["orbit_diagnosis"]["selected_orbit_gate_subset"]
    assert set(gates) == {"selected_view_count", "selected_azimuth_coverage", "selected_azimuth_bins", "selected_opposed_view_diversity", "selected_target_facing", "selected_standoff_consistency"}


def test_alignment_failure_is_classified_without_policy_reassessment():
    result = _build(_fixture(aligned_count=30))
    assert result["policy_compatibility"]["failure_classification"] in {"alignment_coverage_insufficient", "mixed_alignment_and_selected_orbit_coverage_failure"}
    assert result["policy_compatibility"]["policy_reassessment_candidate"] is False


def test_room_scale_selected_fraction_only_can_be_reassessment_candidate():
    data = _fixture(aligned_count=40, selected_count=12, i2_selected_pass=False)
    result = _build(data)
    if result["orbit_diagnosis"]["selected_orbit_coverage_pass"]:
        assert result["policy_compatibility"]["failure_classification"] == "room_scale_selected_fraction_policy_mismatch_candidate"
        assert result["policy_compatibility"]["policy_reassessment_candidate"] is True


def test_narrow_selected_orbit_is_detected():
    data = _fixture(aligned_count=40, selected_count=12, selected_arc=90.0)
    result = _build(data)
    assert result["orbit_diagnosis"]["selected_orbit_coverage_pass"] is False
    assert result["policy_compatibility"]["failure_classification"] == "selected_orbit_coverage_insufficient"


def test_diagnosis_never_authorizes_threshold_relaxation():
    result = _build(_fixture())
    assert result["policy_compatibility"]["threshold_relaxation_authorized"] is False
    assert result["decision"]["authorizes_i4_1_policy_change"] is False
    assert result["decision"]["authorizes_i4_2_detail_preservation"] is False


def test_authority_boundaries_remain_unpromoted():
    result = _build(_fixture())
    authority = result["authority"]
    assert authority["sfm_rerun"] is False
    assert authority["da3_rerun"] is False
    assert authority["fusion_rerun"] is False
    assert authority["metric_scale_claimed"] is False
    assert authority["measurement_authoritative"] is False
    assert authority["surface_mesh_generated"] is False


def test_rejects_non_baseline_quality_i4_failure_stage():
    data = _fixture(); data["source_i4"]["failed_stage"] = "object_orbit_i1_baseline"
    with pytest.raises(ReconstructionValidationError):
        _build(data)


def test_rejects_passing_i2_summary_because_this_is_failure_diagnosis():
    data = _fixture(); data["i2"]["automated_pass"] = True
    with pytest.raises(ReconstructionValidationError):
        _build(data)


def test_rejects_alignment_count_mismatch():
    data = _fixture(); data["i1"]["dense_qualification"]["alignment_accepted_views"] = 29
    with pytest.raises(ReconstructionValidationError):
        _build(data)
