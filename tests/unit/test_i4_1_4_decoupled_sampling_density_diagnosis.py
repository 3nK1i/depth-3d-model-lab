from __future__ import annotations

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.offline_dense_i4_sampling_density_diagnosis import (
    build_i4_1_4_diagnosis,
    build_pair_observations,
    classify_sampling_density_evidence,
    summarize_pair_observations,
)


def _poses(n=5):
    result={}
    for i in range(n):
        a=np.deg2rad(i*8.0); m=np.eye(4); m[:3,3]=[4*np.cos(a),4*np.sin(a),1.0]; result[i]=m
    return result


def _obs(statuses, spacing=.5):
    poses=_poses(6); ts={i:int(i*spacing*1e9) for i in poses}; src={i:i*15 for i in poses}
    return build_pair_observations(edge_statuses=statuses,order_to_timestamp_ns=ts,order_to_source_index=src,order_to_camera_matrix=poses,target_reconstruction=[0,0,0])


def test_pair_observation_temporal_and_geometry_are_finite():
    obs=_obs({(0,1):"pass",(0,2):"degraded"})
    assert obs[0].delta_t_seconds == pytest.approx(.5)
    assert obs[1].delta_t_seconds == pytest.approx(1.0)
    assert obs[0].camera_baseline_over_median_standoff > 0
    assert obs[0].target_view_angle_degrees > 0


def test_pair_observation_rejects_missing_metadata():
    poses=_poses(3)
    with pytest.raises(ReconstructionValidationError):
        build_pair_observations(edge_statuses={(0,2):"pass"},order_to_timestamp_ns={0:0},order_to_source_index={0:0},order_to_camera_matrix=poses,target_reconstruction=[0,0,0])


def test_summary_counts_pass_degraded_inconclusive():
    summary=summarize_pair_observations(_obs({(0,1):"pass",(0,2):"degraded",(1,2):"inconclusive"}))
    assert summary["pass_count"] == 1
    assert summary["degraded_count"] == 1
    assert summary["inconclusive_count"] == 1
    assert summary["pass_fraction"] == .5


def test_time_buckets_distinguish_local_density():
    summary=summarize_pair_observations(_obs({(0,1):"pass",(0,2):"degraded",(0,4):"degraded"},spacing=.5))
    assert summary["time_buckets"]["le_0_75s"]["pass_fraction"] == 1.0
    assert summary["time_buckets"]["gt_0_75_le_1_5s"]["pass_fraction"] == 0.0


def _summary(pf):
    return {"pass_fraction":pf,"time_buckets":{"le_0_75s":{"pass_fraction":pf}}}


def test_classifies_local_density_dominant():
    d=classify_sampling_density_evidence(current=_summary(.05),same_source_dense_prefix=_summary(.30),short_control=_summary(.35))
    assert d["classification"] == "local_density_dominant"
    assert d["decoupled_sampling_architecture_candidate"] is True


def test_classifies_capture_regime_dominant():
    d=classify_sampling_density_evidence(current=_summary(.04),same_source_dense_prefix=_summary(.08),short_control=_summary(.45))
    assert d["classification"] == "capture_regime_dominant_local_density_secondary"
    assert d["cleaner_static_capture_recommended"] is True


def test_classifies_mixed_capture_and_density():
    d=classify_sampling_density_evidence(current=_summary(.03),same_source_dense_prefix=_summary(.18),short_control=_summary(.50))
    assert d["classification"] == "mixed_capture_regime_and_local_density"
    assert d["decoupled_sampling_architecture_candidate"] is True
    assert d["cleaner_static_capture_recommended"] is True


def test_same_source_poor_even_at_density_is_explicit():
    d=classify_sampling_density_evidence(current=_summary(.02),same_source_dense_prefix=_summary(.09),short_control=None)
    assert d["classification"] == "same_source_dense_consistency_poor_even_at_higher_local_density"


def test_no_policy_authorization_from_diagnosis():
    d=classify_sampling_density_evidence(current=_summary(.02),same_source_dense_prefix=_summary(.30),short_control=_summary(.60))
    assert d["sampling_policy_change_authorized"] is False
    assert d["dense_consistency_threshold_relaxation_authorized"] is False


def test_build_diagnosis_preserves_authority_boundaries():
    d=build_i4_1_4_diagnosis(current_summary=_summary(.1),same_source_prefix_summary=None,short_control_summary=None)
    assert d["status"] == "diagnosis_ready"
    assert d["authority"]["sampling_policy_changed"] is False
    assert d["authority"]["surface_mesh_generated"] is False
    assert d["decision"]["i4_2_detail_preservation_authorized"] is False
