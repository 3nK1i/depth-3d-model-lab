from __future__ import annotations

import numpy as np

from core.reconstruction.offline_dense_i4_sampling_density_diagnosis import build_pair_observations, summarize_pair_observations


def _poses(n):
    result={}
    for i in range(n):
        a=2*np.pi*i/n; m=np.eye(4); m[:3,3]=[5*np.cos(a),5*np.sin(a),1]; result[i]=m
    return result


def test_large_pair_set_is_deterministic_and_accounted():
    n=80; poses=_poses(n); ts={i:int(i*.4e9) for i in range(n)}; src={i:i*12 for i in range(n)}
    edges={}
    for i in range(n):
        for gap in (1,2,3):
            if i+gap<n: edges[(i,i+gap)]="pass" if gap==1 and i%3 else "degraded"
    a=summarize_pair_observations(build_pair_observations(edge_statuses=edges,order_to_timestamp_ns=ts,order_to_source_index=src,order_to_camera_matrix=poses,target_reconstruction=[0,0,0]))
    b=summarize_pair_observations(build_pair_observations(edge_statuses=dict(reversed(list(edges.items()))),order_to_timestamp_ns=ts,order_to_source_index=src,order_to_camera_matrix=poses,target_reconstruction=[0,0,0]))
    assert a == b
    assert a["decisive_edge_count"] == len(edges)


def test_extreme_scale_camera_coordinates_remain_finite():
    poses=_poses(20)
    for m in poses.values(): m[:3,3]*=1e6
    edges={(i,i+1):"pass" for i in range(19)}
    s=summarize_pair_observations(build_pair_observations(edge_statuses=edges,order_to_timestamp_ns={i:i*1_000_000_000 for i in poses},order_to_source_index={i:i for i in poses},order_to_camera_matrix=poses,target_reconstruction=[0,0,0]))
    assert np.isfinite(s["baseline_over_median_standoff"]["all"]["p90"])


def test_repeated_diagnosis_has_no_mutation():
    poses=_poses(10); edges={(i,i+1):"degraded" for i in range(9)}; ts={i:i*500_000_000 for i in poses}; src={i:i*15 for i in poses}
    first=build_pair_observations(edge_statuses=edges,order_to_timestamp_ns=ts,order_to_source_index=src,order_to_camera_matrix=poses,target_reconstruction=[0,0,0])
    for _ in range(50):
        assert build_pair_observations(edge_statuses=edges,order_to_timestamp_ns=ts,order_to_source_index=src,order_to_camera_matrix=poses,target_reconstruction=[0,0,0]) == first
