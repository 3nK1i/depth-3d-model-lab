from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from core.reconstruction.offline_dense_i4_sampling_density_diagnosis import build_pair_observations, summarize_pair_observations


def test_real_runner_module_imports_without_optional_ui_dependencies():
    path=Path(__file__).resolve().parents[2]/"scripts/run_i4_1_4_decoupled_sampling_density_diagnosis.py"
    spec=importlib.util.spec_from_file_location("i414_runner", path); assert spec and spec.loader
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module._EDGE_RE.match("  edge 37<->39 status=pass")


def test_pair_contract_uses_real_sfm_style_timestamps_and_source_indices():
    poses={}
    for i in range(4):
        m=np.eye(4); m[:3,3]=[float(i),0,5]; poses[i]=m
    obs=build_pair_observations(
        edge_statuses={(0,1):"pass",(0,2):"degraded",(1,3):"degraded"},
        order_to_timestamp_ns={0:0,1:500_000_000,2:1_000_000_000,3:1_500_000_000},
        order_to_source_index={0:0,1:15,2:30,3:45},
        order_to_camera_matrix=poses,
        target_reconstruction=[1.5,0,0],
    )
    summary=summarize_pair_observations(obs)
    assert summary["edge_count"] == 3
    assert summary["delta_t_seconds"]["pass"]["p50"] == .5


def test_diagnosis_is_read_only_by_contract():
    from core.reconstruction.offline_dense_i4_sampling_density_diagnosis import build_i4_1_4_diagnosis
    s={"pass_fraction":0.1,"time_buckets":{"le_0_75s":{"pass_fraction":0.1}}}
    result=build_i4_1_4_diagnosis(current_summary=s,same_source_prefix_summary=s,short_control_summary=s)
    assert all(result["authority"][k] is False for k in ("sfm_rerun","da3_rerun","dense_pair_rerun","fusion_rerun","sampling_policy_changed","dense_consistency_policy_changed"))
