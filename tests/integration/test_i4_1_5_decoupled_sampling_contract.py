from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from core.reconstruction.offline_dense_i4_decoupled_sampling_counterfactual import build_i4_1_5_counterfactual


def _summary():
    def s(a,b,c,d):
        return {"pass_fraction":.1,"time_buckets":{"le_0_75s":{"pass_fraction":a},"gt_0_75_le_1_5s":{"pass_fraction":b},"gt_1_5_le_3_0s":{"pass_fraction":c},"gt_3_0s":{"pass_fraction":d}}}
    return {"diagnosis":{"classification":"mixed_capture_regime_and_local_density","local_density_signal":True,"capture_regime_signal":True},"current_full_span":s(.03,.03,.03,0.0),"same_source_dense_prefix":s(.269,.08,0.0,None),"short_control":s(.679,.577,.208,None)}


def test_runner_imports_without_optional_ui_dependencies():
    path=Path(__file__).resolve().parents[2]/"scripts/run_i4_1_5_decoupled_sampling_counterfactual.py"
    spec=importlib.util.spec_from_file_location("i415_runner",path); assert spec and spec.loader
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert callable(module._duration)


def test_contract_keeps_one_authoritative_sfm_frame():
    result=build_i4_1_5_counterfactual(duration_seconds=50.6,i4_1_4_summary=_summary())
    g=result["architecture_guardrails"]
    assert g["single_authoritative_sfm_frame"] is True
    assert g["extra_dense_candidates_are_observations_not_second_map"] is True
    assert g["extra_dense_candidates_require_pose_localization"] is True


def test_final_fusion_cap_is_preserved_across_all_plans():
    result=build_i4_1_5_counterfactual(duration_seconds=50.6,i4_1_4_summary=_summary(),final_fusion_cap=40)
    assert all(row["plan"]["final_fusion_cap"] == 40 for row in result["plans"])


def test_summary_is_json_serializable_without_nan(tmp_path):
    result=build_i4_1_5_counterfactual(duration_seconds=50.6,i4_1_4_summary=_summary())
    path=tmp_path/"summary.json"; path.write_text(json.dumps(result,allow_nan=False),encoding="utf-8")
    assert json.loads(path.read_text())["status"] == "counterfactual_ready"
