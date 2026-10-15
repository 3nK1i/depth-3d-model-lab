from __future__ import annotations

from copy import deepcopy

from core.reconstruction.i5_difficult_handheld_robustness import build_i5_1b_difficult_handheld_replay
from tests.i5_1b_helpers import (
    BASELINE_SHA,
    dense_ab_summary,
    i1_summary,
    localization_summary,
    passing_i5_1a,
    source_identity,
)

I51A_SHA = BASELINE_SHA
I51B_SHA = "c" * 64
OOM_SHA = "d" * 64
RECEIPT_SHA = "e" * 64


def clone(value):
    return deepcopy(value)


def replay_parameters() -> dict:
    return {
        "profile": "efficient",
        "matching": "exhaustive",
        "max_keyframes": 40,
        "max_dense_views": 40,
        "dense_candidate_cap": 120,
        "video_target_fps": 2.0,
        "local_target_fps": 2.0,
        "process_res": 392,
    }


def degraded_i5_1b() -> dict:
    dense = dense_ab_summary(dense_gate=False, selected=7)
    dense["baseline_global_only"].update({
        "pair_pass_fraction": 0.021,
        "selected_qualified_views": 2,
        "selected_path": {"view_count": 2, "orders": [37, 39]},
    })
    uniform, anchors = dense["plans"]
    uniform.update({
        "alignment_accepted": 93,
        "alignment_acceptance_fraction": 93 / 102,
        "pair_edges": 252,
        "pair_pass": 32,
        "pair_degraded": 220,
        "pair_pass_fraction": 32 / 252,
        "selected_qualified_views": 7,
        "dense_gate_pass": False,
    })
    anchors.update({
        "alignment_accepted": 109,
        "alignment_acceptance_fraction": 109 / 118,
        "pair_edges": 309,
        "pair_pass": 43,
        "pair_degraded": 266,
        "pair_pass_fraction": 43 / 309,
        "selected_qualified_views": 6,
        "dense_gate_pass": False,
    })
    dense["decision"].update({
        "recommended_plan": "uniform_localized_dense_pool",
        "real_dense_ab_validation_pass": False,
        "i4_1_6c_production_binding_authorized": False,
    })
    result = build_i5_1b_difficult_handheld_replay(
        i5_1a_result=passing_i5_1a(),
        i5_1a_sha256=I51A_SHA,
        source_identity=source_identity(),
        i1_summary=i1_summary(selected_views=2, status="degraded"),
        localization_summary=localization_summary(gate=True),
        dense_ab_summary=dense,
        stage_return_codes={"i1": 2, "localization": 0, "dense_ab": 0},
        replay_parameters=replay_parameters(),
    )
    assert result["quality"]["status"] == "degraded"
    assert result["quality"]["fresh_global_selected_qualified_views"] == 2
    assert result["quality"]["fresh_best_selected_qualified_views"] == 7
    return result


def passing_i5_1c_receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.1C_QUALIFICATION_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "focused_i5_1c_tests": True,
            "predecessor_i5_1_regression": True,
            "oom_correction_regression": True,
            "import_probe": True,
            "runner_help_probe": True,
            "provider_execution_started": False,
            "sfm_rerun": False,
            "dense_pair_rerun": False,
            "quality_replay_rerun": False,
            "thresholds_modified": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "counts": {
            "focused_tests": 30,
            "predecessor_unit_tests": 50,
            "predecessor_integration_tests": 20,
            "predecessor_stress_tests": 15,
            "predecessor_total_tests": 85,
            "combined_exercised_tests": 115,
        },
        "return_codes": {
            "focused": 0,
            "predecessor_unit": 0,
            "predecessor_integration": 0,
            "predecessor_stress": 0,
            "import_probe": 0,
            "runner_help_probe": 0,
        },
    }


def oom_diagnostic_text() -> str:
    return """===== REAL_DIR =====
/tmp/depth3d/i5_1b_difficult_handheld_replay_20260902_133628
I4.1.6A localizing extra candidate 176/176 source_index=1501...
I4.1.6A Decoupled Dense-Candidate Localization: LOCALIZATION_READY
I4.1.6B aligning/projecting extra 147/176 source_index=1259
Sep 02 13:56:18 host kernel: oom-kill: task=python,pid=190953,uid=1000
Sep 02 13:56:18 host kernel: Out of memory: Killed process 190953 (python) total-vm:13506776kB, anon-rss:10942216kB
"""
