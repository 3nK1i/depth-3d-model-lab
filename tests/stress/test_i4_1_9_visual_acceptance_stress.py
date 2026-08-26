from __future__ import annotations

import random

from core.reconstruction.offline_dense_i4_visual_acceptance import build_visual_binding, choose_visual_plan


def _dense(aligned_a: int, aligned_b: int) -> dict:
    def plan(name: str, planned: int, aligned: int, offset: int) -> dict:
        rows = []
        for i in range(planned):
            rows.append({
                "plan_order": i,
                "source_sequence_index": offset + i * 3,
                "alignment_accepted": i < aligned,
                "cloud_npz": f"/tmp/{name}-{i}.npz" if i < aligned else None,
            })
        return {
            "plan_name": name,
            "planned_candidates": planned,
            "alignment_accepted": aligned,
            "full_span": True,
            "pair_pass_fraction": 0.99 if name == "global" else 0.01,
            "selected_qualified_views": 3,
            "candidate_rows": rows,
        }

    return {
        "phase": "I4.1.6B",
        "plans": [plan("uniform", 84, aligned_a, 0), plan("global", 118, aligned_b, 1)],
        "decision": {
            "single_authoritative_sfm_frame": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "production_default_changed": False,
            "recommended_plan": "global",
            "real_dense_ab_validation_pass": False,
        },
        "thresholds": {"threshold_relaxation": False},
    }


def test_visual_selection_is_deterministic_across_repeated_runs() -> None:
    dense = _dense(70, 91)
    selections = [choose_visual_plan(dense, visual_fusion_cap=40) for _ in range(100)]
    assert {item["selected_plan"] for item in selections} == {"uniform"}
    bindings = [build_visual_binding(item, visual_fusion_cap=40) for item in selections]
    assert len({binding.selected_positions for binding in bindings}) == 1


def test_pair_score_never_overrides_failed_alignment_gate_under_many_inputs() -> None:
    rng = random.Random(419)
    for _ in range(200):
        aligned_a = rng.randint(68, 84)  # often >=80%
        aligned_b = rng.randint(60, 94)  # strict pair favorite may fail visual alignment
        dense = _dense(aligned_a, aligned_b)
        selected = choose_visual_plan(dense, visual_fusion_cap=40)
        assessments = {item["plan_name"]: item for item in selected["plan_assessments"]}
        passing = [item for item in assessments.values() if item["visual_gate_pass"]]
        if passing:
            assert assessments[selected["selected_plan"]]["visual_gate_pass"] is True
        else:
            assert selected["visual_plan_available"] is False


def test_binding_never_exceeds_preserved_cap() -> None:
    dense = _dense(84, 118)
    for cap in (10, 17, 23, 40):
        selection = choose_visual_plan(dense, visual_fusion_cap=cap)
        binding = build_visual_binding(selection, visual_fusion_cap=cap)
        assert binding.selected_view_count == cap
        assert binding.selected_positions[0] == 0
        assert binding.selected_positions[-1] == binding.alignment_accepted_count - 1
