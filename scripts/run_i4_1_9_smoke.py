#!/usr/bin/env python3
"""Non-heavy I4.1.9 visual-plan selection/authority smoke."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.offline_dense_i4_visual_acceptance import (
    build_visual_acceptance_decision,
    build_visual_binding,
    choose_visual_plan,
)


def _plan(name: str, planned: int, aligned: int, *, pair: float, selected: int) -> dict:
    rows = []
    for i in range(planned):
        accepted = i < aligned
        rows.append({
            "plan_order": i,
            "source_sequence_index": i * 10,
            "alignment_accepted": accepted,
            "cloud_npz": f"/tmp/fake-{name}-{i}.npz" if accepted else None,
        })
    return {
        "plan_name": name,
        "planned_candidates": planned,
        "alignment_accepted": aligned,
        "full_span": True,
        "pair_pass_fraction": pair,
        "selected_qualified_views": selected,
        "candidate_rows": rows,
    }


def main() -> int:
    dense = {
        "phase": "I4.1.6B",
        "plans": [
            _plan("uniform_localized_dense_pool", 84, 70, pair=0.068, selected=4),
            _plan("global_anchors_plus_local_neighbors", 118, 91, pair=0.157, selected=4),
        ],
        "decision": {
            "single_authoritative_sfm_frame": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
            "production_default_changed": False,
            "recommended_plan": "global_anchors_plus_local_neighbors",
            "real_dense_ab_validation_pass": False,
        },
        "thresholds": {"threshold_relaxation": False},
    }
    source = {
        "phase": "I4.1.7",
        "authority": {
            "single_authoritative_sfm_frame": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "clean_static_orbit_qualified": False,
            "i4_1_8_production_binding_authorized": False,
            "dense_consistency_threshold_relaxation_authorized": False,
        },
    }
    selection = choose_visual_plan(dense, visual_fusion_cap=40)
    binding = build_visual_binding(selection, visual_fusion_cap=40)
    decision = build_visual_acceptance_decision(
        source_summary=source,
        visual_plan_selection=selection,
        binding=binding,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=True,
    )
    assert selection["selected_plan"] == "uniform_localized_dense_pool"
    assert binding.selected_view_count == 40
    assert decision["automated_visual_reconstruction_pass"] is True
    assert decision["geometry_qualification"]["strict_geometry_qualification_passed"] is False
    assert decision["authority"]["metric_scale_claimed"] is False
    print("I4.1.9 smoke: PASS")
    print("visual_plan=uniform_localized_dense_pool aligned=70/84 bound=40")
    print("strict_geometry_qualification_passed=False visual_reconstruction_automated_pass=True")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
