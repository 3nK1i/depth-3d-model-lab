from __future__ import annotations

from core.reconstruction.mvs_da3_g2c_review import (
    build_blind_plan,
    build_manual_review_template,
    resolve_equal_render_point_count,
)


def test_many_blind_plans_are_deterministic_and_public_metadata_never_leaks_mapping():
    for index in range(500):
        sha = f"{index:064x}"[-64:]
        first = build_blind_plan(sha)
        second = build_blind_plan(sha)
        assert first.assignment == second.assignment
        assert first.commitment_sha256 == second.commitment_sha256
        assert "mapping" not in first.public_dict()
        review = build_manual_review_template(
            scenario_id=first.scenario_id,
            views=["oblique", "front", "side", "top"],
            framings=["detail", "context"],
            assignment_commitment_sha256=first.commitment_sha256,
        )
        assert len(review["pair_observations"]) == 8
        assert review["blind_key_revealed"] is False


def test_equal_render_budget_stress_never_exceeds_either_provider_or_requested_budget():
    for mvs in range(1, 512, 7):
        for da3 in range(1, 512, 11):
            for requested in (1, 17, 64, 250, 1000):
                draw = resolve_equal_render_point_count(
                    mvs_source_count=mvs,
                    da3_source_count=da3,
                    requested_point_budget=requested,
                )
                assert draw == min(mvs, da3, requested)
                assert 0 < draw <= mvs
                assert 0 < draw <= da3
                assert draw <= requested
