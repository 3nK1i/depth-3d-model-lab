from __future__ import annotations

from core.reconstruction import (
    ReconstructionEnduranceCycleEvidence,
    ReconstructionEnduranceScenario,
    build_reconstruction_endurance_campaign,
)


def _clean(index: int) -> ReconstructionEnduranceCycleEvidence:
    return ReconstructionEnduranceCycleEvidence(
        cycle_index=index,
        scenario=ReconstructionEnduranceScenario.CLEAN_RECOVERY,
        passed=True,
        process_group_clean=True,
        stale_state_detected=False,
        pipeline_status="completed",
        persistent_scene_state="ready",
        scene_id=f"prs-phase8-9e-{index:02d}",
        canonical_geometry_authoritative=True,
        gaussian_authority="appearance_only",
        nerf_authority="appearance_only",
    )


def test_e_existing_g3_contract_handles_release_scale_repeated_recovery_without_stale_state() -> None:
    cycles = tuple(_clean(index) for index in range(12))
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed
    payload = report.as_dict()
    assert payload["unique_clean_scene_ids"] is True
    assert payload["stale_state_detected"] is False
    assert payload["all_process_groups_clean"] is True
