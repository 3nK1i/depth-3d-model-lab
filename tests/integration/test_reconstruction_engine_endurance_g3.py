from pathlib import Path

from core.reconstruction import build_reconstruction_endurance_campaign
from tests.reconstruction_g3_helpers import (
    checkpoint_rejection_evidence,
    clean_evidence,
    optional_degradation_evidence,
    required_failure_evidence,
)


def test_g3_mixed_campaign_preserves_authority_and_failure_boundaries(tmp_path: Path):
    cycles = (
        clean_evidence(0),
        clean_evidence(1),
        optional_degradation_evidence(2, tmp_path),
        required_failure_evidence(3),
        checkpoint_rejection_evidence(4),
    )
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed
    payload = report.as_dict()
    assert payload["stale_state_detected"] is False
    assert payload["unique_clean_scene_ids"] is True
    assert payload["full_regression_deferred_to"] == "8.7G4"


def test_g3_repeated_degradation_does_not_poison_later_clean_run(tmp_path: Path):
    degraded = optional_degradation_evidence(0, tmp_path)
    clean = clean_evidence(1)
    assert degraded.pipeline_status == "degraded"
    assert clean.pipeline_status == "completed"
    assert clean.canonical_geometry_authoritative is True
