from pathlib import Path

from core.reconstruction import build_reconstruction_endurance_campaign
from tests.reconstruction_g3_helpers import clean_evidence, optional_degradation_evidence


def test_repeated_clean_cycles_keep_unique_scene_identity():
    cycles = tuple(clean_evidence(i) for i in range(20))
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed
    assert report.as_dict()["unique_clean_scene_ids"] is True


def test_repeated_degraded_then_clean_sequences_do_not_leak_state(tmp_path: Path):
    cycles = []
    index = 0
    for _ in range(5):
        cycles.append(optional_degradation_evidence(index, tmp_path)); index += 1
        cycles.append(clean_evidence(index)); index += 1
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=True,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    assert report.passed
