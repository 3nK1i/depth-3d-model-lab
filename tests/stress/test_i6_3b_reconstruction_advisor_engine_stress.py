from __future__ import annotations

from itertools import permutations

from core.reconstruction.dense_provider_architecture import DenseProviderFamily, DenseProviderImplementation
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_advisor_engine import evaluate_reconstruction_advisor


def _obs(provider_id: str, impl: DenseProviderImplementation, outcome: AdvisorEvidenceOutcome, sha: str) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=(
            DenseProviderFamily.MVS
            if impl is DenseProviderImplementation.COLMAP_MVS
            else DenseProviderFamily.TEMPORAL_DEPTH
        ),
        provider_implementation=impl,
        outcome=outcome,
        evidence_context_id="ctx",
        matched_scenario_ids=("scenario",),
        reason_codes=(f"{outcome.value}_evidence",),
        limitations=("bounded",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id=provider_id,
                phase="I5.14B",
                path=f"/frozen/{provider_id}.json",
                sha256=sha * 64,
            ),
        ),
    )


def _request(evidence, available, *, gpu="RTX 2060", free=5 * 1024**3, total=6 * 1024**3):
    return ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence",
        source_ready=True,
        source_item_count=80,
        source_fingerprint="fp",
        workspace_state="partial_result",
        authoritative_sfm_usable=True,
        runtime_integrity_clean=True,
        evidence_context_resolution=AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
        evidence_context_id="ctx",
        available_provider_ids=tuple(available),
        resources=AdvisorResourceObservation(
            gpu_name=gpu,
            cuda_ready=True,
            total_vram_bytes=total,
            free_vram_bytes=free,
            available_ram_bytes=8 * 1024**3,
        ),
        provider_evidence=tuple(evidence),
    )


def test_i6_3b_same_input_same_decision_repeated_250_times() -> None:
    request = _request(
        (_obs("colmap-mvs-external", DenseProviderImplementation.COLMAP_MVS, AdvisorEvidenceOutcome.PASS, "a"),),
        ("colmap-mvs-external",),
    )
    expected = evaluate_reconstruction_advisor(request).as_dict()
    for _ in range(250):
        assert evaluate_reconstruction_advisor(request).as_dict() == expected


def test_i6_3b_evidence_and_availability_permutations_are_order_invariant() -> None:
    evidence = (
        _obs("rolling-depth-temporal", DenseProviderImplementation.ROLLING_DEPTH, AdvisorEvidenceOutcome.PASS, "a"),
        _obs("video-depth-anything-temporal", DenseProviderImplementation.VIDEO_DEPTH_ANYTHING, AdvisorEvidenceOutcome.DEGRADED, "b"),
        _obs("colmap-mvs-external", DenseProviderImplementation.COLMAP_MVS, AdvisorEvidenceOutcome.REJECTED, "c"),
    )
    available = tuple(item.provider_id for item in evidence)
    baseline = evaluate_reconstruction_advisor(_request(evidence, available)).as_dict()
    for evidence_order in permutations(evidence):
        for available_order in permutations(available):
            assert evaluate_reconstruction_advisor(_request(evidence_order, available_order)).as_dict() == baseline


def test_i6_3b_hardware_profiles_do_not_resolve_multiple_pass_ambiguity() -> None:
    evidence = (
        _obs("rolling-depth-temporal", DenseProviderImplementation.ROLLING_DEPTH, AdvisorEvidenceOutcome.PASS, "a"),
        _obs("video-depth-anything-temporal", DenseProviderImplementation.VIDEO_DEPTH_ANYTHING, AdvisorEvidenceOutcome.PASS, "b"),
    )
    profiles = (
        ("NVIDIA GeForce RTX 2060", 6 * 1024**3, 5 * 1024**3),
        ("NVIDIA GeForce RTX 4090", 24 * 1024**3, 20 * 1024**3),
        ("NVIDIA GeForce RTX 5090", 32 * 1024**3, 30 * 1024**3),
    )
    decisions = []
    for gpu, total, free in profiles:
        decision = evaluate_reconstruction_advisor(
            _request(evidence, tuple(item.provider_id for item in evidence), gpu=gpu, total=total, free=free)
        )
        assert decision.disposition is AdvisorDisposition.ABSTAIN
        decisions.append(decision.as_dict())
    assert decisions[0] == decisions[1] == decisions[2]


def test_i6_3b_negative_only_evidence_never_becomes_recommendation_under_repetition() -> None:
    outcomes = (AdvisorEvidenceOutcome.DEGRADED, AdvisorEvidenceOutcome.REJECTED, AdvisorEvidenceOutcome.UNQUALIFIED)
    for outcome in outcomes:
        request = _request(
            (_obs("rolling-depth-temporal", DenseProviderImplementation.ROLLING_DEPTH, outcome, "d"),),
            ("rolling-depth-temporal",),
        )
        for _ in range(50):
            assert evaluate_reconstruction_advisor(request).disposition is AdvisorDisposition.ABSTAIN
