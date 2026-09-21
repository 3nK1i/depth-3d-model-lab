from __future__ import annotations

from app.reconstruction_advisor_state import (
    ReconstructionAdvisorEvidenceBinding,
    ReconstructionAdvisorLiveState,
)
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
)
from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorEvidenceSupport,
    AdvisorResourceObservation,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)

SHA = "d" * 64


def resources() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="NVIDIA GeForce RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def observation(
    provider_id: str = "colmap-mvs-external",
    implementation: DenseProviderImplementation = DenseProviderImplementation.COLMAP_MVS,
    family: DenseProviderFamily = DenseProviderFamily.MVS,
    *,
    context: str = "ctx-i64d",
    outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS,
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=family,
        provider_implementation=implementation,
        outcome=outcome,
        evidence_context_id=context,
        matched_scenario_ids=("scenario-i64d",),
        reason_codes=("frozen_exact_evidence",),
        limitations=("exact_context_only",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id=f"evidence-{provider_id}",
                phase="I5.14B",
                path=f"/evidence/{provider_id}.json",
                sha256=SHA,
            ),
        ),
    )


def binding(
    *,
    context: str = "ctx-i64d",
    available: tuple[str, ...] = ("colmap-mvs-external",),
    observations: tuple[AdvisorEvidenceObservation, ...] | None = None,
) -> ReconstructionAdvisorEvidenceBinding:
    resolution = (
        AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN
        if context
        else AdvisorContextResolution.UNRESOLVED
    )
    evidence = observations if observations is not None else ((observation(context=context),) if context else ())
    return ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=resolution,
        evidence_context_id=context,
        available_provider_ids=available,
        resources=resources(),
        provider_evidence=evidence,
    )


def advisor_state(
    *,
    source_ready: bool = True,
    context: str = "ctx-i64d",
    available: tuple[str, ...] = ("colmap-mvs-external",),
    observations: tuple[AdvisorEvidenceObservation, ...] | None = None,
    recommend: str = "",
) -> ReconstructionAdvisorLiveState:
    evidence_binding = binding(
        context=context,
        available=available,
        observations=observations,
    )
    evidence = evidence_binding.provider_evidence
    resolution = evidence_binding.evidence_context_resolution
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence" if source_ready else "none",
        source_ready=source_ready,
        source_item_count=4 if source_ready else 0,
        source_fingerprint="",
        workspace_state="ready" if source_ready else "idle",
        authoritative_sfm_usable=True if source_ready else False,
        runtime_integrity_clean=True,
        evidence_context_resolution=resolution,
        evidence_context_id=context,
        available_provider_ids=available,
        resources=resources(),
        provider_evidence=evidence,
    )
    if recommend:
        item = next(value for value in evidence if value.provider_id == recommend)
        decision = ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.RECOMMEND,
            evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
            reason_codes=("unique_exact_pass",),
            explanations=("exact evidence",),
            recommended_provider_id=recommend,
            recommended_family=item.provider_family,
            recommended_implementation=item.provider_implementation,
            matched_scenario_ids=item.matched_scenario_ids,
            evidence_bindings=item.bindings,
            considered_provider_ids=(recommend,),
        )
    else:
        decision = ReconstructionAdvisorDecision(
            disposition=AdvisorDisposition.ABSTAIN,
            evidence_support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("test_abstention",),
            explanations=("no recommendation",),
        )
    return ReconstructionAdvisorLiveState(
        request=request,
        decision=decision,
        evidence_binding=evidence_binding,
        source_matches_admitted_result=source_ready,
    )
