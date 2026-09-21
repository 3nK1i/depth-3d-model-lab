from __future__ import annotations

from app.reconstruction_advisor_presentation import (
    I63D_BOUNDARY_TEXT,
    present_reconstruction_advisor,
)
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
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorResourceObservation,
    ReconstructionAdvisorRequest,
)
from core.reconstruction.reconstruction_advisor_engine import evaluate_reconstruction_advisor

SHA = "a" * 64


def _resources() -> AdvisorResourceObservation:
    return AdvisorResourceObservation(
        gpu_name="NVIDIA GeForce RTX 2060",
        cuda_ready=True,
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=5 * 1024**3,
        available_ram_bytes=8 * 1024**3,
    )


def _observation(
    provider_id: str,
    implementation: DenseProviderImplementation,
    family: DenseProviderFamily,
    *,
    outcome: AdvisorEvidenceOutcome = AdvisorEvidenceOutcome.PASS,
) -> AdvisorEvidenceObservation:
    return AdvisorEvidenceObservation(
        provider_id=provider_id,
        provider_family=family,
        provider_implementation=implementation,
        outcome=outcome,
        evidence_context_id="ctx-clean",
        matched_scenario_ids=(f"scenario-{provider_id}",),
        reason_codes=(f"{outcome.value}_evidence",),
        limitations=("frozen_context_only",),
        bindings=(
            AdvisorEvidenceBinding(
                evidence_id=f"evidence-{provider_id}",
                phase="I5.14B",
                path=f"/evidence/{provider_id}.json",
                sha256=SHA,
            ),
        ),
    )


def _state(
    *,
    source_ready: bool = True,
    sfm_usable: bool = True,
    context_resolution: AdvisorContextResolution = AdvisorContextResolution.EXPLICIT_FROZEN_TOKEN,
    available_provider_ids: tuple[str, ...] = ("colmap-mvs-external",),
    evidence: tuple[AdvisorEvidenceObservation, ...] | None = None,
    runtime_integrity_clean: bool = True,
) -> ReconstructionAdvisorLiveState:
    provider_evidence = evidence if evidence is not None else (
        _observation(
            "colmap-mvs-external",
            DenseProviderImplementation.COLMAP_MVS,
            DenseProviderFamily.MVS,
        ),
    )
    context_id = "" if context_resolution is AdvisorContextResolution.UNRESOLVED else "ctx-clean"
    request = ReconstructionAdvisorRequest(
        reconstruction_mode="offline_reconstruction",
        source_workflow="image_sequence" if source_ready else "none",
        source_ready=source_ready,
        source_item_count=128 if source_ready else 0,
        source_fingerprint="fp" if sfm_usable else "",
        workspace_state="partial_result" if sfm_usable else "ready",
        authoritative_sfm_usable=sfm_usable,
        runtime_integrity_clean=runtime_integrity_clean,
        evidence_context_resolution=context_resolution,
        evidence_context_id=context_id,
        available_provider_ids=available_provider_ids,
        resources=_resources(),
        provider_evidence=provider_evidence,
    )
    binding = ReconstructionAdvisorEvidenceBinding(
        evidence_context_resolution=context_resolution,
        evidence_context_id=context_id,
        available_provider_ids=available_provider_ids,
        resources=_resources(),
        provider_evidence=provider_evidence,
        runtime_integrity_clean=runtime_integrity_clean,
    )
    return ReconstructionAdvisorLiveState(
        request=request,
        decision=evaluate_reconstruction_advisor(request),
        evidence_binding=binding,
        source_matches_admitted_result=sfm_usable,
    )


def test_recommendation_is_presented_as_context_specific_guidance() -> None:
    presentation = present_reconstruction_advisor(_state())

    assert presentation.status_text == "Recommendation available"
    assert presentation.status_kind == "ready"
    assert presentation.guidance_text == "COLMAP MVS"
    assert presentation.evidence_text == "Exact qualified evidence"
    assert "not selected or activated automatically" in presentation.guidance_tooltip
    assert presentation.boundary_text == I63D_BOUNDARY_TEXT


def test_source_not_ready_abstention_is_intentional_neutral_state() -> None:
    presentation = present_reconstruction_advisor(
        _state(source_ready=False, sfm_usable=False)
    )

    assert presentation.status_text == "No recommendation"
    assert presentation.status_kind == "neutral"
    assert presentation.guidance_text == "Waiting for source"
    assert "not a reconstruction runtime failure" in presentation.guidance_tooltip


def test_ready_source_without_authoritative_sfm_waits_neutrally() -> None:
    presentation = present_reconstruction_advisor(_state(sfm_usable=False))

    assert presentation.status_kind == "neutral"
    assert presentation.guidance_text == "Waiting for qualified sparse SfM"


def test_unresolved_evidence_context_is_visible_without_claiming_failure() -> None:
    presentation = present_reconstruction_advisor(
        _state(context_resolution=AdvisorContextResolution.UNRESOLVED)
    )

    assert presentation.status_text == "No recommendation"
    assert presentation.status_kind == "neutral"
    assert presentation.guidance_text == "Evidence context unresolved"
    assert "Evidence context: unresolved" in presentation.evidence_tooltip


def test_multiple_exact_pass_candidates_are_warning_abstention_not_ranking() -> None:
    evidence = (
        _observation(
            "colmap-mvs-external",
            DenseProviderImplementation.COLMAP_MVS,
            DenseProviderFamily.MVS,
        ),
        _observation(
            "depth-pro",
            DenseProviderImplementation.DEPTH_PRO,
            DenseProviderFamily.SINGLE_IMAGE_DEPTH,
        ),
    )
    presentation = present_reconstruction_advisor(
        _state(
            available_provider_ids=("colmap-mvs-external", "depth-pro"),
            evidence=evidence,
        )
    )

    assert presentation.status_text == "No recommendation"
    assert presentation.status_kind == "warning"
    assert presentation.guidance_text == "Advisor abstained"
    assert presentation.evidence_text == "Conflicting evidence"
    assert "No global ranking or scalar provider score" in presentation.evidence_tooltip


def test_supported_but_unavailable_provider_is_explicit_warning() -> None:
    presentation = present_reconstruction_advisor(
        _state(available_provider_ids=("depth-pro",))
    )

    assert presentation.status_kind == "warning"
    assert presentation.guidance_text == "Supported provider unavailable"
    assert presentation.evidence_text == "Bounded evidence"


def test_runtime_integrity_blocker_is_warning_and_never_error_styled() -> None:
    presentation = present_reconstruction_advisor(
        _state(runtime_integrity_clean=False)
    )

    assert presentation.status_text == "No recommendation"
    assert presentation.status_kind == "warning"
    assert presentation.guidance_text == "Guidance withheld"
    assert presentation.status_kind != "error"


def test_boundary_tooltip_states_all_non_effects_and_pycolmap_authority() -> None:
    presentation = present_reconstruction_advisor(_state())

    text = presentation.boundary_tooltip
    assert "does not select or activate" in text
    assert "change settings" in text
    assert "press Start" in text
    assert "production Auto" in text
    assert "PyCOLMAP" in text
