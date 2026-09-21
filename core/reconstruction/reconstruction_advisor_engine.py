"""I6.3B deterministic evidence-bound Reconstruction Advisor reasoning engine.

This engine is intentionally passive.  It consumes the immutable I6.3A
ReconstructionAdvisorRequest and returns an immutable ReconstructionAdvisorDecision.
It does not classify a live scene, select or activate a provider in production,
execute reconstruction, mutate settings, press Start, rank providers, compute a
scalar provider score, or use hardware resources as a tie-break.

The governing principle is conservative and deterministic:

    recommend only on exact, unique, clean PASS evidence support; otherwise abstain.

Availability can block a recommendation, but it never resolves evidence ambiguity.
Negative/degraded/unqualified evidence remains explicit in provider exclusions.
"""
from __future__ import annotations

from collections.abc import Iterable

from core.reconstruction.reconstruction_advisor_contract import (
    AdvisorContextResolution,
    AdvisorDisposition,
    AdvisorEvidenceBinding,
    AdvisorEvidenceObservation,
    AdvisorEvidenceOutcome,
    AdvisorEvidenceSupport,
    AdvisorProviderExclusion,
    ReconstructionAdvisorDecision,
    ReconstructionAdvisorRequest,
)

I63B_SCHEMA_VERSION = 1
I63B_PHASE = "I6.3B"
I63B_ENGINE_PROFILE = "evidence_bound_advisory_v1"


def _unique_sorted(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(value).strip() for value in values if str(value).strip()}))


def _bindings(observations: Iterable[AdvisorEvidenceObservation]) -> tuple[AdvisorEvidenceBinding, ...]:
    by_key: dict[tuple[str, str, str, str], AdvisorEvidenceBinding] = {}
    for observation in observations:
        for binding in observation.bindings:
            key = (binding.evidence_id, binding.phase, binding.path, binding.sha256)
            by_key[key] = binding
    return tuple(by_key[key] for key in sorted(by_key))


def _scenarios(observations: Iterable[AdvisorEvidenceObservation]) -> tuple[str, ...]:
    return _unique_sorted(
        scenario
        for observation in observations
        for scenario in observation.matched_scenario_ids
    )


def _limitations(observations: Iterable[AdvisorEvidenceObservation], *extra: str) -> tuple[str, ...]:
    return _unique_sorted(
        [
            limitation
            for observation in observations
            for limitation in observation.limitations
        ]
        + [value for value in extra if value]
    )


def _observation_by_provider(
    observations: Iterable[AdvisorEvidenceObservation],
) -> dict[str, AdvisorEvidenceObservation]:
    return {observation.provider_id: observation for observation in observations}


def _outcome_reason(outcome: AdvisorEvidenceOutcome) -> str | None:
    return {
        AdvisorEvidenceOutcome.PASS: None,
        AdvisorEvidenceOutcome.DEGRADED: "evidence_status_degraded",
        AdvisorEvidenceOutcome.REJECTED: "evidence_status_rejected",
        AdvisorEvidenceOutcome.UNQUALIFIED: "evidence_status_unqualified",
    }[outcome]


def _considered_provider_ids(
    request: ReconstructionAdvisorRequest,
    exact: tuple[AdvisorEvidenceObservation, ...],
) -> tuple[str, ...]:
    return _unique_sorted(
        (*request.available_provider_ids, *(observation.provider_id for observation in exact))
    )


def _exclusions(
    *,
    considered: tuple[str, ...],
    available: set[str],
    exact_by_provider: dict[str, AdvisorEvidenceObservation],
    selected_provider_id: str = "",
    ambiguity: bool = False,
) -> tuple[AdvisorProviderExclusion, ...]:
    result: list[AdvisorProviderExclusion] = []
    for provider_id in considered:
        if provider_id == selected_provider_id:
            continue
        reasons: list[str] = []
        observation = exact_by_provider.get(provider_id)
        if observation is None:
            reasons.append("no_exact_context_evidence")
        else:
            outcome_reason = _outcome_reason(observation.outcome)
            if outcome_reason:
                reasons.append(outcome_reason)
            elif ambiguity:
                reasons.append("multiple_exact_pass_candidates_no_tiebreak")
        if provider_id not in available:
            reasons.append("provider_unavailable")
        if ambiguity and observation is not None and observation.outcome is AdvisorEvidenceOutcome.PASS:
            if "multiple_exact_pass_candidates_no_tiebreak" not in reasons:
                reasons.append("multiple_exact_pass_candidates_no_tiebreak")
        if reasons:
            result.append(AdvisorProviderExclusion(provider_id, tuple(reasons)))
    return tuple(result)


def _abstain(
    *,
    support: AdvisorEvidenceSupport,
    reason_codes: tuple[str, ...],
    explanation: str,
    considered: tuple[str, ...] = (),
    exclusions: tuple[AdvisorProviderExclusion, ...] = (),
    observations: tuple[AdvisorEvidenceObservation, ...] = (),
    limitations: tuple[str, ...] = (),
) -> ReconstructionAdvisorDecision:
    combined_limitations = _unique_sorted(
        (*limitations, *_limitations(observations, "advisor_does_not_change_production_execution"))
    )
    return ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.ABSTAIN,
        evidence_support=support,
        reason_codes=reason_codes,
        explanations=(explanation,),
        limitations=combined_limitations,
        matched_scenario_ids=_scenarios(observations),
        evidence_bindings=_bindings(observations),
        considered_provider_ids=considered,
        excluded_providers=exclusions,
    )


def evaluate_reconstruction_advisor(
    request: ReconstructionAdvisorRequest,
) -> ReconstructionAdvisorDecision:
    """Evaluate one passive I6.3 advisory request deterministically.

    The function has no side effects.  Its output is independent of GPU/RAM values
    except that those observations remain available to higher presentation layers as
    descriptive context.  Hardware resources never decide or break a provider tie.
    """

    if not isinstance(request, ReconstructionAdvisorRequest):
        raise TypeError("request must be ReconstructionAdvisorRequest.")

    available = set(request.available_provider_ids)
    all_considered = _unique_sorted(request.available_provider_ids)

    if not request.source_ready:
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("advisor_abstain", "source_not_ready"),
            explanation="The source is not ready, so the Advisor cannot issue evidence-bound provider guidance.",
            considered=all_considered,
            limitations=("source_must_be_ready_before_advisory_guidance",),
        )

    if not request.authoritative_sfm_usable:
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("advisor_abstain", "authoritative_sfm_not_usable"),
            explanation="Authoritative PyCOLMAP SfM is not usable; dense-provider guidance is therefore withheld.",
            considered=all_considered,
            limitations=("pycolmap_sfm_must_remain_authoritative_and_usable",),
        )

    if not request.runtime_integrity_clean:
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("advisor_abstain", "runtime_integrity_not_clean"),
            explanation="Runtime integrity is not clean, so the Advisor fails closed instead of recommending a provider.",
            considered=all_considered,
            limitations=("runtime_integrity_must_be_clean",),
        )

    if not request.available_provider_ids:
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=("advisor_abstain", "no_concrete_provider_available"),
            explanation="No concrete provider is currently available for an advisory recommendation.",
            limitations=("advisor_never_treats_auto_as_an_executable_provider",),
        )

    if request.evidence_context_resolution is AdvisorContextResolution.UNRESOLVED:
        exclusions = tuple(
            AdvisorProviderExclusion(provider_id, ("evidence_context_unresolved",))
            for provider_id in all_considered
        )
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=(
                "advisor_abstain",
                "evidence_context_unresolved",
                "live_context_resolver_not_qualified",
            ),
            explanation=(
                "The current source is not bound to an explicit frozen evidence context; "
                "the Advisor will not classify it or generalize from another context."
            ),
            considered=all_considered,
            exclusions=exclusions,
            limitations=("live_context_resolution_remains_unqualified",),
        )

    exact = tuple(
        sorted(
            (
                observation
                for observation in request.provider_evidence
                if observation.evidence_context_id == request.evidence_context_id
            ),
            key=lambda observation: (
                observation.provider_id,
                observation.provider_implementation.value,
                observation.outcome.value,
            ),
        )
    )
    considered = _considered_provider_ids(request, exact)
    exact_by_provider = _observation_by_provider(exact)

    if not exact:
        exclusions = tuple(
            AdvisorProviderExclusion(provider_id, ("no_exact_context_evidence",))
            for provider_id in considered
        )
        return _abstain(
            support=AdvisorEvidenceSupport.INSUFFICIENT,
            reason_codes=(
                "advisor_abstain",
                "no_exact_context_evidence",
                "cross_context_generalization_forbidden",
            ),
            explanation=(
                "Frozen evidence exists only outside the requested context or is absent; "
                "cross-context provider generalization is forbidden."
            ),
            considered=considered,
            exclusions=exclusions,
            limitations=("exact_frozen_context_evidence_required",),
        )

    pass_observations = tuple(
        observation for observation in exact if observation.outcome is AdvisorEvidenceOutcome.PASS
    )

    if len(pass_observations) > 1:
        exclusions = _exclusions(
            considered=considered,
            available=available,
            exact_by_provider=exact_by_provider,
            ambiguity=True,
        )
        return _abstain(
            support=AdvisorEvidenceSupport.CONFLICTING,
            reason_codes=(
                "advisor_abstain",
                "multiple_exact_pass_candidates_no_tiebreak",
                "availability_and_hardware_tiebreak_forbidden",
            ),
            explanation=(
                "More than one provider has clean PASS evidence in the exact frozen context; "
                "the Advisor abstains rather than ranking or tie-breaking them."
            ),
            considered=considered,
            exclusions=exclusions,
            observations=exact,
            limitations=(
                "global_or_cross_family_provider_ranking_forbidden",
                "hardware_resources_are_descriptive_not_a_tiebreak",
            ),
        )

    if not pass_observations:
        exclusions = _exclusions(
            considered=considered,
            available=available,
            exact_by_provider=exact_by_provider,
        )
        return _abstain(
            support=AdvisorEvidenceSupport.BOUNDED,
            reason_codes=("advisor_abstain", "no_exact_clean_pass_provider"),
            explanation=(
                "The exact frozen context contains evidence, but no provider has a clean PASS outcome; "
                "degraded, rejected, and unqualified evidence remains visible."
            ),
            considered=considered,
            exclusions=exclusions,
            observations=exact,
            limitations=("negative_or_degraded_evidence_cannot_be_promoted",),
        )

    selected = pass_observations[0]
    if selected.provider_id not in available:
        exclusions = _exclusions(
            considered=considered,
            available=available,
            exact_by_provider=exact_by_provider,
        )
        return _abstain(
            support=AdvisorEvidenceSupport.BOUNDED,
            reason_codes=(
                "advisor_abstain",
                "unique_exact_pass_provider_unavailable",
                "fallback_or_retry_until_pass_forbidden",
            ),
            explanation=(
                "The exact frozen context uniquely supports one provider, but that provider is unavailable; "
                "the Advisor will not substitute another provider."
            ),
            considered=considered,
            exclusions=exclusions,
            observations=exact,
            limitations=("provider_availability_blocks_but_does_not_rewrite_evidence",),
        )

    exclusions = _exclusions(
        considered=considered,
        available=available,
        exact_by_provider=exact_by_provider,
        selected_provider_id=selected.provider_id,
    )
    return ReconstructionAdvisorDecision(
        disposition=AdvisorDisposition.RECOMMEND,
        evidence_support=AdvisorEvidenceSupport.EXACT_QUALIFIED,
        reason_codes=(
            "advisor_exact_unique_evidence_recommendation",
            "advisory_only_no_production_effect",
        ),
        explanations=(
            "One concrete provider has the only clean PASS observation in the exact frozen evidence context; "
            "the recommendation is advisory only.",
        ),
        limitations=_limitations(
            (selected,),
            "recommendation_is_context_specific_not_a_global_provider_winner",
            "advisor_does_not_change_production_execution",
        ),
        recommended_provider_id=selected.provider_id,
        recommended_family=selected.provider_family,
        recommended_implementation=selected.provider_implementation,
        matched_scenario_ids=_scenarios((selected,)),
        evidence_bindings=_bindings((selected,)),
        considered_provider_ids=considered,
        excluded_providers=exclusions,
    )


__all__ = [
    "I63B_SCHEMA_VERSION",
    "I63B_PHASE",
    "I63B_ENGINE_PROFILE",
    "evaluate_reconstruction_advisor",
]
