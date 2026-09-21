"""I5.13B deterministic evidence-bound Auto policy engine (offline/shadow only).

The engine implements the conservative I5.13A design without integrating Auto into
production execution.  It consumes explicit *frozen evidence context tokens*; it
never classifies a live scene, executes a provider, changes the SfM matcher, ranks
providers globally, or changes a production default.

Automatic selection is permitted only when all hard gates pass and the exact frozen
evidence context contains exactly one clean PASS provider.  Otherwise the engine
returns a valid abstention decision with explicit reason codes.  Engineer override
remains separate, explicit, and non-automatic.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Mapping

from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.errors import ReconstructionValidationError

I513B_SCHEMA_VERSION = 1
I513B_PHASE = "I5.13B"
I513B_RECEIPT_PHASE = "I5.13B_ENGINE_RECEIPT"
I513B_ENGINE_PROFILE = "evidence_bound_conservative_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

I513B_PROVIDER_ID_BY_IMPLEMENTATION: dict[DenseProviderImplementation, str] = {
    DenseProviderImplementation.COLMAP_MVS: "colmap-mvs-external",
    DenseProviderImplementation.DA3: "depth-anything-3-external",
    DenseProviderImplementation.ROLLING_DEPTH: "rolling-depth-temporal",
    DenseProviderImplementation.VIDEO_DEPTH_ANYTHING: "video-depth-anything-temporal",
    DenseProviderImplementation.METRIC3D_V2: "metric3d-v2-single-image",
    DenseProviderImplementation.UNIDEPTH_V2: "unidepth-v2-single-image",
    DenseProviderImplementation.MOGE_2: "moge-2-single-image",
    DenseProviderImplementation.DEPTH_PRO: "depth-pro-single-image",
    DenseProviderImplementation.DEPTH_ANYTHING_V2: "depth-anything-v2-single-image",
    DenseProviderImplementation.MARIGOLD: "marigold-single-image",
}
I513B_IMPLEMENTATION_BY_PROVIDER_ID = {value: key for key, value in I513B_PROVIDER_ID_BY_IMPLEMENTATION.items()}
I513B_FAMILY_BY_IMPLEMENTATION = {
    DenseProviderImplementation.COLMAP_MVS: DenseProviderFamily.MVS,
    DenseProviderImplementation.DA3: DenseProviderFamily.MULTI_VIEW_AI_DEPTH,
    DenseProviderImplementation.ROLLING_DEPTH: DenseProviderFamily.TEMPORAL_DEPTH,
    DenseProviderImplementation.VIDEO_DEPTH_ANYTHING: DenseProviderFamily.TEMPORAL_DEPTH,
    DenseProviderImplementation.METRIC3D_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    DenseProviderImplementation.UNIDEPTH_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    DenseProviderImplementation.MOGE_2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    DenseProviderImplementation.DEPTH_PRO: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    DenseProviderImplementation.DEPTH_ANYTHING_V2: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
    DenseProviderImplementation.MARIGOLD: DenseProviderFamily.SINGLE_IMAGE_DEPTH,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I5.13B {label} is missing or malformed.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I5.13B {label} contract drift: {key}={mapping.get(key)!r}, expected={expected!r}."
        )


def _sha(value: Any, *, label: str) -> str:
    digest = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(digest):
        raise ReconstructionValidationError(f"I5.13B {label} is not a SHA-256 digest.")
    return digest


def _provider_id(implementation: DenseProviderImplementation) -> str:
    return I513B_PROVIDER_ID_BY_IMPLEMENTATION[implementation]


def _family_provider_ids(family: DenseProviderFamily) -> tuple[str, ...]:
    return tuple(
        _provider_id(implementation)
        for implementation in DenseProviderImplementation
        if I513B_FAMILY_BY_IMPLEMENTATION[implementation] is family
    )


@dataclass(frozen=True, slots=True)
class I513BOfflineAutoContext:
    """Explicit evidence token consumed only by the offline/shadow engine."""

    sparse_context_id: str
    evidence_context_id: str
    provider_family: DenseProviderFamily
    authoritative_sfm_production_usable: bool
    runtime_integrity_clean: bool
    available_provider_ids: tuple[str, ...]
    schema_version: int = I513B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        sparse = str(self.sparse_context_id).strip()
        evidence = str(self.evidence_context_id).strip()
        if not sparse or not evidence:
            raise ReconstructionValidationError("I5.13B offline context IDs must be non-empty.")
        if not isinstance(self.provider_family, DenseProviderFamily):
            raise TypeError("provider_family must be DenseProviderFamily.")
        if not isinstance(self.authoritative_sfm_production_usable, bool):
            raise TypeError("authoritative_sfm_production_usable must be bool.")
        if not isinstance(self.runtime_integrity_clean, bool):
            raise TypeError("runtime_integrity_clean must be bool.")
        providers = tuple(str(value).strip() for value in self.available_provider_ids)
        if any(not value for value in providers) or len(set(providers)) != len(providers):
            raise ReconstructionValidationError("I5.13B available_provider_ids must be unique and non-empty values.")
        if any(value == "auto" for value in providers):
            raise ReconstructionValidationError("I5.13B Auto cannot be an available executable provider.")
        if int(self.schema_version) != I513B_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I5.13B offline-context schema version.")
        object.__setattr__(self, "sparse_context_id", sparse)
        object.__setattr__(self, "evidence_context_id", evidence)
        object.__setattr__(self, "available_provider_ids", providers)
        object.__setattr__(self, "schema_version", I513B_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sparse_context_id": self.sparse_context_id,
            "evidence_context_id": self.evidence_context_id,
            "provider_family": self.provider_family.value,
            "authoritative_sfm_production_usable": self.authoritative_sfm_production_usable,
            "runtime_integrity_clean": self.runtime_integrity_clean,
            "available_provider_ids": list(self.available_provider_ids),
        }


@dataclass(frozen=True, slots=True)
class I513BCompiledPolicy:
    profile: str
    sparse_context_status: Mapping[str, str]
    dense_contexts: Mapping[str, Mapping[str, Any]]
    evidence_sha256_bindings: Mapping[str, str]
    production_auto_authorized: bool = False
    provider_execution_authorized: bool = False
    schema_version: int = I513B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.profile != I513B_ENGINE_PROFILE:
            raise ReconstructionValidationError("I5.13B policy profile drifted.")
        if self.production_auto_authorized or self.provider_execution_authorized:
            raise ReconstructionValidationError("I5.13B engine must remain offline/shadow only.")
        if int(self.schema_version) != I513B_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I5.13B policy schema version.")


@dataclass(frozen=True, slots=True)
class I513BShadowDecision:
    decision_class: str
    decision: DenseProviderSelectionDecision
    policy_profile: str
    sparse_context_id: str
    evidence_context_id: str
    provider_family: DenseProviderFamily
    evidence_provider_status: Mapping[str, str]
    evidence_eligible_provider_ids: tuple[str, ...]
    excluded_provider_reasons: Mapping[str, tuple[str, ...]]
    matched_evidence_scenario_ids: tuple[str, ...]
    evidence_sha256_bindings: Mapping[str, str]
    production_effect: bool = False
    provider_execution: bool = False
    sfm_execution: bool = False
    mvs_execution: bool = False
    schema_version: int = I513B_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.decision_class not in {"select", "abstain", "engineer_override"}:
            raise ReconstructionValidationError("I5.13B decision_class is invalid.")
        if self.policy_profile != I513B_ENGINE_PROFILE:
            raise ReconstructionValidationError("I5.13B result policy profile drifted.")
        if not isinstance(self.decision, DenseProviderSelectionDecision):
            raise TypeError("decision must be DenseProviderSelectionDecision.")
        if self.production_effect or self.provider_execution or self.sfm_execution or self.mvs_execution:
            raise ReconstructionValidationError("I5.13B shadow decision cannot have production/execution effects.")
        if not self.matched_evidence_scenario_ids:
            raise ReconstructionValidationError("I5.13B decisions require matched evidence scenario/context IDs.")
        if not self.evidence_sha256_bindings:
            raise ReconstructionValidationError("I5.13B decisions require evidence SHA-256 provenance.")
        for sha in self.evidence_sha256_bindings.values():
            _sha(sha, label="decision evidence binding")
        if int(self.schema_version) != I513B_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported I5.13B result schema version.")

    @property
    def considered_provider_ids(self) -> tuple[str, ...]:
        return self.decision.considered_provider_ids

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return self.decision.reason_codes

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decision_class": self.decision_class,
            "policy_profile": self.policy_profile,
            "sparse_context_id": self.sparse_context_id,
            "evidence_context_id": self.evidence_context_id,
            "provider_family": self.provider_family.value,
            "decision": self.decision.as_dict(),
            "considered_provider_ids": list(self.decision.considered_provider_ids),
            "reason_codes": list(self.decision.reason_codes),
            "evidence_provider_status": dict(self.evidence_provider_status),
            "evidence_eligible_provider_ids": list(self.evidence_eligible_provider_ids),
            "excluded_provider_reasons": {key: list(value) for key, value in self.excluded_provider_reasons.items()},
            "matched_evidence_scenario_ids": list(self.matched_evidence_scenario_ids),
            "evidence_sha256_bindings": dict(self.evidence_sha256_bindings),
            "production_effect": self.production_effect,
            "provider_execution": self.provider_execution,
            "sfm_execution": self.sfm_execution,
            "mvs_execution": self.mvs_execution,
        }


def validate_i5_13a_for_i5_13b(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="I5.13A admission")
    _require(root, "phase", "I5.13A", label="I5.13A admission")
    _require(root, "status", "auto_policy_evidence_synthesis_and_design_admitted", label="I5.13A admission")
    _require(root, "pass", True, label="I5.13A admission")
    _require(root, "quality_status", "auto_policy_design_admitted", label="I5.13A admission")
    _require(root, "policy_profile", I513B_ENGINE_PROFILE, label="I5.13A admission")

    architecture = _mapping(root.get("architecture_contract"), label="I5.13A architecture contract")
    _require(architecture, "auto_is_selection_policy_only", True, label="I5.13A architecture contract")
    _require(architecture, "auto_is_executable_provider", False, label="I5.13A architecture contract")
    _require(architecture, "camera_pose_authority_owner", I43A_AUTHORITATIVE_FRAME_OWNER, label="I5.13A architecture contract")

    design = _mapping(root.get("auto_policy_v1_design"), label="I5.13A policy design")
    _require(design, "profile", I513B_ENGINE_PROFILE, label="I5.13A policy design")
    _require(design, "primary_principle", "select_only_on_exact_unique_evidence_support_else_abstain", label="I5.13A policy design")
    boundary = _mapping(design.get("production_context_boundary"), label="I5.13A production boundary")
    _require(boundary, "i5_13b_engine_is_offline_shadow_only", True, label="I5.13A production boundary")
    _require(boundary, "production_auto_integration_deferred_to_i5_13d", True, label="I5.13A production boundary")

    replay = _mapping(root.get("offline_replay_contract"), label="I5.13A replay contract")
    _require(replay, "policy_output_deterministic", True, label="I5.13A replay contract")
    _require(replay, "same_input_same_decision_required", True, label="I5.13A replay contract")
    _require(replay, "provider_execution", False, label="I5.13A replay contract")
    _require(replay, "sfm_execution", False, label="I5.13A replay contract")
    _require(replay, "mvs_execution", False, label="I5.13A replay contract")

    decision = _mapping(root.get("decision"), label="I5.13A decision")
    _require(decision, "i5_13a_complete", True, label="I5.13A decision")
    _require(decision, "i5_13b_authorized", True, label="I5.13A decision")
    _require(decision, "i5_13b_started", False, label="I5.13A decision")
    _require(decision, "i5_13c_authorized", False, label="I5.13A decision")
    _require(decision, "i5_13d_authorized", False, label="I5.13A decision")
    _require(decision, "production_auto_authorized", False, label="I5.13A decision")
    _require(decision, "production_default_modified", False, label="I5.13A decision")
    _require(decision, "provider_selected", False, label="I5.13A decision")
    _require(decision, "provider_promoted", False, label="I5.13A decision")
    return deepcopy(dict(root))


def _compile_dense_contexts(admission: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    synthesis = _mapping(admission.get("evidence_synthesis"), label="I5.13A evidence synthesis")
    providers = _mapping(synthesis.get("provider_operating_envelope"), label="provider operating envelope")

    mvs = _mapping(providers.get("mvs"), label="MVS envelope")
    rolling = _mapping(providers.get("rolling_depth"), label="RollingDepth envelope")
    vda = _mapping(providers.get("video_depth_anything"), label="VDA envelope")
    single = _mapping(providers.get("single_image_depth"), label="single-image envelope")

    contexts: dict[str, dict[str, Any]] = {
        "i5_6_current_mvs_envelope": {
            "family": DenseProviderFamily.MVS,
            "statuses": {_provider_id(DenseProviderImplementation.COLMAP_MVS): str(mvs.get("quality_status", "unknown"))},
            "source_bindings": ("i5_6c_mvs",),
        },
        "i5_current_castle_temporal_envelope": {
            "family": DenseProviderFamily.TEMPORAL_DEPTH,
            "statuses": {
                _provider_id(DenseProviderImplementation.ROLLING_DEPTH): str(rolling.get("current_castle_quality_status", "unknown")),
                _provider_id(DenseProviderImplementation.VIDEO_DEPTH_ANYTHING): str(vda.get("quality_status", "unknown")),
            },
            "source_bindings": ("i5_7c_rolling_depth", "i5_8c_video_depth_anything"),
        },
    }

    terminal_events = providers.get("cross_provider_terminal_quality_events", [])
    if not isinstance(terminal_events, list):
        raise ReconstructionValidationError("I5.13B terminal quality events are malformed.")
    event_status: dict[tuple[str, str], str] = {}
    for raw in terminal_events:
        event = _mapping(raw, label="terminal quality event")
        family = str(event.get("family_id", ""))
        scenario = str(event.get("scenario_id", ""))
        implementation = str(event.get("provider_implementation", ""))
        status = str(event.get("production_status", ""))
        event_status[(scenario, implementation)] = status
        if family == "temporal_depth" and scenario in {"g3_temporal_motion", "g3_temporal_static_sweep"}:
            ctx = contexts.setdefault(
                scenario,
                {
                    "family": DenseProviderFamily.TEMPORAL_DEPTH,
                    "statuses": {
                        _provider_id(DenseProviderImplementation.ROLLING_DEPTH): "unknown",
                        _provider_id(DenseProviderImplementation.VIDEO_DEPTH_ANYTHING): "unknown",
                    },
                    "source_bindings": ("i4_3g5b_cross_provider_closure",),
                },
            )
            try:
                impl = DenseProviderImplementation(implementation)
            except ValueError as exc:
                raise ReconstructionValidationError("I5.13B terminal event provider implementation is unknown.") from exc
            if I513B_FAMILY_BY_IMPLEMENTATION[impl] is DenseProviderFamily.TEMPORAL_DEPTH:
                ctx["statuses"][_provider_id(impl)] = status

    historical = _mapping(single.get("historical_g4_anchors"), label="historical G4 anchors")
    scenario_key_map = {
        "complete_room": "g4_single_image_complete_room",
        "low_texture": "g4_single_image_low_texture",
        "weak_overlap": "g4_single_image_weak_overlap",
    }
    for short_name, scenario_id in scenario_key_map.items():
        raw_statuses = _mapping(historical.get(short_name), label=f"G4 {short_name} statuses")
        statuses: dict[str, str] = {}
        for impl_name, status in raw_statuses.items():
            try:
                impl = DenseProviderImplementation(str(impl_name))
            except ValueError as exc:
                raise ReconstructionValidationError(f"I5.13B unknown single-image implementation {impl_name!r}.") from exc
            if I513B_FAMILY_BY_IMPLEMENTATION[impl] is not DenseProviderFamily.SINGLE_IMAGE_DEPTH:
                raise ReconstructionValidationError("I5.13B G4 provider-family drifted.")
            statuses[_provider_id(impl)] = str(status)
        contexts[scenario_id] = {
            "family": DenseProviderFamily.SINGLE_IMAGE_DEPTH,
            "statuses": statuses,
            "source_bindings": ("i5_9c_single_image_depth", "i4_3g5b_cross_provider_closure"),
        }
    return contexts


def compile_i5_13b_policy(i5_13a_admission: Mapping[str, Any]) -> I513BCompiledPolicy:
    admission = validate_i5_13a_for_i5_13b(i5_13a_admission)
    synthesis = _mapping(admission.get("evidence_synthesis"), label="I5.13A evidence synthesis")
    sparse = _mapping(synthesis.get("sparse_sfm_operating_envelope"), label="sparse SfM envelope")
    sparse_status = {
        str(key): str(_mapping(value, label=f"sparse context {key}").get("quality_status", "unknown"))
        for key, value in sparse.items()
    }
    expected_sparse = {"difficult_handheld", "low_overlap_weak_parallax", "visual_stress", "forward_motion", "dynamic_scene_proxy"}
    if set(sparse_status) != expected_sparse:
        raise ReconstructionValidationError("I5.13B sparse context coverage drifted.")

    bindings = _mapping(admission.get("evidence_bindings"), label="I5.13A evidence bindings")
    sha_bindings: dict[str, str] = {}
    for name, raw in bindings.items():
        item = _mapping(raw, label=f"I5.13A binding {name}")
        sha_bindings[str(name)] = _sha(item.get("sha256"), label=f"I5.13A binding {name}")

    return I513BCompiledPolicy(
        profile=I513B_ENGINE_PROFILE,
        sparse_context_status=sparse_status,
        dense_contexts=_compile_dense_contexts(admission),
        evidence_sha256_bindings=sha_bindings,
    )


def _decision(
    *,
    selected_provider_id: str,
    considered: tuple[str, ...],
    request: DenseProviderSelectionRequest,
    automatic: bool,
    engineer_override: bool,
    fallback_used: bool,
    reason_codes: tuple[str, ...],
) -> DenseProviderSelectionDecision:
    implementation = I513B_IMPLEMENTATION_BY_PROVIDER_ID.get(selected_provider_id) if selected_provider_id else None
    return DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.AUTO,
        requested_provider_id="",
        selected_provider_id=selected_provider_id,
        selected_implementation=implementation,
        considered_provider_ids=considered,
        automatic=automatic,
        engineer_override_applied=engineer_override,
        fallback_used=fallback_used,
        reason_codes=reason_codes,
    )


def _result(
    *,
    policy: I513BCompiledPolicy,
    request: DenseProviderSelectionRequest,
    context: I513BOfflineAutoContext,
    decision_class: str,
    selected_provider_id: str = "",
    considered: tuple[str, ...] = (),
    reason_codes: tuple[str, ...],
    evidence_status: Mapping[str, str] | None = None,
    eligible: tuple[str, ...] = (),
    excluded: Mapping[str, tuple[str, ...]] | None = None,
    matched: tuple[str, ...] | None = None,
    engineer_override: bool = False,
) -> I513BShadowDecision:
    decision = _decision(
        selected_provider_id=selected_provider_id,
        considered=considered,
        request=request,
        automatic=decision_class != "engineer_override",
        engineer_override=engineer_override,
        fallback_used=False,
        reason_codes=reason_codes,
    )
    return I513BShadowDecision(
        decision_class=decision_class,
        decision=decision,
        policy_profile=policy.profile,
        sparse_context_id=context.sparse_context_id,
        evidence_context_id=context.evidence_context_id,
        provider_family=context.provider_family,
        evidence_provider_status=dict(evidence_status or {}),
        evidence_eligible_provider_ids=eligible,
        excluded_provider_reasons=dict(excluded or {}),
        matched_evidence_scenario_ids=matched or (context.sparse_context_id, context.evidence_context_id),
        evidence_sha256_bindings=dict(policy.evidence_sha256_bindings),
    )


def evaluate_i5_13b_auto_policy(
    policy: I513BCompiledPolicy,
    request: DenseProviderSelectionRequest,
    context: I513BOfflineAutoContext,
) -> I513BShadowDecision:
    if not isinstance(policy, I513BCompiledPolicy):
        raise TypeError("policy must be I513BCompiledPolicy.")
    if not isinstance(request, DenseProviderSelectionRequest):
        raise TypeError("request must be DenseProviderSelectionRequest.")
    if not isinstance(context, I513BOfflineAutoContext):
        raise TypeError("context must be I513BOfflineAutoContext.")
    if request.mode is not DenseProviderSelectionMode.AUTO:
        raise ReconstructionValidationError("I5.13B engine accepts Auto selection requests only.")

    available = set(context.available_provider_ids)
    family_ids = _family_provider_ids(context.provider_family) if context.provider_family is not DenseProviderFamily.HYBRID else ()
    considered = tuple(family_ids)

    # Non-negotiable execution/authority gates apply even to engineer override.
    if not request.authoritative_sfm_available:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "authoritative_sfm_unavailable"))
    if not context.authoritative_sfm_production_usable:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "authoritative_sfm_not_production_usable"))
    if not context.runtime_integrity_clean:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "runtime_integrity_not_clean"))
    if context.provider_family is DenseProviderFamily.HYBRID:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=(),
                       reason_codes=("auto_abstain", "hybrid_family_forbidden"))

    override = request.engineer_override_provider_id
    if override:
        implementation = I513B_IMPLEMENTATION_BY_PROVIDER_ID.get(override)
        if implementation is None:
            return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                           reason_codes=("auto_abstain", "engineer_override_provider_unknown"))
        if I513B_FAMILY_BY_IMPLEMENTATION[implementation] is not context.provider_family:
            return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                           reason_codes=("auto_abstain", "engineer_override_family_mismatch"))
        if override not in available:
            return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                           reason_codes=("auto_abstain", "engineer_override_provider_unavailable"))
        return _result(policy=policy, request=request, context=context, decision_class="engineer_override",
                       selected_provider_id=override, considered=considered,
                       reason_codes=("engineer_override_explicit", "override_does_not_rewrite_evidence"),
                       engineer_override=True)

    sparse_status = policy.sparse_context_status.get(context.sparse_context_id)
    if sparse_status is None:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "sparse_context_unrecognized"))
    if sparse_status == "rejected":
        code = "sparse_context_rejected_hard_abstention" if context.sparse_context_id == "forward_motion" else "sparse_context_rejected_auto_abstention"
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", code))
    if sparse_status == "degraded":
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "sparse_context_degraded_auto_abstention"))
    if sparse_status != "pass":
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "sparse_context_not_clean_pass"))

    dense = policy.dense_contexts.get(context.evidence_context_id)
    if dense is None:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "evidence_context_unrecognized"))
    expected_family = dense.get("family")
    if expected_family is not context.provider_family:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "evidence_family_mismatch"))

    statuses = dict(_mapping(dense.get("statuses"), label="compiled dense context statuses"))
    considered = tuple(statuses)
    excluded: dict[str, tuple[str, ...]] = {}
    pass_ids: list[str] = []
    for provider_id, status in statuses.items():
        reasons: list[str] = []
        if status == "pass":
            pass_ids.append(provider_id)
        elif status == "degraded":
            reasons.append("evidence_status_degraded")
        elif status == "rejected":
            reasons.append("evidence_status_rejected")
        else:
            reasons.append("evidence_status_unknown")
        if provider_id not in available:
            reasons.append("provider_unavailable")
        if reasons:
            excluded[provider_id] = tuple(reasons)

    eligible = tuple(pass_ids)
    source_bindings = tuple(str(x) for x in dense.get("source_bindings", ()))
    matched = (context.sparse_context_id, context.evidence_context_id, *source_bindings)

    if not pass_ids:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "no_clean_evidence_eligible_provider"),
                       evidence_status=statuses, eligible=eligible, excluded=excluded, matched=matched)
    if len(pass_ids) > 1:
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "multiple_clean_evidence_eligible_providers", "evidence_not_unique_even_after_availability"),
                       evidence_status=statuses, eligible=eligible, excluded=excluded, matched=matched)

    selected = pass_ids[0]
    if selected not in available:
        excluded[selected] = tuple(dict.fromkeys((*excluded.get(selected, ()), "provider_unavailable")))
        return _result(policy=policy, request=request, context=context, decision_class="abstain", considered=considered,
                       reason_codes=("auto_abstain", "unique_evidence_provider_unavailable"),
                       evidence_status=statuses, eligible=eligible, excluded=excluded, matched=matched)

    return _result(
        policy=policy,
        request=request,
        context=context,
        decision_class="select",
        selected_provider_id=selected,
        considered=considered,
        reason_codes=("auto_exact_unique_evidence_select", "offline_shadow_only"),
        evidence_status=statuses,
        eligible=eligible,
        excluded=excluded,
        matched=matched,
    )


__all__ = [
    "I513B_SCHEMA_VERSION",
    "I513B_PHASE",
    "I513B_RECEIPT_PHASE",
    "I513B_ENGINE_PROFILE",
    "I513B_PROVIDER_ID_BY_IMPLEMENTATION",
    "I513BOfflineAutoContext",
    "I513BCompiledPolicy",
    "I513BShadowDecision",
    "validate_i5_13a_for_i5_13b",
    "compile_i5_13b_policy",
    "evaluate_i5_13b_auto_policy",
]


def _binding(value: Mapping[str, Any], *, label: str) -> dict[str, str]:
    root = _mapping(value, label=f"{label} binding")
    path = str(root.get("path", "")).strip()
    sha = _sha(root.get("sha256"), label=f"{label} binding")
    if not path:
        raise ReconstructionValidationError(f"I5.13B {label} binding path is empty.")
    return {"path": path, "sha256": sha}


def validate_i5_13b_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, label="engine receipt")
    _require(root, "phase", I513B_RECEIPT_PHASE, label="engine receipt")
    _require(root, "status", "pass", label="engine receipt")
    _require(root, "pass", True, label="engine receipt")
    scope = _mapping(root.get("scope"), label="engine receipt scope")
    for key in (
        "focused_i5_13b_tests",
        "i5_13a_regression",
        "dense_selection_contract_regression",
        "determinism_stress",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="engine receipt scope")
    for key in (
        "provider_execution",
        "sfm_execution",
        "mvs_execution",
        "production_auto_integration",
        "production_default_modified",
        "provider_promoted",
        "scene_classifier_implemented",
    ):
        _require(scope, key, False, label="engine receipt scope")
    counts = _mapping(root.get("counts"), label="engine receipt counts")
    focused = counts.get("focused_tests")
    predecessor = counts.get("predecessor_total_tests")
    combined = counts.get("combined_exercised_tests")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (focused, predecessor, combined)):
        raise ReconstructionValidationError("I5.13B receipt test counts are invalid.")
    if focused < 25 or predecessor < 20 or combined != focused + predecessor:
        raise ReconstructionValidationError("I5.13B receipt coverage is below the engine freeze floor.")
    rcs = _mapping(root.get("return_codes"), label="engine receipt return codes")
    for key in ("focused", "predecessor_unit", "predecessor_integration", "predecessor_stress", "import_probe", "runner_help_probe"):
        if rcs.get(key) != 0:
            raise ReconstructionValidationError(f"I5.13B receipt {key} return code is not zero.")
    return deepcopy(dict(root))


def _shadow_request(*, override: str = "") -> DenseProviderSelectionRequest:
    return DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.AUTO,
        available_view_count=19,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id="i5_13b_offline_shadow_sfm",
        allow_fallback=False,
        engineer_override_provider_id=override,
    )


def _shadow_context(
    sparse: str,
    evidence: str,
    family: DenseProviderFamily,
    providers: tuple[str, ...],
) -> I513BOfflineAutoContext:
    return I513BOfflineAutoContext(
        sparse_context_id=sparse,
        evidence_context_id=evidence,
        provider_family=family,
        authoritative_sfm_production_usable=True,
        runtime_integrity_clean=True,
        available_provider_ids=providers,
    )


def _deterministic_self_probes(policy: I513BCompiledPolicy) -> dict[str, dict[str, Any]]:
    all_single = _family_provider_ids(DenseProviderFamily.SINGLE_IMAGE_DEPTH)
    all_temporal = _family_provider_ids(DenseProviderFamily.TEMPORAL_DEPTH)
    probes = {
        "unique_mvs_select": (
            _shadow_request(),
            _shadow_context("low_overlap_weak_parallax", "i5_6_current_mvs_envelope", DenseProviderFamily.MVS, _family_provider_ids(DenseProviderFamily.MVS)),
        ),
        "unique_temporal_select": (
            _shadow_request(),
            _shadow_context("dynamic_scene_proxy", "i5_current_castle_temporal_envelope", DenseProviderFamily.TEMPORAL_DEPTH, all_temporal),
        ),
        "single_image_low_texture_ambiguity_abstain": (
            _shadow_request(),
            _shadow_context("low_overlap_weak_parallax", "g4_single_image_low_texture", DenseProviderFamily.SINGLE_IMAGE_DEPTH, all_single),
        ),
        "forward_motion_hard_abstain": (
            _shadow_request(),
            _shadow_context("forward_motion", "i5_6_current_mvs_envelope", DenseProviderFamily.MVS, _family_provider_ids(DenseProviderFamily.MVS)),
        ),
        "unknown_context_abstain": (
            _shadow_request(),
            _shadow_context("low_overlap_weak_parallax", "unknown_future_context", DenseProviderFamily.MVS, _family_provider_ids(DenseProviderFamily.MVS)),
        ),
        "explicit_engineer_override": (
            _shadow_request(override="colmap-mvs-external"),
            _shadow_context("difficult_handheld", "i5_6_current_mvs_envelope", DenseProviderFamily.MVS, _family_provider_ids(DenseProviderFamily.MVS)),
        ),
    }
    expected = {
        "unique_mvs_select": ("select", "colmap-mvs-external"),
        "unique_temporal_select": ("select", "rolling-depth-temporal"),
        "single_image_low_texture_ambiguity_abstain": ("abstain", ""),
        "forward_motion_hard_abstain": ("abstain", ""),
        "unknown_context_abstain": ("abstain", ""),
        "explicit_engineer_override": ("engineer_override", "colmap-mvs-external"),
    }
    results: dict[str, dict[str, Any]] = {}
    for name, (request, context) in probes.items():
        first = evaluate_i5_13b_auto_policy(policy, request, context).as_dict()
        second = evaluate_i5_13b_auto_policy(policy, request, context).as_dict()
        if first != second:
            raise ReconstructionValidationError(f"I5.13B deterministic self-probe {name} changed across identical inputs.")
        expected_class, expected_provider = expected[name]
        if first["decision_class"] != expected_class or first["decision"]["selected_provider_id"] != expected_provider:
            raise ReconstructionValidationError(f"I5.13B deterministic self-probe {name} outcome drifted.")
        results[name] = first
    return results


def build_i5_13b_engine_freeze(
    *,
    i5_13a_admission: Mapping[str, Any],
    i5_13a_binding: Mapping[str, Any],
    post_i5_13a_checkpoint_binding: Mapping[str, Any],
    engine_module_binding: Mapping[str, Any],
    engine_receipt: Mapping[str, Any],
    engine_receipt_binding: Mapping[str, Any],
) -> dict[str, Any]:
    policy = compile_i5_13b_policy(i5_13a_admission)
    receipt = validate_i5_13b_receipt(engine_receipt)
    probes = _deterministic_self_probes(policy)
    class_counts = {"select": 0, "abstain": 0, "engineer_override": 0}
    for result in probes.values():
        class_counts[result["decision_class"]] += 1

    contexts = {
        key: {
            "family": value["family"].value,
            "provider_status": dict(value["statuses"]),
            "source_bindings": list(value["source_bindings"]),
        }
        for key, value in policy.dense_contexts.items()
    }
    return {
        "schema_version": I513B_SCHEMA_VERSION,
        "phase": I513B_PHASE,
        "status": "deterministic_auto_policy_engine_frozen_offline_shadow",
        "pass": True,
        "quality_status": "offline_shadow_engine_pass",
        "policy_profile": policy.profile,
        "evidence_bindings": {
            "i5_13a_admission": _binding(i5_13a_binding, label="I5.13A admission"),
            "post_i5_13a_source_checkpoint": _binding(post_i5_13a_checkpoint_binding, label="post-I5.13A source checkpoint"),
            "i5_13b_engine_module": _binding(engine_module_binding, label="I5.13B engine module"),
            "i5_13b_engine_receipt": _binding(engine_receipt_binding, label="I5.13B engine receipt"),
        },
        "engine_contract": {
            "offline_shadow_only": True,
            "selection_contract": "DenseProviderSelectionDecision",
            "auto_is_executable_provider": False,
            "explicit_frozen_context_tokens_required": True,
            "production_scene_classifier_implemented": False,
            "same_input_same_decision_required": True,
            "abstention_is_valid_terminal_decision": True,
            "engineer_override_is_separate_nonautomatic_decision": True,
            "cross_family_rank": False,
            "global_scalar_score": False,
            "runtime_or_gpu_tiebreak": False,
            "fallback_retry_until_pass": False,
            "hybrid_selection": False,
            "matching_policy_change": False,
            "camera_pose_authority_owner": I43A_AUTHORITATIVE_FRAME_OWNER,
        },
        "compiled_policy": {
            "sparse_context_status": dict(policy.sparse_context_status),
            "dense_evidence_contexts": contexts,
            "canonical_provider_ids": {implementation.value: provider_id for implementation, provider_id in I513B_PROVIDER_ID_BY_IMPLEMENTATION.items()},
            "evidence_binding_count": len(policy.evidence_sha256_bindings),
        },
        "deterministic_self_probes": probes,
        "probe_summary": {
            "probe_count": len(probes),
            "decision_class_counts": class_counts,
            "all_same_input_same_decision": True,
            "unique_mvs_selection_verified": True,
            "unique_temporal_selection_verified": True,
            "ambiguous_single_image_abstention_verified": True,
            "forward_motion_hard_abstention_verified": True,
            "unknown_context_abstention_verified": True,
            "engineer_override_nonautomatic_verified": True,
        },
        "i5_13c_replay_preconditions": {
            "frozen_engine_module_hash_required": True,
            "frozen_i5_13a_design_hash_required": True,
            "expected_decisions_declared_before_replay": True,
            "frozen_evidence_only": True,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scenario_rematerialization": False,
            "no_result_driven_rule_tuning": True,
        },
        "scope": {
            "auto_policy_engine_implemented": True,
            "offline_shadow_only": True,
            "auto_provider_selection_performed_in_production": False,
            "production_auto_integration": False,
            "provider_execution": False,
            "sfm_execution": False,
            "mvs_execution": False,
            "scene_classifier_implemented": False,
            "hybrid_execution": False,
            "matching_policy_modified": False,
            "provider_promoted": False,
            "production_default_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
        "claim_boundaries": {
            "production_auto_qualified": False,
            "universal_provider_winner": False,
            "global_provider_ranking": False,
            "global_scalar_score": False,
            "cross_family_rank": False,
            "runtime_performance_rank": False,
            "cross_gpu_equivalence": False,
            "universal_sfm_robustness": False,
            "universal_dense_provider_robustness": False,
        },
        "regression_summary": {
            "focused_i5_13b_tests": receipt["counts"]["focused_tests"],
            "predecessor_tests": receipt["counts"]["predecessor_total_tests"],
            "combined_exercised_tests": receipt["counts"]["combined_exercised_tests"],
            "test_counts_are_not_policy_scores": True,
        },
        "decision": {
            "i5_13_started": True,
            "i5_13a_complete": True,
            "i5_13b_complete": True,
            "i5_13c_authorized": True,
            "i5_13c_started": False,
            "i5_13d_authorized": False,
            "i5_13d_started": False,
            "i5_14_started": False,
            "auto_policy_engine_implemented": True,
            "production_auto_authorized": False,
            "production_default_modified": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "global_scalar_score_created": False,
            "hybrid_authorized": False,
            "matching_policy_change_authorized": False,
        },
        "next": "I5.13C frozen-evidence offline Auto policy replay (no provider/SfM/MVS execution).",
    }


__all__.extend([
    "validate_i5_13b_receipt",
    "build_i5_13b_engine_freeze",
])
