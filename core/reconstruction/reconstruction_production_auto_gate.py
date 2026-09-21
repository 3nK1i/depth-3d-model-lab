"""I6.4C production Auto gate and qualification boundary.

This module deliberately keeps the evidence-qualified I5 offline/shadow Auto
policy separate from production authorization.  It consumes the descriptive
I6.4A/I6.4B provider-selection snapshot and exposes one immutable product gate
for later UI presentation.

The bounded conclusion is unchanged from I5.13D/I5.14B:

* offline/shadow Auto policy: qualified;
* production Auto: not qualified, not authorized, not selectable;
* live-context -> frozen-evidence resolution: not qualified;
* production Auto orchestration: not implemented/qualified;
* no provider selection/activation/execution/settings/start effect is granted.

This module does not import or execute the I5 Auto engine.  It does not add an
Auto orchestration entry point, alter production defaults, create provider
ranks/scores, or change PyCOLMAP camera/global-frame authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.reconstruction.i5_auto_policy_production_qualification import (
    I513D_POLICY_PROFILE,
)
from core.reconstruction.i5_final_closure import I514B_PHASE
from core.reconstruction.reconstruction_provider_selection_contract import (
    I64A_CAMERA_AUTHORITY_OWNER,
    ProductionAutoGate,
    ProductionProviderSelectionMode,
    ProductionProviderSelectionSnapshot,
)

I64C_SCHEMA_VERSION = 1
I64C_PHASE = "I6.4C"
I64C_POLICY_PROFILE = I513D_POLICY_PROFILE
I64C_I5_AUTHORITY_PHASE = I514B_PHASE
I64C_DECISION = "defer_production_auto"
I64C_DECISION_REASON = (
    "offline_policy_is_qualified_but_live_context_resolution_and_"
    "production_orchestration_are_not_qualified"
)

I64C_REQUIRED_BLOCKING_GAPS: tuple[str, ...] = (
    "live_context_to_evidence_token_mapping_not_qualified",
    "production_scene_or_context_resolver_not_implemented",
    "adaptive_production_planning_explicit_provider_only",
    "production_ui_auto_activation_not_implemented",
    "production_auto_engine_not_wired_into_app",
    "production_auto_orchestration_entrypoint_missing",
)

I64C_REQUIRED_FUTURE_PREREQUISITES: tuple[str, ...] = (
    "qualified_live_context_resolver",
    "context_resolver_must_abstain_outside_frozen_evidence_envelope",
    "production_orchestration_binding_with_full_decision_provenance",
    "production_shadow_validation_before_live_activation",
    "no_result_driven_rule_tuning",
    "no_global_score_or_cross_family_rank",
    "pycolmap_camera_pose_authority_preserved",
)


class ProductionAutoBoundaryStatus(str, Enum):
    """Current product qualification state for Auto."""

    DEFERRED_NOT_QUALIFIED = "deferred_not_qualified"


@dataclass(frozen=True, slots=True)
class ProductionAutoQualificationBoundary:
    """Read-only production Auto qualification state with zero execution effect."""

    gate: ProductionAutoGate
    current_selection_mode: ProductionProviderSelectionMode
    manual_selected_provider_id: str = ""
    status: ProductionAutoBoundaryStatus = ProductionAutoBoundaryStatus.DEFERRED_NOT_QUALIFIED
    policy_profile: str = I64C_POLICY_PROFILE
    qualification_authority_phase: str = I64C_I5_AUTHORITY_PHASE
    offline_shadow_qualified: bool = True
    offline_replay_qualified: bool = True
    production_auto_qualified: bool = False
    production_auto_authorized: bool = False
    selectable: bool = False
    decision: str = I64C_DECISION
    decision_reason: str = I64C_DECISION_REASON
    blocking_gaps: tuple[str, ...] = I64C_REQUIRED_BLOCKING_GAPS
    required_future_prerequisites: tuple[str, ...] = I64C_REQUIRED_FUTURE_PREREQUISITES
    provider_selection_effect: bool = False
    provider_activation_authorized: bool = False
    provider_execution_authorized: bool = False
    settings_mutation_authorized: bool = False
    start_action_authorized: bool = False
    production_default_modified: bool = False
    schema_version: int = I64C_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.gate, ProductionAutoGate):
            raise TypeError("gate must be ProductionAutoGate.")
        if not isinstance(self.current_selection_mode, ProductionProviderSelectionMode):
            raise TypeError("current_selection_mode must be ProductionProviderSelectionMode.")
        if self.current_selection_mode is ProductionProviderSelectionMode.AUTO:
            raise ValueError("I6.4C cannot admit production Auto selection mode.")
        selected = str(self.manual_selected_provider_id or "").strip()
        if self.current_selection_mode is ProductionProviderSelectionMode.MANUAL:
            if not selected:
                raise ValueError("Manual selection mode requires manual_selected_provider_id.")
        elif selected:
            raise ValueError("Only manual selection mode may carry manual_selected_provider_id.")
        if self.status is not ProductionAutoBoundaryStatus.DEFERRED_NOT_QUALIFIED:
            raise ValueError("I6.4C production Auto status must remain deferred/not qualified.")
        if str(self.policy_profile).strip() != I64C_POLICY_PROFILE:
            raise ValueError("I6.4C policy profile drifted from the qualified I5 Auto profile.")
        if str(self.qualification_authority_phase).strip() != I64C_I5_AUTHORITY_PHASE:
            raise ValueError("I6.4C qualification authority must remain the final I5 closure.")

        if self.gate.offline_shadow_qualified is not True:
            raise ValueError("I6.4C must preserve qualified offline/shadow Auto evidence.")
        if self.gate.production_qualified or self.gate.production_authorized or self.gate.selectable:
            raise ValueError("I6.4C received a broadened I6.4A production Auto gate.")
        if self.offline_shadow_qualified is not True or self.offline_replay_qualified is not True:
            raise ValueError("I6.4C must preserve the qualified I5 offline/shadow policy and replay.")
        if self.production_auto_qualified or self.production_auto_authorized or self.selectable:
            raise ValueError("I6.4C production Auto must remain not qualified, unauthorized and non-selectable.")
        if str(self.decision).strip() != I64C_DECISION:
            raise ValueError("I6.4C decision must remain defer_production_auto.")
        if str(self.decision_reason).strip() != I64C_DECISION_REASON:
            raise ValueError("I6.4C decision reason drifted.")
        if tuple(self.blocking_gaps) != I64C_REQUIRED_BLOCKING_GAPS:
            raise ValueError("I6.4C blocking gaps must preserve the I5 production-Auto boundary.")
        if tuple(self.required_future_prerequisites) != I64C_REQUIRED_FUTURE_PREREQUISITES:
            raise ValueError("I6.4C future prerequisites must preserve the I5 production-Auto boundary.")
        if any(
            (
                self.provider_selection_effect,
                self.provider_activation_authorized,
                self.provider_execution_authorized,
                self.settings_mutation_authorized,
                self.start_action_authorized,
                self.production_default_modified,
            )
        ):
            raise ValueError("I6.4C production Auto gate cannot grant production effects.")
        if int(self.schema_version) != I64C_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.4C production-Auto schema version.")

        object.__setattr__(self, "manual_selected_provider_id", selected)
        object.__setattr__(self, "policy_profile", I64C_POLICY_PROFILE)
        object.__setattr__(self, "qualification_authority_phase", I64C_I5_AUTHORITY_PHASE)
        object.__setattr__(self, "decision", I64C_DECISION)
        object.__setattr__(self, "decision_reason", I64C_DECISION_REASON)
        object.__setattr__(self, "blocking_gaps", I64C_REQUIRED_BLOCKING_GAPS)
        object.__setattr__(
            self,
            "required_future_prerequisites",
            I64C_REQUIRED_FUTURE_PREREQUISITES,
        )
        object.__setattr__(self, "schema_version", I64C_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I64C_PHASE,
            "status": self.status.value,
            "policy_profile": self.policy_profile,
            "qualification_authority_phase": self.qualification_authority_phase,
            "production_auto": {
                "policy_id": self.gate.policy_id,
                "offline_shadow_qualified": self.offline_shadow_qualified,
                "offline_replay_qualified": self.offline_replay_qualified,
                "production_qualified": self.production_auto_qualified,
                "production_authorized": self.production_auto_authorized,
                "selectable": self.selectable,
                "decision": self.decision,
                "decision_reason": self.decision_reason,
                "reason_codes": list(self.gate.reason_codes),
                "blocking_gaps": list(self.blocking_gaps),
                "required_future_prerequisites": list(self.required_future_prerequisites),
            },
            "current_selection": {
                "mode": self.current_selection_mode.value,
                "manual_selected_provider_id": self.manual_selected_provider_id,
            },
            "effects": {
                "provider_selection_effect": self.provider_selection_effect,
                "provider_activation_authorized": self.provider_activation_authorized,
                "provider_execution_authorized": self.provider_execution_authorized,
                "settings_mutation_authorized": self.settings_mutation_authorized,
                "start_action_authorized": self.start_action_authorized,
                "production_default_modified": self.production_default_modified,
            },
            "authority": {
                "camera_authority_owner": I64A_CAMERA_AUTHORITY_OWNER,
                "dense_provider_establishes_metric_scale": False,
                "certified_metrology": False,
            },
            "claim_boundaries": {
                "offline_shadow_qualification_is_production_authorization": False,
                "manual_selection_is_auto": False,
                "auto_gate_is_execution": False,
                "global_provider_ranking": False,
                "global_scalar_score": False,
                "universal_provider_winner": False,
                "certified_metric_scale": False,
            },
        }


def build_production_auto_qualification_boundary(
    snapshot: ProductionProviderSelectionSnapshot,
) -> ProductionAutoQualificationBoundary:
    """Compose the fail-closed I6.4C gate from the current I6.4 selection snapshot."""

    if not isinstance(snapshot, ProductionProviderSelectionSnapshot):
        raise TypeError("snapshot must be ProductionProviderSelectionSnapshot.")
    if snapshot.selection_mode is ProductionProviderSelectionMode.AUTO:
        raise ValueError("I6.4C cannot compose a boundary from production Auto selection mode.")
    selected = (
        snapshot.selected_provider_id
        if snapshot.selection_mode is ProductionProviderSelectionMode.MANUAL
        else ""
    )
    return ProductionAutoQualificationBoundary(
        gate=snapshot.auto_gate,
        current_selection_mode=snapshot.selection_mode,
        manual_selected_provider_id=selected,
    )


__all__ = [
    "I64C_SCHEMA_VERSION",
    "I64C_PHASE",
    "I64C_POLICY_PROFILE",
    "I64C_I5_AUTHORITY_PHASE",
    "I64C_DECISION",
    "I64C_DECISION_REASON",
    "I64C_REQUIRED_BLOCKING_GAPS",
    "I64C_REQUIRED_FUTURE_PREREQUISITES",
    "ProductionAutoBoundaryStatus",
    "ProductionAutoQualificationBoundary",
    "build_production_auto_qualification_boundary",
]
