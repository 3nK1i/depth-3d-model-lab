"""I6.3A execution-free Reconstruction Advisor contract and evidence model.

The Advisor is a product guidance surface, not production Auto.  This module
freezes immutable request/evidence/decision contracts that later I6.3 steps may
populate and present.  It intentionally does not classify live scenes, run
reconstruction, select or activate a provider, mutate settings, press Start, or
change reconstruction authority.

The contract preserves the I5 evidence boundaries:

* PyCOLMAP SfM remains the sole canonical camera/global-frame authority.
* provider observations remain scenario/context specific and content-addressed;
* PASS, DEGRADED and REJECTED evidence are preserved rather than averaged;
* recommendations require exact qualified evidence support and provenance;
* otherwise the Advisor must abstain rather than invent a tie-break;
* global provider ranking, cross-family ranking and scalar quality/confidence
  scores are outside scope;
* production Auto remains unqualified and unauthorized.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    dense_provider_implementation_contracts,
)

I63A_SCHEMA_VERSION = 1
I63A_PHASE = "I6.3A"
I63A_ADVISOR_SCOPE = "offline_reconstruction_advisory_only"
ADVISOR_CAMERA_AUTHORITY_OWNER = I43A_AUTHORITATIVE_FRAME_OWNER
ADVISOR_PRODUCTION_AUTO_SCOPE = "offline_shadow_only"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IMPLEMENTATION_FAMILY = {
    item.implementation: item.family for item in dense_provider_implementation_contracts()
}


class AdvisorContextResolution(str, Enum):
    """How the current source was bound to a frozen evidence context."""

    UNRESOLVED = "unresolved"
    EXPLICIT_FROZEN_TOKEN = "explicit_frozen_token"


class AdvisorDisposition(str, Enum):
    """Whether guidance names one provider or explicitly declines to do so."""

    RECOMMEND = "recommend"
    ABSTAIN = "abstain"


class AdvisorEvidenceOutcome(str, Enum):
    """Frozen evidence outcome; never converted into a global score."""

    PASS = "pass"
    DEGRADED = "degraded"
    REJECTED = "rejected"
    UNQUALIFIED = "unqualified"


class AdvisorEvidenceSupport(str, Enum):
    """Qualitative support category for one advisory decision."""

    EXACT_QUALIFIED = "exact_qualified"
    BOUNDED = "bounded"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


def _nonempty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _unique_text(values: tuple[str, ...] | list[str], *, field_name: str, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(str(value or "").strip() for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} must be non-empty.")
    if any(not value for value in result):
        raise ValueError(f"{field_name} must not contain empty values.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values.")
    return result


@dataclass(frozen=True, slots=True)
class AdvisorProviderExclusion:
    """Explain why one considered provider was not recommended."""

    provider_id: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        provider_id = _nonempty(self.provider_id, field_name="provider_id")
        if provider_id == "auto":
            raise ValueError("provider_id must identify a concrete provider, never Auto.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(
            self,
            "reason_codes",
            _unique_text(self.reason_codes, field_name="reason_codes"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"provider_id": self.provider_id, "reason_codes": list(self.reason_codes)}


@dataclass(frozen=True, slots=True)
class AdvisorEvidenceBinding:
    """Content-addressed provenance for one frozen qualification artifact."""

    evidence_id: str
    phase: str
    path: str
    sha256: str
    schema_version: int = I63A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _nonempty(self.evidence_id, field_name="evidence_id"))
        object.__setattr__(self, "phase", _nonempty(self.phase, field_name="phase"))
        object.__setattr__(self, "path", _nonempty(self.path, field_name="path"))
        digest = str(self.sha256 or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError("sha256 must be a lowercase 64-character SHA-256 digest.")
        object.__setattr__(self, "sha256", digest)
        if int(self.schema_version) != I63A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3A evidence-binding schema version.")
        object.__setattr__(self, "schema_version", I63A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "phase": self.phase,
            "path": self.path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class AdvisorResourceObservation:
    """Observed resources for context only; never a provider-quality score."""

    gpu_name: str
    cuda_ready: bool
    total_vram_bytes: int
    free_vram_bytes: int
    available_ram_bytes: int
    schema_version: int = I63A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "gpu_name", _nonempty(self.gpu_name, field_name="gpu_name"))
        if not isinstance(self.cuda_ready, bool):
            raise TypeError("cuda_ready must be bool.")
        total = int(self.total_vram_bytes)
        free = int(self.free_vram_bytes)
        ram = int(self.available_ram_bytes)
        if total < 0:
            raise ValueError("total_vram_bytes must be non-negative.")
        if free < 0 or free > total:
            raise ValueError("free_vram_bytes must be within [0, total_vram_bytes].")
        if ram < 0:
            raise ValueError("available_ram_bytes must be non-negative.")
        object.__setattr__(self, "total_vram_bytes", total)
        object.__setattr__(self, "free_vram_bytes", free)
        object.__setattr__(self, "available_ram_bytes", ram)
        if int(self.schema_version) != I63A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3A resource-observation schema version.")
        object.__setattr__(self, "schema_version", I63A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gpu_name": self.gpu_name,
            "cuda_ready": self.cuda_ready,
            "total_vram_bytes": self.total_vram_bytes,
            "free_vram_bytes": self.free_vram_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "is_quality_score": False,
        }


@dataclass(frozen=True, slots=True)
class AdvisorEvidenceObservation:
    """One provider outcome bound to one exact frozen evidence context."""

    provider_id: str
    provider_family: DenseProviderFamily
    provider_implementation: DenseProviderImplementation
    outcome: AdvisorEvidenceOutcome
    evidence_context_id: str
    matched_scenario_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    limitations: tuple[str, ...]
    bindings: tuple[AdvisorEvidenceBinding, ...]
    schema_version: int = I63A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider_id = _nonempty(self.provider_id, field_name="provider_id")
        if provider_id == "auto":
            raise ValueError("provider_id must identify a concrete provider, never Auto.")
        object.__setattr__(self, "provider_id", provider_id)
        if not isinstance(self.provider_family, DenseProviderFamily):
            raise TypeError("provider_family must be DenseProviderFamily.")
        if not isinstance(self.provider_implementation, DenseProviderImplementation):
            raise TypeError("provider_implementation must be DenseProviderImplementation.")
        expected_family = _IMPLEMENTATION_FAMILY[self.provider_implementation]
        if self.provider_family is not expected_family:
            raise ValueError("provider family does not match the canonical implementation contract.")
        if not isinstance(self.outcome, AdvisorEvidenceOutcome):
            raise TypeError("outcome must be AdvisorEvidenceOutcome.")
        object.__setattr__(
            self,
            "evidence_context_id",
            _nonempty(self.evidence_context_id, field_name="evidence_context_id"),
        )
        object.__setattr__(
            self,
            "matched_scenario_ids",
            _unique_text(self.matched_scenario_ids, field_name="matched_scenario_ids"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _unique_text(self.reason_codes, field_name="reason_codes"),
        )
        object.__setattr__(
            self,
            "limitations",
            _unique_text(self.limitations, field_name="limitations", allow_empty=True),
        )
        bindings = tuple(self.bindings)
        if not bindings or any(not isinstance(item, AdvisorEvidenceBinding) for item in bindings):
            raise ValueError("bindings must contain at least one AdvisorEvidenceBinding.")
        binding_keys = {(item.evidence_id, item.sha256) for item in bindings}
        if len(binding_keys) != len(bindings):
            raise ValueError("bindings must not contain duplicate evidence provenance.")
        object.__setattr__(self, "bindings", bindings)
        if int(self.schema_version) != I63A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3A provider-evidence schema version.")
        object.__setattr__(self, "schema_version", I63A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_family": self.provider_family.value,
            "provider_implementation": self.provider_implementation.value,
            "outcome": self.outcome.value,
            "evidence_context_id": self.evidence_context_id,
            "matched_scenario_ids": list(self.matched_scenario_ids),
            "reason_codes": list(self.reason_codes),
            "limitations": list(self.limitations),
            "bindings": [item.as_dict() for item in self.bindings],
            "global_rank": None,
            "scalar_score": None,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionAdvisorRequest:
    """Immutable passive context supplied to the future I6.3B advisor engine."""

    reconstruction_mode: str
    source_workflow: str
    source_ready: bool
    source_item_count: int
    source_fingerprint: str
    workspace_state: str
    authoritative_sfm_usable: bool
    runtime_integrity_clean: bool
    evidence_context_resolution: AdvisorContextResolution
    evidence_context_id: str
    available_provider_ids: tuple[str, ...]
    resources: AdvisorResourceObservation
    provider_evidence: tuple[AdvisorEvidenceObservation, ...]
    camera_authority_owner: str = ADVISOR_CAMERA_AUTHORITY_OWNER
    production_auto_policy_scope: str = ADVISOR_PRODUCTION_AUTO_SCOPE
    production_auto_qualified: bool = False
    production_auto_authorized: bool = False
    provider_activation_authorized: bool = False
    execution_authorized: bool = False
    settings_mutation_authorized: bool = False
    schema_version: int = I63A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        mode = _nonempty(self.reconstruction_mode, field_name="reconstruction_mode")
        if mode != "offline_reconstruction":
            raise ValueError("I6.3A only admits offline_reconstruction advisory context.")
        object.__setattr__(self, "reconstruction_mode", mode)
        object.__setattr__(self, "source_workflow", _nonempty(self.source_workflow, field_name="source_workflow"))
        if not isinstance(self.source_ready, bool):
            raise TypeError("source_ready must be bool.")
        count = int(self.source_item_count)
        if count < 0:
            raise ValueError("source_item_count must be non-negative.")
        if self.source_ready and count <= 0:
            raise ValueError("source_ready requires at least one source item.")
        object.__setattr__(self, "source_item_count", count)
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint or "").strip())
        object.__setattr__(self, "workspace_state", _nonempty(self.workspace_state, field_name="workspace_state"))
        if not isinstance(self.authoritative_sfm_usable, bool):
            raise TypeError("authoritative_sfm_usable must be bool.")
        if not isinstance(self.runtime_integrity_clean, bool):
            raise TypeError("runtime_integrity_clean must be bool.")
        if not isinstance(self.evidence_context_resolution, AdvisorContextResolution):
            raise TypeError("evidence_context_resolution must be AdvisorContextResolution.")
        context_id = str(self.evidence_context_id or "").strip()
        if self.evidence_context_resolution is AdvisorContextResolution.UNRESOLVED:
            if context_id:
                raise ValueError("UNRESOLVED advisory context must not carry evidence_context_id.")
        elif not context_id:
            raise ValueError("EXPLICIT_FROZEN_TOKEN requires evidence_context_id.")
        object.__setattr__(self, "evidence_context_id", context_id)
        available = _unique_text(
            self.available_provider_ids,
            field_name="available_provider_ids",
            allow_empty=True,
        )
        if "auto" in available:
            raise ValueError("available_provider_ids must contain concrete providers only.")
        object.__setattr__(self, "available_provider_ids", available)
        if not isinstance(self.resources, AdvisorResourceObservation):
            raise TypeError("resources must be AdvisorResourceObservation.")
        evidence = tuple(self.provider_evidence)
        if any(not isinstance(item, AdvisorEvidenceObservation) for item in evidence):
            raise TypeError("provider_evidence must contain AdvisorEvidenceObservation values.")
        identities = [(item.provider_id, item.evidence_context_id) for item in evidence]
        if len(set(identities)) != len(identities):
            raise ValueError("provider_evidence contains duplicate provider/evidence context identity.")
        object.__setattr__(self, "provider_evidence", evidence)
        if self.camera_authority_owner != ADVISOR_CAMERA_AUTHORITY_OWNER:
            raise ValueError("I6.3A must preserve PyCOLMAP SfM as canonical camera authority.")
        if self.production_auto_policy_scope != ADVISOR_PRODUCTION_AUTO_SCOPE:
            raise ValueError("I6.3A must preserve the offline/shadow Auto-policy scope.")
        for field_name in (
            "production_auto_qualified",
            "production_auto_authorized",
            "provider_activation_authorized",
            "execution_authorized",
            "settings_mutation_authorized",
        ):
            if bool(getattr(self, field_name)):
                raise ValueError(f"I6.3A advisory request cannot set {field_name}=True.")
            object.__setattr__(self, field_name, False)
        if int(self.schema_version) != I63A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3A advisor-request schema version.")
        object.__setattr__(self, "schema_version", I63A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "advisor_scope": I63A_ADVISOR_SCOPE,
            "context": {
                "reconstruction_mode": self.reconstruction_mode,
                "source_workflow": self.source_workflow,
                "source_ready": self.source_ready,
                "source_item_count": self.source_item_count,
                "source_fingerprint": self.source_fingerprint,
                "workspace_state": self.workspace_state,
                "authoritative_sfm_usable": self.authoritative_sfm_usable,
                "runtime_integrity_clean": self.runtime_integrity_clean,
                "evidence_context_resolution": self.evidence_context_resolution.value,
                "evidence_context_id": self.evidence_context_id,
                "available_provider_ids": list(self.available_provider_ids),
                "resources": self.resources.as_dict(),
            },
            "provider_evidence": [item.as_dict() for item in self.provider_evidence],
            "authority": {
                "camera_authority_owner": self.camera_authority_owner,
            },
            "production_auto": {
                "policy_scope": self.production_auto_policy_scope,
                "qualified": False,
                "authorized": False,
            },
            "effects": {
                "provider_activation_authorized": False,
                "execution_authorized": False,
                "settings_mutation_authorized": False,
            },
            "claim_boundaries": {
                "universal_provider_winner": False,
                "global_provider_ranking": False,
                "global_scalar_score": False,
                "cross_family_rank": False,
                "runtime_performance_rank": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ReconstructionAdvisorDecision:
    """Immutable advisory output; never an execution or provider-selection command."""

    disposition: AdvisorDisposition
    evidence_support: AdvisorEvidenceSupport
    reason_codes: tuple[str, ...]
    explanations: tuple[str, ...]
    limitations: tuple[str, ...] = ()
    recommended_provider_id: str = ""
    recommended_family: DenseProviderFamily | None = None
    recommended_implementation: DenseProviderImplementation | None = None
    matched_scenario_ids: tuple[str, ...] = ()
    evidence_bindings: tuple[AdvisorEvidenceBinding, ...] = ()
    considered_provider_ids: tuple[str, ...] = ()
    excluded_providers: tuple[AdvisorProviderExclusion, ...] = ()
    production_effect: bool = False
    provider_selection_effect: bool = False
    provider_execution: bool = False
    settings_mutation: bool = False
    start_action: bool = False
    production_auto_qualified: bool = False
    production_auto_authorized: bool = False
    global_provider_ranking_created: bool = False
    global_scalar_score_created: bool = False
    camera_authority_owner: str = ADVISOR_CAMERA_AUTHORITY_OWNER
    schema_version: int = I63A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, AdvisorDisposition):
            raise TypeError("disposition must be AdvisorDisposition.")
        if not isinstance(self.evidence_support, AdvisorEvidenceSupport):
            raise TypeError("evidence_support must be AdvisorEvidenceSupport.")
        object.__setattr__(self, "reason_codes", _unique_text(self.reason_codes, field_name="reason_codes"))
        object.__setattr__(self, "explanations", _unique_text(self.explanations, field_name="explanations"))
        object.__setattr__(
            self,
            "limitations",
            _unique_text(self.limitations, field_name="limitations", allow_empty=True),
        )
        scenarios = _unique_text(
            self.matched_scenario_ids,
            field_name="matched_scenario_ids",
            allow_empty=True,
        )
        object.__setattr__(self, "matched_scenario_ids", scenarios)
        considered = _unique_text(
            self.considered_provider_ids,
            field_name="considered_provider_ids",
            allow_empty=True,
        )
        if "auto" in considered:
            raise ValueError("considered_provider_ids must contain concrete providers only.")
        object.__setattr__(self, "considered_provider_ids", considered)
        exclusions = tuple(self.excluded_providers)
        if any(not isinstance(item, AdvisorProviderExclusion) for item in exclusions):
            raise TypeError("excluded_providers must contain AdvisorProviderExclusion values.")
        excluded_ids = tuple(item.provider_id for item in exclusions)
        if len(set(excluded_ids)) != len(excluded_ids):
            raise ValueError("excluded_providers must contain unique provider IDs.")
        if considered and any(provider_id not in considered for provider_id in excluded_ids):
            raise ValueError("excluded_providers must be a subset of considered_provider_ids.")
        object.__setattr__(self, "excluded_providers", exclusions)
        bindings = tuple(self.evidence_bindings)
        if any(not isinstance(item, AdvisorEvidenceBinding) for item in bindings):
            raise TypeError("evidence_bindings must contain AdvisorEvidenceBinding values.")
        object.__setattr__(self, "evidence_bindings", bindings)

        provider_id = str(self.recommended_provider_id or "").strip()
        if self.disposition is AdvisorDisposition.RECOMMEND:
            if self.evidence_support is not AdvisorEvidenceSupport.EXACT_QUALIFIED:
                raise ValueError("Advisor recommendations require EXACT_QUALIFIED evidence support.")
            if not provider_id or provider_id == "auto":
                raise ValueError("Advisor recommendations require one concrete recommended provider.")
            if not isinstance(self.recommended_family, DenseProviderFamily):
                raise ValueError("Advisor recommendations require recommended_family.")
            if not isinstance(self.recommended_implementation, DenseProviderImplementation):
                raise ValueError("Advisor recommendations require recommended_implementation.")
            if _IMPLEMENTATION_FAMILY[self.recommended_implementation] is not self.recommended_family:
                raise ValueError("recommended provider family does not match implementation contract.")
            if considered and provider_id not in considered:
                raise ValueError("recommended provider must be present in considered_provider_ids.")
            if provider_id in {item.provider_id for item in exclusions}:
                raise ValueError("recommended provider cannot also be excluded.")
            if not scenarios:
                raise ValueError("Advisor recommendations require matched evidence scenarios.")
            if not bindings:
                raise ValueError("Advisor recommendations require evidence bindings.")
        else:
            if provider_id or self.recommended_family is not None or self.recommended_implementation is not None:
                raise ValueError("Advisor abstention must not carry a provider recommendation.")
        object.__setattr__(self, "recommended_provider_id", provider_id)

        for field_name in (
            "production_effect",
            "provider_selection_effect",
            "provider_execution",
            "settings_mutation",
            "start_action",
            "production_auto_qualified",
            "production_auto_authorized",
            "global_provider_ranking_created",
            "global_scalar_score_created",
        ):
            if bool(getattr(self, field_name)):
                raise ValueError(f"I6.3A advisory decision cannot set {field_name}=True.")
            object.__setattr__(self, field_name, False)
        if self.camera_authority_owner != ADVISOR_CAMERA_AUTHORITY_OWNER:
            raise ValueError("I6.3A must preserve PyCOLMAP SfM as canonical camera authority.")
        if int(self.schema_version) != I63A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.3A advisor-decision schema version.")
        object.__setattr__(self, "schema_version", I63A_SCHEMA_VERSION)

    @property
    def recommends_provider(self) -> bool:
        return self.disposition is AdvisorDisposition.RECOMMEND

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "advisor_scope": I63A_ADVISOR_SCOPE,
            "disposition": self.disposition.value,
            "confidence": {
                "evidence_support": self.evidence_support.value,
                "numeric_score": None,
                "scalar_score_used": False,
            },
            "recommendation": {
                "provider_id": self.recommended_provider_id,
                "provider_family": self.recommended_family.value if self.recommended_family else "",
                "provider_implementation": (
                    self.recommended_implementation.value if self.recommended_implementation else ""
                ),
            },
            "reason_codes": list(self.reason_codes),
            "explanations": list(self.explanations),
            "limitations": list(self.limitations),
            "matched_scenario_ids": list(self.matched_scenario_ids),
            "evidence_bindings": [item.as_dict() for item in self.evidence_bindings],
            "considered_provider_ids": list(self.considered_provider_ids),
            "excluded_providers": [item.as_dict() for item in self.excluded_providers],
            "authority": {
                "camera_authority_owner": self.camera_authority_owner,
            },
            "effects": {
                "production_effect": False,
                "provider_selection_effect": False,
                "provider_execution": False,
                "settings_mutation": False,
                "start_action": False,
            },
            "production_auto": {
                "qualified": False,
                "authorized": False,
            },
            "claim_boundaries": {
                "global_provider_ranking_created": False,
                "global_scalar_score_created": False,
                "cross_family_rank": False,
                "universal_provider_winner": False,
            },
        }


__all__ = [
    "ADVISOR_CAMERA_AUTHORITY_OWNER",
    "ADVISOR_PRODUCTION_AUTO_SCOPE",
    "I63A_ADVISOR_SCOPE",
    "I63A_PHASE",
    "I63A_SCHEMA_VERSION",
    "AdvisorContextResolution",
    "AdvisorDisposition",
    "AdvisorEvidenceBinding",
    "AdvisorEvidenceObservation",
    "AdvisorEvidenceOutcome",
    "AdvisorEvidenceSupport",
    "AdvisorProviderExclusion",
    "AdvisorResourceObservation",
    "ReconstructionAdvisorDecision",
    "ReconstructionAdvisorRequest",
]
