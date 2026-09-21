"""I6.4A production provider availability and selection contracts.

This module is deliberately execution-free.  It provides the product-facing
contract that later I6.4 steps may populate with explicit runtime availability,
context-qualified evidence, Advisor guidance and user selection intent.

The contract keeps five concepts separate:

* provider identity -- one of the ten canonical concrete dense providers;
* runtime availability -- whether that concrete provider is observed runnable;
* context qualification -- whether frozen evidence qualifies it for one exact
  evidence context (never a universal provider claim);
* recommendation -- passive Advisor guidance;
* selection -- explicit user intent, still not provider activation/execution.

``Auto`` is not a concrete provider.  It has a separate fail-closed gate.  The
I5 offline/shadow Auto policy remains qualified, while production Auto remains
unqualified, unauthorized and non-selectable.

Nothing in this module probes a runtime, selects/activates a provider, changes a
setting, starts reconstruction, modifies production defaults, creates a global
provider rank/score, or changes PyCOLMAP SfM camera/global-frame authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from core.reconstruction.dense_provider_architecture import (
    I43A_AUTHORITATIVE_FRAME_OWNER,
    DenseProviderFamily,
    DenseProviderImplementation,
    dense_provider_implementation_contracts,
)

I64A_SCHEMA_VERSION = 1
I64A_PHASE = "I6.4A"
I64A_PRODUCTION_AUTO_POLICY_ID = "auto"
I64A_CANONICAL_PROVIDER_COUNT = 10
I64A_SELECTION_SCOPE = "offline_dense_provider_user_intent_only"
I64A_CAMERA_AUTHORITY_OWNER = I43A_AUTHORITATIVE_FRAME_OWNER
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION: dict[DenseProviderImplementation, str] = {
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

_IMPLEMENTATION_CONTRACT = {
    item.implementation: item for item in dense_provider_implementation_contracts()
}

if len(PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION) != I64A_CANONICAL_PROVIDER_COUNT:
    raise RuntimeError("I6.4A canonical provider count drifted.")
if set(PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION) != set(DenseProviderImplementation):
    raise RuntimeError("I6.4A canonical provider identities do not cover every implementation.")


class ProductionProviderAvailability(str, Enum):
    """Observed runtime availability; UNKNOWN means no production probe claim."""

    UNKNOWN = "unknown"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ProductionProviderQualification(str, Enum):
    """Qualification for one explicit evidence context, never a global rank."""

    UNRESOLVED = "unresolved"
    QUALIFIED_FOR_CONTEXT = "qualified_for_context"
    NOT_QUALIFIED_FOR_CONTEXT = "not_qualified_for_context"


class ProductionProviderSelectionMode(str, Enum):
    """Product selection intent.  Selection remains distinct from execution."""

    NONE = "none"
    MANUAL = "manual"
    AUTO = "auto"


def _nonempty(value: Any, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty.")
    return text


def _unique_reason_codes(values: tuple[str, ...] | list[str], *, field_name: str) -> tuple[str, ...]:
    result = tuple(str(value or "").strip() for value in values)
    if any(not value for value in result):
        raise ValueError(f"{field_name} must not contain empty values.")
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} must contain unique values.")
    return result


def _sha256_values(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    result = tuple(str(value or "").strip().lower() for value in values)
    if any(not _SHA256_RE.fullmatch(value) for value in result):
        raise ValueError("qualification_evidence_sha256 values must be lowercase SHA-256 digests.")
    if len(set(result)) != len(result):
        raise ValueError("qualification_evidence_sha256 values must be unique.")
    return result


@dataclass(frozen=True, slots=True)
class ProductionProviderCatalogEntry:
    """One concrete production-provider observation with no execution effect."""

    provider_id: str
    label: str
    implementation: DenseProviderImplementation
    family: DenseProviderFamily
    availability: ProductionProviderAvailability = ProductionProviderAvailability.UNKNOWN
    availability_reason_codes: tuple[str, ...] = ()
    qualification: ProductionProviderQualification = ProductionProviderQualification.UNRESOLVED
    qualification_context_id: str = ""
    qualification_evidence_sha256: tuple[str, ...] = ()
    manual_selection_allowed: bool = False
    schema_version: int = I64A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        provider_id = _nonempty(self.provider_id, field_name="provider_id")
        label = _nonempty(self.label, field_name="label")
        if provider_id == I64A_PRODUCTION_AUTO_POLICY_ID:
            raise ValueError("Auto is a policy gate and must never be a concrete provider entry.")
        if not isinstance(self.implementation, DenseProviderImplementation):
            raise TypeError("implementation must be DenseProviderImplementation.")
        if not isinstance(self.family, DenseProviderFamily):
            raise TypeError("family must be DenseProviderFamily.")
        expected_contract = _IMPLEMENTATION_CONTRACT[self.implementation]
        expected_id = PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[self.implementation]
        if provider_id != expected_id:
            raise ValueError("provider_id does not match the canonical implementation identity.")
        if self.family is not expected_contract.family:
            raise ValueError("provider family does not match the canonical implementation contract.")
        if not isinstance(self.availability, ProductionProviderAvailability):
            raise TypeError("availability must be ProductionProviderAvailability.")
        availability_reasons = _unique_reason_codes(
            self.availability_reason_codes,
            field_name="availability_reason_codes",
        )
        if self.availability is ProductionProviderAvailability.AVAILABLE and availability_reasons:
            raise ValueError("An available provider cannot carry unavailability reason codes.")
        if self.availability is ProductionProviderAvailability.UNAVAILABLE and not availability_reasons:
            raise ValueError("An unavailable provider requires at least one reason code.")
        if not isinstance(self.qualification, ProductionProviderQualification):
            raise TypeError("qualification must be ProductionProviderQualification.")
        context_id = str(self.qualification_context_id or "").strip()
        evidence_sha256 = _sha256_values(self.qualification_evidence_sha256)
        if self.qualification is ProductionProviderQualification.UNRESOLVED:
            if context_id or evidence_sha256:
                raise ValueError("An unresolved qualification cannot carry context/evidence claims.")
        else:
            if not context_id:
                raise ValueError("qualification_context_id is required for a resolved qualification claim.")
            if not evidence_sha256:
                raise ValueError("Resolved qualification requires SHA-256 evidence provenance.")
        if not isinstance(self.manual_selection_allowed, bool):
            raise TypeError("manual_selection_allowed must be bool.")
        if self.manual_selection_allowed and (
            self.availability is not ProductionProviderAvailability.AVAILABLE
            or self.qualification is not ProductionProviderQualification.QUALIFIED_FOR_CONTEXT
        ):
            raise ValueError(
                "manual selection is allowed only for an available provider qualified for the exact context."
            )
        if int(self.schema_version) != I64A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.4A provider-catalog schema version.")
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "availability_reason_codes", availability_reasons)
        object.__setattr__(self, "qualification_context_id", context_id)
        object.__setattr__(self, "qualification_evidence_sha256", evidence_sha256)
        object.__setattr__(self, "schema_version", I64A_SCHEMA_VERSION)

    @property
    def runtime_available(self) -> bool:
        return self.availability is ProductionProviderAvailability.AVAILABLE

    @property
    def context_qualified(self) -> bool:
        return self.qualification is ProductionProviderQualification.QUALIFIED_FOR_CONTEXT

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "label": self.label,
            "implementation": self.implementation.value,
            "family": self.family.value,
            "availability": self.availability.value,
            "runtime_available": self.runtime_available,
            "availability_reason_codes": list(self.availability_reason_codes),
            "qualification": self.qualification.value,
            "context_qualified": self.context_qualified,
            "qualification_context_id": self.qualification_context_id,
            "qualification_evidence_sha256": list(self.qualification_evidence_sha256),
            "manual_selection_allowed": self.manual_selection_allowed,
            "claims": {
                "global_provider_qualification": False,
                "metric_scale": False,
                "camera_pose_authority": False,
                "certified_metrology": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ProductionAutoGate:
    """Fail-closed product representation of Auto at the I6.4A boundary."""

    policy_id: str = I64A_PRODUCTION_AUTO_POLICY_ID
    offline_shadow_qualified: bool = True
    production_qualified: bool = False
    production_authorized: bool = False
    selectable: bool = False
    reason_codes: tuple[str, ...] = (
        "production_auto_not_qualified",
        "production_auto_not_authorized",
    )
    schema_version: int = I64A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if str(self.policy_id).strip() != I64A_PRODUCTION_AUTO_POLICY_ID:
            raise ValueError("I6.4A Auto gate policy_id must remain 'auto'.")
        if self.offline_shadow_qualified is not True:
            raise ValueError("I6.4A must preserve the qualified I5 offline/shadow Auto policy.")
        if self.production_qualified or self.production_authorized or self.selectable:
            raise ValueError("I6.4A production Auto must remain fail-closed and non-selectable.")
        reasons = _unique_reason_codes(self.reason_codes, field_name="reason_codes")
        required = {"production_auto_not_qualified", "production_auto_not_authorized"}
        if not required.issubset(set(reasons)):
            raise ValueError("I6.4A Auto gate must preserve qualification/authorization reason codes.")
        if int(self.schema_version) != I64A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.4A Auto-gate schema version.")
        object.__setattr__(self, "policy_id", I64A_PRODUCTION_AUTO_POLICY_ID)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "schema_version", I64A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "offline_shadow_qualified": self.offline_shadow_qualified,
            "production_qualified": self.production_qualified,
            "production_authorized": self.production_authorized,
            "selectable": self.selectable,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True, slots=True)
class ProductionProviderSelectionSnapshot:
    """Product-facing selection state with recommendation/selection kept separate."""

    providers: tuple[ProductionProviderCatalogEntry, ...]
    recommended_provider_id: str = ""
    selection_mode: ProductionProviderSelectionMode = ProductionProviderSelectionMode.NONE
    selected_provider_id: str = ""
    auto_gate: ProductionAutoGate = field(default_factory=ProductionAutoGate)
    provider_selection_effect: bool = False
    provider_activation_authorized: bool = False
    provider_execution_authorized: bool = False
    settings_mutation_authorized: bool = False
    start_action_authorized: bool = False
    schema_version: int = I64A_SCHEMA_VERSION

    def __post_init__(self) -> None:
        providers = tuple(self.providers)
        if not providers or any(not isinstance(item, ProductionProviderCatalogEntry) for item in providers):
            raise ValueError("providers must contain at least one ProductionProviderCatalogEntry.")
        ids = tuple(item.provider_id for item in providers)
        if len(set(ids)) != len(ids):
            raise ValueError("providers must contain unique provider IDs.")
        if I64A_PRODUCTION_AUTO_POLICY_ID in ids:
            raise ValueError("Auto must remain separate from the concrete provider catalog.")
        if not isinstance(self.selection_mode, ProductionProviderSelectionMode):
            raise TypeError("selection_mode must be ProductionProviderSelectionMode.")
        if not isinstance(self.auto_gate, ProductionAutoGate):
            raise TypeError("auto_gate must be ProductionAutoGate.")
        recommended = str(self.recommended_provider_id or "").strip()
        selected = str(self.selected_provider_id or "").strip()
        by_id = {item.provider_id: item for item in providers}
        if recommended:
            if recommended not in by_id:
                raise ValueError("recommended_provider_id must reference the concrete provider catalog.")
            item = by_id[recommended]
            if not item.runtime_available or not item.context_qualified:
                raise ValueError("Recommended provider must be available and qualified for the exact context.")
        if self.selection_mode is ProductionProviderSelectionMode.NONE:
            if selected:
                raise ValueError("selection_mode=none cannot carry selected_provider_id.")
        elif self.selection_mode is ProductionProviderSelectionMode.MANUAL:
            if not selected or selected not in by_id:
                raise ValueError("Manual selection requires a concrete catalog selected_provider_id.")
            if not by_id[selected].manual_selection_allowed:
                raise ValueError("Selected provider is not manually selectable.")
        elif self.selection_mode is ProductionProviderSelectionMode.AUTO:
            if selected:
                raise ValueError("Auto selection does not carry a concrete selected_provider_id before policy resolution.")
            if not self.auto_gate.selectable:
                raise ValueError("Auto selection is unavailable while the production Auto gate is fail-closed.")
        forbidden_effects = (
            self.provider_selection_effect,
            self.provider_activation_authorized,
            self.provider_execution_authorized,
            self.settings_mutation_authorized,
            self.start_action_authorized,
        )
        if any(forbidden_effects):
            raise ValueError("I6.4A selection snapshot is descriptive only and cannot have production effects.")
        if int(self.schema_version) != I64A_SCHEMA_VERSION:
            raise ValueError("Unsupported I6.4A selection-snapshot schema version.")
        object.__setattr__(self, "providers", providers)
        object.__setattr__(self, "recommended_provider_id", recommended)
        object.__setattr__(self, "selected_provider_id", selected)
        object.__setattr__(self, "schema_version", I64A_SCHEMA_VERSION)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": I64A_PHASE,
            "scope": I64A_SELECTION_SCOPE,
            "providers": [item.as_dict() for item in self.providers],
            "recommended_provider_id": self.recommended_provider_id,
            "selection_mode": self.selection_mode.value,
            "selected_provider_id": self.selected_provider_id,
            "auto": self.auto_gate.as_dict(),
            "effects": {
                "provider_selection_effect": self.provider_selection_effect,
                "provider_activation_authorized": self.provider_activation_authorized,
                "provider_execution_authorized": self.provider_execution_authorized,
                "settings_mutation_authorized": self.settings_mutation_authorized,
                "start_action_authorized": self.start_action_authorized,
            },
            "authority": {
                "camera_authority_owner": I64A_CAMERA_AUTHORITY_OWNER,
                "dense_provider_establishes_metric_scale": False,
                "certified_metrology": False,
            },
            "claim_boundaries": {
                "availability_is_qualification": False,
                "qualification_is_recommendation": False,
                "recommendation_is_selection": False,
                "selection_is_execution": False,
                "global_provider_ranking": False,
                "global_scalar_score": False,
                "universal_provider_winner": False,
                "production_auto_qualified": False,
                "production_auto_authorized": False,
            },
        }


def canonical_production_provider_catalog() -> tuple[ProductionProviderCatalogEntry, ...]:
    """Return all canonical provider identities without making runtime/evidence claims."""

    entries: list[ProductionProviderCatalogEntry] = []
    for implementation in DenseProviderImplementation:
        contract = _IMPLEMENTATION_CONTRACT[implementation]
        entries.append(
            ProductionProviderCatalogEntry(
                provider_id=PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION[implementation],
                label=contract.label,
                implementation=implementation,
                family=contract.family,
                availability=ProductionProviderAvailability.UNKNOWN,
                availability_reason_codes=(),
                qualification=ProductionProviderQualification.UNRESOLVED,
                qualification_context_id="",
                qualification_evidence_sha256=(),
                manual_selection_allowed=False,
            )
        )
    catalog = tuple(entries)
    if len(catalog) != I64A_CANONICAL_PROVIDER_COUNT:
        raise RuntimeError("I6.4A canonical production provider catalog count drifted.")
    return catalog


__all__ = [
    "I64A_SCHEMA_VERSION",
    "I64A_PHASE",
    "I64A_PRODUCTION_AUTO_POLICY_ID",
    "I64A_CANONICAL_PROVIDER_COUNT",
    "I64A_SELECTION_SCOPE",
    "I64A_CAMERA_AUTHORITY_OWNER",
    "PRODUCTION_PROVIDER_ID_BY_IMPLEMENTATION",
    "ProductionProviderAvailability",
    "ProductionProviderQualification",
    "ProductionProviderSelectionMode",
    "ProductionProviderCatalogEntry",
    "ProductionAutoGate",
    "ProductionProviderSelectionSnapshot",
    "canonical_production_provider_catalog",
]
