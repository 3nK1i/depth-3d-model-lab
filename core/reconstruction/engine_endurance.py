"""Phase 8.7G3 whole-engine endurance, failure, and degradation contracts.

G3 composes the already-qualified G1 lifecycle containment and G2 checkpoint/resume
capabilities into a bounded campaign report. It intentionally does not reimplement
backend quality/performance qualification, and it does not own the final full Phase
8.7 regression/UI-readiness gate (G4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

from core.reconstruction.errors import ReconstructionValidationError

ENGINE_ENDURANCE_SCHEMA_VERSION = 1
ENGINE_ENDURANCE_PHASE = "8.7G3"


class ReconstructionEnduranceScenario(str, Enum):
    CLEAN_RECOVERY = "clean_recovery"
    OPTIONAL_DEGRADATION = "optional_degradation"
    REQUIRED_FAILURE = "required_failure"
    CHECKPOINT_REJECTION = "checkpoint_rejection"


@dataclass(frozen=True, slots=True)
class ReconstructionEnduranceCycleEvidence:
    cycle_index: int
    scenario: ReconstructionEnduranceScenario
    passed: bool
    process_group_clean: bool = True
    stale_state_detected: bool = False
    pipeline_status: str = ""
    persistent_scene_state: str = ""
    scene_id: str = ""
    canonical_geometry_authoritative: bool = False
    gaussian_authority: str = ""
    nerf_authority: str = ""
    checkpoint_rejected_safely: bool = False
    required_failure_fail_closed: bool = False
    reason: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    schema_version: int = ENGINE_ENDURANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        index = int(self.cycle_index)
        if index < 0:
            raise ReconstructionValidationError("cycle_index must be non-negative.")
        if not isinstance(self.scenario, ReconstructionEnduranceScenario):
            raise TypeError("scenario must be ReconstructionEnduranceScenario.")
        if not isinstance(self.passed, bool):
            raise TypeError("passed must be bool.")
        if not isinstance(self.process_group_clean, bool):
            raise TypeError("process_group_clean must be bool.")
        if not isinstance(self.stale_state_detected, bool):
            raise TypeError("stale_state_detected must be bool.")
        if int(self.schema_version) != ENGINE_ENDURANCE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported G3 endurance schema version.")
        meta = tuple((str(k).strip(), str(v).strip()) for k, v in self.metadata)
        if any(not k for k, _ in meta) or len({k for k, _ in meta}) != len(meta):
            raise ReconstructionValidationError("metadata keys must be unique and non-empty.")
        object.__setattr__(self, "cycle_index", index)
        object.__setattr__(self, "pipeline_status", str(self.pipeline_status).strip())
        object.__setattr__(self, "persistent_scene_state", str(self.persistent_scene_state).strip())
        object.__setattr__(self, "scene_id", str(self.scene_id).strip())
        object.__setattr__(self, "gaussian_authority", str(self.gaussian_authority).strip())
        object.__setattr__(self, "nerf_authority", str(self.nerf_authority).strip())
        object.__setattr__(self, "reason", str(self.reason).strip())
        object.__setattr__(self, "metadata", meta)
        object.__setattr__(self, "schema_version", ENGINE_ENDURANCE_SCHEMA_VERSION)
        self._validate_scenario_invariants()

    def _validate_scenario_invariants(self) -> None:
        if self.passed and (not self.process_group_clean or self.stale_state_detected):
            raise ReconstructionValidationError(
                "A passing G3 cycle must have a clean process boundary and no stale state."
            )
        if self.scenario is ReconstructionEnduranceScenario.CLEAN_RECOVERY and self.passed:
            if self.pipeline_status != "completed" or self.persistent_scene_state != "ready":
                raise ReconstructionValidationError("Clean recovery requires completed pipeline and ready scene.")
            if not self.canonical_geometry_authoritative:
                raise ReconstructionValidationError("Clean recovery must preserve canonical geometry authority.")
            if self.gaussian_authority != "appearance_only" or self.nerf_authority != "appearance_only":
                raise ReconstructionValidationError("Clean recovery must preserve appearance-only neural authority.")
            if not self.scene_id:
                raise ReconstructionValidationError("Clean recovery requires a persistent scene identity.")
        elif self.scenario is ReconstructionEnduranceScenario.OPTIONAL_DEGRADATION and self.passed:
            if self.pipeline_status != "degraded":
                raise ReconstructionValidationError("Optional degradation must be explicit in pipeline status.")
            if not self.canonical_geometry_authoritative:
                raise ReconstructionValidationError("Optional degradation cannot invalidate canonical geometry.")
            allowed = {"appearance_only", "missing"}
            if self.gaussian_authority not in allowed or self.nerf_authority not in allowed:
                raise ReconstructionValidationError("Optional degradation neural authority must remain bounded.")
            if self.gaussian_authority == "missing" and self.nerf_authority == "missing":
                raise ReconstructionValidationError("The bounded degradation fixture must retain one appearance backend.")
        elif self.scenario is ReconstructionEnduranceScenario.REQUIRED_FAILURE and self.passed:
            if self.pipeline_status != "failed" or not self.required_failure_fail_closed:
                raise ReconstructionValidationError("Required-stage failure must fail closed.")
            if self.persistent_scene_state:
                raise ReconstructionValidationError("Required-stage failure cannot publish a persistent scene.")
        elif self.scenario is ReconstructionEnduranceScenario.CHECKPOINT_REJECTION and self.passed:
            if not self.checkpoint_rejected_safely:
                raise ReconstructionValidationError("Checkpoint rejection scenario must reject safely.")

    @property
    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": ENGINE_ENDURANCE_PHASE,
            "cycle_index": self.cycle_index,
            "scenario": self.scenario.value,
            "pass": self.passed,
            "process_group_clean": self.process_group_clean,
            "stale_state_detected": self.stale_state_detected,
            "pipeline_status": self.pipeline_status,
            "persistent_scene_state": self.persistent_scene_state,
            "scene_id": self.scene_id,
            "canonical_geometry_authoritative": self.canonical_geometry_authoritative,
            "gaussian_authority": self.gaussian_authority,
            "nerf_authority": self.nerf_authority,
            "checkpoint_rejected_safely": self.checkpoint_rejected_safely,
            "required_failure_fail_closed": self.required_failure_fail_closed,
            "reason": self.reason,
            "metadata": self.metadata_dict,
        }


@dataclass(frozen=True, slots=True)
class ReconstructionEnduranceCampaignReport:
    cycles: tuple[ReconstructionEnduranceCycleEvidence, ...]
    backward_compatibility_verified: bool
    resource_cleanup_verified: bool
    authority_stability_verified: bool
    checkpoint_safety_verified: bool
    full_regression_deferred_to: str = "8.7G4"
    reconstruction_ui_enabled: bool = False
    schema_version: int = ENGINE_ENDURANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        cycles = tuple(self.cycles)
        if not cycles or any(not isinstance(c, ReconstructionEnduranceCycleEvidence) for c in cycles):
            raise ReconstructionValidationError("G3 campaign requires endurance cycle evidence.")
        indices = [c.cycle_index for c in cycles]
        if len(set(indices)) != len(indices):
            raise ReconstructionValidationError("G3 cycle indices must be unique.")
        if int(self.schema_version) != ENGINE_ENDURANCE_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported G3 endurance schema version.")
        if str(self.full_regression_deferred_to).strip() != "8.7G4":
            raise ReconstructionValidationError("Full Phase 8.7 regression remains owned by 8.7G4.")
        if self.reconstruction_ui_enabled:
            raise ReconstructionValidationError("Reconstruction UI remains disabled until Phase 8.8.")
        object.__setattr__(self, "cycles", cycles)
        object.__setattr__(self, "full_regression_deferred_to", "8.7G4")
        object.__setattr__(self, "schema_version", ENGINE_ENDURANCE_SCHEMA_VERSION)

    @property
    def passed(self) -> bool:
        return bool(
            all(c.passed for c in self.cycles)
            and all(c.process_group_clean for c in self.cycles)
            and not any(c.stale_state_detected for c in self.cycles)
            and self.backward_compatibility_verified
            and self.resource_cleanup_verified
            and self.authority_stability_verified
            and self.checkpoint_safety_verified
            and self._unique_clean_scene_ids()
        )

    def _unique_clean_scene_ids(self) -> bool:
        ids = [
            c.scene_id
            for c in self.cycles
            if c.scenario is ReconstructionEnduranceScenario.CLEAN_RECOVERY and c.scene_id
        ]
        return len(ids) == len(set(ids))

    @property
    def scenario_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for cycle in self.cycles:
            result[cycle.scenario.value] = result.get(cycle.scenario.value, 0) + 1
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "phase": ENGINE_ENDURANCE_PHASE,
            "pass": self.passed,
            "cycle_count": len(self.cycles),
            "scenario_counts": self.scenario_counts,
            "backward_compatibility_verified": self.backward_compatibility_verified,
            "resource_cleanup_verified": self.resource_cleanup_verified,
            "authority_stability_verified": self.authority_stability_verified,
            "checkpoint_safety_verified": self.checkpoint_safety_verified,
            "stale_state_detected": any(c.stale_state_detected for c in self.cycles),
            "all_process_groups_clean": all(c.process_group_clean for c in self.cycles),
            "unique_clean_scene_ids": self._unique_clean_scene_ids(),
            "cycles": [c.as_dict() for c in self.cycles],
            "full_regression_deferred_to": self.full_regression_deferred_to,
            "reconstruction_ui_enabled": self.reconstruction_ui_enabled,
        }


def build_reconstruction_endurance_campaign(
    cycles: Iterable[ReconstructionEnduranceCycleEvidence],
    *,
    backward_compatibility_verified: bool,
    resource_cleanup_verified: bool,
    authority_stability_verified: bool,
    checkpoint_safety_verified: bool,
) -> ReconstructionEnduranceCampaignReport:
    return ReconstructionEnduranceCampaignReport(
        cycles=tuple(cycles),
        backward_compatibility_verified=bool(backward_compatibility_verified),
        resource_cleanup_verified=bool(resource_cleanup_verified),
        authority_stability_verified=bool(authority_stability_verified),
        checkpoint_safety_verified=bool(checkpoint_safety_verified),
    )
