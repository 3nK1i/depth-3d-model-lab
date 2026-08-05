"""Immutable Step 6 measurement-computation result contracts."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from core.contracts.common import FrameReference, ProcessingStep
from core.contracts.validation import ContractValidationError, require_nonempty
from core.measurement.contracts import MeasurementKind, ObjectMeasurementResult

class MeasurementComputationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    REJECTED = "rejected"

@dataclass(frozen=True, slots=True)
class MeasurementDiagnostics:
    observation_id: str
    selected_box_id: str = ""
    selected_box_method: str = ""
    perturbation_sample_count: int = 0
    available_kinds: tuple[MeasurementKind, ...] = ()
    unavailable_kinds: tuple[MeasurementKind, ...] = ()
    rejected_kinds: tuple[MeasurementKind, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id"))
        for name in ("selected_box_id", "selected_box_method"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        count = int(self.perturbation_sample_count)
        if count < 0:
            raise ContractValidationError("perturbation_sample_count must be non-negative.")
        object.__setattr__(self, "perturbation_sample_count", count)
        all_kinds = tuple(self.available_kinds) + tuple(self.unavailable_kinds) + tuple(self.rejected_kinds)
        if len(all_kinds) != len(set(all_kinds)):
            raise ContractValidationError("Measurement diagnostics may classify each kind only once.")
        object.__setattr__(self, "warnings", tuple(str(v).strip() for v in self.warnings if str(v).strip()))

@dataclass(frozen=True, slots=True)
class ObjectMeasurementComputationResult:
    observation_id: str
    frame_reference: FrameReference
    status: MeasurementComputationStatus
    measurement_result: ObjectMeasurementResult
    diagnostics: MeasurementDiagnostics
    processing_steps: tuple[ProcessingStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", require_nonempty(self.observation_id, field_name="observation_id"))
        if self.measurement_result.observation_id != self.observation_id:
            raise ContractValidationError("measurement_result observation mismatch.")
        if self.measurement_result.frame_reference != self.frame_reference:
            raise ContractValidationError("measurement_result frame mismatch.")
        if self.diagnostics.observation_id != self.observation_id:
            raise ContractValidationError("diagnostics observation mismatch.")
        available = sum(e.status.value == "available" for e in self.measurement_result.estimates)
        if self.status is MeasurementComputationStatus.REJECTED and available:
            raise ContractValidationError("Rejected computation cannot contain available estimates.")
        if self.status is MeasurementComputationStatus.SUCCESS and available != len(self.measurement_result.estimates):
            raise ContractValidationError("Successful computation requires all requested estimates available.")
        object.__setattr__(self, "processing_steps", tuple(self.processing_steps))
        object.__setattr__(self, "warnings", tuple(str(v).strip() for v in self.warnings if str(v).strip()))

@dataclass(frozen=True, slots=True)
class MeasurementBatchResult:
    frame_reference: FrameReference
    object_results: tuple[ObjectMeasurementComputationResult, ...]
    status: MeasurementComputationStatus
    processing_steps: tuple[ProcessingStep, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        results = tuple(self.object_results)
        ids = [r.observation_id for r in results]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("Measurement batch observation IDs must be unique.")
        if any(r.frame_reference != self.frame_reference for r in results):
            raise ContractValidationError("Measurement batch frame mismatch.")
        expected = derive_measurement_batch_status(results)
        if self.status is not expected:
            raise ContractValidationError("Measurement batch status disagrees with object results.")
        object.__setattr__(self, "object_results", results)
        object.__setattr__(self, "processing_steps", tuple(self.processing_steps))
        object.__setattr__(self, "warnings", tuple(str(v).strip() for v in self.warnings if str(v).strip()))

def derive_measurement_batch_status(results) -> MeasurementComputationStatus:
    results = tuple(results)
    if not results or all(r.status is MeasurementComputationStatus.REJECTED for r in results):
        return MeasurementComputationStatus.REJECTED
    if all(r.status is MeasurementComputationStatus.SUCCESS for r in results):
        return MeasurementComputationStatus.SUCCESS
    return MeasurementComputationStatus.PARTIAL
