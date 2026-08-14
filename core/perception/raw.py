"""Dependency-light raw outputs returned by perception adapters."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np
from core.contracts.common import FrameReference, ModelProvenance
from core.contracts.validation import ContractValidationError, JsonValue, normalize_metadata, readonly_array, require_finite, require_nonempty, require_probability
from core.models.metadata import ConfidenceKind

@dataclass(frozen=True, slots=True)
class RawPerceptionInstance:
    instance_id: str
    class_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    mask_confidence: float | None = None
    mask_confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    metadata: Mapping[str, JsonValue] | tuple[tuple[str, JsonValue], ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'instance_id', require_nonempty(self.instance_id, field_name='instance_id'))
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ContractValidationError('class_id must be a non-negative integer.')
        object.__setattr__(self, 'class_id', int(self.class_id))
        object.__setattr__(self, 'class_name', require_nonempty(self.class_name, field_name='class_name'))
        object.__setattr__(self, 'confidence', require_probability(self.confidence, field_name='confidence'))
        box = tuple((float(v) for v in self.box_xyxy))
        if len(box) != 4 or not np.isfinite(np.asarray(box)).all():
            raise ContractValidationError('box_xyxy must contain four finite coordinates.')
        object.__setattr__(self, 'box_xyxy', box)
        if self.mask is not None:
            object.__setattr__(self, 'mask', readonly_array(self.mask, field_name='raw mask', dtype=np.bool_, ndim=2))
        if self.mask_confidence_kind is ConfidenceKind.NONE:
            if self.mask_confidence is not None:
                raise ContractValidationError('mask_confidence must be None when kind is NONE.')
        elif self.mask_confidence is None:
            raise ContractValidationError('mask_confidence is required when kind is declared.')
        elif self.mask_confidence_kind is ConfidenceKind.CONFIDENCE:
            object.__setattr__(self, 'mask_confidence', require_probability(self.mask_confidence, field_name='mask_confidence'))
        else:
            value = require_finite(self.mask_confidence, field_name='mask_uncertainty')
            if value < 0:
                raise ContractValidationError('mask uncertainty must not be negative.')
            object.__setattr__(self, 'mask_confidence', value)
        object.__setattr__(self, 'metadata', normalize_metadata(self.metadata))
        object.__setattr__(self, 'warnings', tuple((str(v).strip() for v in self.warnings)))

    def __reduce__(self):
        return (type(self), (self.instance_id, self.class_id, self.class_name, self.confidence, self.box_xyxy, self.mask, self.mask_confidence, self.mask_confidence_kind, self.metadata, self.warnings))

@dataclass(frozen=True, slots=True)
class RawPerceptionBatch:
    frame_reference: FrameReference
    pixel_space_id: str
    instances: tuple[RawPerceptionInstance, ...]
    model_provenance: ModelProvenance
    processing_ms: float = 0.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'pixel_space_id', require_nonempty(self.pixel_space_id, field_name='pixel_space_id'))
        ids = [v.instance_id for v in self.instances]
        if len(ids) != len(set(ids)):
            raise ContractValidationError('Raw instance IDs must be unique.')
        value = require_finite(self.processing_ms, field_name='processing_ms')
        if value < 0:
            raise ContractValidationError('processing_ms must not be negative.')
        object.__setattr__(self, 'processing_ms', value)
        object.__setattr__(self, 'warnings', tuple((str(v).strip() for v in self.warnings)))

@dataclass(frozen=True, slots=True)
class RawPromptedMask:
    frame_reference: FrameReference
    parent_detection_id: str
    parent_mask_id: str
    mask: np.ndarray
    model_provenance: ModelProvenance
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.parent_detection_id).strip() and (not str(self.parent_mask_id).strip()):
            raise ContractValidationError('A prompted mask requires a parent detection or mask ID.')
        object.__setattr__(self, 'parent_detection_id', str(self.parent_detection_id).strip())
        object.__setattr__(self, 'parent_mask_id', str(self.parent_mask_id).strip())
        object.__setattr__(self, 'mask', readonly_array(self.mask, field_name='prompted mask', dtype=np.bool_, ndim=2))
        if self.confidence_kind is ConfidenceKind.NONE:
            if self.confidence is not None:
                raise ContractValidationError('confidence must be None when kind is NONE.')
        elif self.confidence is None:
            raise ContractValidationError('confidence is required when kind is declared.')
        elif self.confidence_kind is ConfidenceKind.CONFIDENCE:
            object.__setattr__(
                self,
                'confidence',
                require_probability(self.confidence, field_name='confidence'),
            )
        else:
            uncertainty = require_finite(self.confidence, field_name='uncertainty')
            if uncertainty < 0.0:
                raise ContractValidationError('Prompted-mask uncertainty must not be negative.')
            object.__setattr__(self, 'confidence', uncertainty)
        object.__setattr__(self, 'warnings', tuple((str(v).strip() for v in self.warnings)))

    def __reduce__(self):
        return (type(self), (self.frame_reference, self.parent_detection_id, self.parent_mask_id, self.mask, self.model_provenance, self.confidence, self.confidence_kind, self.warnings))


@dataclass(frozen=True, slots=True)
class RawAutomaticMaskProposal:
    """One unlabeled automatic-segmentation mask proposal."""

    proposal_id: str
    mask: np.ndarray
    confidence: float | None = None
    confidence_kind: ConfidenceKind = ConfidenceKind.NONE
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_id",
            require_nonempty(self.proposal_id, field_name="proposal_id"),
        )
        object.__setattr__(
            self,
            "mask",
            readonly_array(self.mask, field_name="automatic proposal mask", dtype=np.bool_, ndim=2),
        )
        if self.confidence_kind is ConfidenceKind.NONE:
            if self.confidence is not None:
                raise ContractValidationError(
                    "confidence must be None when confidence_kind is NONE."
                )
        elif self.confidence is None:
            raise ContractValidationError(
                "confidence is required when confidence_kind is declared."
            )
        elif self.confidence_kind is ConfidenceKind.CONFIDENCE:
            object.__setattr__(
                self,
                "confidence",
                require_probability(self.confidence, field_name="confidence"),
            )
        else:
            uncertainty = require_finite(self.confidence, field_name="uncertainty")
            if uncertainty < 0.0:
                raise ContractValidationError(
                    "Automatic proposal uncertainty must not be negative."
                )
            object.__setattr__(self, "confidence", uncertainty)
        object.__setattr__(
            self, "warnings", tuple(str(value).strip() for value in self.warnings)
        )

    def __reduce__(self):
        return (
            type(self),
            (
                self.proposal_id,
                self.mask,
                self.confidence,
                self.confidence_kind,
                self.warnings,
            ),
        )


@dataclass(frozen=True, slots=True)
class RawAutomaticMaskBatch:
    """Dependency-light unlabeled mask proposals from a segment-everything adapter."""

    frame_reference: FrameReference
    pixel_space_id: str
    proposals: tuple[RawAutomaticMaskProposal, ...]
    model_provenance: ModelProvenance
    processing_ms: float = 0.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixel_space_id",
            require_nonempty(self.pixel_space_id, field_name="pixel_space_id"),
        )
        proposal_ids = [proposal.proposal_id for proposal in self.proposals]
        if len(proposal_ids) != len(set(proposal_ids)):
            raise ContractValidationError(
                "Automatic mask proposal IDs must be unique within a batch."
            )
        value = require_finite(self.processing_ms, field_name="processing_ms")
        if value < 0.0:
            raise ContractValidationError("processing_ms must not be negative.")
        object.__setattr__(self, "processing_ms", value)
        object.__setattr__(
            self, "warnings", tuple(str(value).strip() for value in self.warnings)
        )
