"""Phase 9.2C automatic segmentation proposal engine.

The engine deliberately produces *unlabeled* semantic workspace proposals.  It
reuses the existing perception adapter registry and `InstanceMask` contract, then
hands review/class assignment to the Phase 9.2A/B semantic workspace/project
layers.  Automatic proposals never become geometry, metric or measurement
authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Iterable

import numpy as np

from core.contracts.validation import ContractValidationError, require_nonempty
from core.perception.base import AutomaticSegmentationAdapter
from core.perception.messages import FramePerceptionRequest
from core.perception.metadata import AdapterCapability, PerceptionDescriptor, PerceptionTask
from core.perception.result import DenseMaskPayload, InstanceMask, MaskSource
from core.perception.semantic_project import SegmentationProject
from core.perception.semantic_workspace import SemanticClassCatalog, SemanticWorkspace, workspace_from_mask_proposals


AUTO_SEGMENT_SCHEMA_VERSION = 1


class AutoSegmentGenerationStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    PARTIAL = "partial"


@dataclass(frozen=True, slots=True)
class AutoSegmentPolicy:
    """Deterministic post-processing for segment-everything proposal batches."""

    minimum_mask_pixels: int = 64
    max_proposals: int = 128
    remove_exact_duplicates: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.minimum_mask_pixels, bool) or int(self.minimum_mask_pixels) < 1:
            raise ContractValidationError("minimum_mask_pixels must be a positive integer.")
        if isinstance(self.max_proposals, bool) or int(self.max_proposals) < 1:
            raise ContractValidationError("max_proposals must be a positive integer.")
        object.__setattr__(self, "minimum_mask_pixels", int(self.minimum_mask_pixels))
        object.__setattr__(self, "max_proposals", int(self.max_proposals))
        object.__setattr__(self, "remove_exact_duplicates", bool(self.remove_exact_duplicates))


@dataclass(frozen=True, slots=True)
class AutoSegmentGenerationResult:
    workspace: SemanticWorkspace
    status: AutoSegmentGenerationStatus
    model_id: str
    generated_proposals: int
    kept_proposals: int
    dropped_small: int = 0
    dropped_duplicates: int = 0
    dropped_over_limit: int = 0
    processing_ms: float = 0.0
    warnings: tuple[str, ...] = ()
    schema_version: int = AUTO_SEGMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, SemanticWorkspace):
            raise TypeError("workspace must be SemanticWorkspace.")
        if not isinstance(self.status, AutoSegmentGenerationStatus):
            raise TypeError("status must be AutoSegmentGenerationStatus.")
        object.__setattr__(self, "model_id", require_nonempty(self.model_id, field_name="model_id"))
        for field_name in (
            "generated_proposals",
            "kept_proposals",
            "dropped_small",
            "dropped_duplicates",
            "dropped_over_limit",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 0:
                raise ContractValidationError(f"{field_name} must be a non-negative integer.")
            object.__setattr__(self, field_name, int(value))
        if self.kept_proposals > self.generated_proposals:
            raise ContractValidationError("kept_proposals cannot exceed generated_proposals.")
        if float(self.processing_ms) < 0.0 or not np.isfinite(float(self.processing_ms)):
            raise ContractValidationError("processing_ms must be finite and non-negative.")
        object.__setattr__(self, "processing_ms", float(self.processing_ms))
        object.__setattr__(self, "warnings", tuple(str(value).strip() for value in self.warnings))
        if int(self.schema_version) != AUTO_SEGMENT_SCHEMA_VERSION:
            raise ContractValidationError(
                f"Unsupported auto-segment schema version: {self.schema_version}."
            )
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def is_geometry_authority(self) -> bool:
        return False

    @property
    def is_metric_authority(self) -> bool:
        return False

    @property
    def is_measurement_authority(self) -> bool:
        return False


class AutoSegmentEngine:
    """Synchronous core engine for automatic unlabeled mask proposal generation.

    UI integration will execute this service off the GUI thread.  Keeping the
    engine itself synchronous makes it deterministic, dependency-light, and easy
    to validate independently of Qt/process orchestration.
    """

    def __init__(
        self,
        adapter: AutomaticSegmentationAdapter,
        descriptor: PerceptionDescriptor,
        *,
        policy: AutoSegmentPolicy = AutoSegmentPolicy(),
    ) -> None:
        if not isinstance(adapter, AutomaticSegmentationAdapter):
            raise TypeError("adapter must implement AutomaticSegmentationAdapter.")
        if not isinstance(descriptor, PerceptionDescriptor):
            raise TypeError("descriptor must be PerceptionDescriptor.")
        if PerceptionTask.AUTOMATIC_SEGMENTATION not in descriptor.capabilities.tasks:
            raise ContractValidationError(
                "Auto Segment requires a descriptor with AUTOMATIC_SEGMENTATION task support."
            )
        if AdapterCapability.AUTOMATIC_SEGMENTATION not in descriptor.capabilities.adapter_capabilities:
            raise ContractValidationError(
                "Auto Segment requires the AUTOMATIC_SEGMENTATION adapter capability."
            )
        self.adapter = adapter
        self.descriptor = descriptor
        self.policy = policy

    @staticmethod
    def _mask_digest(mask: np.ndarray) -> bytes:
        packed = np.packbits(np.asarray(mask, dtype=np.bool_), axis=None)
        return sha256(packed.tobytes()).digest()

    def _normalize_masks(self, request: FramePerceptionRequest, batch) -> tuple[tuple[InstanceMask, ...], int, int, int]:
        expected_shape = (request.rgb_space.size.height, request.rgb_space.size.width)
        candidates: list[tuple[int, str, object]] = []
        dropped_small = 0
        dropped_duplicates = 0
        seen: set[bytes] = set()
        for proposal in batch.proposals:
            if proposal.mask.shape != expected_shape:
                raise ContractValidationError(
                    "Automatic segmentation proposal shape does not match RGB pixel space."
                )
            area = int(np.count_nonzero(proposal.mask))
            if area < self.policy.minimum_mask_pixels:
                dropped_small += 1
                continue
            if self.policy.remove_exact_duplicates:
                digest = self._mask_digest(proposal.mask)
                if digest in seen:
                    dropped_duplicates += 1
                    continue
                seen.add(digest)
            candidates.append((area, proposal.proposal_id, proposal))

        # Largest masks first, then stable source ID.  This makes truncation
        # deterministic regardless of adapter iteration details.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        kept = candidates[: self.policy.max_proposals]
        dropped_over_limit = max(0, len(candidates) - len(kept))
        prefix = f"{request.frame_reference.session_id}:{request.frame_reference.frame_id}:auto"
        masks = tuple(
            InstanceMask(
                mask_id=f"{prefix}:{proposal.proposal_id}",
                frame_reference=request.frame_reference,
                pixel_space=request.rgb_space,
                payload=DenseMaskPayload(proposal.mask),
                source=MaskSource.NATIVE_INSTANCE,
                model_provenance=batch.model_provenance,
                confidence=proposal.confidence,
                confidence_kind=proposal.confidence_kind,
                foreground_valid=True,
            )
            for _area, _proposal_id, proposal in kept
        )
        return masks, dropped_small, dropped_duplicates, dropped_over_limit

    def generate(
        self,
        request: FramePerceptionRequest,
        *,
        workspace_id: str,
        project: SegmentationProject | None = None,
    ) -> AutoSegmentGenerationResult:
        if PerceptionTask.AUTOMATIC_SEGMENTATION not in request.requested_tasks:
            raise ContractValidationError(
                "Auto Segment requests must include PerceptionTask.AUTOMATIC_SEGMENTATION."
            )
        workspace_key = require_nonempty(workspace_id, field_name="workspace_id")
        batch = self.adapter.propose_masks(request)
        if batch.frame_reference != request.frame_reference:
            raise ContractValidationError("Auto Segment batch frame does not match the request.")
        if batch.pixel_space_id != request.rgb_space.space_id:
            raise ContractValidationError("Auto Segment batch pixel space does not match RGB space.")
        masks, dropped_small, dropped_duplicates, dropped_over_limit = self._normalize_masks(
            request, batch
        )
        catalog = project.class_catalog if project is not None else SemanticClassCatalog()
        if masks:
            workspace = workspace_from_mask_proposals(
                masks,
                workspace_id=workspace_key,
                class_catalog=catalog,
            )
        else:
            workspace = SemanticWorkspace(
                workspace_id=workspace_key,
                frame_reference=request.frame_reference,
                class_catalog=catalog,
                objects=(),
            )
        warnings = list(batch.warnings)
        if dropped_small:
            warnings.append(f"Filtered {dropped_small} proposal(s) below minimum mask area.")
        if dropped_duplicates:
            warnings.append(f"Filtered {dropped_duplicates} exact duplicate proposal(s).")
        if dropped_over_limit:
            warnings.append(f"Truncated {dropped_over_limit} proposal(s) above max_proposals.")
        if not masks:
            status = AutoSegmentGenerationStatus.EMPTY
        elif warnings:
            status = AutoSegmentGenerationStatus.PARTIAL
        else:
            status = AutoSegmentGenerationStatus.SUCCESS
        return AutoSegmentGenerationResult(
            workspace=workspace,
            status=status,
            model_id=batch.model_provenance.model_id,
            generated_proposals=len(batch.proposals),
            kept_proposals=len(masks),
            dropped_small=dropped_small,
            dropped_duplicates=dropped_duplicates,
            dropped_over_limit=dropped_over_limit,
            processing_ms=batch.processing_ms,
            warnings=tuple(warnings),
        )
