"""Segmentation-first perception contracts, registry, pipeline, and optional adapters."""
from core.perception.base import BasePerceptionAdapter, FramePerceptionAdapter, PerceptionAdapterError, PerceptionDependencyError, PerceptionModelLoadError, PromptedSegmentationAdapter, SequencePerceptionAdapter, UnsupportedPerceptionOperation
from core.perception.metadata import AdapterCapability, CoResidencyPolicy, PerceptionCapabilities, PerceptionDescriptor, PerceptionResourceMetadata, PerceptionTask, PreferredDevice
from core.perception.messages import FramePerceptionRequest, PerceptionPipelineRequest, PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady, PerceptionWorkerResponse, PromptedSegmentationRequest, SegmentationPrompt, SegmentationPromptKind, SequenceResetRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY, DuplicatePerceptionModelError, PerceptionAdapterContractError, PerceptionRegistry, PerceptionRegistryError, UnknownPerceptionModelError
from core.perception.result import ApprovalStatus, BoundingBox2D, DenseMaskPayload, Detection2D, InstanceMask, MaskRepresentation, MaskSource, ObjectObservation, PerceptionFrameResult, PerceptionResultStatus, PolygonMaskPayload, PromptedMaskRefinement, RLEMaskPayload, SharedMemoryMaskPayload, UnsupportedMaskRepresentationError, normalize_detection_box, normalize_mask_payload, select_authoritative_mask
from core.perception.pipeline import PerceptionPipeline, PerceptionPipelinePolicy
from core.perception.raw import RawPerceptionBatch, RawPerceptionInstance, RawPromptedMask
from core.perception.semantic_workspace import (
    SEMANTIC_WORKSPACE_SCHEMA_VERSION,
    SemanticClassCatalog,
    SemanticClassDefinition,
    SemanticClassOrigin,
    SemanticLabelSource,
    SemanticReviewStatus,
    SemanticWorkspace,
    SemanticWorkspaceObject,
    accepted_observations,
    assign_workspace_class,
    set_workspace_review_status,
    workspace_from_mask_proposals,
    workspace_from_perception_result,
)

from core.perception.semantic_project import (
    SEGMENTATION_PROJECT_SCHEMA_VERSION,
    SegmentationProject,
    SegmentationProjectStore,
    SemanticInstanceNamingPolicy,
    add_project_class,
    create_segmentation_project,
    create_segmentation_project_from_model_names,
    load_segmentation_project,
    merge_project_catalog_into_workspace,
    rename_project_class,
    save_segmentation_project,
    segmentation_project_to_json,
)
__all__ = [name for name in globals() if not name.startswith('_')]
