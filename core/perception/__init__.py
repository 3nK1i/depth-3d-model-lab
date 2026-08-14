"""Segmentation-first perception contracts, registry, pipeline, and optional adapters."""
from core.perception.base import AutomaticSegmentationAdapter, BasePerceptionAdapter, FramePerceptionAdapter, PerceptionAdapterError, PerceptionDependencyError, PerceptionModelLoadError, PromptedSegmentationAdapter, SequencePerceptionAdapter, UnsupportedPerceptionOperation
from core.perception.metadata import AdapterCapability, CoResidencyPolicy, PerceptionCapabilities, PerceptionDescriptor, PerceptionResourceMetadata, PerceptionTask, PreferredDevice
from core.perception.messages import FramePerceptionRequest, PerceptionPipelineRequest, PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady, PerceptionWorkerResponse, PromptedSegmentationRequest, SegmentationPrompt, SegmentationPromptKind, SequenceResetRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY, DuplicatePerceptionModelError, PerceptionAdapterContractError, PerceptionRegistry, PerceptionRegistryError, UnknownPerceptionModelError
from core.perception.result import ApprovalStatus, BoundingBox2D, DenseMaskPayload, Detection2D, InstanceMask, MaskRepresentation, MaskSource, ObjectObservation, PerceptionFrameResult, PerceptionResultStatus, PolygonMaskPayload, PromptedMaskRefinement, RLEMaskPayload, SharedMemoryMaskPayload, UnsupportedMaskRepresentationError, normalize_detection_box, normalize_mask_payload, select_authoritative_mask
from core.perception.pipeline import PerceptionPipeline, PerceptionPipelinePolicy
from core.perception.raw import RawAutomaticMaskBatch, RawAutomaticMaskProposal, RawPerceptionBatch, RawPerceptionInstance, RawPromptedMask
from core.perception.auto_segment import (
    AUTO_SEGMENT_SCHEMA_VERSION,
    AutoSegmentEngine,
    AutoSegmentGenerationResult,
    AutoSegmentGenerationStatus,
    AutoSegmentPolicy,
)

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
    append_workspace_mask_proposal,
    assign_workspace_class,
    clear_workspace_class_assignment,
    merge_workspace_objects,
    replace_workspace_object_mask,
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
    retire_project_class,
    rebind_project_catalog_to_workspace,
    save_segmentation_project,
    segmentation_project_to_json,
)

from core.perception.semantic_workspace_persistence import (
    SEMANTIC_WORKSPACE_PERSISTENCE_SCHEMA_VERSION,
    PersistedSemanticWorkspace,
    SegmentationWorkspaceStore,
)

from core.perception.reviewed_analysis import (
    ReviewedSemanticObject,
    ReviewedSemanticSelection,
    load_reviewed_semantic_selection,
    perception_from_reviewed_selection,
    reviewed_selection_from_workspace,
    save_reviewed_semantic_selection,
    source_image_sha256,
)

__all__ = [name for name in globals() if not name.startswith('_')]
