"""Segmentation-first perception contracts, registry, pipeline, and optional adapters."""
from core.perception.base import BasePerceptionAdapter, FramePerceptionAdapter, PerceptionAdapterError, PerceptionDependencyError, PerceptionModelLoadError, PromptedSegmentationAdapter, SequencePerceptionAdapter, UnsupportedPerceptionOperation
from core.perception.metadata import AdapterCapability, CoResidencyPolicy, PerceptionCapabilities, PerceptionDescriptor, PerceptionResourceMetadata, PerceptionTask, PreferredDevice
from core.perception.messages import FramePerceptionRequest, PerceptionPipelineRequest, PerceptionShutdownRequest, PerceptionWorkerConfiguration, PerceptionWorkerReady, PerceptionWorkerResponse, PromptedSegmentationRequest, SegmentationPrompt, SegmentationPromptKind, SequenceResetRequest
from core.perception.registry import PERCEPTION_MODEL_REGISTRY, DuplicatePerceptionModelError, PerceptionAdapterContractError, PerceptionRegistry, PerceptionRegistryError, UnknownPerceptionModelError
from core.perception.result import ApprovalStatus, BoundingBox2D, DenseMaskPayload, Detection2D, InstanceMask, MaskRepresentation, MaskSource, ObjectObservation, PerceptionFrameResult, PerceptionResultStatus, PolygonMaskPayload, PromptedMaskRefinement, RLEMaskPayload, SharedMemoryMaskPayload, UnsupportedMaskRepresentationError, normalize_detection_box, normalize_mask_payload, select_authoritative_mask
from core.perception.pipeline import PerceptionPipeline, PerceptionPipelinePolicy
from core.perception.raw import RawPerceptionBatch, RawPerceptionInstance, RawPromptedMask
__all__ = [name for name in globals() if not name.startswith('_')]
