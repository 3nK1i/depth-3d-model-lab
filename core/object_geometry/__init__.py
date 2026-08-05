"""Segmentation-first object-specific 3D point extraction."""

from .contracts import (
    ExtractionStageCounts,
    ObjectPointExtractionBatchResult,
    ObjectPointExtractionResult,
    ObjectPointExtractionStatus,
    PointCoordinateSemantics,
)
from .factories import (
    coordinate_semantics_for_depth_scale,
    make_depth_camera_frame,
    make_extraction_request,
)
from .pipeline import ObjectPointExtractionRequest, extract_object_point_clouds
from .policy import (
    BackgroundPlanePolicy,
    ClusteringPolicy,
    ConfidenceFilterPolicy,
    DepthDiscontinuityPolicy,
    DepthRangePolicy,
    MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy,
    SamplingPolicy,
    StatisticalOutlierPolicy,
)

__all__ = [
    "BackgroundPlanePolicy",
    "ClusteringPolicy",
    "ConfidenceFilterPolicy",
    "DepthDiscontinuityPolicy",
    "DepthRangePolicy",
    "ExtractionStageCounts",
    "MaskPreprocessingPolicy",
    "ObjectPointExtractionBatchResult",
    "ObjectPointExtractionPolicy",
    "ObjectPointExtractionRequest",
    "ObjectPointExtractionResult",
    "ObjectPointExtractionStatus",
    "PointCoordinateSemantics",
    "SamplingPolicy",
    "StatisticalOutlierPolicy",
    "coordinate_semantics_for_depth_scale",
    "extract_object_point_clouds",
    "make_depth_camera_frame",
    "make_extraction_request",
]
