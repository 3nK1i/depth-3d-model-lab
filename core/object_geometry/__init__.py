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
from .diagnostics import SupportRetentionDiagnostics, support_retention_diagnostics
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
    SupportRetentionPolicy,
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
    "SupportRetentionDiagnostics",
    "StatisticalOutlierPolicy",
    "SupportRetentionPolicy",
    "coordinate_semantics_for_depth_scale",
    "extract_object_point_clouds",
    "make_depth_camera_frame",
    "make_extraction_request",
    "support_retention_diagnostics",
]

from .fitting_contracts import (
    BoxFitDiagnostics,
    GeometryDegeneracy,
    ObjectFrameEstimate,
    ObjectGeometryFitBatchResult,
    ObjectGeometryFitResult,
    ObjectGeometryFitStatus,
    OrientationStability,
    PrincipalAxesEstimate,
)
from .fitting_factories import make_fitting_request
from .fitting_pipeline import ObjectGeometryFitRequest, fit_object_geometry
from .fitting_policy import (
    BoxFitPolicy,
    GenericShapeFitPolicy,
    ObjectGeometryFittingPolicy,
    RobustPcaPolicy,
)

__all__ += [
    "BoxFitDiagnostics", "BoxFitPolicy", "GenericShapeFitPolicy",
    "GeometryDegeneracy", "ObjectFrameEstimate", "ObjectGeometryFitBatchResult",
    "ObjectGeometryFitRequest", "ObjectGeometryFitResult", "ObjectGeometryFitStatus",
    "ObjectGeometryFittingPolicy", "OrientationStability", "PrincipalAxesEstimate",
    "RobustPcaPolicy", "fit_object_geometry", "make_fitting_request",
]
