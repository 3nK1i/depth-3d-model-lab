"""Authoritative object-specific 3D point extraction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation
from core.contracts.common import ProcessingStep
from core.contracts.spatial import AxisConvention, CoordinateFrame, CoordinateFrameKind
from core.contracts.validation import ContractValidationError
from core.measurement.contracts import ObjectPointCloud, RejectionReason, ScaleRecoveryStatus
from core.models.metadata import ConfidenceKind, DepthScale
from core.object_geometry.clustering import select_voxel_cluster
from core.object_geometry.contracts import (
    ExtractionStageCounts,
    ObjectPointExtractionBatchResult,
    ObjectPointExtractionResult,
    ObjectPointExtractionStatus,
    PointCoordinateSemantics,
    derive_batch_status,
)
from core.object_geometry.filters import (
    background_plane_keep_mask,
    depth_discontinuity_keep_mask,
    robust_xyz_outlier_keep_mask,
)
from core.object_geometry.masks import (
    erode_mask_conservatively,
    exterior_ring,
    largest_connected_component,
    mask_centroid_xy,
)
from core.object_geometry.policy import ObjectPointExtractionPolicy
from core.object_geometry.projection import pixel_indices_from_mask, project_selected_pixels
from core.object_geometry.sampling import deterministic_spatial_sample_indices
from core.object_geometry.validation import assert_native_depth_alignment, validate_observation_selection


@dataclass(frozen=True, slots=True)
class ObjectPointExtractionRequest:
    aligned_bundle: AlignedFrameBundle
    observation_ids: tuple[str, ...] = ()
    policy: ObjectPointExtractionPolicy = field(default_factory=ObjectPointExtractionPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.aligned_bundle, AlignedFrameBundle):
            raise TypeError("aligned_bundle must be an AlignedFrameBundle.")
        if not isinstance(self.policy, ObjectPointExtractionPolicy):
            raise TypeError("policy must be an ObjectPointExtractionPolicy.")
        selection = validate_observation_selection(self.aligned_bundle, tuple(self.observation_ids))
        object.__setattr__(self, "observation_ids", selection)


def _camera_frame(bundle: AlignedFrameBundle) -> CoordinateFrame:
    return CoordinateFrame(
        frame_id=f"camera-depth:s{bundle.frame_reference.session_id}:f{bundle.frame_reference.frame_id}",
        kind=CoordinateFrameKind.CAMERA_DEPTH,
        axis_convention=AxisConvention.X_RIGHT_Y_DOWN_Z_FORWARD,
        source_frame=bundle.frame_reference,
    )


def _coordinate_semantics(depth_scale: DepthScale) -> PointCoordinateSemantics:
    return {
        DepthScale.METRIC: PointCoordinateSemantics.METRIC_METRES,
        DepthScale.RELATIVE: PointCoordinateSemantics.RELATIVE_MODEL_UNITS,
        DepthScale.SYNTHETIC: PointCoordinateSemantics.SYNTHETIC_UNITS,
    }.get(depth_scale, PointCoordinateSemantics.UNKNOWN)


def _scale_recovery_status(depth_scale: DepthScale) -> ScaleRecoveryStatus:
    return ScaleRecoveryStatus.NOT_REQUIRED if depth_scale is DepthScale.METRIC else ScaleRecoveryStatus.NOT_ATTEMPTED


def _step(observation_id: str, stage: str, method: str, before: int, after: int, **parameters) -> ProcessingStep:
    return ProcessingStep(
        step_id=f"{observation_id}:{stage}",
        method=method,
        input_ids=(f"{observation_id}:{stage}:input",),
        output_ids=(f"{observation_id}:{stage}:output",),
        parameters={"before": before, "after": after, **parameters},
    )


def _reject(
    observation: AlignedObjectObservation,
    bundle: AlignedFrameBundle,
    coordinate_frame: CoordinateFrame,
    semantics: PointCoordinateSemantics,
    reason: RejectionReason,
    counts: ExtractionStageCounts,
    steps: list[ProcessingStep],
    warnings: list[str],
    raw_cloud: ObjectPointCloud | None = None,
) -> ObjectPointExtractionResult:
    return ObjectPointExtractionResult(
        observation_id=observation.observation_id,
        frame_reference=bundle.frame_reference,
        status=ObjectPointExtractionStatus.REJECTED,
        coordinate_frame=coordinate_frame,
        coordinate_semantics=semantics,
        raw_cloud=raw_cloud,
        filtered_cloud=None,
        stage_counts=counts,
        rejection_reasons=(reason,),
        processing_steps=tuple(steps),
        warnings=tuple(warnings),
    )


def _weights_for_pixels(bundle: AlignedFrameBundle, pixels_xy: np.ndarray, policy: ObjectPointExtractionPolicy) -> np.ndarray | None:
    if bundle.confidence is None or bundle.confidence_kind is ConfidenceKind.NONE:
        return None
    values = bundle.confidence[pixels_xy[:, 1], pixels_xy[:, 0]].astype(np.float64)
    if bundle.confidence_kind is ConfidenceKind.CONFIDENCE:
        return values
    weights = 1.0 / (values + policy.confidence.inverse_uncertainty_epsilon)
    maximum = float(np.max(weights)) if weights.size else 0.0
    if maximum > 0.0:
        weights /= maximum
    return weights


def _cloud(
    observation: AlignedObjectObservation,
    bundle: AlignedFrameBundle,
    coordinate_frame: CoordinateFrame,
    points: np.ndarray,
    pixels: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray | None,
    *,
    raw_count: int,
    steps: tuple[ProcessingStep, ...],
) -> ObjectPointCloud:
    return ObjectPointCloud(
        observation_id=observation.observation_id,
        frame_reference=bundle.frame_reference,
        coordinate_frame_id=coordinate_frame.frame_id,
        points_xyz_m=points,
        source_pixel_space_id=bundle.target_depth_space.space_id,
        source_pixel_space_size=bundle.target_depth_space.size,
        source_pixel_indices_xy=pixels,
        colors_rgb=colors,
        point_weights=weights,
        raw_point_count=raw_count,
        retained_point_count=len(points),
        mask_id=observation.source_mask_id,
        depth_model_provenance=bundle.depth_model_provenance,
        depth_scale=bundle.depth_scale,
        intrinsics_source=bundle.intrinsics.source,
        scale_recovery_status=_scale_recovery_status(bundle.depth_scale),
        processing_steps=steps,
    )


def _extract_one(
    observation: AlignedObjectObservation,
    bundle: AlignedFrameBundle,
    coordinate_frame: CoordinateFrame,
    policy: ObjectPointExtractionPolicy,
) -> ObjectPointExtractionResult:
    semantics = _coordinate_semantics(bundle.depth_scale)
    warnings = list(observation.warnings)
    if observation.border_contacts:
        warnings.append("object_touches_image_border")
    steps: list[ProcessingStep] = []
    zero = ExtractionStageCounts()
    if observation.aligned_mask is None or not np.any(observation.aligned_mask):
        return _reject(observation, bundle, coordinate_frame, semantics, RejectionReason.MASK_UNAVAILABLE, zero, steps, warnings)

    selected = np.asarray(observation.aligned_mask, dtype=bool)
    selected_count = int(np.count_nonzero(selected))
    if selected_count < policy.mask.minimum_mask_pixels:
        counts = ExtractionStageCounts(selected_mask_pixels=selected_count)
        return _reject(observation, bundle, coordinate_frame, semantics, RejectionReason.TOO_FEW_VALID_POINTS, counts, steps, warnings)

    component = largest_connected_component(selected, connectivity=policy.mask.connectivity) if policy.mask.keep_largest_component else np.array(selected, copy=True)
    component_count = int(np.count_nonzero(component))
    steps.append(_step(
        observation.observation_id,
        "largest-component",
        "connected_components",
        selected_count,
        component_count,
        connectivity=policy.mask.connectivity,
        mask_id=observation.source_mask_id,
        mask_source=observation.mask_source.value if observation.mask_source is not None else "unknown",
        mask_revision=observation.mask_revision,
        approval_status=observation.approval_status.value,
        border_contacts=observation.border_contacts,
    ))

    eroded, erosion_skipped = erode_mask_conservatively(
        component,
        pixels=policy.mask.erosion_pixels,
        minimum_pixels=policy.mask.minimum_mask_pixels,
        connectivity=policy.mask.connectivity,
    )
    if erosion_skipped:
        warnings.append("boundary_erosion_skipped_to_preserve_support")
    eroded_count = int(np.count_nonzero(eroded))
    steps.append(_step(observation.observation_id, "erosion", "conservative_boundary_erosion", component_count, eroded_count, pixels=policy.mask.erosion_pixels, skipped=erosion_skipped))

    depth_eligible = eroded & bundle.rgb_support_mask & bundle.depth_valid_mask
    depth_count = int(np.count_nonzero(depth_eligible))
    steps.append(_step(observation.observation_id, "depth-validity", "mask_rgb_depth_intersection", eroded_count, depth_count))

    confidence_eligible = depth_eligible & bundle.confidence_valid_mask
    if bundle.confidence is not None:
        if bundle.confidence_kind is ConfidenceKind.CONFIDENCE and policy.confidence.minimum_confidence is not None:
            confidence_eligible &= bundle.confidence >= policy.confidence.minimum_confidence
        elif bundle.confidence_kind is ConfidenceKind.UNCERTAINTY and policy.confidence.maximum_uncertainty is not None:
            confidence_eligible &= bundle.confidence <= policy.confidence.maximum_uncertainty
    confidence_count = int(np.count_nonzero(confidence_eligible))
    steps.append(_step(observation.observation_id, "confidence", "semantic_confidence_filter", depth_count, confidence_count, confidence_kind=bundle.confidence_kind.value, minimum_confidence=policy.confidence.minimum_confidence, maximum_uncertainty=policy.confidence.maximum_uncertainty))

    ranged = np.array(confidence_eligible, copy=True)
    if policy.depth_range.minimum_depth_value is not None:
        ranged &= bundle.depth >= policy.depth_range.minimum_depth_value
    if policy.depth_range.maximum_depth_value is not None:
        ranged &= bundle.depth <= policy.depth_range.maximum_depth_value
    range_count = int(np.count_nonzero(ranged))
    steps.append(_step(observation.observation_id, "depth-range", "explicit_depth_range", confidence_count, range_count, depth_scale=bundle.depth_scale.value, minimum=policy.depth_range.minimum_depth_value, maximum=policy.depth_range.maximum_depth_value))

    discontinuity = depth_discontinuity_keep_mask(bundle.depth, ranged, policy.discontinuity)
    discontinuity_count = int(np.count_nonzero(discontinuity.keep))
    if discontinuity.warning:
        warnings.append(discontinuity.warning)
    steps.append(_step(observation.observation_id, "discontinuity", discontinuity.method, range_count, discontinuity_count, threshold=discontinuity.threshold))

    if discontinuity_count < policy.minimum_retained_points:
        counts = ExtractionStageCounts(
            selected_count, component_count, eroded_count, depth_count, confidence_count,
            range_count, discontinuity_count, 0, 0, 0, 0, 0,
        )
        return _reject(observation, bundle, coordinate_frame, semantics, RejectionReason.TOO_FEW_VALID_POINTS, counts, steps, warnings)

    pixels = pixel_indices_from_mask(discontinuity.keep)
    points = project_selected_pixels(bundle.depth, pixels, bundle.intrinsics)
    colors = bundle.aligned_rgb[pixels[:, 1], pixels[:, 0]].astype(np.uint8, copy=True)
    weights = _weights_for_pixels(bundle, pixels, policy)
    raw_count = len(points)
    raw_cloud = _cloud(observation, bundle, coordinate_frame, points, pixels, colors, weights, raw_count=raw_count, steps=tuple(steps))

    outlier = robust_xyz_outlier_keep_mask(points, policy.outliers)
    if outlier.warning:
        warnings.append(outlier.warning)
    points_f = points[outlier.keep]
    pixels_f = pixels[outlier.keep]
    colors_f = colors[outlier.keep]
    weights_f = weights[outlier.keep] if weights is not None else None
    outlier_count = len(points_f)
    steps.append(_step(observation.observation_id, "outliers", outlier.method, raw_count, outlier_count, threshold=outlier.threshold))

    plane_keep = np.ones(outlier_count, dtype=bool)
    plane_method = "disabled"
    plane_warning = ""
    if policy.background_plane.enabled and outlier_count:
        ring_mask = exterior_ring(component, pixels=policy.background_plane.exterior_ring_pixels)
        ring_mask &= bundle.depth_valid_mask & bundle.rgb_support_mask
        ring_pixels = pixel_indices_from_mask(ring_mask)
        ring_points = project_selected_pixels(bundle.depth, ring_pixels, bundle.intrinsics) if len(ring_pixels) else np.empty((0, 3))
        plane = background_plane_keep_mask(points_f, ring_points, policy.background_plane)
        plane_keep = plane.keep
        plane_method = plane.method
        plane_warning = plane.warning
        if plane_warning:
            warnings.append(plane_warning)
    points_f = points_f[plane_keep]
    pixels_f = pixels_f[plane_keep]
    colors_f = colors_f[plane_keep]
    weights_f = weights_f[plane_keep] if weights_f is not None else None
    plane_count = len(points_f)
    steps.append(_step(observation.observation_id, "background-plane", plane_method, outlier_count, plane_count, enabled=policy.background_plane.enabled))

    centroid = mask_centroid_xy(eroded)
    cluster = select_voxel_cluster(points_f, pixels_f, mask_centroid_xy=centroid, policy=policy.clustering)
    points_f = points_f[cluster.keep]
    pixels_f = pixels_f[cluster.keep]
    colors_f = colors_f[cluster.keep]
    weights_f = weights_f[cluster.keep] if weights_f is not None else None
    cluster_count = len(points_f)
    if cluster.ambiguous:
        warnings.append("ambiguous_3d_clusters")
    steps.append(_step(observation.observation_id, "clustering", cluster.method, plane_count, cluster_count, voxel_size=cluster.voxel_size, cluster_count=cluster.cluster_count, ambiguous=cluster.ambiguous))

    if cluster_count < policy.minimum_retained_points:
        counts = ExtractionStageCounts(
            selected_count, component_count, eroded_count, depth_count, confidence_count,
            range_count, discontinuity_count, raw_count, outlier_count, plane_count, cluster_count, 0,
        )
        return _reject(observation, bundle, coordinate_frame, semantics, RejectionReason.TOO_FEW_VALID_POINTS, counts, steps, warnings, raw_cloud=raw_cloud)

    sample_indices = deterministic_spatial_sample_indices(points_f, policy.sampling.maximum_points)
    points_f = points_f[sample_indices]
    pixels_f = pixels_f[sample_indices]
    colors_f = colors_f[sample_indices]
    weights_f = weights_f[sample_indices] if weights_f is not None else None
    output_count = len(points_f)
    steps.append(_step(observation.observation_id, "point-budget", "deterministic_spatial_sampling", cluster_count, output_count, maximum_points=policy.sampling.maximum_points))

    counts = ExtractionStageCounts(
        selected_count, component_count, eroded_count, depth_count, confidence_count,
        range_count, discontinuity_count, raw_count, outlier_count, plane_count, cluster_count, output_count,
    )
    filtered_cloud = _cloud(observation, bundle, coordinate_frame, points_f, pixels_f, colors_f, weights_f, raw_count=raw_count, steps=tuple(steps))
    status = ObjectPointExtractionStatus.PARTIAL if (observation.border_contacts or cluster.ambiguous) else ObjectPointExtractionStatus.SUCCESS
    return ObjectPointExtractionResult(
        observation.observation_id,
        bundle.frame_reference,
        status,
        coordinate_frame,
        semantics,
        raw_cloud,
        filtered_cloud,
        counts,
        (),
        tuple(steps),
        tuple(dict.fromkeys(warnings)),
    )


def extract_object_point_clouds(
    request: ObjectPointExtractionRequest,
) -> ObjectPointExtractionBatchResult:
    """Extract deterministic object-specific point clouds from one aligned frame."""

    started = time.perf_counter()
    bundle = request.aligned_bundle
    assert_native_depth_alignment(bundle)
    coordinate_frame = _camera_frame(bundle)
    selected = set(request.observation_ids)
    observations = tuple(
        value for value in bundle.aligned_observations
        if not selected or value.observation_id in selected
    )
    results = tuple(_extract_one(value, bundle, coordinate_frame, request.policy) for value in observations)
    status = derive_batch_status(results)
    duration_ms = (time.perf_counter() - started) * 1000.0
    step = ProcessingStep(
        step_id=f"frame:{bundle.frame_reference.frame_id}:object-point-extraction",
        method="segmentation_first_direct_back_projection",
        input_ids=(bundle.target_depth_space.space_id,),
        output_ids=tuple(result.observation_id for result in results),
        duration_ms=duration_ms,
        parameters={
            "objects_requested": len(observations),
            "objects_successful": sum(result.status is not ObjectPointExtractionStatus.REJECTED for result in results),
            "depth_resampled": False,
            "viewer_point_cloud_reused": False,
        },
    )
    warnings = tuple(
        f"{result.observation_id}:{warning}"
        for result in results
        for warning in result.warnings
    )
    return ObjectPointExtractionBatchResult(
        bundle.frame_reference,
        coordinate_frame,
        results,
        status,
        (step,),
        warnings,
    )
