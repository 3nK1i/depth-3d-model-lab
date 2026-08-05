from __future__ import annotations

import pickle

import numpy as np

from core.measurement.contracts import RejectionReason
from core.models.metadata import ConfidenceKind, DepthScale
from core.object_geometry import (
    ClusteringPolicy,
    ConfidenceFilterPolicy,
    DepthDiscontinuityPolicy,
    DepthRangePolicy,
    MaskPreprocessingPolicy,
    ObjectPointExtractionPolicy,
    ObjectPointExtractionRequest,
    ObjectPointExtractionStatus,
    PointCoordinateSemantics,
    SamplingPolicy,
    StatisticalOutlierPolicy,
    extract_object_point_clouds,
)
from tests.object_geometry_helpers import make_bundle


def _simple_policy(**changes):
    values = dict(
        mask=MaskPreprocessingPolicy(erosion_pixels=0, minimum_mask_pixels=4),
        discontinuity=DepthDiscontinuityPolicy(enabled=False),
        outliers=StatisticalOutlierPolicy(enabled=False),
        clustering=ClusteringPolicy(enabled=False),
        minimum_retained_points=4,
    )
    values.update(changes)
    return ObjectPointExtractionPolicy(**values)


def test_pipeline_extracts_object_only_points_with_pixels_colors_and_provenance():
    bundle = make_bundle()
    result = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy()))
    item = result.object_results[0]
    assert item.status is ObjectPointExtractionStatus.SUCCESS
    assert item.filtered_cloud is not None
    cloud = item.filtered_cloud
    assert cloud.metric_ready
    assert cloud.source_pixel_space_id == "depth"
    assert cloud.source_pixel_indices_xy.shape == (cloud.retained_point_count, 2)
    assert cloud.colors_rgb.shape == (cloud.retained_point_count, 3)
    assert not cloud.points_xyz_m.flags.writeable
    restored = pickle.loads(pickle.dumps(item))
    assert restored.stage_counts == item.stage_counts
    assert not restored.filtered_cloud.points_xyz_m.flags.writeable
    assert not restored.filtered_cloud.source_pixel_indices_xy.flags.writeable


def test_pipeline_rejects_missing_or_too_small_mask_without_empty_cloud():
    empty = np.zeros((36, 48), bool)
    bundle = make_bundle(masks=(empty,))
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
    assert item.status is ObjectPointExtractionStatus.REJECTED
    assert RejectionReason.MASK_UNAVAILABLE in item.rejection_reasons
    assert item.filtered_cloud is None


def test_confidence_and_depth_range_are_per_policy_not_model_specific():
    confidence = np.ones((36, 48), np.float32)
    confidence[:, :24] = 0.2
    bundle = make_bundle(confidence_kind=ConfidenceKind.CONFIDENCE, confidence=confidence)
    policy = _simple_policy(
        confidence=ConfidenceFilterPolicy(minimum_confidence=0.8),
        depth_range=DepthRangePolicy(minimum_depth_value=1.02, maximum_depth_value=1.07),
    )
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy)).object_results[0]
    assert item.stage_counts.confidence_valid_pixels < item.stage_counts.depth_valid_pixels
    assert item.stage_counts.depth_range_pixels <= item.stage_counts.confidence_valid_pixels


def test_uncertainty_is_converted_to_normalized_inverse_weights():
    uncertainty = np.full((36, 48), 0.5, np.float32)
    uncertainty[:, 20:30] = 0.1
    bundle = make_bundle(confidence_kind=ConfidenceKind.UNCERTAINTY, confidence=uncertainty)
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
    weights = item.filtered_cloud.point_weights
    assert weights is not None
    assert np.isclose(np.max(weights), 1.0)
    assert np.min(weights) > 0.0


def test_relative_depth_remains_nonmetric_and_synthetic_is_explicit():
    for scale, semantics in (
        (DepthScale.RELATIVE, PointCoordinateSemantics.RELATIVE_MODEL_UNITS),
        (DepthScale.SYNTHETIC, PointCoordinateSemantics.SYNTHETIC_UNITS),
    ):
        bundle = make_bundle(depth_scale=scale)
        item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
        assert item.coordinate_semantics is semantics
        assert not item.filtered_cloud.metric_ready


def test_multiple_overlapping_objects_remain_independent():
    first = np.zeros((36, 48), bool); first[5:25, 5:30] = True
    second = np.zeros((36, 48), bool); second[12:32, 20:44] = True
    bundle = make_bundle(masks=(first, second))
    batch = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy()))
    assert len(batch.object_results) == 2
    assert batch.object_results[0].filtered_cloud.observation_id != batch.object_results[1].filtered_cloud.observation_id


def test_border_contact_is_preserved_as_partial_not_global_rejection():
    mask = np.zeros((36, 48), bool); mask[5:30, 0:20] = True
    bundle = make_bundle(masks=(mask,), border_contacts=(("left",),))
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
    assert item.status is ObjectPointExtractionStatus.PARTIAL
    assert item.filtered_cloud is not None
    assert "object_touches_image_border" in item.warnings


def test_deterministic_point_budget_preserves_raw_count():
    bundle = make_bundle(width=200, height=160, masks=(np.ones((160, 200), bool),))
    policy = _simple_policy(
        mask=MaskPreprocessingPolicy(erosion_pixels=0, minimum_mask_pixels=4),
        sampling=SamplingPolicy(maximum_points=250),
    )
    first = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy)).object_results[0]
    second = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=policy)).object_results[0]
    assert first.filtered_cloud.retained_point_count == 250
    assert first.filtered_cloud.raw_point_count > 250
    assert np.array_equal(first.filtered_cloud.points_xyz_m, second.filtered_cloud.points_xyz_m)


def test_manual_mask_provenance_is_preserved():
    bundle = make_bundle(manual=True)
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
    assert item.filtered_cloud.mask_id == "mask-0"


def test_pipeline_intersects_rgb_support_and_depth_validity_without_mutating_inputs():
    mask = np.zeros((36, 48), bool); mask[4:32, 6:42] = True
    valid = np.ones((36, 48), bool); valid[:, :12] = False
    support = np.ones((36, 48), bool); support[:, 36:] = False
    bundle = make_bundle(masks=(mask,), valid_mask=valid, rgb_support=support)
    depth_before = np.array(bundle.depth, copy=True)
    mask_before = np.array(bundle.aligned_observations[0].aligned_mask, copy=True)
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy())).object_results[0]
    assert item.stage_counts.depth_valid_pixels < item.stage_counts.eroded_mask_pixels
    assert np.array_equal(bundle.depth, depth_before)
    assert np.array_equal(bundle.aligned_observations[0].aligned_mask, mask_before)


def test_explicit_observation_selection_processes_only_requested_object():
    first = np.zeros((36, 48), bool); first[4:20, 4:20] = True
    second = np.zeros((36, 48), bool); second[16:34, 25:46] = True
    bundle = make_bundle(masks=(first, second))
    batch = extract_object_point_clouds(
        ObjectPointExtractionRequest(bundle, observation_ids=("object-1",), policy=_simple_policy())
    )
    assert [item.observation_id for item in batch.object_results] == ["object-1"]


def test_processing_provenance_records_manual_authority_and_no_viewer_reuse():
    bundle = make_bundle(manual=True)
    batch = extract_object_point_clouds(ObjectPointExtractionRequest(bundle, policy=_simple_policy()))
    first_step = batch.object_results[0].processing_steps[0].as_dict()
    assert first_step["parameters"]["mask_source"] == "manual_correction"
    assert first_step["parameters"]["approval_status"] == "approved"
    assert batch.processing_steps[0].as_dict()["parameters"]["viewer_point_cloud_reused"] is False
