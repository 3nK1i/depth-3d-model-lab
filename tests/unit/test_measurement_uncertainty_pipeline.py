from __future__ import annotations
import pickle
import pytest
from core.measurement import (
    MeasurementComputationPolicy, MeasurementComputationStatus, MeasurementKind,
    MeasurementStatus, PerturbationPolicy, ReliabilityPolicy, ScaleRecovery,
    ScaleRecoveryStatus, ReliabilityAssessment, ReliabilityLevel, UncertaintyBasis, compute_measurements,
    make_measurement_request,
)
from core.measurement.contracts import RejectionReason, UncertaintyComponentKind
from core.models.metadata import DepthScale
from tests.measurement_computation_helpers import make_chain

def _result(**kwargs):
    b,e,f=make_chain(**kwargs)
    return compute_measurements(make_measurement_request(b,e,f))

def _by_kind(result, kind):
    estimates=result.object_results[0].measurement_result.estimates
    return next(x for x in estimates if x.kind is kind)

def test_metric_pipeline_generates_values_uncertainty_bounds_and_provenance():
    result=_result()
    width=_by_kind(result,MeasurementKind.WIDTH)
    assert width.status is MeasurementStatus.AVAILABLE
    assert width.value_si==pytest.approx(2.0,rel=.03)
    assert width.uncertainty.basis is UncertaintyBasis.HEURISTIC_PERTURBATION
    assert width.uncertainty.coverage_probability is None
    assert width.lower_bound_si==pytest.approx(width.value_si-width.uncertainty.expanded_uncertainty_si)
    assert {p.model_id for p in width.model_provenance} >= {'depth','seg'}
    assert width.calibration_status.value=='not_calibrated'

def test_uncertainty_contains_explainable_components():
    result=_result(outliers=12)
    width=_by_kind(result,MeasurementKind.WIDTH)
    kinds={c.kind for c in width.uncertainty.components}
    assert UncertaintyComponentKind.FITTING in kinds
    assert UncertaintyComponentKind.DEPTH in kinds
    assert UncertaintyComponentKind.MASK_BOUNDARY in kinds
    assert UncertaintyComponentKind.INTRINSICS in kinds
    assert UncertaintyComponentKind.POINT_CLOUD_OUTLIERS in kinds

def test_valid_observation_count_is_frame_observation_not_perturbation_count():
    width=_by_kind(_result(),MeasurementKind.WIDTH)
    assert width.valid_observation_count==1
    assert width.uncertainty.perturbation_sample_count>1

def test_relative_depth_blocks_metric_dimensions_but_allows_stable_orientation():
    result=_result(depth_scale=DepthScale.RELATIVE)
    width=_by_kind(result,MeasurementKind.WIDTH)
    yaw=_by_kind(result,MeasurementKind.ORIENTATION_YAW)
    assert width.status is MeasurementStatus.UNAVAILABLE
    assert RejectionReason.RELATIVE_DEPTH_WITHOUT_SCALE in width.rejection_reasons
    assert yaw.status is MeasurementStatus.AVAILABLE

def test_successful_scale_recovery_enables_relative_metric_measurements():
    b,e,f=make_chain(depth_scale=DepthScale.RELATIVE)
    recovery=ScaleRecovery(ScaleRecoveryStatus.SUCCEEDED,'known_reference',2.0,('object-0',),ReliabilityLevel.MEDIUM)
    result=compute_measurements(make_measurement_request(b,e,f,scale_recovery_by_observation=(('object-0',recovery),)))
    width=_by_kind(result,MeasurementKind.WIDTH)
    assert width.status is MeasurementStatus.AVAILABLE
    assert width.value_si==pytest.approx(4.0,rel=.04)
    assert any(c.kind is UncertaintyComponentKind.SCALE for c in width.uncertainty.components)

def test_synthetic_depth_produces_no_available_measurements():
    result=_result(depth_scale=DepthScale.SYNTHETIC)
    assert result.status is MeasurementComputationStatus.REJECTED
    assert all(x.status is not MeasurementStatus.AVAILABLE for x in result.object_results[0].measurement_result.estimates)

def test_left_border_rejects_width_but_keeps_height_and_distance():
    result=_result(border_contacts=('left',))
    assert _by_kind(result,MeasurementKind.WIDTH).status is MeasurementStatus.UNAVAILABLE
    assert _by_kind(result,MeasurementKind.HEIGHT).status is MeasurementStatus.AVAILABLE
    assert _by_kind(result,MeasurementKind.CAMERA_DISTANCE).status is MeasurementStatus.AVAILABLE
    assert _by_kind(result,MeasurementKind.VOLUME).status is MeasurementStatus.UNAVAILABLE

def test_top_border_rejects_height_but_keeps_width():
    result=_result(border_contacts=('top',))
    assert _by_kind(result,MeasurementKind.HEIGHT).status is MeasurementStatus.UNAVAILABLE
    assert _by_kind(result,MeasurementKind.WIDTH).status is MeasurementStatus.AVAILABLE

def test_symmetric_object_suppresses_ambiguous_orientation_but_keeps_dimensions():
    result=_result(extents=(1.,1.,1.),angle=.2,outliers=0)
    assert _by_kind(result,MeasurementKind.WIDTH).status is MeasurementStatus.AVAILABLE
    assert _by_kind(result,MeasurementKind.ORIENTATION_YAW).status is MeasurementStatus.UNAVAILABLE

def test_configured_uncertainty_limit_rejects_only_affected_measurements():
    b,e,f=make_chain(outliers=20)
    policy=MeasurementComputationPolicy(reliability=ReliabilityPolicy(high_relative_uncertainty=.001,medium_relative_uncertainty=.002,maximum_relative_uncertainty=.003))
    result=compute_measurements(make_measurement_request(b,e,f,policy=policy))
    estimates=result.object_results[0].measurement_result.estimates
    assert any(x.status is MeasurementStatus.REJECTED and RejectionReason.EXCESSIVE_UNCERTAINTY in x.rejection_reasons for x in estimates)
    assert len({x.kind for x in estimates})==len(estimates)

def test_result_is_pickle_safe():
    result=_result()
    restored=pickle.loads(pickle.dumps(result))
    assert restored.frame_reference==result.frame_reference
    assert restored.object_results[0].diagnostics.selected_box_id

def test_policy_validation_rejects_invalid_values():
    with pytest.raises(ValueError): PerturbationPolicy(point_subset_strides=(1,))
    with pytest.raises(ValueError): ReliabilityPolicy(high_relative_uncertainty=.2,medium_relative_uncertainty=.1)



def test_available_volume_is_suppressed_when_required_depth_is_rejected():
    from dataclasses import replace
    from core.measurement.pipeline import _enforce_derived_volume_dependency

    b,e,f=make_chain()
    result=compute_measurements(make_measurement_request(b,e,f))
    original=list(result.object_results[0].measurement_result.estimates)
    depth=next(x for x in original if x.kind is MeasurementKind.DEPTH)
    changed_depth=replace(
        depth,
        status=MeasurementStatus.REJECTED,
        value_si=None,
        uncertainty=None,
        lower_bound_si=None,
        upper_bound_si=None,
        reliability=ReliabilityAssessment(ReliabilityLevel.REJECTED, policy_id="test"),
        rejection_reasons=(RejectionReason.EXCESSIVE_UNCERTAINTY,),
    )
    inputs=[changed_depth if x.kind is MeasurementKind.DEPTH else x for x in original]
    updated,changed=_enforce_derived_volume_dependency(
        estimates=inputs,
        observation_id='object-0',
        bundle=b,
        scale_recovery=ScaleRecovery(ScaleRecoveryStatus.NOT_REQUIRED),
        method='test',
    )
    volume=next(x for x in updated if x.kind is MeasurementKind.VOLUME)
    assert changed
    assert volume.status is MeasurementStatus.REJECTED
    assert RejectionReason.MEASUREMENT_INPUT_UNAVAILABLE in volume.rejection_reasons
