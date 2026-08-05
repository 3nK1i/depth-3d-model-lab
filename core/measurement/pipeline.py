"""Phase 7.5 Step 6 measurement and uncertainty computation pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
import time
from core.alignment.contracts import AlignedFrameBundle, unique_model_provenance
from core.contracts.common import ProcessingStep
from core.contracts.validation import ContractValidationError
from core.measurement.computation_contracts import (
    MeasurementBatchResult, MeasurementComputationStatus, MeasurementDiagnostics,
    ObjectMeasurementComputationResult, derive_measurement_batch_status,
)
from core.measurement.contracts import (
    MeasurementEstimate, MeasurementKind, MeasurementStatus, ObjectMeasurementResult,
    RejectionReason, ReliabilityAssessment, ReliabilityLevel, ScaleRecovery,
    ScaleRecoveryStatus,
)
from core.measurement.eligibility import eligibility_for_kind
from core.measurement.geometry import apply_scale_to_values, scale_factor_for_kind, select_measurement_box, unit_for_kind, values_from_box
from core.measurement.perturbation import generate_perturbation_samples
from core.measurement.policy import MeasurementComputationPolicy
from core.measurement.reliability import assess_reliability
from core.measurement.uncertainty import estimate_uncertainty
from core.models.metadata import DepthScale
from core.object_geometry.contracts import ObjectPointExtractionBatchResult
from core.object_geometry.fitting_contracts import ObjectGeometryFitBatchResult, ObjectGeometryFitStatus

@dataclass(frozen=True, slots=True)
class MeasurementComputationRequest:
    aligned_bundle: AlignedFrameBundle
    extraction_batch: ObjectPointExtractionBatchResult
    fit_batch: ObjectGeometryFitBatchResult
    observation_ids: tuple[str,...] = ()
    policy: MeasurementComputationPolicy = field(default_factory=MeasurementComputationPolicy)
    scale_recovery_by_observation: tuple[tuple[str, ScaleRecovery], ...] = ()

    def __post_init__(self) -> None:
        frame=self.aligned_bundle.frame_reference
        if self.extraction_batch.frame_reference != frame or self.fit_batch.frame_reference != frame:
            raise ContractValidationError("Measurement inputs must reference the same frame.")
        known={o.observation_id for o in self.aligned_bundle.aligned_observations}
        selected=tuple(self.observation_ids)
        if len(selected)!=len(set(selected)) or any(v not in known for v in selected):
            raise ContractValidationError("observation_ids must be unique and present in the aligned bundle.")
        recovery = tuple(self.scale_recovery_by_observation)
        recovery_ids = [oid for oid, _ in recovery]
        if len(recovery_ids) != len(set(recovery_ids)) or any(oid not in known for oid in recovery_ids):
            raise ContractValidationError("scale recovery entries must be unique and reference aligned observations.")
        if any(not isinstance(value, ScaleRecovery) for _, value in recovery):
            raise TypeError("scale recovery values must be ScaleRecovery contracts.")
        if self.aligned_bundle.depth_scale is DepthScale.METRIC and any(value.status is not ScaleRecoveryStatus.NOT_REQUIRED for _, value in recovery):
            raise ContractValidationError("Metric depth must use ScaleRecoveryStatus.NOT_REQUIRED.")
        object.__setattr__(self,"observation_ids",selected)
        object.__setattr__(self,"scale_recovery_by_observation",recovery)

def _scale_recovery(depth_scale: DepthScale) -> ScaleRecovery:
    if depth_scale is DepthScale.METRIC:
        return ScaleRecovery(ScaleRecoveryStatus.NOT_REQUIRED)
    return ScaleRecovery(ScaleRecoveryStatus.NOT_ATTEMPTED)

def _unavailable(*, observation_id, kind, status, reasons, warnings, bundle, scale_recovery, method="eligibility_policy"):
    level=ReliabilityLevel.REJECTED if status is MeasurementStatus.REJECTED else ReliabilityLevel.UNAVAILABLE
    return MeasurementEstimate(
        measurement_id=f"{observation_id}:{kind.value}", observation_id=observation_id, kind=kind,
        status=status, method=method, unit_si=unit_for_kind(kind),
        reliability=ReliabilityAssessment(level,policy_id="phase75-step6-eligibility-v1"),
        depth_scale=bundle.depth_scale, intrinsics_source=bundle.intrinsics.source,
        calibration_status=bundle.intrinsics.calibration_status, scale_recovery=scale_recovery,
        rejection_reasons=tuple(reasons), warnings=tuple(warnings),
    )

def _enforce_derived_volume_dependency(
    *,
    estimates: list[MeasurementEstimate],
    observation_id: str,
    bundle: AlignedFrameBundle,
    scale_recovery: ScaleRecovery,
    method: str,
) -> tuple[list[MeasurementEstimate], bool]:
    """Suppress derived volume when a required dimension is unavailable.

    Volume is computed from width × height × depth in Step 6. It therefore
    cannot remain available when any of those authoritative estimates is not
    available. Existing unavailable/rejected volume results are preserved.
    """
    by_kind = {estimate.kind: estimate for estimate in estimates}
    volume = by_kind.get(MeasurementKind.VOLUME)
    if volume is None or volume.status is not MeasurementStatus.AVAILABLE:
        return estimates, False

    required = (MeasurementKind.WIDTH, MeasurementKind.HEIGHT, MeasurementKind.DEPTH)
    missing = tuple(
        kind
        for kind in required
        if by_kind.get(kind) is None or by_kind[kind].status is not MeasurementStatus.AVAILABLE
    )
    if not missing:
        return estimates, False

    rejected_dependency = any(
        by_kind.get(kind) is not None
        and by_kind[kind].status is MeasurementStatus.REJECTED
        for kind in missing
    )
    replacement_status = (
        MeasurementStatus.REJECTED if rejected_dependency else MeasurementStatus.UNAVAILABLE
    )
    replacement = _unavailable(
        observation_id=observation_id,
        kind=MeasurementKind.VOLUME,
        status=replacement_status,
        reasons=(RejectionReason.MEASUREMENT_INPUT_UNAVAILABLE,),
        warnings=(
            "volume requires available width, height, and depth estimates; "
            "unavailable=" + ",".join(kind.value for kind in missing),
        ),
        bundle=bundle,
        scale_recovery=scale_recovery,
        method=method,
    )
    return [
        replacement if estimate.kind is MeasurementKind.VOLUME else estimate
        for estimate in estimates
    ], True


def compute_measurements(request: MeasurementComputationRequest) -> MeasurementBatchResult:
    started=time.perf_counter(); bundle=request.aligned_bundle; selected=set(request.observation_ids)
    extraction={r.observation_id:r for r in request.extraction_batch.object_results}
    fits={r.observation_id:r for r in request.fit_batch.object_results}
    results=[]
    recovery_map=dict(request.scale_recovery_by_observation)
    for observation in bundle.aligned_observations:
        oid=observation.observation_id
        if selected and oid not in selected: continue
        ext=extraction.get(oid); fit=fits.get(oid); estimates=[]; object_warnings=list(observation.warnings)
        if observation.border_contacts: object_warnings.append("object_touches_image_border")
        scale=recovery_map.get(oid, _scale_recovery(bundle.depth_scale))
        if ext is None or fit is None or ext.filtered_cloud is None or fit.status is ObjectGeometryFitStatus.REJECTED:
            for kind in request.policy.selection.requested_kinds:
                estimates.append(_unavailable(observation_id=oid,kind=kind,status=MeasurementStatus.REJECTED,reasons=(RejectionReason.MEASUREMENT_INPUT_UNAVAILABLE,),warnings=("required object geometry is unavailable",),bundle=bundle,scale_recovery=scale))
            result=ObjectMeasurementResult(oid,bundle.frame_reference,tuple(estimates),warnings=tuple(object_warnings))
            diag=MeasurementDiagnostics(oid,rejected_kinds=request.policy.selection.requested_kinds,warnings=("required object geometry is unavailable",))
            results.append(ObjectMeasurementComputationResult(oid,bundle.frame_reference,MeasurementComputationStatus.REJECTED,result,diag)); continue
        box, box_reason=select_measurement_box(fit,prefer_stable_oriented=request.policy.selection.prefer_stable_oriented_box)
        if box is None:
            for kind in request.policy.selection.requested_kinds:
                estimates.append(_unavailable(observation_id=oid,kind=kind,status=MeasurementStatus.REJECTED,reasons=(RejectionReason.FITTING_FAILED,),warnings=("no measurement box is available",),bundle=bundle,scale_recovery=scale))
            result=ObjectMeasurementResult(oid,bundle.frame_reference,tuple(estimates),warnings=tuple(object_warnings))
            diag=MeasurementDiagnostics(oid,rejected_kinds=request.policy.selection.requested_kinds,warnings=("no measurement box is available",))
            results.append(ObjectMeasurementComputationResult(oid,bundle.frame_reference,MeasurementComputationStatus.REJECTED,result,diag)); continue
        scale_factor = scale.scale_factor if scale.status is ScaleRecoveryStatus.SUCCEEDED and scale.scale_factor is not None else 1.0
        baseline=apply_scale_to_values(values_from_box(box), scale_factor)
        samples=generate_perturbation_samples(baseline_box=box,fit_result=fit,extraction_result=ext,bundle=bundle,observation=observation,policy=request.policy.perturbation)
        if scale_factor != 1.0:
            samples={kind: tuple(v * scale_factor_for_kind(kind, scale_factor) for v in values) for kind, values in samples.items()}
        provenance=unique_model_provenance((bundle.depth_model_provenance,*bundle.perception_model_provenance,*(p for p in (observation.mask_model_provenance,observation.detection_model_provenance) if p is not None)))
        available=[]; unavailable=[]; rejected=[]
        for kind in request.policy.selection.requested_kinds:
            status,reasons,eligibility_warnings=eligibility_for_kind(kind=kind,depth_scale=bundle.depth_scale,scale_recovery_succeeded=scale.status is ScaleRecoveryStatus.SUCCEEDED,border_contacts=observation.border_contacts,fit=fit,policy=request.policy.selection)
            if status is not MeasurementStatus.AVAILABLE:
                estimates.append(_unavailable(observation_id=oid,kind=kind,status=status,reasons=reasons,warnings=eligibility_warnings,bundle=bundle,scale_recovery=scale)); (rejected if status is MeasurementStatus.REJECTED else unavailable).append(kind); continue
            value=baseline[kind]
            uncertainty=estimate_uncertainty(kind=kind,value=value,samples=samples.get(kind,(value,)),bundle=bundle,observation=observation,extraction=ext,fit=fit,scale_recovery=scale,policy=request.policy.perturbation)
            if kind in {MeasurementKind.CENTRE_X, MeasurementKind.CENTRE_Y, MeasurementKind.CENTRE_Z}:
                reference=max(abs(baseline[MeasurementKind.CAMERA_DISTANCE]),1e-6)
            elif kind in {MeasurementKind.ORIENTATION_ROLL, MeasurementKind.ORIENTATION_PITCH, MeasurementKind.ORIENTATION_YAW}:
                reference=max(abs(value),0.25)
            else:
                reference=max(abs(value),1e-6)
            reliability=assess_reliability(kind=kind,value=value,uncertainty=uncertainty.expanded_uncertainty_si,support_points=ext.filtered_cloud.retained_point_count,fit=fit,warnings=tuple(eligibility_warnings),policy=request.policy.reliability,reference_scale=reference)
            if uncertainty.expanded_uncertainty_si/reference > request.policy.reliability.maximum_relative_uncertainty:
                estimates.append(_unavailable(observation_id=oid,kind=kind,status=MeasurementStatus.REJECTED,reasons=(RejectionReason.EXCESSIVE_UNCERTAINTY,),warnings=("uncertainty exceeds configured threshold",),bundle=bundle,scale_recovery=scale,method=box.fit_method)); rejected.append(kind); continue
            estimates.append(MeasurementEstimate(
                measurement_id=f"{oid}:{kind.value}",observation_id=oid,kind=kind,status=MeasurementStatus.AVAILABLE,
                method=box.fit_method,unit_si=unit_for_kind(kind),reliability=reliability,depth_scale=bundle.depth_scale,
                intrinsics_source=bundle.intrinsics.source,calibration_status=bundle.intrinsics.calibration_status,
                scale_recovery=scale,value_si=value,uncertainty=uncertainty,
                lower_bound_policy=request.policy.lower_bound_policy,valid_observation_count=1,
                model_provenance=provenance,processing_steps=tuple(ext.processing_steps)+tuple(fit.processing_steps),
                warnings=tuple(eligibility_warnings),
            )); available.append(kind)
        estimates, volume_dependency_changed = _enforce_derived_volume_dependency(
            estimates=estimates,
            observation_id=oid,
            bundle=bundle,
            scale_recovery=scale,
            method=box.fit_method,
        )
        if volume_dependency_changed:
            available = [kind for kind in available if kind is not MeasurementKind.VOLUME]
            volume_status = next(
                estimate.status
                for estimate in estimates
                if estimate.kind is MeasurementKind.VOLUME
            )
            unavailable = [kind for kind in unavailable if kind is not MeasurementKind.VOLUME]
            rejected = [kind for kind in rejected if kind is not MeasurementKind.VOLUME]
            if volume_status is MeasurementStatus.REJECTED:
                rejected.append(MeasurementKind.VOLUME)
            else:
                unavailable.append(MeasurementKind.VOLUME)

        boxes=tuple(b for b in (fit.axis_aligned_box,fit.camera_aligned_box,fit.oriented_box) if b is not None)
        shapes=(fit.generic_shape_model,) if fit.generic_shape_model is not None else ()
        measurement_result=ObjectMeasurementResult(oid,bundle.frame_reference,tuple(estimates),boxes,shapes,warnings=tuple(dict.fromkeys(object_warnings)))
        status=MeasurementComputationStatus.SUCCESS if len(available)==len(estimates) else MeasurementComputationStatus.PARTIAL if available else MeasurementComputationStatus.REJECTED
        diag=MeasurementDiagnostics(oid,box.box_id,box_reason,max((len(v) for v in samples.values()),default=0),tuple(available),tuple(unavailable),tuple(rejected),tuple(object_warnings))
        results.append(ObjectMeasurementComputationResult(oid,bundle.frame_reference,status,measurement_result,diag,warnings=tuple(object_warnings)))
    out=tuple(results); duration=(time.perf_counter()-started)*1000
    step=ProcessingStep(step_id=f"frame:{bundle.frame_reference.frame_id}:measurement",method="heuristic_perturbation_measurement",input_ids=tuple(r.observation_id for r in out),output_ids=tuple(f"{r.observation_id}:measurements" for r in out),duration_ms=duration,parameters={"object_count":len(out),"uncertainty_basis":"heuristic_perturbation","empirically_calibrated":False})
    return MeasurementBatchResult(bundle.frame_reference,out,derive_measurement_batch_status(out),(step,),tuple(f"{r.observation_id}:{w}" for r in out for w in r.warnings))
