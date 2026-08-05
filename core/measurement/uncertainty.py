"""Explainable component-wise heuristic uncertainty estimation."""
from __future__ import annotations
import math
import numpy as np
from core.alignment.contracts import AlignedFrameBundle, AlignedObjectObservation
from core.measurement.contracts import (
    MeasurementKind, ReliabilityLevel, ScaleRecovery, ScaleRecoveryStatus, UncertaintyBasis, UncertaintyComponent,
    UncertaintyComponentKind, UncertaintyEstimate,
)
from core.measurement.geometry import is_orientation_kind
from core.measurement.policy import PerturbationPolicy
from core.object_geometry.contracts import ObjectPointExtractionResult
from core.object_geometry.fitting_contracts import ObjectGeometryFitResult
from core.models.metadata import ConfidenceKind


def _component(kind, contribution, method, count=0, note=""):
    if not np.isfinite(contribution) or contribution <= 0:
        return None
    return UncertaintyComponent(kind, float(contribution), method, sample_count=int(count), note=note)

def _sample_spread(samples: tuple[float,...], baseline: float) -> float:
    if len(samples) <= 1: return 0.0
    values=np.asarray(samples,float)
    return float(max(np.percentile(np.abs(values-baseline), 90), np.std(values)))

def estimate_uncertainty(
    *, kind: MeasurementKind, value: float, samples: tuple[float,...], bundle: AlignedFrameBundle,
    observation: AlignedObjectObservation, extraction: ObjectPointExtractionResult,
    fit: ObjectGeometryFitResult, scale_recovery: ScaleRecovery, policy: PerturbationPolicy,
) -> UncertaintyEstimate:
    components=[]
    fitting=_sample_spread(samples,value)
    c=_component(UncertaintyComponentKind.FITTING,fitting,"deterministic_repeated_box_fitting",len(samples));
    if c: components.append(c)
    if not is_orientation_kind(kind):
        if bundle.confidence_kind is ConfidenceKind.UNCERTAINTY and bundle.confidence is not None and observation.aligned_mask is not None:
            vals=bundle.confidence[observation.aligned_mask & bundle.confidence_valid_mask]
            depth=float(np.median(vals)) if vals.size else abs(value)*policy.fallback_depth_relative_uncertainty
            method="median_model_uncertainty"
        elif bundle.confidence_kind is ConfidenceKind.CONFIDENCE and bundle.confidence is not None and observation.aligned_mask is not None:
            vals=bundle.confidence[observation.aligned_mask & bundle.confidence_valid_mask]
            conf=float(np.median(vals)) if vals.size else 0.0
            depth=abs(value)*(1-conf)*policy.confidence_relative_scale
            method="confidence_to_uncertainty_heuristic"
        else:
            depth=abs(value)*policy.fallback_depth_relative_uncertainty
            method="fallback_relative_depth_perturbation"
        c=_component(UncertaintyComponentKind.DEPTH,depth,method)
        if c: components.append(c)
    if kind in {MeasurementKind.WIDTH,MeasurementKind.HEIGHT,MeasurementKind.DEPTH,MeasurementKind.VOLUME}:
        mask_scale=0.0
        z=float(np.median(extraction.filtered_cloud.points_xyz_m[:,2])) if extraction.filtered_cloud is not None else 0.0
        fx,fy=float(bundle.intrinsics.matrix[0,0]),float(bundle.intrinsics.matrix[1,1])
        if kind is MeasurementKind.WIDTH: mask_scale=2*policy.mask_boundary_pixels*z/fx
        elif kind is MeasurementKind.HEIGHT: mask_scale=2*policy.mask_boundary_pixels*z/fy
        elif kind is MeasurementKind.DEPTH and extraction.filtered_cloud is not None: mask_scale=float(np.median(np.abs(extraction.filtered_cloud.points_xyz_m[:,2]-z)))
        elif kind is MeasurementKind.VOLUME: mask_scale=abs(value)*policy.mask_boundary_pixels*0.01
        c=_component(UncertaintyComponentKind.MASK_BOUNDARY,mask_scale,"one_pixel_mask_boundary_perturbation")
        if c: components.append(c)
    if kind in {MeasurementKind.WIDTH,MeasurementKind.HEIGHT,MeasurementKind.CENTRE_X,MeasurementKind.CENTRE_Y,MeasurementKind.VOLUME}:
        intr=abs(value)*policy.focal_length_relative_perturbation
        c=_component(UncertaintyComponentKind.INTRINSICS,intr,"estimated_intrinsics_parameter_perturbation")
        if c: components.append(c)
    raw,filtered=extraction.raw_cloud,extraction.filtered_cloud
    if raw is not None and filtered is not None and raw.retained_point_count>filtered.retained_point_count:
        fraction=(raw.retained_point_count-filtered.retained_point_count)/raw.retained_point_count
        c=_component(UncertaintyComponentKind.POINT_CLOUD_OUTLIERS,abs(value)*min(fraction,0.25),"raw_vs_filtered_support_sensitivity")
        if c: components.append(c)
    if scale_recovery.status is ScaleRecoveryStatus.SUCCEEDED and not is_orientation_kind(kind):
        fraction = {ReliabilityLevel.HIGH: 0.01, ReliabilityLevel.MEDIUM: 0.03, ReliabilityLevel.LOW: 0.08}.get(scale_recovery.reliability, 0.10)
        c=_component(UncertaintyComponentKind.SCALE,abs(value)*fraction,"scale_recovery_reliability_heuristic",note=scale_recovery.method)
        if c: components.append(c)
    if observation.border_contacts and kind not in {MeasurementKind.WIDTH,MeasurementKind.HEIGHT,MeasurementKind.VOLUME}:
        c=_component(UncertaintyComponentKind.BORDER_CONTACT,abs(value)*policy.border_contact_relative_uncertainty,"visible_geometry_border_contact",note=','.join(observation.border_contacts))
        if c: components.append(c)
    if any('occlusion' in w.lower() for w in observation.warnings):
        c=_component(UncertaintyComponentKind.OCCLUSION,abs(value)*policy.occlusion_relative_uncertainty,"observation_occlusion_warning")
        if c: components.append(c)
    expanded=math.sqrt(sum(c.contribution_si**2 for c in components))
    expanded=max(expanded,policy.minimum_uncertainty_si)
    return UncertaintyEstimate(
        basis=UncertaintyBasis.HEURISTIC_PERTURBATION,
        expanded_uncertainty_si=expanded,
        components=tuple(components),
        perturbation_sample_count=len(samples),
        note="Estimated from deterministic perturbation and sensitivity components; not empirically calibrated.",
    )
