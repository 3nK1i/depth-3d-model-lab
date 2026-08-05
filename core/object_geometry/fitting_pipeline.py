"""Step 5 object-geometry fitting pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from core.measurement.contracts import RejectionReason
from .contracts import ObjectPointExtractionBatchResult, ObjectPointExtractionStatus
from .fitting_contracts import ObjectGeometryFitBatchResult,ObjectGeometryFitResult,ObjectGeometryFitStatus,OrientationStability,derive_fit_batch_status
from .fitting_policy import ObjectGeometryFittingPolicy
from .principal_axes import fit_principal_axes
from .frames import make_object_frame
from .boxes import fit_axis_aligned,fit_camera_aligned,fit_oriented
from .shape_fitting import make_generic_box_shape

@dataclass(frozen=True, slots=True)
class ObjectGeometryFitRequest:
    extraction_batch: ObjectPointExtractionBatchResult
    observation_ids: tuple[str,...]=()
    policy: ObjectGeometryFittingPolicy=field(default_factory=ObjectGeometryFittingPolicy)

def fit_object_geometry(request):
    wanted=set(request.observation_ids); results=[]
    for extracted in request.extraction_batch.object_results:
        if wanted and extracted.observation_id not in wanted: continue
        if extracted.status is ObjectPointExtractionStatus.REJECTED or extracted.filtered_cloud is None:
            results.append(ObjectGeometryFitResult(extracted.observation_id,extracted.frame_reference,ObjectGeometryFitStatus.REJECTED,extracted.coordinate_semantics,False,rejection_reasons=(RejectionReason.FITTING_FAILED,),warnings=("point extraction was rejected",))); continue
        cloud=extracted.filtered_cloud
        try:
            axes=fit_principal_axes(cloud,request.policy.pca); frame=make_object_frame(extracted.coordinate_frame,axes); boxes=[]; diags=[]
            aabb=cabb=obb=None
            if request.policy.boxes.fit_axis_aligned: aabb,d=fit_axis_aligned(cloud.points_xyz_m,cloud.observation_id,cloud.coordinate_frame_id); boxes.append(aabb); diags.append(d)
            if request.policy.boxes.fit_camera_aligned: cabb,d=fit_camera_aligned(cloud.points_xyz_m,cloud.observation_id,cloud.coordinate_frame_id,request.policy.boxes.lower_percentile,request.policy.boxes.upper_percentile); boxes.append(cabb); diags.append(d)
            overall=frame.orientation_stability
            if request.policy.boxes.fit_oriented and (request.policy.boxes.allow_ambiguous_oriented_box or overall is not OrientationStability.AMBIGUOUS): obb,d=fit_oriented(cloud.points_xyz_m,cloud.observation_id,cloud.coordinate_frame_id,axes.orientation_matrix,request.policy.boxes.lower_percentile,request.policy.boxes.upper_percentile); boxes.append(obb); diags.append(d)
            preferred=obb if obb is not None and overall is OrientationStability.STABLE else (cabb or aabb or obb)
            if preferred is None: raise ValueError("no valid box fitted")
            status=ObjectGeometryFitStatus.SUCCESS if obb is not None and overall is OrientationStability.STABLE else ObjectGeometryFitStatus.PARTIAL
            reason="stable oriented box" if preferred is obb and status is ObjectGeometryFitStatus.SUCCESS else "camera-aligned fallback for ambiguous or partial orientation"
            shape=make_generic_box_shape(preferred,coordinate_semantics=extracted.coordinate_semantics,metric_ready=cloud.metric_ready,degeneracy=axes.degeneracy,orientation_stability=overall) if request.policy.shape.enabled else None
            results.append(ObjectGeometryFitResult(cloud.observation_id,cloud.frame_reference,status,extracted.coordinate_semantics,cloud.metric_ready,axes,frame,aabb,cabb,obb,preferred.box_id,reason,shape,tuple(diags),warnings=axes.warnings))
        except Exception as exc:
            results.append(ObjectGeometryFitResult(extracted.observation_id,extracted.frame_reference,ObjectGeometryFitStatus.REJECTED,extracted.coordinate_semantics,False,rejection_reasons=(RejectionReason.FITTING_FAILED,),warnings=(str(exc),)))
    out=tuple(results); return ObjectGeometryFitBatchResult(request.extraction_batch.frame_reference,out,derive_fit_batch_status(out))
