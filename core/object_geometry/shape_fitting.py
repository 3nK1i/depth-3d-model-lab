"""Generic box shape-model construction."""
from __future__ import annotations
from core.measurement.contracts import ShapeKind, ShapeModel

def make_generic_box_shape(box, *, coordinate_semantics, metric_ready, degeneracy, orientation_stability):
    return ShapeModel(shape_model_id=f"{box.observation_id}:generic-box",observation_id=box.observation_id,coordinate_frame_id=box.coordinate_frame_id,kind=ShapeKind.GENERIC_BOX,fit_method="preferred_3d_box",fit_status="success",support_point_count=box.support_point_count,parameters={"selected_box_id":box.box_id,"selected_box_kind":box.kind.value,"coordinate_semantics":coordinate_semantics.value,"metric_ready":bool(metric_ready),"centre_xyz":box.centre_xyz_m.tolist(),"extents_xyz":box.extents_xyz_m.tolist(),"orientation":box.orientation_matrix.tolist(),"degeneracy":degeneracy.value,"orientation_stability":orientation_stability.value},fit_residual_m=box.fit_residual_m)
