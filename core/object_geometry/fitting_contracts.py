"""Immutable Step 5 fitting contracts."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from core.contracts.common import FrameReference, ProcessingStep
from core.contracts.spatial import CoordinateFrame, RigidTransform3D
from core.contracts.validation import ContractValidationError, readonly_array, require_nonempty
from core.measurement.contracts import BoundingBox3D, ShapeModel, RejectionReason
from core.object_geometry.contracts import PointCoordinateSemantics

class ObjectGeometryFitStatus(str, Enum):
    SUCCESS="success"; PARTIAL="partial"; REJECTED="rejected"
class GeometryDegeneracy(str, Enum):
    NONE="none"; PLANAR="planar"; LINEAR="linear"; POINT_LIKE="point_like"; UNKNOWN="unknown"
class OrientationStability(str, Enum):
    STABLE="stable"; WEAK="weak"; AMBIGUOUS="ambiguous"; UNAVAILABLE="unavailable"

@dataclass(frozen=True, slots=True)
class PrincipalAxesEstimate:
    observation_id: str
    coordinate_frame_id: str
    robust_centre_xyz: np.ndarray
    centroid_xyz: np.ndarray
    orientation_matrix: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_ratios: np.ndarray
    eigenvalue_gap_ratios: np.ndarray
    axis_stability: tuple[OrientationStability, OrientationStability, OrientationStability]
    degeneracy: GeometryDegeneracy
    support_point_count: int
    method: str = "robust_weighted_pca"
    processing_steps: tuple[ProcessingStep,...]=()
    warnings: tuple[str,...]=()
    def __post_init__(self):
        object.__setattr__(self,"observation_id",require_nonempty(self.observation_id,field_name="observation_id"))
        object.__setattr__(self,"coordinate_frame_id",require_nonempty(self.coordinate_frame_id,field_name="coordinate_frame_id"))
        for n,s in (("robust_centre_xyz",(3,)),("centroid_xyz",(3,)),("eigenvalues",(3,)),("explained_variance_ratios",(3,)),("eigenvalue_gap_ratios",(2,))):
            a=readonly_array(getattr(self,n),field_name=n,dtype=np.float64,shape=s,finite=True); object.__setattr__(self,n,a)
        R=readonly_array(self.orientation_matrix,field_name="orientation_matrix",dtype=np.float64,shape=(3,3),finite=True)
        if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1,atol=1e-6): raise ContractValidationError("orientation_matrix must be right-handed orthonormal")
        if np.any(self.eigenvalues < -1e-12) or np.any(np.diff(self.eigenvalues)>1e-10): raise ContractValidationError("eigenvalues must be nonnegative and descending")
        if not np.isclose(float(self.explained_variance_ratios.sum()),1.0,atol=1e-6): raise ContractValidationError("explained variance ratios must sum to one")
        if self.support_point_count < 3: raise ContractValidationError("support_point_count must be >= 3")
        object.__setattr__(self,"orientation_matrix",R); object.__setattr__(self,"warnings",tuple(str(x) for x in self.warnings))

@dataclass(frozen=True, slots=True)
class ObjectFrameEstimate:
    camera_frame: CoordinateFrame
    object_frame: CoordinateFrame
    object_to_camera_transform: RigidTransform3D
    frame_method: str
    orientation_stability: OrientationStability
    degeneracy: GeometryDegeneracy
    def __post_init__(self):
        if self.object_frame.parent_frame_id != self.camera_frame.frame_id: raise ContractValidationError("object frame parent must be camera frame")
        if self.object_to_camera_transform.source_frame_id != self.object_frame.frame_id or self.object_to_camera_transform.destination_frame_id != self.camera_frame.frame_id: raise ContractValidationError("object_to_camera transform frame mismatch")

@dataclass(frozen=True, slots=True)
class BoxFitDiagnostics:
    box_id: str
    lower_percentile: float
    upper_percentile: float
    support_point_count: int
    inside_point_count: int
    excluded_tail_count: int
    fit_residual: float | None
    minimum_local_xyz: np.ndarray
    maximum_local_xyz: np.ndarray
    warnings: tuple[str,...]=()
    def __post_init__(self):
        object.__setattr__(self,"box_id",require_nonempty(self.box_id,field_name="box_id"))
        lo=readonly_array(self.minimum_local_xyz,field_name="minimum_local_xyz",dtype=np.float64,shape=(3,),finite=True)
        hi=readonly_array(self.maximum_local_xyz,field_name="maximum_local_xyz",dtype=np.float64,shape=(3,),finite=True)
        if np.any(hi<=lo): raise ContractValidationError("box local bounds must have positive extents")
        if self.inside_point_count+self.excluded_tail_count != self.support_point_count: raise ContractValidationError("diagnostic counts disagree")
        if self.fit_residual is not None and self.fit_residual < 0: raise ContractValidationError("fit_residual must be nonnegative")
        object.__setattr__(self,"minimum_local_xyz",lo); object.__setattr__(self,"maximum_local_xyz",hi)

@dataclass(frozen=True, slots=True)
class ObjectGeometryFitResult:
    observation_id: str
    frame_reference: FrameReference
    status: ObjectGeometryFitStatus
    coordinate_semantics: PointCoordinateSemantics
    metric_ready: bool
    principal_axes: PrincipalAxesEstimate|None=None
    object_frame: ObjectFrameEstimate|None=None
    axis_aligned_box: BoundingBox3D|None=None
    camera_aligned_box: BoundingBox3D|None=None
    oriented_box: BoundingBox3D|None=None
    preferred_box_id: str=""
    preferred_box_reason: str=""
    generic_shape_model: ShapeModel|None=None
    diagnostics: tuple[BoxFitDiagnostics,...]=()
    rejection_reasons: tuple[RejectionReason,...]=()
    processing_steps: tuple[ProcessingStep,...]=()
    warnings: tuple[str,...]=()
    def __post_init__(self):
        object.__setattr__(self,"observation_id",require_nonempty(self.observation_id,field_name="observation_id"))
        boxes=tuple(b for b in (self.axis_aligned_box,self.camera_aligned_box,self.oriented_box) if b is not None)
        if self.status is ObjectGeometryFitStatus.REJECTED:
            if boxes or self.preferred_box_id or self.generic_shape_model is not None: raise ContractValidationError("rejected fit cannot expose geometry")
            if not self.rejection_reasons: raise ContractValidationError("rejected fit requires reasons")
        else:
            if not boxes: raise ContractValidationError("successful/partial fit requires a box")
            if self.preferred_box_id not in {b.box_id for b in boxes}: raise ContractValidationError("preferred_box_id must reference a fitted box")
        object.__setattr__(self,"diagnostics",tuple(self.diagnostics)); object.__setattr__(self,"warnings",tuple(str(x) for x in self.warnings))

@dataclass(frozen=True, slots=True)
class ObjectGeometryFitBatchResult:
    frame_reference: FrameReference
    object_results: tuple[ObjectGeometryFitResult,...]
    status: ObjectGeometryFitStatus
    warnings: tuple[str,...]=()
    def __post_init__(self):
        ids=[r.observation_id for r in self.object_results]
        if len(ids)!=len(set(ids)): raise ContractValidationError("fit batch observation IDs must be unique")
        if any(r.frame_reference != self.frame_reference for r in self.object_results): raise ContractValidationError("fit batch frame mismatch")
        expected=derive_fit_batch_status(self.object_results)
        if self.status is not expected: raise ContractValidationError("fit batch status disagrees with object results")

def derive_fit_batch_status(results):
    results=tuple(results)
    if not results or all(r.status is ObjectGeometryFitStatus.REJECTED for r in results): return ObjectGeometryFitStatus.REJECTED
    if all(r.status is ObjectGeometryFitStatus.SUCCESS for r in results): return ObjectGeometryFitStatus.SUCCESS
    return ObjectGeometryFitStatus.PARTIAL
