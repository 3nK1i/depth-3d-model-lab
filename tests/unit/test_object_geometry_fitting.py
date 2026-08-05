import pickle
import numpy as np
from core.models.metadata import DepthScale
from core.object_geometry import *
from core.object_geometry.boxes import fit_axis_aligned,fit_camera_aligned,fit_oriented
from core.object_geometry.principal_axes import fit_principal_axes
from core.object_geometry.robust_statistics import robust_centre,trim_extreme_tails
from tests.object_geometry_fitting_helpers import cuboid_points,extraction_batch,rotation_z

def test_policy_validation():
    try: BoxFitPolicy(lower_percentile=90,upper_percentile=10); assert False
    except ValueError: pass

def test_robust_centre_ignores_outlier():
    p=np.vstack([np.zeros((20,3)),[[100,100,100]]]); assert np.allclose(robust_centre(p),0)

def test_tail_trim():
    p=np.vstack([np.zeros((40,3)),[[100,0,0]]]); assert trim_extreme_tails(p).sum()==40

def test_axis_aligned_exact_bounds():
    p=cuboid_points(angle=0); b,d=fit_axis_aligned(p,"o","c"); assert np.allclose(np.sort(b.extents_xyz_m),np.sort([2,1,.5]),atol=1e-8); assert d.excluded_tail_count==0

def test_pca_orientation_right_handed():
    cloud=extraction_batch().object_results[0].filtered_cloud; a=fit_principal_axes(cloud); assert np.allclose(a.orientation_matrix.T@a.orientation_matrix,np.eye(3)); assert np.linalg.det(a.orientation_matrix)>0

def test_deterministic_pca():
    c=extraction_batch().object_results[0].filtered_cloud; a=fit_principal_axes(c); b=fit_principal_axes(c); assert np.allclose(a.orientation_matrix,b.orientation_matrix)

def test_oriented_box_recovers_extents():
    batch=extraction_batch(cuboid_points(angle=.6)); r=fit_object_geometry(make_fitting_request(batch)).object_results[0]; assert r.oriented_box is not None; assert np.allclose(np.sort(r.oriented_box.extents_xyz_m),np.sort([2,1,.5]),atol=.08)

def test_pipeline_metric_ready():
    r=fit_object_geometry(make_fitting_request(extraction_batch())).object_results[0]; assert r.metric_ready and r.generic_shape_model is not None

def test_relative_not_metric_ready():
    b=extraction_batch(depth_scale=DepthScale.RELATIVE,semantics=PointCoordinateSemantics.RELATIVE_MODEL_UNITS); r=fit_object_geometry(make_fitting_request(b)).object_results[0]; assert not r.metric_ready

def test_pickle_roundtrip():
    r=fit_object_geometry(make_fitting_request(extraction_batch())); q=pickle.loads(pickle.dumps(r)); assert q.object_results[0].preferred_box_id==r.object_results[0].preferred_box_id

def test_readonly_arrays():
    r=fit_object_geometry(make_fitting_request(extraction_batch())).object_results[0]; assert not r.principal_axes.orientation_matrix.flags.writeable

def test_object_transform_round_trip():
    r=fit_object_geometry(make_fitting_request(extraction_batch())).object_results[0]; M=r.object_frame.object_to_camera_transform.matrix; local=np.array([0.,0.,0.,1.]); assert np.allclose((M@local)[:3],r.principal_axes.robust_centre_xyz)

def test_planar_degeneracy():
    x,y=np.meshgrid(np.linspace(-1,1,20),np.linspace(-1,1,20)); p=np.c_[x.ravel(),y.ravel(),np.zeros(x.size)]; a=fit_principal_axes(extraction_batch(p).object_results[0].filtered_cloud); assert a.degeneracy is GeometryDegeneracy.PLANAR

def test_linear_degeneracy():
    p=np.c_[np.linspace(0,1,100),np.zeros(100),np.zeros(100)]; a=fit_principal_axes(extraction_batch(p).object_results[0].filtered_cloud); assert a.degeneracy is GeometryDegeneracy.LINEAR

def test_too_few_rejected():
    b=extraction_batch(np.random.default_rng(0).normal(size=(10,3))); r=fit_object_geometry(make_fitting_request(b)).object_results[0]; assert r.status is ObjectGeometryFitStatus.REJECTED

def test_multi_object_filter_request():
    b=extraction_batch(); req=make_fitting_request(b,observation_ids=("missing",)); out=fit_object_geometry(req); assert out.status is ObjectGeometryFitStatus.REJECTED and not out.object_results

def test_no_input_mutation():
    b=extraction_batch(); before=b.object_results[0].filtered_cloud.points_xyz_m.copy(); fit_object_geometry(make_fitting_request(b)); assert np.array_equal(before,b.object_results[0].filtered_cloud.points_xyz_m)
