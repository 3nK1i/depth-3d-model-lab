import numpy as np
from core.object_geometry import fit_object_geometry,make_fitting_request
from tests.object_geometry_fitting_helpers import extraction_batch,cuboid_points

def test_repeated_determinism():
    b=extraction_batch(); values=[fit_object_geometry(make_fitting_request(b)).object_results[0].preferred_box_id for _ in range(30)]; assert len(set(values))==1

def test_rotations():
    for a in np.linspace(0,1.4,20):
        r=fit_object_geometry(make_fitting_request(extraction_batch(cuboid_points(angle=float(a))))).object_results[0]
        assert r.preferred_box_id

def test_outlier_load():
    r=fit_object_geometry(make_fitting_request(extraction_batch(cuboid_points(outliers=100)))).object_results[0]; assert r.preferred_box_id
