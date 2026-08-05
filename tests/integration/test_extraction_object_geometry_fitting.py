from core.object_geometry import fit_object_geometry,make_fitting_request,ObjectGeometryFitStatus
from tests.object_geometry_fitting_helpers import extraction_batch

def test_extraction_to_fitting():
    result=fit_object_geometry(make_fitting_request(extraction_batch()))
    assert result.status in {ObjectGeometryFitStatus.SUCCESS,ObjectGeometryFitStatus.PARTIAL}
    assert result.object_results[0].preferred_box_id
