from __future__ import annotations

from core.models.metadata import DepthScale
from core.object_geometry.contracts import PointCoordinateSemantics
from core.object_geometry.factories import (
    coordinate_semantics_for_depth_scale,
    make_depth_camera_frame,
    make_extraction_request,
)
from tests.object_geometry_helpers import make_bundle


def test_factories_create_depth_camera_request_and_semantics():
    bundle = make_bundle()
    frame = make_depth_camera_frame(bundle)
    request = make_extraction_request(bundle)
    assert frame.source_frame == bundle.frame_reference
    assert request.aligned_bundle is bundle
    assert coordinate_semantics_for_depth_scale(DepthScale.METRIC) is PointCoordinateSemantics.METRIC_METRES
    assert coordinate_semantics_for_depth_scale(DepthScale.RELATIVE) is PointCoordinateSemantics.RELATIVE_MODEL_UNITS
