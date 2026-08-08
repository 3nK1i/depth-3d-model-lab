from __future__ import annotations

import numpy as np

from core.reconstruction import (
    FeatureTolerance,
    InspectionDecision,
    SemanticGeometryRole,
    build_reference_assembly,
    inspect_semantic_geometry,
)
from scripts.run_registration_inspection_smoke import (
    _rot_z,
    _semantic_fixture,
    _transform_candidate,
)


def test_semantic_surface_to_reference_registration_and_connector_qc_pipeline() -> None:
    surface, semantic, _ = _semantic_fixture()
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="integration-golden",
        default_tolerance=FeatureTolerance(
            position_warn=0.08,
            position_reject=0.16,
            orientation_warn_degrees=5.0,
            orientation_reject_degrees=12.0,
            extent_relative_warn=0.08,
            extent_relative_reject=0.16,
        ),
        registration_anchor_feature_ids=("connector-main",),
    )
    candidate_surface, candidate_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=1.2,
        rotation=_rot_z(15.0),
        translation=np.asarray([0.5, -0.2, 0.2]),
        bend_top_degrees=18.0,
        shift_top_reference=np.asarray([0.0, -0.22, 0.0]),
        remove_bottom=True,
    )
    report = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic)
    top = next(result for result in report.feature_results if result.reference_feature_id == "pin-top")
    bottom = next(result for result in report.feature_results if result.reference_feature_id == "pin-bottom")
    body = next(result for result in report.feature_results if result.semantic_role is SemanticGeometryRole.CONNECTOR_BODY)

    assert report.registration.status.value == "success"
    assert body.decision is InspectionDecision.PASS
    assert top.decision is InspectionDecision.REJECT
    assert set(top.connector_defects) >= {"bent_pin", "misaligned_pin"}
    assert bottom.decision is InspectionDecision.REJECT
    assert "missing_pin" in bottom.connector_defects
    assert report.decision is InspectionDecision.REJECT
    assert not report.certified_metrology
