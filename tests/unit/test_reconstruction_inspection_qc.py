from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest

from core.reconstruction import (
    FeatureTolerance,
    InspectionDecision,
    InspectionFindingKind,
    MetricGeometryStatus,
    ReferenceGeometrySource,
    RegistrationInspectionPolicy,
    RegistrationMethod,
    RegistrationStatus,
    ReconstructionValidationError,
    SemanticGeometryRole,
    SfmScaleStatus,
    build_reference_assembly,
    inspect_semantic_geometry,
)
from scripts.run_registration_inspection_smoke import (
    _metric_variant,
    _rot_z,
    _semantic_fixture,
    _transform_candidate,
)


@lru_cache(maxsize=1)
def _fixture():
    return _semantic_fixture()


def _tolerance() -> FeatureTolerance:
    return FeatureTolerance(
        position_warn=0.08,
        position_reject=0.16,
        orientation_warn_degrees=5.0,
        orientation_reject_degrees=12.0,
        extent_relative_warn=0.08,
        extent_relative_reject=0.16,
        minimum_geometry_quality=0.60,
        maximum_uncertainty_fraction=0.30,
    )


def _reference():
    surface, semantic, _ = _fixture()
    return build_reference_assembly(
        surface,
        semantic,
        reference_id="unit-golden",
        source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
    )


def _candidate(**kwargs):
    surface, semantic, _ = _fixture()
    return _transform_candidate(
        surface,
        semantic,
        scale_factor=kwargs.pop("scale_factor", 1.25),
        rotation=kwargs.pop("rotation", _rot_z(13.0)),
        translation=kwargs.pop("translation", np.asarray([0.3, -0.2, 0.1])),
        **kwargs,
    )


def test_feature_tolerance_rejects_bad_threshold_order() -> None:
    with pytest.raises(ReconstructionValidationError):
        FeatureTolerance(position_warn=0.2, position_reject=0.1)
    with pytest.raises(ReconstructionValidationError):
        FeatureTolerance(position_warn=0.1, position_reject=0.2, minimum_geometry_quality=1.1)


def test_registration_policy_is_bounded() -> None:
    policy = RegistrationInspectionPolicy()
    assert policy.maximum_reference_features >= 1
    assert policy.maximum_candidate_objects >= 1
    assert 0.0 <= policy.minimum_registration_anchor_quality <= 1.0


def test_registration_policy_rejects_invalid_extra_decision() -> None:
    with pytest.raises(ReconstructionValidationError):
        RegistrationInspectionPolicy(extra_feature_decision=InspectionDecision.PASS)


def test_reference_builder_uses_track_ids_as_default_feature_keys() -> None:
    reference = _reference()
    assert {feature.feature_id for feature in reference.features} == {
        "connector-main",
        "pin-top",
        "pin-bottom",
    }


def test_reference_builder_marks_explicit_anchor() -> None:
    reference = _reference()
    anchor = next(feature for feature in reference.features if feature.registration_anchor)
    assert anchor.feature_id == "connector-main"
    assert anchor.orientation_reliable
    assert len(anchor.registration_points_reference) >= 3


def test_reference_builder_preserves_arbitrary_scale_semantics() -> None:
    reference = _reference()
    assert reference.metric_geometry_status is MetricGeometryStatus.ARBITRARY
    assert reference.unit_to_metre is None
    assert reference.relative_scale_uncertainty is None
    assert not reference.certified_metrology


def test_reference_builder_rejects_unknown_anchor_id() -> None:
    surface, semantic, _ = _fixture()
    with pytest.raises(ReconstructionValidationError):
        build_reference_assembly(
            surface,
            semantic,
            reference_id="bad-anchor",
            default_tolerance=_tolerance(),
            registration_anchor_feature_ids=("missing",),
        )


def test_rigidly_transformed_arbitrary_candidate_passes_relative_qc() -> None:
    surface, semantic = _candidate()
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert report.decision is InspectionDecision.PASS
    assert report.registration.status is RegistrationStatus.SUCCESS
    assert report.registration.method is RegistrationMethod.ORIENTED_SEMANTIC_ANCHOR
    assert not report.comparison_metric_available
    assert report.reference_unit_to_metre is None


def test_arbitrary_registration_estimates_relative_scale() -> None:
    scale_factor = 1.40
    surface, semantic = _candidate(scale_factor=scale_factor)
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert report.registration.scale_candidate_to_reference == pytest.approx(1.0 / scale_factor, rel=0.03)


def test_clean_candidate_has_only_within_tolerance_findings() -> None:
    surface, semantic = _candidate()
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert all(result.decision is InspectionDecision.PASS for result in report.feature_results)
    assert all(result.findings == (InspectionFindingKind.WITHIN_TOLERANCE,) for result in report.feature_results)


def test_missing_required_pin_is_rejected() -> None:
    surface, semantic = _candidate(remove_bottom=True)
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    bottom = next(result for result in report.feature_results if result.reference_feature_id == "pin-bottom")
    assert report.decision is InspectionDecision.REJECT
    assert bottom.decision is InspectionDecision.REJECT
    assert InspectionFindingKind.MISSING_FEATURE in bottom.findings
    assert "missing_pin" in bottom.connector_defects


def test_misaligned_pin_is_rejected_and_labeled() -> None:
    surface, semantic = _candidate(shift_top_reference=np.asarray([0.0, -0.24, 0.0]))
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    top = next(result for result in report.feature_results if result.reference_feature_id == "pin-top")
    assert top.decision is InspectionDecision.REJECT
    assert InspectionFindingKind.POSITION_DEVIATION in top.findings
    assert "misaligned_pin" in top.connector_defects


def test_bent_pin_is_rejected_and_labeled() -> None:
    surface, semantic = _candidate(bend_top_degrees=20.0)
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    top = next(result for result in report.feature_results if result.reference_feature_id == "pin-top")
    assert top.decision is InspectionDecision.REJECT
    assert InspectionFindingKind.ORIENTATION_DEVIATION in top.findings
    assert "bent_pin" in top.connector_defects


def test_low_quality_required_feature_is_insufficient_evidence_not_pass() -> None:
    surface, semantic = _candidate(low_quality_top=True)
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    top = next(result for result in report.feature_results if result.reference_feature_id == "pin-top")
    assert report.decision is InspectionDecision.INSUFFICIENT_EVIDENCE
    assert top.decision is InspectionDecision.INSUFFICIENT_EVIDENCE
    assert InspectionFindingKind.LOW_GEOMETRY_QUALITY in top.findings


def test_explicit_candidate_key_override_supports_cross_session_identity() -> None:
    surface, semantic = _candidate()
    top = next(obj for obj in semantic.objects if "pin-top" in obj.track_ids)
    changed = replace(top, track_ids=("new-session-track",))
    semantic = replace(
        semantic,
        objects=tuple(changed if obj.object_id == top.object_id else obj for obj in semantic.objects),
    )
    report = inspect_semantic_geometry(
        _reference(),
        surface,
        semantic,
        candidate_key_by_object_id={top.object_id: "pin-top"},
    )
    assert report.decision is InspectionDecision.PASS


def test_duplicate_candidate_association_keys_are_rejected() -> None:
    surface, semantic = _candidate()
    objects = semantic.objects
    with pytest.raises(ReconstructionValidationError):
        inspect_semantic_geometry(
            _reference(),
            surface,
            semantic,
            candidate_key_by_object_id={objects[0].object_id: "same", objects[1].object_id: "same"},
        )


def test_metric_reference_and_candidate_emit_metric_deviations_without_metrology_claim() -> None:
    surface, semantic, _ = _fixture()
    metric_surface, metric_semantic = _metric_variant(surface, semantic, profile_id="metric-unit")
    reference = build_reference_assembly(
        metric_surface,
        metric_semantic,
        reference_id="metric-reference",
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
    )
    candidate_surface, candidate_semantic = _transform_candidate(
        metric_surface,
        metric_semantic,
        scale_factor=1.0,
        rotation=_rot_z(-11.0),
        translation=np.asarray([-0.2, 0.25, 0.1]),
    )
    report = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic)
    assert report.decision is InspectionDecision.PASS
    assert report.comparison_metric_available
    assert report.reference_unit_to_metre == pytest.approx(0.01)
    assert report.relative_scale_uncertainty is not None
    assert all(result.position_error_m is not None for result in report.feature_results)
    assert not report.certified_metrology


def test_metric_candidate_with_different_coordinate_unit_is_rescaled() -> None:
    surface, semantic, _ = _fixture()
    ref_surface, ref_semantic = _metric_variant(surface, semantic, profile_id="metric-ref")
    reference = build_reference_assembly(
        ref_surface,
        ref_semantic,
        reference_id="metric-reference",
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
    )
    cand_surface, cand_semantic = _metric_variant(surface, semantic, profile_id="metric-cand")
    cand_surface = replace(cand_surface, unit_to_metre=0.02)
    cand_semantic = replace(
        cand_semantic,
        unit_to_metre=0.02,
        objects=tuple(replace(obj, unit_to_metre=0.02) for obj in cand_semantic.objects),
    )
    # Coordinates represent half as many candidate units for the same metric geometry.
    mesh = replace(cand_surface.mesh, vertices_reconstruction=cand_surface.mesh.vertices_reconstruction * 0.5)
    cand_surface = replace(cand_surface, mesh=mesh, reconstruction_frame_id="metric-candidate")
    cand_semantic = replace(cand_semantic, reconstruction_frame_id="metric-candidate")
    report = inspect_semantic_geometry(reference, cand_surface, cand_semantic)
    assert report.registration.status is RegistrationStatus.SUCCESS
    assert report.registration.scale_candidate_to_reference == pytest.approx(2.0)


def test_mixed_arbitrary_and_metric_comparison_is_rejected() -> None:
    surface, semantic = _candidate()
    ref_surface, ref_semantic, _ = _fixture()
    metric_surface, metric_semantic = _metric_variant(ref_surface, ref_semantic, profile_id="mixed-ref")
    reference = build_reference_assembly(
        metric_surface,
        metric_semantic,
        reference_id="mixed-reference",
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
    )
    with pytest.raises(ReconstructionValidationError):
        inspect_semantic_geometry(reference, surface, semantic)


def test_optional_missing_feature_warns_instead_of_rejects() -> None:
    surface, semantic, _ = _fixture()
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="optional-reference",
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
        optional_feature_ids=("pin-bottom",),
    )
    candidate_surface, candidate_semantic = _candidate(remove_bottom=True)
    report = inspect_semantic_geometry(reference, candidate_surface, candidate_semantic)
    bottom = next(result for result in report.feature_results if result.reference_feature_id == "pin-bottom")
    assert bottom.decision is InspectionDecision.WARN
    assert report.decision is InspectionDecision.WARN


def test_extra_candidate_object_is_reported_as_warning() -> None:
    surface, semantic = _candidate()
    extra = replace(
        semantic.objects[-1],
        object_id="extra-pin-object",
        track_ids=("extra-pin",),
    )
    semantic = replace(
        semantic,
        objects=semantic.objects + (extra,),
        statistics=replace(semantic.statistics, persistent_object_count=len(semantic.objects) + 1),
    )
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert "extra-pin-object" in report.extra_candidate_object_ids
    assert report.decision is InspectionDecision.WARN


def test_low_quality_registration_anchor_produces_insufficient_evidence() -> None:
    surface, semantic = _candidate()
    body = next(obj for obj in semantic.objects if obj.semantic_role is SemanticGeometryRole.CONNECTOR_BODY)
    weak_body = replace(body, mean_vertex_quality=0.1)
    semantic = replace(
        semantic,
        objects=tuple(weak_body if obj.object_id == body.object_id else obj for obj in semantic.objects),
    )
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert report.registration.status is RegistrationStatus.INSUFFICIENT_EVIDENCE
    assert report.decision is InspectionDecision.INSUFFICIENT_EVIDENCE


def test_reference_can_declare_cad_feature_source_without_parsing_cad_in_core() -> None:
    surface, semantic, _ = _fixture()
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="cad-derived",
        source=ReferenceGeometrySource.CAD_FEATURE_MODEL,
        default_tolerance=_tolerance(),
        registration_anchor_feature_ids=("connector-main",),
    )
    assert reference.source is ReferenceGeometrySource.CAD_FEATURE_MODEL


def test_report_never_claims_certified_metrology() -> None:
    surface, semantic = _candidate()
    report = inspect_semantic_geometry(_reference(), surface, semantic)
    assert report.as_dict()["certified_metrology"] is False
    assert report.as_dict()["inspection_qc_enabled"] is True
