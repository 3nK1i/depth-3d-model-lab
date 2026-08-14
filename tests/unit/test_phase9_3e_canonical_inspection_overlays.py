from __future__ import annotations

import numpy as np
import pytest

from app.reconstruction_canonical_inspection_overlays import build_canonical_object_inspection_overlay
from app.reconstruction_canonical_semantics import (
    CanonicalSemanticObjectBinding,
    CanonicalSemanticScenePresentation,
)
from app.reconstruction_inspection_qc_presentation import ReconstructionInspectionQcPresentation
from app.reconstruction_inspection_selection import (
    ReconstructionInspectionSelection,
    ReconstructionInspectionSelectionScope,
)
from app.reconstruction_measurement_presentation import (
    ReconstructionMeasurementPresentation,
    ReconstructionMeasurementValue,
)
from app.reconstruction_measurement_validity_presentation import (
    ReconstructionMeasurementValidityContext,
    ReconstructionMeasurementValidityEntry,
    ReconstructionMeasurementValidityPresentation,
)


def _selection(object_id: str = "obj:1") -> ReconstructionInspectionSelection:
    return ReconstructionInspectionSelection(
        scope=ReconstructionInspectionSelectionScope.OBJECT,
        scene_id="scene:1",
        session_id="session:1",
        scene_source="persistent_scene",
        scene_availability="ready",
        canonical_geometry_authoritative=True,
        object_id=object_id,
        class_name="connector",
        semantic_role="digital_twin_entity",
        semantic_source="semantic_geometry",
        metric_geometry_status="arbitrary",
    )


def _semantic_scene(object_id: str = "obj:1") -> CanonicalSemanticScenePresentation:
    obj = CanonicalSemanticObjectBinding(
        object_id=object_id,
        class_id=1,
        class_name="connector",
        semantic_role="digital_twin_entity",
        confidence=0.93,
        centroid_viewer=np.array([0.5, 0.5, 0.5]),
        vertex_indices=np.array([0, 1, 2], dtype=np.uint32),
        face_indices=np.array([0], dtype=np.uint32),
        surface_vertices_viewer=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.float32),
        surface_faces=np.array([[0, 1, 2]], dtype=np.uint32),
        observation_ids=("obs:1",), frame_ids=("f:1",), mask_ids=("m:1",),
        track_ids=(), camera_profile_ids=(), pose_layers=(),
    )
    return CanonicalSemanticScenePresentation(
        scene_id="scene:1", session_id="session:1",
        semantic_artifact_id="sem:1", semantic_provider_id="semantic-geometry-core",
        canonical_artifact_id="mesh:1", reconstruction_frame_id="reconstruction",
        metric_geometry_status="arbitrary", source_surface_schema_version=1,
        total_object_count=1, objects=(obj,), detailed_snapshot=True,
    )


def _e2(object_id: str = "obj:1", observation_id: str = "obs:measurement") -> ReconstructionMeasurementPresentation:
    return ReconstructionMeasurementPresentation(
        session_id="session:1", target_object_id=object_id, target_label="connector",
        source="phase7_5_measurement", source_observation_id=observation_id,
        values=(ReconstructionMeasurementValue("width", 0.042, "m", "source"),),
        source_measurement_count=2,
    )


def _e3(*, object_id: str = "obj:1", observation_id: str = "obs:measurement", rejected: bool = False) -> ReconstructionMeasurementValidityPresentation:
    entry = (
        ReconstructionMeasurementValidityEntry(
            kind="depth", status="rejected", unit_si="m", method="source",
            reliability="rejected", rejection_reasons=("excessive_uncertainty",),
            valid_observation_count=2,
        )
        if rejected else
        ReconstructionMeasurementValidityEntry(
            kind="width", status="available", unit_si="m", method="source",
            reliability="high", value_si=0.042, expanded_uncertainty_si=0.0015,
            lower_bound_si=0.0405, upper_bound_si=0.0435, reliability_score=0.92,
            valid_observation_count=5, uncertainty_qualification="Expanded source uncertainty",
        )
    )
    return ReconstructionMeasurementValidityPresentation(
        session_id="session:1", target_object_id=object_id, target_label="connector",
        source="phase7_5_measurement", source_observation_id=observation_id,
        entries=(entry,),
        context=ReconstructionMeasurementValidityContext(
            depth_scale="metric", intrinsics_source="calibrated",
            calibration_status="calibrated", confidence_kind="uncertainty",
        ),
        scene_metric_geometry_status="arbitrary",
    )


def _qc(*, available: bool = True, object_id: str = "obj:1") -> ReconstructionInspectionQcPresentation:
    return ReconstructionInspectionQcPresentation(
        session_id="session:1", reference_id="ref:1", source="phase8_7c.inspection_report",
        target_scope="object", target_object_id=object_id, target_label="connector",
        overall_decision="warn", target_result_available=available,
        target_decision="warn" if available else "",
        findings=("width_near_tolerance",) if available else (), registration_status="success",
        registration_confidence=0.9, registration_reason="source", comparison_metric_available=False,
        candidate_metric_geometry_status="arbitrary", reference_metric_geometry_status="validated_metric",
        reference_feature_count=1, matched_feature_count=1 if available else 0,
        missing_required_count=0 if available else 1, extra_candidate_count=0,
        pass_count=0, warn_count=1 if available else 0, reject_count=0,
        insufficient_evidence_count=0 if available else 1,
    )


def test_e2_alone_does_not_put_unqualified_numbers_in_viewport() -> None:
    assert build_canonical_object_inspection_overlay(
        selection=_selection(), canonical_semantics=_semantic_scene(), measurement=_e2()
    ) is None


def test_available_e3_value_is_qualified_with_uncertainty_and_scene_scale() -> None:
    presentation = build_canonical_object_inspection_overlay(
        selection=_selection(), canonical_semantics=_semantic_scene(), measurement=_e2(), validity=_e3()
    )
    assert presentation is not None
    assert "42.0 mm ± 1.5 mm" in presentation.overlay_text
    assert "High 92%" in presentation.overlay_text
    assert "metric depth" in presentation.overlay_text
    assert "arbitrary scene gauge" in presentation.overlay_text
    assert "metrology not certified" in presentation.overlay_text
    assert presentation.numeric_measurement_count == 1
    assert presentation.certified_metrology is False
    # Readability refinement keeps the callout clearly above the selected
    # surface instead of crowding its top edge.
    assert presentation.anchor_viewer[2] >= 1.18


def test_rejected_measurement_shows_reason_without_numeric_value() -> None:
    presentation = build_canonical_object_inspection_overlay(
        selection=_selection(), canonical_semantics=_semantic_scene(), measurement=_e2(), validity=_e3(rejected=True)
    )
    assert presentation is not None
    assert "Depth: REJECTED" in presentation.overlay_text
    assert "Excessive uncertainty" in presentation.overlay_text
    assert "42.0 mm" not in presentation.overlay_text
    assert "±" not in presentation.overlay_text
    assert presentation.rejected_measurement_count == 1


def test_exact_measurement_binding_is_required() -> None:
    with pytest.raises(ValueError, match="observation"):
        build_canonical_object_inspection_overlay(
            selection=_selection(), canonical_semantics=_semantic_scene(),
            measurement=_e2(observation_id="obs:a"), validity=_e3(observation_id="obs:b")
        )


def test_qc_exact_object_result_and_not_evaluated_are_kept_separate() -> None:
    exact = build_canonical_object_inspection_overlay(
        selection=_selection(), canonical_semantics=_semantic_scene(), inspection_qc=_qc(available=True)
    )
    assert exact is not None
    assert "QC: WARN" in exact.overlay_text
    assert "Width near tolerance" in exact.overlay_text

    missing = build_canonical_object_inspection_overlay(
        selection=_selection(), canonical_semantics=_semantic_scene(), inspection_qc=_qc(available=False)
    )
    assert missing is not None
    assert "QC: NOT EVALUATED" in missing.overlay_text
    assert "QC: WARN" not in missing.overlay_text


def test_canonical_scene_identity_mismatch_fails_closed() -> None:
    selection = _selection()
    semantic = _semantic_scene()
    bad = CanonicalSemanticScenePresentation(
        scene_id="scene:other", session_id=semantic.session_id,
        semantic_artifact_id=semantic.semantic_artifact_id,
        semantic_provider_id=semantic.semantic_provider_id,
        canonical_artifact_id=semantic.canonical_artifact_id,
        reconstruction_frame_id=semantic.reconstruction_frame_id,
        metric_geometry_status=semantic.metric_geometry_status,
        source_surface_schema_version=semantic.source_surface_schema_version,
        total_object_count=semantic.total_object_count, objects=semantic.objects, detailed_snapshot=True,
    )
    with pytest.raises(ValueError, match="scene"):
        build_canonical_object_inspection_overlay(selection=selection, canonical_semantics=bad, measurement=_e2(), validity=_e3())
