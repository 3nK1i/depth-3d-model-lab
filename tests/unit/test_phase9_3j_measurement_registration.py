from __future__ import annotations

import numpy as np

from app.measurement_canonical_registration_presentation import (
    build_measurement_canonical_registration_presentation,
)
from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.reconstruction.measurement_registration import CanonicalMeasurementRegistrationState
from tests.phase9_3j_helpers import (
    SOURCE_SHA256,
    make_phase9_3j_inspection_bundle,
    make_phase9_3j_qualified_scene,
)


def _presentations(scene):
    cameras = load_persistent_canonical_camera_trajectory(scene)
    mesh = load_canonical_viewer_mesh(scene)
    assert cameras is not None and mesh is not None
    semantics = load_persistent_canonical_semantics(scene, mesh)
    assert semantics is not None
    return cameras, semantics


def test_3j_qualified_registration_emits_real_dimension_endpoints_and_uncertainty(tmp_path):
    scene, authority = make_phase9_3j_qualified_scene(tmp_path)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path)
    cameras, semantics = _presentations(scene)
    presentation = build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=authority,
        inspection_summary=summary,
        inspection_object=obj,
    )
    assert presentation.state is CanonicalMeasurementRegistrationState.QUALIFIED
    assert presentation.canonical_camera_id == "cam-source"
    assert presentation.canonical_object_id == "semantic-0000-0000"
    assert len(presentation.dimensions) == 1
    width = presentation.dimensions[0]
    assert width.kind == "width"
    assert np.isclose(width.value_si, 0.042)
    assert np.allclose(width.start_reconstruction, [0.079, 0.2, 1.0])
    assert np.allclose(width.end_reconstruction, [0.121, 0.2, 1.0])
    assert width.registered_dimension_uncertainty_si > width.source_expanded_uncertainty_si
    assert width.endpoint_placement_uncertainty_si > 0.0
    assert presentation.certified_metrology is False


def test_3j_missing_metric_authority_is_unavailable_without_breaking_measurement(tmp_path):
    scene, _authority = make_phase9_3j_qualified_scene(tmp_path)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path)
    cameras, semantics = _presentations(scene)
    presentation = build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=None,
        inspection_summary=summary,
        inspection_object=obj,
    )
    assert presentation.state is CanonicalMeasurementRegistrationState.UNAVAILABLE
    assert presentation.dimensions == ()
    assert "metric authority" in presentation.reason_text.lower()
    assert obj.measurements[0]["value_si"] == 0.042


def test_3j_metric_scene_without_quantified_pose_uncertainty_is_provisional(tmp_path):
    scene, authority = make_phase9_3j_qualified_scene(tmp_path, with_pose_uncertainty=False)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path)
    cameras, semantics = _presentations(scene)
    presentation = build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=authority,
        inspection_summary=summary,
        inspection_object=obj,
    )
    assert presentation.state is CanonicalMeasurementRegistrationState.PROVISIONAL
    assert presentation.dimensions == ()
    assert "pose uncertainty" in presentation.reason_text.lower()


def test_3j_source_fingerprint_must_match_exact_persisted_camera(tmp_path):
    scene, authority = make_phase9_3j_qualified_scene(tmp_path)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path, source_sha256="d" * 64)
    cameras, semantics = _presentations(scene)
    assert cameras.camera_index_for_source_frame(SOURCE_SHA256) == 0
    presentation = build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=authority,
        inspection_summary=summary,
        inspection_object=obj,
    )
    assert presentation.state is CanonicalMeasurementRegistrationState.UNAVAILABLE
    assert presentation.dimensions == ()
    assert "fingerprinted canonical camera" in presentation.reason_text.lower()


def test_3j_measurement_method_must_match_persisted_box_method(tmp_path):
    from dataclasses import replace

    scene, authority = make_phase9_3j_qualified_scene(tmp_path)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path)
    bad_measurement = dict(obj.measurements[0])
    bad_measurement["method"] = "different_geometry_fit"
    obj = replace(obj, measurements=(bad_measurement,))
    cameras, semantics = _presentations(scene)
    presentation = build_measurement_canonical_registration_presentation(
        scene=scene,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=authority,
        inspection_summary=summary,
        inspection_object=obj,
    )
    assert presentation.state is CanonicalMeasurementRegistrationState.UNAVAILABLE
    assert presentation.dimensions == ()
    assert "method" in presentation.reason_text.lower()
