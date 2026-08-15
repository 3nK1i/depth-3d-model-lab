from __future__ import annotations

from app.measurement_canonical_registration_presentation import (
    build_measurement_canonical_registration_presentation,
)
from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_semantics import load_persistent_canonical_semantics
from core.reconstruction.measurement_registration import CanonicalMeasurementRegistrationState
from core.reconstruction.persistent_authority import (
    read_persistent_scene_authority_snapshot,
    write_persistent_scene_authority_snapshot,
)
from core.reconstruction.persistent_scene import (
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.phase9_3j_helpers import (
    make_phase9_3j_inspection_bundle,
    make_phase9_3j_qualified_scene,
)


def test_3j_persisted_scene_authority_and_camera_fingerprint_rebind_qualified_measurement(tmp_path) -> None:
    scene, authority = make_phase9_3j_qualified_scene(tmp_path / "fixture")
    scene_root = (tmp_path / "persistent-scene").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, scene_root)
    write_persistent_scene_authority_snapshot(authority, scene_root)

    reloaded = read_persistent_reconstruction_scene(scene_manifest)
    verify_persistent_scene_file_references(reloaded)
    reloaded_authority = read_persistent_scene_authority_snapshot(
        scene_root,
        expected_scene=reloaded,
    )
    mesh = load_canonical_viewer_mesh(reloaded)
    cameras = load_persistent_canonical_camera_trajectory(reloaded)
    semantics = load_persistent_canonical_semantics(reloaded, mesh)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path / "inspection")

    assert mesh is not None
    assert cameras is not None
    assert semantics is not None
    presentation = build_measurement_canonical_registration_presentation(
        scene=reloaded,
        canonical_cameras=cameras,
        canonical_semantics=semantics,
        authority=reloaded_authority,
        inspection_summary=summary,
        inspection_object=obj,
    )

    assert presentation.state is CanonicalMeasurementRegistrationState.QUALIFIED
    assert presentation.canonical_camera_id == "cam-source"
    assert len(presentation.dimensions) == 1
    assert presentation.dimensions[0].kind == "width"
    assert presentation.dimensions[0].source_expanded_uncertainty_si == 0.0015
    assert presentation.dimensions[0].registered_dimension_uncertainty_si > 0.0015
    assert presentation.certified_metrology is False


def test_3j_uncalibrated_or_unqualified_scene_stays_usable_and_emits_no_dimension_geometry(tmp_path) -> None:
    scene, _authority = make_phase9_3j_qualified_scene(tmp_path / "fixture")
    mesh = load_canonical_viewer_mesh(scene)
    cameras = load_persistent_canonical_camera_trajectory(scene)
    semantics = load_persistent_canonical_semantics(scene, mesh)
    summary, obj = make_phase9_3j_inspection_bundle(tmp_path / "inspection")

    assert mesh is not None and cameras is not None and semantics is not None
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
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False


def test_3j_validate_only_harness_verifies_persistent_da3_scene_without_claiming_real_execution(tmp_path) -> None:
    import json
    import cv2

    from app.da3_canonical_scene_persistence import materialize_da3_canonical_persistent_scene
    from core.perception.reviewed_analysis import source_image_sha256
    from core.reconstruction.da3_canonical_scene import integrate_da3_result_to_canonical_scene
    from core.reconstruction.da3_provider import read_da3_input_manifest
    from scripts.run_phase9_3j_real_scene_qualification import validate_worker_result
    from tests.phase9_3h_helpers import make_phase9_3h_da3_fixture, rgb_frames_for_manifest

    evidence = tmp_path / "da3"
    input_path, result_path, manifest = make_phase9_3h_da3_fixture(evidence)
    input_manifest = read_da3_input_manifest(input_path)
    fingerprints = []
    for value in input_manifest.image_paths:
        image = cv2.imread(str(value), cv2.IMREAD_COLOR)
        assert image is not None
        fingerprints.append(source_image_sha256(image))
    integration = integrate_da3_result_to_canonical_scene(
        manifest,
        session_id="6b7bd762-5fd8-4d30-b7df-1ac8f4b0f07e",
        request_id="phase9-3j-validate-only",
        rgb_frames=rgb_frames_for_manifest(input_path, manifest),
        source_frame_sha256=tuple(fingerprints),
        pixel_stride=2,
    )
    scene, scene_manifest, _surface = materialize_da3_canonical_persistent_scene(
        integration,
        (tmp_path / "persistent-scene").resolve(),
        input_manifest_path=input_path,
        result_manifest_path=result_path,
    )
    result = {
        "pass": True,
        "status": "completed",
        "persistent_scene_manifest": str(scene_manifest),
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "da3_supporting_authority_only": True,
        "da3_runtime_probed": False,
        "da3_backend_started": False,
        "da3_runtime_lock_valid": False,
        "da3_turing_compatibility_patch_applied": False,
        "da3_import_source": "",
        "da3_source_revision": manifest.source_revision,
        "da3_model_id": manifest.model_id,
        "da3_model_revision": manifest.model_revision,
        "da3_peak_vram_mib": manifest.peak_vram_mib,
        "da3_gpu_name": manifest.gpu_name,
        "semantic_context_available": False,
        "semantic_geometry_fabricated": False,
        "inspection_qc_available": False,
        "inspection_qc_fabricated": False,
    }
    result_path_out = (tmp_path / "offline_result.json").resolve()
    result_path_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = validate_worker_result(
        result_path_out,
        real_execution=False,
        elapsed_seconds=None,
    )
    assert summary["status"] == "PASS_VALIDATE_ONLY"
    assert summary["real_execution_performed"] is False
    assert summary["persistent_scene"]["scene_id"] == scene.scene_id
    assert summary["persistent_scene"]["source_frame_fingerprint_count"] == 3
    assert summary["authority"]["metric_geometry_status"] == "arbitrary"
    assert summary["measurement_registration"]["state"] == "UNAVAILABLE"
    assert summary["deferred"]["real_physical_checkerboard_charuco_calibration"] is True
