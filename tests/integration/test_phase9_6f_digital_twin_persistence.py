from __future__ import annotations

from pathlib import Path

from app.reconstruction_digital_twin_persistence import (
    build_digital_twin_workspace_state,
    export_digital_twin_state_bundle,
    read_optional_digital_twin_workspace_state,
    verify_digital_twin_state_bundle,
    write_digital_twin_workspace_state,
)
from app.reconstruction_estimated_metric_measurement import (
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from core.reconstruction import (
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    return persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-16T10:20:00+00:00",
    )


def test_f_save_reload_preserves_scene_identity_and_original_estimated_ruler_provenance(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    root = (tmp_path / "scene").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    hint = estimated_scale_from_ruler_reference(
        scene_id=scene.scene_id,
        distance_reconstruction_units=4.0,
        approximate_distance_m=2.0,
        reference_relative_uncertainty=0.15,
    )
    ruler = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(4.0, 0.0, 0.0),
        canonical_scene=None,
        estimated_scale_hint=hint,
    )
    state = build_digital_twin_workspace_state(
        scene,
        estimated_metric_scale_hint=hint,
        ruler_metric_record=ruler,
        created_at_utc="2026-08-16T10:21:00+00:00",
    )
    write_digital_twin_workspace_state(state, root)

    reloaded_scene = read_persistent_reconstruction_scene(scene_manifest)
    verify_persistent_scene_file_references(reloaded_scene)
    reloaded_state, _ = read_optional_digital_twin_workspace_state(
        scene_manifest,
        expected_scene=reloaded_scene,
    )
    assert reloaded_scene.scene_id == scene.scene_id
    assert reloaded_scene.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256
    assert reloaded_scene.canonical_geometry_artifact_id == scene.canonical_geometry_artifact_id
    assert reloaded_state is not None
    assert reloaded_state.ruler_metric_record is not None
    assert reloaded_state.ruler_metric_record.original_estimate is not None
    assert reloaded_state.ruler_metric_record.original_estimate.distance_m == 2.0
    assert reloaded_state.certified_metrology is False


def test_f_state_export_is_integrity_checked_and_explicitly_does_not_embed_heavy_payloads(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    root = (tmp_path / "scene").resolve()
    scene_manifest = write_persistent_reconstruction_scene(scene, root)
    state = build_digital_twin_workspace_state(
        scene,
        created_at_utc="2026-08-16T10:25:00+00:00",
    )
    export_manifest = export_digital_twin_state_bundle(
        scene_manifest_path=scene_manifest,
        workspace_state=state,
        output_dir=(tmp_path / "state-export").resolve(),
    )
    payload = verify_digital_twin_state_bundle(export_manifest.parent)
    assert payload["scene_id"] == scene.scene_id
    assert payload["artifact_payloads_embedded"] is False
    assert payload["portable_payload_bundle"] is False
    assert payload["restore_requires_referenced_files"] is True
    assert payload["referenced_payload_count"] == len(scene.file_references)
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["certified_metrology"] is False


def test_f_legacy_scene_without_workspace_sidecar_remains_loadable(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene_manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "legacy").resolve())
    loaded = read_persistent_reconstruction_scene(scene_manifest)
    state, manifest = read_optional_digital_twin_workspace_state(
        scene_manifest,
        expected_scene=loaded,
    )
    assert state is None
    assert manifest is None
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.certified_metrology is False
