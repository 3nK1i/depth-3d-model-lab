from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.reconstruction_digital_twin_persistence import (
    DIGITAL_TWIN_WORKSPACE_CHECKSUM,
    DigitalTwinProcessingHistoryRecord,
    build_digital_twin_workspace_state,
    read_optional_digital_twin_workspace_state,
    write_digital_twin_workspace_state,
)
from app.reconstruction_estimated_metric_measurement import (
    build_reconstruction_ruler_metric_record,
    estimated_scale_from_ruler_reference,
)
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
    DigitalTwinProcessingSnapshot,
)
from core.reconstruction import (
    ReconstructionValidationError,
    persistent_scene_from_pipeline_result,
    write_persistent_reconstruction_scene,
)
from tests.reconstruction_f5_helpers import run_f5_fixture


def _scene(tmp_path: Path):
    _fixture, _providers, result, snapshots = run_f5_fixture(tmp_path / "run")
    return persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-16T10:00:00+00:00",
    )


def _workspace_inputs(scene, tmp_path: Path):
    hint = estimated_scale_from_ruler_reference(
        scene_id=scene.scene_id,
        distance_reconstruction_units=2.0,
        approximate_distance_m=1.0,
        reference_relative_uncertainty=0.20,
    )
    ruler = build_reconstruction_ruler_metric_record(
        scene_id=scene.scene_id,
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(2.0, 0.0, 0.0),
        canonical_scene=None,
        estimated_scale_hint=hint,
    )
    spec = DigitalTwinProcessingJobSpec(
        job_id="aa1c0430-aa4c-4f26-a649-296718c41b33",
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title="Rebuild NeRF",
        backend_label="Nerfstudio Nerfacto",
        runtime_label="eia-nerf-nerfstudio",
        output_dir=(tmp_path / "job").resolve(),
        expected_protocol="depth3d.test.nerf.v1",
        scene_id=scene.scene_id,
        profile_label="Efficient • 64 steps",
    )
    snapshot = DigitalTwinProcessingSnapshot(
        lifecycle=DigitalTwinProcessingLifecycle.COMPLETED,
        spec=spec,
        stage="completed",
        message="NeRF appearance ready.",
        progress_fraction=1.0,
        elapsed_seconds=12.5,
        result_path=str((tmp_path / "job" / "summary.json").resolve()),
    )
    assert scene.nerf_artifact is not None
    return hint, ruler, snapshot, scene.nerf_artifact.artifact_id


def test_f_workspace_sidecar_round_trips_estimate_ruler_selection_and_terminal_history(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene_manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    hint, ruler, snapshot, selected = _workspace_inputs(scene, tmp_path)
    state = build_digital_twin_workspace_state(
        scene,
        estimated_metric_scale_hint=hint,
        ruler_metric_record=ruler,
        selected_appearance_artifact_id=selected,
        processing_snapshot=snapshot,
        created_at_utc="2026-08-16T10:05:00+00:00",
    )
    state_manifest = write_digital_twin_workspace_state(state, scene_manifest.parent)
    loaded, loaded_manifest = read_optional_digital_twin_workspace_state(
        scene_manifest,
        expected_scene=scene,
    )

    assert loaded_manifest == state_manifest
    assert loaded is not None
    assert loaded.workspace_fingerprint_sha256 == state.workspace_fingerprint_sha256
    assert loaded.estimated_metric_scale_hint == hint
    assert loaded.ruler_metric_record is not None
    assert loaded.ruler_metric_record.original_estimate is not None
    assert loaded.ruler_metric_record.original_estimate.distance_m == pytest.approx(1.0)
    assert loaded.selected_appearance_artifact_id == selected
    assert loaded.processing_history is not None
    assert loaded.processing_history.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED
    assert loaded.processing_history.certified_metrology is False
    assert loaded.certified_metrology is False


def test_f_active_heavy_job_is_never_persisted_as_resumable_state(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    _hint, _ruler, completed, _selected = _workspace_inputs(scene, tmp_path)
    assert completed.spec is not None
    running = DigitalTwinProcessingSnapshot(
        lifecycle=DigitalTwinProcessingLifecycle.RUNNING,
        spec=completed.spec,
        stage="training",
        progress_fraction=0.5,
    )
    with pytest.raises(ReconstructionValidationError, match="cannot be persisted or resumed"):
        build_digital_twin_workspace_state(scene, processing_snapshot=running)


def test_f_workspace_sidecar_tamper_fails_closed_without_touching_scene_manifest(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    scene_manifest = write_persistent_reconstruction_scene(scene, (tmp_path / "scene").resolve())
    state = build_digital_twin_workspace_state(
        scene,
        created_at_utc="2026-08-16T10:10:00+00:00",
    )
    state_manifest = write_digital_twin_workspace_state(state, scene_manifest.parent)
    raw = json.loads(state_manifest.read_text(encoding="utf-8"))
    raw["certified_metrology"] = True
    state_manifest.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Recompute only the file checksum to prove semantic validation still rejects
    # the forbidden authority claim.
    import hashlib
    digest = hashlib.sha256(state_manifest.read_bytes()).hexdigest()
    (scene_manifest.parent / DIGITAL_TWIN_WORKSPACE_CHECKSUM).write_text(
        f"{digest}  {state_manifest.name}\n", encoding="utf-8"
    )
    with pytest.raises(ReconstructionValidationError, match="forbidden authority claim"):
        read_optional_digital_twin_workspace_state(scene_manifest, expected_scene=scene)


def test_f_selected_workspace_appearance_must_be_appearance_only(tmp_path: Path) -> None:
    scene = _scene(tmp_path)
    with pytest.raises(ReconstructionValidationError, match="appearance-only"):
        build_digital_twin_workspace_state(
            scene,
            selected_appearance_artifact_id=scene.canonical_geometry_artifact_id,
        )
