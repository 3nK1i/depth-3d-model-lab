from __future__ import annotations

from pathlib import Path

from app.reconstruction_gaussian_visualization import build_gaussian_visualization_presentation
from core.reconstruction import (
    PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION,
    current_process_gaussian_heavy_modules,
    persist_gaussian_representation,
    read_persistent_reconstruction_scene,
    reload_persistent_gaussian_representation,
    verify_persistent_scene_file_references,
)
from tests.phase9_4d_helpers import make_phase9_4d_source


def _persist(tmp_path: Path):
    base, artifact, preview, metrics, manifest, summary = make_phase9_4d_source(tmp_path)
    result = persist_gaussian_representation(
        base_scene=base,
        gaussian_artifact=artifact,
        preview_path=preview,
        qualification_metrics_path=metrics,
        source_artifact_manifest_path=manifest,
        source_optimization_summary_path=summary,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    return base, result


def test_4d_reuses_existing_persistent_scene_and_static_gaussian_presentation_contract(tmp_path: Path) -> None:
    base, result = _persist(tmp_path)
    loaded = read_persistent_reconstruction_scene(result.scene_manifest_path)
    assert loaded.schema_version == PERSISTENT_RECONSTRUCTION_SCENE_SCHEMA_VERSION == 1
    assert loaded.canonical_geometry_artifact_id == base.canonical_geometry_artifact_id
    assert loaded.gaussian_artifact is not None
    presentation = build_gaussian_visualization_presentation(loaded)
    assert presentation is not None
    assert presentation.ready is True
    assert presentation.canonical_geometry_authoritative is False
    assert presentation.certified_metrology is False
    assert Path(presentation.preview_path).is_file()
    assert Path(presentation.preview_path).parent == Path(result.artifact.artifact_dir)


def test_4d_restart_reload_is_backend_free_and_integrity_bound(tmp_path: Path) -> None:
    _base, result = _persist(tmp_path)
    reloaded = reload_persistent_gaussian_representation(result.scene_manifest_path)
    refs = verify_persistent_scene_file_references(reloaded.scene)
    assert reloaded.restart_reload_verified is True
    assert reloaded.scene.gaussian_artifact is not None
    assert len(reloaded.gaussian_file_references) >= 5
    assert len(refs) >= len(reloaded.gaussian_file_references)
    assert current_process_gaussian_heavy_modules() == ()
    assert reloaded.backend_started is False
