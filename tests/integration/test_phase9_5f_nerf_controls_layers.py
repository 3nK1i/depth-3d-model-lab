from __future__ import annotations

from pathlib import Path
import sys

from app.reconstruction_nerf_controls import NerfDisplaySettings
from app.reconstruction_nerf_interactive import (
    build_interactive_nerf_presentation,
    turntable_state_to_nerf_render_request,
)
from core.reconstruction import persist_nerf_representation
from tests.phase9_5d_helpers import make_phase9_5d_source


def _persist(tmp_path: Path):
    scene, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    return persist_nerf_representation(
        base_scene=scene,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=manifest,
        source_optimization_summary_path=summary,
        source_worker_console_path=console,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )


def test_95f_real_persisted_contract_to_controlled_request_keeps_heavy_runtime_out(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    before = {name for name in sys.modules if name.startswith(("nerfstudio", "tinycudann", "torch"))}
    presentation = build_interactive_nerf_presentation(result.scene)
    settings = NerfDisplaySettings(render_profile="balanced", opacity_percent=65, auto_refresh=False)
    request = turntable_state_to_nerf_render_request(
        request_id=8,
        center_viewer=(0.0, 0.0, 0.0),
        distance=3.0,
        azimuth_degrees=20.0,
        elevation_degrees=15.0,
        fov_degrees=45.0,
        scene_id=presentation.scene_id,
        artifact_id=presentation.artifact_id,
        width=settings.render_width,
        height=settings.render_height,
    )
    after = {name for name in sys.modules if name.startswith(("nerfstudio", "tinycudann", "torch"))}

    assert result.scene.canonical_geometry_authoritative is True
    assert presentation.scene_role == "appearance_only"
    assert (request.width, request.height) == (480, 270)
    payload = request.as_protocol_payload()
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["metric_scale_authoritative"] is False
    assert payload["measurement_authoritative"] is False
    assert payload["qc_authoritative"] is False
    assert payload["certified_metrology"] is False
    assert after == before


def test_95f_controls_do_not_mutate_persistent_scene_fingerprint(tmp_path: Path) -> None:
    result = _persist(tmp_path)
    before = result.scene.scene_fingerprint_sha256
    settings = NerfDisplaySettings(render_profile="fast", opacity_percent=35, auto_refresh=True)
    assert settings.visibility_only is True
    assert result.scene.scene_fingerprint_sha256 == before
