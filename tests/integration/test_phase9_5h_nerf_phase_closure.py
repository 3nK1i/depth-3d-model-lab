from __future__ import annotations

from pathlib import Path

from app.reconstruction_nerf_controls import NerfDisplaySettings
from app.reconstruction_nerf_interactive import build_interactive_nerf_presentation
from app.reconstruction_nerf_resources import LatestRenderRequestQueue, build_nerf_interactive_resource_policy
from core.measurement.interactive_distance import InteractiveDistanceState, estimate_reconstruction_distance
from core.reconstruction import persist_nerf_representation, verify_persistent_scene_file_references
from scripts.run_phase9_5h_real_scene_qualification import validate_phase9_5_chain
from tests.phase9_5d_helpers import make_phase9_5d_source
from tests.phase9_5h_helpers import make_valid_phase9_5h_evidence


def test_5h_synthetic_persisted_nerf_keeps_canonical_measurement_and_bounded_queue(tmp_path: Path) -> None:
    base, artifact, preview, runtime, manifest, summary, console = make_phase9_5d_source(tmp_path)
    persisted = persist_nerf_representation(
        base_scene=base,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=manifest,
        source_optimization_summary_path=summary,
        source_worker_console_path=console,
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        output_dir=(tmp_path / "persistent").resolve(),
    )
    scene = persisted.scene
    assert verify_persistent_scene_file_references(scene)
    presentation = build_interactive_nerf_presentation(scene)
    assert presentation is not None and presentation.ready is True
    assert presentation.scene_role == "appearance_only"
    assert scene.canonical_geometry_authoritative is True

    settings = NerfDisplaySettings(render_profile="balanced", opacity_percent=70, auto_refresh=False)
    assert settings.render_width == 480
    assert settings.render_height == 270

    queue = LatestRenderRequestQueue()
    dispatched, _ = queue.submit({"type": "render", "request_id": 1})
    assert dispatched is True
    for request_id in range(2, 1002):
        dispatched, _ = queue.submit({"type": "render", "request_id": request_id})
        assert dispatched is False
    assert queue.snapshot.render_in_flight is True
    assert queue.snapshot.pending_request_id == 1001
    assert build_nerf_interactive_resource_policy().max_pending_renders == 1

    result = estimate_reconstruction_distance(
        point_a_reconstruction=(0.0, 0.0, 0.0),
        point_b_reconstruction=(1.0, 0.0, 0.0),
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        scale_state="arbitrary",
        calibration_available=False,
    )
    assert result.state is InteractiveDistanceState.UNAVAILABLE
    assert result.distance_m is None

    evidence = make_valid_phase9_5h_evidence()
    evidence["resources"]["cycles"] = 1
    evidence["resources"]["worker_start_count"] = 1
    evidence["resources"]["worker_release_count"] = 1
    evidence["resources"]["total_render_count"] = 3
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is True
    assert reasons == ()
