from __future__ import annotations

import gc
from pathlib import Path
import weakref

from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_gaussian_controls import GaussianDisplaySettings, apply_gaussian_display_settings
from app.reconstruction_gaussian_interactive import load_interactive_gaussian_scene
from app.reconstruction_gaussian_resources import build_gaussian_interactive_resource_policy
from core.measurement.interactive_distance import InteractiveDistanceState, estimate_reconstruction_distance
from core.reconstruction import verify_persistent_scene_file_references
from scripts.run_phase9_4h_real_scene_qualification import validate_phase9_4_chain
from tests.phase9_4e_helpers import make_phase9_4e_persistent_scene
from tests.phase9_4h_helpers import make_valid_phase9_4h_evidence
from tests.reconstruction_helpers import make_hardware


def test_94h_synthetic_persistent_scene_closes_view_controls_resources_and_measurement(tmp_path: Path) -> None:
    scene, _manifest, artifact = make_phase9_4e_persistent_scene(tmp_path)
    refs = verify_persistent_scene_file_references(scene)
    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    assert refs

    policy = build_gaussian_interactive_resource_policy(
        make_hardware(total_gib=6.0, name="NVIDIA GeForce RTX 2060")
    )
    interactive = load_interactive_gaussian_scene(scene, max_gaussians=policy.max_interactive_gaussians)
    buffers = apply_gaussian_display_settings(
        interactive,
        GaussianDisplaySettings(density_percent=50, size_percent=125, opacity_percent=70),
    )
    assert buffers.display_count == 32
    assert scene.gaussian_artifact is not None
    assert interactive.artifact_id == scene.gaussian_artifact.artifact_id
    assert interactive.scene_path == artifact.scene_path

    source_ref = weakref.ref(interactive.positions_reconstruction)
    display_ref = weakref.ref(buffers.positions_reconstruction)
    del buffers
    del interactive
    gc.collect()
    assert source_ref() is None
    assert display_ref() is None

    result = estimate_reconstruction_distance(
        point_a_reconstruction=mesh.vertices_reconstruction[0],
        point_b_reconstruction=mesh.vertices_reconstruction[2],
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        scale_state="arbitrary",
        calibration_available=False,
    )
    assert result.state is InteractiveDistanceState.UNAVAILABLE
    assert result.distance_m is None

    evidence = make_valid_phase9_4h_evidence()
    evidence["resources"]["cycles"] = 1
    evidence["resources"]["release_checks_passed"] = 1
    passed, reasons = validate_phase9_4_chain(evidence)
    assert passed is True
    assert reasons == ()
