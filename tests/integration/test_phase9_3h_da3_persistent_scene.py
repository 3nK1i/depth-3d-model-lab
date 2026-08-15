from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

from app.offline_reconstruction_semantics import semantic_presentation_from_persistent_scene
from app.reconstruction_canonical_cameras import load_persistent_canonical_camera_trajectory
from app.reconstruction_canonical_geometry import load_canonical_viewer_mesh
from app.reconstruction_canonical_scene_presentation import build_canonical_scene_presentation
from app.reconstruction_dense_provider_presentation import build_dense_provider_presentation
from core.contracts.validation import metadata_as_dict
from core.reconstruction import (
    ReconstructionArtifactKind,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
)
from core.reconstruction.da3 import DA3_PROVIDER_ID
from tests.phase9_3h_helpers import integrate_phase9_3h_fixture, make_phase9_3h_da3_fixture


def test_3h_persistent_da3_scene_reloads_canonical_surface_cameras_and_truthful_unavailable_context(tmp_path: Path) -> None:
    integration, manifest, _root = integrate_phase9_3h_fixture(tmp_path)
    scene = read_persistent_reconstruction_scene(manifest)
    verify_persistent_scene_file_references(scene)

    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
    assert scene.state.value == "degraded"

    mesh = load_canonical_viewer_mesh(scene)
    assert mesh is not None
    assert mesh.vertex_count == integration.surface.mesh.vertex_count
    assert mesh.face_count == integration.surface.mesh.face_count

    cameras = load_persistent_canonical_camera_trajectory(scene)
    assert cameras is not None
    assert cameras.persisted_pose_count == integration.da3_manifest.frame_count

    semantics = semantic_presentation_from_persistent_scene(scene)
    assert semantics is not None
    assert semantics.objects == ()
    assert "unavailable" in semantics.structure_text.lower()

    providers = build_dense_provider_presentation(scene, build_canonical_scene_presentation(scene))
    da3 = next(item for item in providers.entries if item.provider_id == DA3_PROVIDER_ID)
    assert da3.scene_bound is True
    assert da3.external is True
    assert "supporting" in da3.authority_text.lower()
    assert "DA3 support bound" in providers.integration_text


def test_3h_persistent_scene_keeps_calibration_and_metric_scale_optional(tmp_path: Path) -> None:
    _integration, manifest, _root = integrate_phase9_3h_fixture(tmp_path)
    scene = read_persistent_reconstruction_scene(manifest)
    snapshots = {
        artifact.kind: metadata_as_dict(artifact.payload_snapshot)
        for artifact in scene.artifacts
    }

    ingestion = snapshots[ReconstructionArtifactKind.INGESTION_RESULT]
    surface = snapshots[ReconstructionArtifactKind.SURFACE_MESH]
    semantic = snapshots[ReconstructionArtifactKind.SEMANTIC_GEOMETRY]
    inspection = snapshots[ReconstructionArtifactKind.INSPECTION_REPORT]
    assert ingestion["calibration_required"] is False
    assert ingestion["metric_scale_qualified"] is False
    assert surface["metric_geometry_status"] == "arbitrary"
    assert surface["unit_to_metre"] is None
    assert semantic["available"] is False and semantic["fabricated"] is False
    assert inspection["available"] is False and inspection["fabricated"] is False


def test_3h_depth_side_worker_reuse_path_materializes_reloadable_scene_without_heavy_backend(tmp_path: Path) -> None:
    input_path, result_path, _manifest = make_phase9_3h_da3_fixture(tmp_path / "worker-evidence")
    output = tmp_path / "worker-output"
    worker = Path(__file__).resolve().parents[2] / "scripts" / "run_phase9_3h_da3_canonical_job.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--session-id",
            "33333333-3333-4333-8333-333333333333",
            "--output-dir",
            str(output),
            "--reuse-da3-input-manifest",
            str(input_path),
            "--reuse-da3-result-manifest",
            str(result_path),
            "--pixel-stride",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((output / "offline_result.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["persistent_scene_ready"] is True
    assert payload["canonical_geometry_authoritative"] is True
    assert payload["certified_metrology"] is False
    assert payload["metric_geometry_status"] == "arbitrary"
    assert payload["unit_to_metre"] is None
    assert payload["calibration_required"] is False
    assert payload["da3_runtime_probed"] is False
    assert payload["da3_backend_started"] is False
    assert payload["semantic_context_available"] is False
    assert payload["inspection_qc_available"] is False
    scene = read_persistent_reconstruction_scene(payload["persistent_scene_manifest"])
    verify_persistent_scene_file_references(scene)
    assert scene.canonical_geometry_authoritative is True
    assert scene.certified_metrology is False
