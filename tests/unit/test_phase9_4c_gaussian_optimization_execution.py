from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.reconstruction import (
    GaussianCanonicalSeedEvidence,
    GaussianInteractiveRuntimeState,
    build_gaussian_seed_from_canonical_arrays,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_4c_helpers import ready_gaussian_runtime, tetrahedral_canonical_arrays


def test_4c_seed_positions_are_exact_canonical_vertices() -> None:
    vertices, faces, colors = tetrahedral_canonical_arrays()
    seed = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors, max_gaussians=4)
    assert seed.gaussian_count == 4
    assert np.array_equal(seed.positions_reconstruction, vertices)
    assert tuple(seed.source_vertex_indices.tolist()) == (0, 1, 2, 3)
    assert seed.as_dict()["positions_are_canonical_mesh_vertices"] is True


def test_4c_seed_is_deterministically_bounded() -> None:
    vertices = np.stack([np.arange(20), np.zeros(20), np.ones(20)], axis=1).astype(np.float32)
    faces = np.asarray([[index, index + 1, index + 2] for index in range(18)], dtype=np.uint32)
    colors = np.ones((20, 4), dtype=np.float32)
    first = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors, max_gaussians=5)
    second = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors, max_gaussians=5)
    assert first.gaussian_count == 5
    assert np.array_equal(first.source_vertex_indices, second.source_vertex_indices)
    assert np.array_equal(first.positions_reconstruction, second.positions_reconstruction)


def test_4c_seed_rejects_invalid_faces() -> None:
    vertices, faces, colors = tetrahedral_canonical_arrays()
    broken = faces.copy()
    broken[0, 2] = 99
    with pytest.raises(ReconstructionValidationError):
        build_gaussian_seed_from_canonical_arrays(vertices, broken, colors)


def test_4c_seed_rejects_nonfinite_geometry() -> None:
    vertices, faces, colors = tetrahedral_canonical_arrays()
    vertices = vertices.copy()
    vertices[0, 0] = np.nan
    with pytest.raises(ReconstructionValidationError):
        build_gaussian_seed_from_canonical_arrays(vertices, faces, colors)


def test_4c_seed_evidence_requires_exact_surface_hash(tmp_path: Path) -> None:
    surface = (tmp_path / "canonical.obj").resolve()
    surface.write_text("v 0 0 0\n", encoding="utf-8")
    evidence = GaussianCanonicalSeedEvidence(
        source_surface_path=str(surface),
        source_surface_sha256="a" * 64,
        gaussian_count=4,
        source_vertex_count=4,
        source_face_count=4,
    )
    assert evidence.positions_are_canonical_mesh_vertices is True
    assert evidence.gaussian_count == 4


def test_4c_seed_evidence_rejects_more_gaussians_than_vertices(tmp_path: Path) -> None:
    surface = (tmp_path / "canonical.obj").resolve()
    surface.write_text("v 0 0 0\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError):
        GaussianCanonicalSeedEvidence(
            source_surface_path=str(surface),
            source_surface_sha256="b" * 64,
            gaussian_count=5,
            source_vertex_count=4,
            source_face_count=4,
        )


def test_4c_ready_runtime_keeps_process_group_policy(tmp_path: Path) -> None:
    report = ready_gaussian_runtime(tmp_path)
    assert report.state is GaussianInteractiveRuntimeState.READY
    assert report.spec.process_policy.start_new_session is True
    assert report.spec.process_policy.shell is False
    assert report.spec.process_policy.terminate_signal == "SIGTERM"
    assert report.spec.process_policy.kill_signal == "SIGKILL"


def test_4c_runtime_report_remains_appearance_only(tmp_path: Path) -> None:
    report = ready_gaussian_runtime(tmp_path)
    assert report.gaussian_scene_role.value == "appearance_only"
    assert report.canonical_geometry_authoritative is False
    assert report.metric_scale_authoritative is False
    assert report.measurement_authoritative is False
    assert report.qc_authoritative is False
    assert report.certified_metrology is False


def test_4c_supervisor_uses_process_group_and_validates_appearance_artifact(tmp_path: Path, monkeypatch) -> None:
    import json
    from core.reconstruction import (
        GaussianCanonicalSeedEvidence,
        GaussianOptimizationProfile,
        materialize_gaussian_input_from_da3_persistent_scene,
        run_supervised_phase9_gaussian_optimization,
    )
    from tests.phase9_4b_helpers import make_phase9_4b_da3_scene
    import core.reconstruction.gaussian_optimization_execution as execution

    runtime = ready_gaussian_runtime(tmp_path / "runtime")
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_gaussian_input_from_da3_persistent_scene(
        scene, output_dir=(tmp_path / "dataset").resolve()
    )
    vertices, faces, colors = tetrahedral_canonical_arrays()
    seed = build_gaussian_seed_from_canonical_arrays(vertices, faces, colors)
    source = (tmp_path / "canonical.obj").resolve()
    source.write_text("# fixture\n", encoding="utf-8")
    seed_evidence = GaussianCanonicalSeedEvidence(
        source_surface_path=str(source),
        source_surface_sha256="d" * 64,
        gaussian_count=seed.gaussian_count,
        source_vertex_count=len(vertices),
        source_face_count=len(faces),
    )
    worker = (Path(__file__).resolve().parents[2] / "scripts/gaussian_workers/gsplat_worker.py").resolve()
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            self.pid = 424242
            self.returncode = 0

        def communicate(self, timeout=None):
            captured["timeout"] = timeout
            command = captured["command"]
            request_path = Path(command[command.index("--request") + 1])
            result_path = Path(command[command.index("--result") + 1])
            request = json.loads(request_path.read_text(encoding="utf-8"))
            dataset = json.loads(Path(request["dataset_manifest_path"]).read_text(encoding="utf-8"))
            artifact_dir = Path(request["output_dir"]) / "gsplat"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            scene_path = artifact_dir / "scene.pt"
            preview = artifact_dir / "preview.png"
            scene_path.write_bytes(b"fixture-scene")
            preview.write_bytes(b"fixture-preview")
            metrics = {
                "backend_kind": "gsplat_external",
                "real_cuda_execution": True,
                "optimization_completed": True,
                "render_written": True,
                "peak_vram_mib": 128.0,
            }
            (artifact_dir / "qualification_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            payload = {
                "schema_version": 1,
                "request_id": request["request_id"],
                "backend_id": "eia-gsplat-external",
                "backend_version": "1.5.3",
                "backend_provenance": request["backend_provenance"],
                "optimization_state": "optimized",
                "scene_role": "appearance_only",
                "artifact_dir": str(artifact_dir.resolve()),
                "scene_path": str(scene_path.resolve()),
                "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
                "input_frame_count": int(dataset["frame_count"]),
                "gaussian_count": seed.gaussian_count,
                "training_steps": 24,
                "training_duration_s": 0.5,
                "render_capable": True,
                "device": "cuda:0 fixture",
                "peak_vram_mib": 128.0,
                "pose_convention": dataset["pose_convention"],
                "reconstruction_frame_id": dataset["reconstruction_frame_id"],
                "metric_geometry_status": dataset["metric_geometry_status"],
                "unit_to_metre": dataset["unit_to_metre"],
                "relative_scale_uncertainty": dataset["relative_scale_uncertainty"],
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            }
            (artifact_dir / "artifact_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return "fixture worker completed\n", None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(execution.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(execution, "current_process_gaussian_heavy_modules", lambda: ())
    report = run_supervised_phase9_gaussian_optimization(
        runtime_report=runtime,
        dataset=bridge.dataset,
        dataset_manifest_path=bridge.manifest_path,
        seed=seed,
        seed_evidence=seed_evidence,
        output_dir=(tmp_path / "optimization").resolve(),
        worker_script=worker,
        request_id="phase9-4c-supervisor",
        profile=GaussianOptimizationProfile.EFFICIENT,
        maximum_input_frames=8,
    )
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert report.state.value == "completed"
    assert report.artifact.scene_role.value == "appearance_only"
    assert report.artifact.canonical_geometry_authoritative is False
    assert report.certified_metrology is False
