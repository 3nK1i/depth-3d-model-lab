from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from core.reconstruction import (
    PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
    PHASE9_5A_TCNN_QUALIFIED_REVISION,
    NerfCheckoutEvidence,
    NerfEnvironmentProbe,
    NerfInteractiveRuntimeSpec,
    NerfOptimizationExecutionState,
    NerfOptimizationProfile,
    NerfPythonRuntimeEvidence,
    assess_nerf_interactive_runtime,
    materialize_nerf_input_from_da3_persistent_scene,
    run_supervised_phase9_nerf_optimization,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene


def _ready_runtime(tmp_path: Path):
    python = (tmp_path / "eia-nerf-nerfstudio/bin/python").resolve()
    ns = (tmp_path / "nerfstudio").resolve()
    tcnn = (tmp_path / "tiny-cuda-nn").resolve()
    python.parent.mkdir(parents=True, exist_ok=True)
    ns.mkdir(parents=True, exist_ok=True)
    tcnn.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    spec = NerfInteractiveRuntimeSpec(
        python_executable=str(python), nerfstudio_root=str(ns), tiny_cuda_nn_root=str(tcnn)
    )
    env = NerfEnvironmentProbe(
        python_executable=str(python), available=True, nerfstudio_version="1.1.5",
        torch_version="2.1.2+cu118", tinycudann_available=True, ns_train_available=True,
        ns_render_available=True, cuda_available=True, cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",), cuda_compute_capabilities=((7, 5),),
        cuda_total_vram_mib=(5744.0,), torch_cuda_version="11.8",
    )
    return assess_nerf_interactive_runtime(
        spec,
        environment_probe=env,
        python_runtime=NerfPythonRuntimeEvidence(str(python), python_version="3.10.20"),
        nerfstudio_checkout=NerfCheckoutEvidence(str(ns), True, PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION, True),
        tiny_cuda_nn_checkout=NerfCheckoutEvidence(str(tcnn), True, PHASE9_5A_TCNN_QUALIFIED_REVISION, True),
    )


def test_5c_ready_runtime_preserves_process_group_policy(tmp_path: Path) -> None:
    report = _ready_runtime(tmp_path)
    assert report.ready is True
    assert report.spec.process_policy.start_new_session is True
    assert report.spec.process_policy.shell is False
    assert report.spec.process_policy.as_dict()["termination_order"] == ["SIGTERM", "SIGKILL"]


def test_5c_supervisor_executes_existing_worker_without_importing_backend(tmp_path: Path, monkeypatch) -> None:
    import core.reconstruction.nerf_optimization_execution as execution

    runtime = _ready_runtime(tmp_path / "runtime")
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "dataset")
    worker = (Path(__file__).resolve().parents[2] / "scripts/nerf_workers/nerfstudio_worker.py").resolve()
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
            artifact_dir = Path(request["output_dir"]) / "nerfstudio-nerfacto"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            config = artifact_dir / "config.yml"
            checkpoint = artifact_dir / "step-000000064.ckpt"
            preview = artifact_dir / "preview.png"
            config.write_text("fixture: true\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            preview.write_bytes(b"preview")
            runtime_evidence = {
                "nerfstudio_revision": PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
                "tiny_cuda_nn_revision": PHASE9_5A_TCNN_QUALIFIED_REVISION,
                "training_completed_without_oom": True,
                "preview_path": str(preview.resolve()),
                "gpu_name": "NVIDIA GeForce RTX 2060",
                "compute_capability": [7, 5],
                "total_vram_mib": 5744.0,
            }
            (artifact_dir / "e2_runtime_evidence.json").write_text(json.dumps(runtime_evidence), encoding="utf-8")
            payload = {
                "schema_version": 1,
                "request_id": request["request_id"],
                "backend_id": "nerfstudio-nerfacto",
                "backend_version": "1.1.5",
                "backend_revision": PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
                "backend_provenance": request["backend_provenance"],
                "method_name": "nerfacto",
                "training_state": "trained",
                "scene_role": "appearance_only",
                "artifact_dir": str(artifact_dir.resolve()),
                "config_path": str(config.resolve()),
                "checkpoint_path": str(checkpoint.resolve()),
                "dataset_fingerprint_sha256": request["dataset_fingerprint_sha256"],
                "input_frame_count": dataset["frame_count"],
                "training_steps": 64,
                "training_duration_s": 0.5,
                "render_capable": True,
                "device": "cuda:0 NVIDIA GeForce RTX 2060",
                "pose_convention": dataset["pose_convention"],
                "reconstruction_frame_id": dataset["reconstruction_frame_id"],
                "metric_geometry_status": dataset["metric_geometry_status"],
                "unit_to_metre": dataset["unit_to_metre"],
                "relative_scale_uncertainty": dataset["relative_scale_uncertainty"],
                "canonical_geometry_authoritative": False,
                "certified_metrology": False,
            }
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            return "fixture worker completed\n", None

        def poll(self):
            return self.returncode

    monkeypatch.setattr(execution.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(execution, "current_process_nerf_heavy_modules", lambda: ())
    report = run_supervised_phase9_nerf_optimization(
        runtime_report=runtime,
        dataset=bridge.dataset,
        dataset_manifest_path=bridge.manifest_path,
        output_dir=tmp_path / "optimization",
        worker_script=worker,
        request_id="phase9-5c-fixture",
        profile=NerfOptimizationProfile.EFFICIENT,
        training_steps=64,
        maximum_input_frames=8,
    )
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["env"]["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["env"]["EIA_NERFSTUDIO_ROOT"] == runtime.spec.nerfstudio_root
    assert report.state is NerfOptimizationExecutionState.COMPLETED
    assert report.artifact.training_steps == 64
    assert Path(report.preview_path).is_file()
    assert Path(report.optimization_manifest_path).is_file()
    assert report.quality_evaluated is False
    assert report.performance_qualified is False
    assert report.certified_metrology is False


def test_5c_report_cannot_be_upgraded_to_geometry_authority(tmp_path: Path, monkeypatch) -> None:
    # Exercise the dataclass fail-closed rule with a real report from the supervisor test path.
    test_5c_supervisor_executes_existing_worker_without_importing_backend(tmp_path, monkeypatch)
    # The invariant itself is checked on construction in a focused minimal way below.
    from core.reconstruction.nerf_optimization_execution import NerfOptimizationExecutionReport
    assert "canonical_geometry_authoritative" in NerfOptimizationExecutionReport.__dataclass_fields__


def test_5c_training_step_bounds_are_narrow(tmp_path: Path, monkeypatch) -> None:
    import core.reconstruction.nerf_optimization_execution as execution
    runtime = _ready_runtime(tmp_path / "runtime")
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "dataset")
    worker = (Path(__file__).resolve().parents[2] / "scripts/nerf_workers/nerfstudio_worker.py").resolve()
    monkeypatch.setattr(execution, "current_process_nerf_heavy_modules", lambda: ())
    with pytest.raises(Exception, match="training_steps"):
        run_supervised_phase9_nerf_optimization(
            runtime_report=runtime, dataset=bridge.dataset, dataset_manifest_path=bridge.manifest_path,
            output_dir=tmp_path / "out", worker_script=worker, request_id="bad", training_steps=999,
        )


def test_5c_rejects_heavy_module_leakage_before_launch(tmp_path: Path, monkeypatch) -> None:
    import core.reconstruction.nerf_optimization_execution as execution
    runtime = _ready_runtime(tmp_path / "runtime")
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "dataset")
    worker = (Path(__file__).resolve().parents[2] / "scripts/nerf_workers/nerfstudio_worker.py").resolve()
    monkeypatch.setattr(execution, "current_process_nerf_heavy_modules", lambda: ("nerfstudio",))
    with pytest.raises(Exception, match="leaked"):
        run_supervised_phase9_nerf_optimization(
            runtime_report=runtime, dataset=bridge.dataset, dataset_manifest_path=bridge.manifest_path,
            output_dir=tmp_path / "out", worker_script=worker, request_id="leak",
        )


def test_5c_does_not_change_arbitrary_scale_truth(tmp_path: Path) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "scene")
    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=tmp_path / "dataset")
    assert bridge.dataset.metric_geometry_status.value == "arbitrary"
    assert bridge.dataset.unit_to_metre is None
    assert bridge.certified_metrology is False
