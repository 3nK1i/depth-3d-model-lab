#!/usr/bin/env python3
"""Isolated Nerfstudio/Nerfacto worker for Phase 8.7E4 hardening.

The worker owns only NeRF-backend execution.  It measures real training/render
performance, reloads and validates artifacts, and exposes test-only lifecycle
fault hooks used by the E4 backend supervisor.  It does not implement whole
reconstruction-engine recovery (Phase 8.7G).
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NERFSTUDIO_METHOD,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfHardwareEvidence,
    NerfHardwareProfile,
    NerfOptimizationProfile,
    NerfPerformanceMetrics,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    nerf_backend_provenance,
    read_neural_scene_dataset_manifest,
    validate_nerf_artifact_files,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset, profile_settings  # noqa: E402
from scripts.nerf_workers.performance import (  # noqa: E402
    ProcessTreeGpuSampler,
    probe_gpu_hardware,
    timed_gpu_phase,
)

FAULT_ENV = "DEPTH3D_NERF_E4_TEST_FAULTS"
FAULT_NONE = "none"
FAULT_HANG = "hang"
FAULT_CRASH = "crash"
FAULT_PARTIAL = "partial_artifact"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read Git revision for {path}: {completed.stderr.strip()}")
    return completed.stdout.strip().lower()


def _require_checkout(
    env_name: str,
    expected_prefix: str,
    *,
    legacy_env_name: str | None = None,
) -> tuple[Path, str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw and legacy_env_name:
        raw = os.environ.get(legacy_env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for reproducible E4 validation")
    root = Path(raw).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"{env_name} is not a Git checkout: {root}")
    revision = _git_revision(root)
    if not revision.startswith(expected_prefix.lower()):
        raise RuntimeError(f"{env_name} revision mismatch: {revision}")
    return root, revision


def _configure_nerfacto(*, dataset_dir: Path, output_dir: Path, profile: NerfOptimizationProfile, steps: int):
    from nerfstudio.configs.method_configs import method_configs

    config = deepcopy(method_configs[NERFSTUDIO_METHOD])
    config.output_dir = output_dir
    config.experiment_name = "eia-phase8-7e4"
    config.timestamp = "performance-hardening"
    config.data = dataset_dir
    config.max_num_iterations = int(steps)
    config.steps_per_save = int(steps)
    disabled_eval = int(steps) + 1000
    config.steps_per_eval_batch = disabled_eval
    config.steps_per_eval_image = disabled_eval
    config.steps_per_eval_all_images = disabled_eval
    config.vis = "tensorboard"
    config.viewer.quit_on_train_completion = True
    config.machine.num_devices = 1
    config.machine.num_machines = 1
    config.machine.device_type = "cuda"
    config.machine.seed = 42
    config.logging.steps_per_log = max(1, min(int(steps), 25))

    settings = profile_settings(profile)
    datamanager = config.pipeline.datamanager
    datamanager.train_num_rays_per_batch = settings["train_rays_per_batch"]
    datamanager.eval_num_rays_per_batch = settings["eval_rays_per_batch"]
    datamanager.dataparser.orientation_method = "none"
    datamanager.dataparser.center_method = "none"
    datamanager.dataparser.auto_scale_poses = False
    datamanager.dataparser.scale_factor = 1.0
    datamanager.dataparser.downscale_factor = 1
    datamanager.dataparser.eval_mode = "filename"

    model = config.pipeline.model
    model.eval_num_rays_per_chunk = settings["eval_num_rays_per_chunk"]
    model.camera_optimizer.mode = "off"
    return config


def _fault_marker(request: dict, mode: str) -> Path | None:
    raw = str(request.get("lifecycle_fault_marker_path", "")).strip()
    if not raw:
        return None
    marker = Path(raw).expanduser().resolve()
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(300)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _write_json(marker, {"mode": mode, "worker_pid": os.getpid(), "child_pid": child.pid})
    return marker


def _maybe_inject_fault(request: dict, result_path: Path) -> None:
    mode = str(request.get("lifecycle_fault_mode", FAULT_NONE)).strip().lower()
    if mode == FAULT_NONE:
        return
    if os.environ.get(FAULT_ENV) != "1":
        raise RuntimeError("E4 lifecycle fault injection is disabled outside explicit tests")
    _fault_marker(request, mode)
    if mode == FAULT_HANG:
        time.sleep(300.0)
        raise RuntimeError("unreachable lifecycle hang")
    if mode == FAULT_CRASH:
        os._exit(86)
    if mode == FAULT_PARTIAL:
        _write_json(result_path, {"phase": "8.7E4", "partial": True, "complete": False})
        raise RuntimeError("intentional E4 partial-artifact fault")
    raise RuntimeError(f"unsupported E4 lifecycle fault mode: {mode}")


def _load_checkpoint_payload(torch, checkpoint_path: Path) -> dict:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("Nerfstudio checkpoint payload is not a mapping")
    if "pipeline" not in payload or "step" not in payload:
        raise RuntimeError("Nerfstudio checkpoint is missing pipeline/step keys")
    return payload


def _fresh_reload_and_render(
    *,
    torch,
    config_path: Path,
    render_repeats: int,
    gpu_uuid: str,
) -> tuple[float, tuple[float, ...], float, int, int, int]:
    from nerfstudio.utils.eval_utils import eval_setup

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    started = time.monotonic()
    _loaded_config, pipeline, _loaded_checkpoint, step = eval_setup(config_path, test_mode="test")
    torch.cuda.synchronize()
    reload_s = float(time.monotonic() - started)
    pipeline.eval()

    loader = pipeline.datamanager.fixed_indices_eval_dataloader
    eval_count = len(pipeline.datamanager.eval_dataset)
    if eval_count < 1:
        raise RuntimeError("E4 checkpoint reload produced no eval images")

    # Warm-up once; the measured loop excludes one-time CUDA/materialization work.
    warm_bundle, _warm_batch = loader.get_data_from_image_idx(0)
    torch.cuda.synchronize()
    warm_outputs = pipeline.model.get_outputs_for_camera_ray_bundle(warm_bundle)
    torch.cuda.synchronize()
    if "rgb" not in warm_outputs:
        raise RuntimeError("E4 render warm-up did not produce rgb")
    del warm_outputs

    sampler = ProcessTreeGpuSampler(root_pid=os.getpid(), gpu_uuid=gpu_uuid, interval_s=0.05).start()
    latencies_ms: list[float] = []
    width = 0
    height = 0
    try:
        for repeat in range(int(render_repeats)):
            image_index = repeat % eval_count
            ray_bundle, _batch = loader.get_data_from_image_idx(image_index)
            torch.cuda.synchronize()
            render_started = time.monotonic()
            outputs = pipeline.model.get_outputs_for_camera_ray_bundle(ray_bundle)
            torch.cuda.synchronize()
            elapsed_ms = float((time.monotonic() - render_started) * 1000.0)
            rgb = outputs.get("rgb")
            if rgb is None:
                raise RuntimeError("E4 native render did not produce rgb")
            if not bool(torch.isfinite(rgb).all().item()):
                raise RuntimeError("E4 native render produced non-finite RGB values")
            shape = tuple(int(value) for value in rgb.shape)
            if len(shape) < 2:
                raise RuntimeError(f"unexpected E4 RGB shape: {shape}")
            height, width = shape[0], shape[1]
            latencies_ms.append(elapsed_ms)
            del outputs
    finally:
        render_peak_vram_mib = sampler.stop()

    if len(latencies_ms) != int(render_repeats):
        raise RuntimeError("E4 render latency sample count mismatch")
    return reload_s, tuple(latencies_ms), render_peak_vram_mib, width, height, int(step)


def _run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("backend_kind") != NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value:
        raise RuntimeError("E4 worker received unsupported backend_kind")
    if request.get("method_name") != NERFSTUDIO_METHOD:
        raise RuntimeError("E4 worker received unsupported method")

    _maybe_inject_fault(request, result_path)

    nerfstudio_root, nerfstudio_revision = _require_checkout(
        "DEPTH3D_NERFSTUDIO_ROOT",
        NERFSTUDIO_RELEASE_REVISION_PREFIX,
        legacy_env_name="EIA_NERFSTUDIO_ROOT",
    )
    _tcnn_root, tcnn_revision = _require_checkout(
        "DEPTH3D_TCNN_ROOT",
        TINY_CUDA_NN_E2_REVISION,
        legacy_env_name="EIA_TCNN_ROOT",
    )

    import torch
    import tinycudann  # noqa: F401
    import nerfstudio  # noqa: F401

    version = md.version("nerfstudio")
    if version != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
        raise RuntimeError(f"Nerfstudio version mismatch: {version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside E4 Nerfstudio worker")

    manifest_path = Path(request["dataset_manifest_path"]).expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(manifest_path)
    training_ids = tuple(str(value) for value in request["training_frame_ids"])
    validation_ids = tuple(str(value) for value in request["validation_frame_ids"])
    steps = int(request["training_steps"])
    render_repeats = int(request.get("render_repeats", 8))
    profile = NerfOptimizationProfile(str(request["profile"]))
    hardware_profile = NerfHardwareProfile(str(request.get("hardware_profile", NerfHardwareProfile.CUSTOM.value)))

    artifact_root = Path(request["output_dir"]).expanduser().resolve() / "nerfstudio-nerfacto-e4"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = artifact_root / "dataset"
    export_nerfstudio_dataset(
        dataset,
        dataset_dir,
        copy_images=False,
        training_frame_ids=training_ids,
        validation_frame_ids=validation_ids,
    )

    ns_output = artifact_root / "nerfstudio-output"
    config = _configure_nerfacto(
        dataset_dir=dataset_dir,
        output_dir=ns_output,
        profile=profile,
        steps=steps,
    )
    from nerfstudio.scripts.train import main as nerfstudio_train

    hardware_probe = probe_gpu_hardware(0)
    torch.cuda.empty_cache()

    def train() -> None:
        nerfstudio_train(config)

    training_duration_s, training_peak_vram_mib, training_vram_samples = timed_gpu_phase(
        train,
        gpu_uuid=hardware_probe.uuid,
        interval_s=0.05,
    )

    base_dir = config.get_base_dir().resolve()
    config_path = base_dir / "config.yml"
    checkpoint_dir = config.get_checkpoint_dir().resolve()
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not checkpoints:
        raise RuntimeError("E4 training completed without checkpoint artifacts")
    checkpoint_path = checkpoints[-1]
    validate_nerf_artifact_files(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        artifact_root=artifact_root,
    )
    _load_checkpoint_payload(torch, checkpoint_path)

    reload_s, latencies_ms, render_peak_vram_mib, width, height, loaded_step = _fresh_reload_and_render(
        torch=torch,
        config_path=config_path,
        render_repeats=render_repeats,
        gpu_uuid=hardware_probe.uuid,
    )

    # Artifact rejection probes operate on copies only; the valid checkpoint is never mutated.
    missing_rejected = False
    try:
        validate_nerf_artifact_files(
            config_path=config_path,
            checkpoint_path=artifact_root / "missing-step.ckpt",
            artifact_root=artifact_root,
        )
    except Exception:
        missing_rejected = True

    malformed_path = artifact_root / "malformed-checkpoint.ckpt"
    malformed_path.write_bytes(b"EIA-NOT-A-NERF-CHECKPOINT\n")
    malformed_rejected = False
    try:
        _load_checkpoint_payload(torch, malformed_path)
    except Exception:
        malformed_rejected = True
    finally:
        malformed_path.unlink(missing_ok=True)

    hardware = NerfHardwareEvidence(
        profile=hardware_profile,
        gpu_name=hardware_probe.name,
        gpu_uuid=hardware_probe.uuid,
        compute_capability=tuple(torch.cuda.get_device_capability(0)),
        total_vram_mib=hardware_probe.total_vram_mib,
        driver_version=hardware_probe.driver_version,
        torch_cuda_version=str(torch.version.cuda or ""),
        measured=True,
    )
    performance = NerfPerformanceMetrics(
        training_steps=steps,
        training_duration_s=training_duration_s,
        training_peak_vram_mib=training_peak_vram_mib,
        checkpoint_reload_s=reload_s,
        render_latency_ms=latencies_ms,
        render_peak_vram_mib=render_peak_vram_mib,
        render_width=width,
        render_height=height,
    )

    evidence = {
        "phase": "8.7E4",
        "nerfstudio_version": version,
        "nerfstudio_revision": nerfstudio_revision,
        "tiny_cuda_nn_revision": tcnn_revision,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "training_vram_sample_count": training_vram_samples,
        "loaded_checkpoint_step": loaded_step,
        "training_completed_without_oom": True,
        "performance": performance.as_dict(),
        "hardware": hardware.as_dict(),
        "artifact": {
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "config_size_bytes": config_path.stat().st_size,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "config_sha256": _sha256(config_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "checkpoint_reload_s": reload_s,
            "render_probe_passed": True,
            "missing_artifact_rejected": missing_rejected,
            "malformed_checkpoint_rejected": malformed_rejected,
        },
        "training_frame_count": len(training_ids),
        "heldout_frame_count": len(validation_ids),
        "backend_process_scope_only": True,
        "whole_engine_hardening_deferred_to": "8.7G",
        "whole_application_gpu_profiling_deferred_to": "8.9D",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    evidence_path = _write_json(artifact_root / "e4_runtime_evidence.json", evidence)

    artifact = NerfSceneArtifact(
        request_id=str(request["request_id"]),
        backend_id="nerfstudio-nerfacto",
        backend_version=version,
        backend_revision=nerfstudio_revision,
        backend_provenance=nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL),
        method_name=NERFSTUDIO_METHOD,
        training_state=NerfTrainingState.TRAINED,
        scene_role=NerfSceneRole.APPEARANCE_ONLY,
        artifact_dir=str(artifact_root),
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        dataset_fingerprint_sha256=str(request["dataset_fingerprint_sha256"]),
        input_frame_count=dataset.frame_count,
        training_steps=steps,
        training_duration_s=training_duration_s,
        render_capable=True,
        device=f"cuda:0 {torch.cuda.get_device_name(0)}",
        pose_convention=dataset.pose_convention,
        reconstruction_frame_id=dataset.reconstruction_frame_id,
        metric_geometry_status=dataset.metric_geometry_status,
        unit_to_metre=dataset.unit_to_metre,
        relative_scale_uncertainty=dataset.relative_scale_uncertainty,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    _write_json(
        result_path,
        {
            "phase": "8.7E4",
            "complete": True,
            "artifact": artifact.as_dict(),
            "runtime_evidence_path": str(evidence_path),
            "performance": performance.as_dict(),
            "hardware": hardware.as_dict(),
            "artifact_integrity": evidence["artifact"],
            "training_frame_ids": list(training_ids),
            "validation_frame_ids": list(validation_ids),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    try:
        _run(args.request.resolve(), args.result.resolve())
    except Exception as exc:
        print(f"EIA Nerfstudio E4 worker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
