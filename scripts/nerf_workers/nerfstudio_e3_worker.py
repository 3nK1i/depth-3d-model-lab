#!/usr/bin/env python3
"""Isolated real Nerfstudio/Nerfacto worker for Phase 8.7E3 quality validation."""

from __future__ import annotations

import argparse
from copy import deepcopy
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
    NerfHeldOutRender,
    NerfHeldOutRenderManifest,
    NerfOptimizationProfile,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    nerf_backend_provenance,
    read_neural_scene_dataset_manifest,
    write_nerf_held_out_render_manifest,
)
from scripts.nerf_workers.common import export_nerfstudio_dataset, profile_settings  # noqa: E402


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
        raise RuntimeError(f"{env_name} is required for reproducible E3 validation")
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
    config.experiment_name = "eia-phase8-7e3"
    config.timestamp = "real-scene-validation"
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


def _render_test_split(config_path: Path, artifact_root: Path) -> Path:
    from nerfstudio.scripts.render import DatasetRender

    render_root = artifact_root / "rendered"
    DatasetRender(
        load_config=config_path,
        output_path=render_root,
        split="test",
        rendered_output_names=["rgb"],
        image_format="png",
    ).main()
    return render_root


def _map_rendered_outputs(
    *,
    render_root: Path,
    transforms_path: Path,
    validation_ids: tuple[str, ...],
    dataset,
) -> tuple[NerfHeldOutRender, ...]:
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    by_id: dict[str, str] = {}
    for item in transforms.get("frames", []):
        frame_id = str(item.get("eia_frame_id", ""))
        file_path = str(item.get("file_path", ""))
        if frame_id and file_path:
            by_id[frame_id] = file_path
    test_root = render_root / "test"
    candidates = [path for path in sorted(test_root.rglob("*.png")) if path.is_file() and path.stat().st_size > 0]
    if len(candidates) < len(validation_ids):
        produced = [str(path.relative_to(render_root)) for path in candidates]
        raise RuntimeError(f"E3 rendered fewer RGB PNGs than held-out views: {produced}")

    frame_by_id = {frame.frame_id: frame for frame in dataset.frames}
    result: list[NerfHeldOutRender] = []
    used: set[Path] = set()
    for frame_id in validation_ids:
        exported = by_id.get(frame_id)
        if not exported:
            raise RuntimeError(f"E3 transforms mapping is missing frame {frame_id}")
        expected_stem = Path(exported).stem
        matches = [path for path in candidates if path.stem == expected_stem and path not in used]
        if len(matches) != 1:
            produced = [str(path.relative_to(render_root)) for path in candidates]
            raise RuntimeError(
                f"E3 could not map held-out render for {frame_id} (expected stem {expected_stem}); "
                f"produced_files={produced}"
            )
        rendered = matches[0].resolve()
        used.add(rendered)
        frame = frame_by_id[frame_id]
        result.append(
            NerfHeldOutRender(
                frame_id=frame_id,
                reference_image_path=frame.image_path,
                rendered_image_path=str(rendered),
                width=frame.width,
                height=frame.height,
            )
        )
    return tuple(result)


def _run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("backend_kind") != NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value:
        raise RuntimeError("E3 worker received unsupported backend_kind")
    if request.get("method_name") != NERFSTUDIO_METHOD:
        raise RuntimeError("E3 worker received unsupported method")

    nerfstudio_root, nerfstudio_revision = _require_checkout(
        "DEPTH3D_NERFSTUDIO_ROOT",
        NERFSTUDIO_RELEASE_REVISION_PREFIX,
        legacy_env_name="EIA_NERFSTUDIO_ROOT",
    )
    tcnn_root, tcnn_revision = _require_checkout(
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
        raise RuntimeError("CUDA is unavailable inside E3 Nerfstudio worker")

    manifest_path = Path(request["dataset_manifest_path"]).expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(manifest_path)
    training_ids = tuple(str(value) for value in request["training_frame_ids"])
    validation_ids = tuple(str(value) for value in request["validation_frame_ids"])
    steps = int(request["training_steps"])
    profile = NerfOptimizationProfile(str(request["profile"]))

    artifact_root = Path(request["output_dir"]).expanduser().resolve() / "nerfstudio-nerfacto-e3"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = artifact_root / "dataset"
    transforms_path = export_nerfstudio_dataset(
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

    torch.cuda.empty_cache()
    started = time.monotonic()
    nerfstudio_train(config)
    training_duration_s = time.monotonic() - started

    base_dir = config.get_base_dir().resolve()
    config_path = base_dir / "config.yml"
    checkpoint_dir = config.get_checkpoint_dir().resolve()
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not config_path.is_file() or not checkpoints:
        raise RuntimeError("E3 training completed without config/checkpoint artifacts")
    checkpoint_path = checkpoints[-1]

    render_root = _render_test_split(config_path, artifact_root)
    renders = _map_rendered_outputs(
        render_root=render_root,
        transforms_path=transforms_path,
        validation_ids=validation_ids,
        dataset=dataset,
    )
    render_manifest = NerfHeldOutRenderManifest(
        validation_fingerprint_sha256=str(request["validation_fingerprint_sha256"]),
        dataset_fingerprint_sha256=str(request["dataset_fingerprint_sha256"]),
        validation_frame_ids=validation_ids,
        renders=renders,
    )
    render_manifest_path = write_nerf_held_out_render_manifest(
        render_manifest, artifact_root / "heldout_render_manifest.json"
    )

    device_index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device_index)
    evidence = {
        "phase": "8.7E3",
        "nerfstudio_version": version,
        "nerfstudio_revision": nerfstudio_revision,
        "tiny_cuda_nn_revision": tcnn_revision,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "gpu_name": str(torch.cuda.get_device_name(device_index)),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "total_vram_mib": float(props.total_memory / (1024 ** 2)),
        "training_steps": steps,
        "training_frame_count": len(training_ids),
        "heldout_frame_count": len(validation_ids),
        "training_completed_without_oom": True,
        "heldout_render_count": len(renders),
        "quality_metrics_measured_in_worker": False,
        "performance_hardening_deferred_to": "8.7E4",
    }
    evidence_path = artifact_root / "e3_runtime_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")

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
        device=f"cuda:{device_index} {torch.cuda.get_device_name(device_index)}",
        pose_convention=dataset.pose_convention,
        reconstruction_frame_id=dataset.reconstruction_frame_id,
        metric_geometry_status=dataset.metric_geometry_status,
        unit_to_metre=dataset.unit_to_metre,
        relative_scale_uncertainty=dataset.relative_scale_uncertainty,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    result = {
        "artifact": artifact.as_dict(),
        "render_manifest_path": str(render_manifest_path),
        "runtime_evidence_path": str(evidence_path),
        "training_frame_ids": list(training_ids),
        "validation_frame_ids": list(validation_ids),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    try:
        _run(args.request.resolve(), args.result.resolve())
    except Exception as exc:
        print(f"Nerfstudio E3 worker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
