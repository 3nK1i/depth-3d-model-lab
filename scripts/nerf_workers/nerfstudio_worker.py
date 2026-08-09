#!/usr/bin/env python3
"""Real isolated Nerfstudio/Nerfacto worker for Phase 8.7E2 qualification."""

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
    NERF_POSE_CONVENTION,
    NERFSTUDIO_METHOD,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfSceneArtifact,
    NerfSceneRole,
    NerfTrainingState,
    nerf_backend_provenance,
    read_neural_scene_dataset_manifest,
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


def _require_checkout(env_name: str, expected_prefix: str) -> tuple[Path, str]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for reproducible E2 qualification")
    root = Path(raw).expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError(f"{env_name} is not a Git checkout: {root}")
    revision = _git_revision(root)
    if not revision.startswith(expected_prefix.lower()):
        raise RuntimeError(
            f"{env_name} revision mismatch: {revision} does not start with {expected_prefix}"
        )
    return root, revision


def _request_training_steps(payload: dict) -> int:
    raw = int(payload.get("qualification_training_steps", 16))
    if raw < 2 or raw > 256:
        raise RuntimeError("qualification_training_steps must be between 2 and 256 for E2")
    return raw


def _configure_nerfacto(*, dataset_dir: Path, output_dir: Path, request: dict, steps: int):
    from nerfstudio.configs.method_configs import method_configs

    config = deepcopy(method_configs[NERFSTUDIO_METHOD])
    config.output_dir = output_dir
    config.experiment_name = "eia-phase8-7e2"
    config.timestamp = "qualification"
    config.data = dataset_dir
    config.max_num_iterations = steps
    config.steps_per_save = max(1, steps)
    # E2 is execution qualification, not quality evaluation.  Avoid expensive
    # periodic eval but still exercise test rendering after checkpoint reload.
    disabled_eval_step = steps + 1000
    config.steps_per_eval_batch = disabled_eval_step
    config.steps_per_eval_image = disabled_eval_step
    config.steps_per_eval_all_images = disabled_eval_step
    config.vis = "tensorboard"
    config.viewer.quit_on_train_completion = True
    config.machine.num_devices = 1
    config.machine.num_machines = 1
    config.machine.device_type = "cuda"
    config.machine.seed = 42
    config.logging.steps_per_log = max(1, min(steps, 4))

    profile = NerfOptimizationProfile(str(request["profile"]))
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


def _find_dataset_render_rgb_png(render_root: Path) -> Path | None:
    """Locate Nerfstudio v1.1.5 dataset-render RGB output.

    Nerfstudio derives the rendered filename relative to
    ``os.path.commonpath(image_filenames)``.  For a one-image test split the
    common path can be the image file itself, which produces the flat layout
    ``test/rgb.png`` instead of the usual ``test/rgb/<image>.png`` directory
    layout.  E2 intentionally uses one held-out camera, so both layouts are
    valid backend artifacts.
    """

    test_root = render_root / "test"
    flat = test_root / "rgb.png"
    if flat.is_file() and flat.stat().st_size > 0:
        return flat

    nested_root = test_root / "rgb"
    if nested_root.is_dir():
        for candidate in sorted(nested_root.rglob("*.png")):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    return None


def _render_checkpoint(config_path: Path, output_dir: Path) -> Path:
    from nerfstudio.scripts.render import DatasetRender

    render_root = output_dir / "rendered"
    DatasetRender(
        load_config=config_path,
        output_path=render_root,
        split="test",
        rendered_output_names=["rgb"],
        image_format="png",
    ).main()
    rendered_rgb = _find_dataset_render_rgb_png(render_root)
    if rendered_rgb is None:
        produced = sorted(
            str(path.relative_to(render_root))
            for path in render_root.rglob("*")
            if path.is_file()
        )
        raise RuntimeError(
            "Nerfstudio qualification render did not produce an RGB PNG; "
            f"produced_files={produced}"
        )
    preview = output_dir / "preview.png"
    shutil.copy2(rendered_rgb, preview)
    return preview


def _run(request_path: Path, result_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("backend_kind") != NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value:
        raise RuntimeError("real Nerfstudio worker received unsupported backend_kind")
    if request.get("method_name") != NERFSTUDIO_METHOD:
        raise RuntimeError("real Nerfstudio worker received unsupported method")

    nerfstudio_root, nerfstudio_revision = _require_checkout(
        "EIA_NERFSTUDIO_ROOT", NERFSTUDIO_RELEASE_REVISION_PREFIX
    )
    tcnn_root, tcnn_revision = _require_checkout("EIA_TCNN_ROOT", TINY_CUDA_NN_E2_REVISION)

    import torch
    import tinycudann  # noqa: F401
    import nerfstudio  # noqa: F401

    version = md.version("nerfstudio")
    if version != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
        raise RuntimeError(
            f"Nerfstudio version mismatch: {version}, expected {NERFSTUDIO_RELEASE_TAG}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside isolated Nerfstudio worker")

    manifest = Path(request["dataset_manifest_path"]).expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(manifest)
    if request["dataset_fingerprint_sha256"] != json.loads(
        manifest.read_text(encoding="utf-8")
    ).get("dataset_fingerprint_sha256"):
        raise RuntimeError("request/manifest dataset fingerprint mismatch")
    max_frames = int(request["maximum_input_frames"])
    if dataset.frame_count > max_frames:
        raise RuntimeError("NeRF E2 dataset exceeds maximum_input_frames")

    artifact_root = Path(request["output_dir"]).expanduser().resolve() / "nerfstudio-nerfacto"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = artifact_root / "dataset"
    export_nerfstudio_dataset(dataset, dataset_dir)

    steps = _request_training_steps(request)
    ns_output = artifact_root / "nerfstudio-output"
    config = _configure_nerfacto(
        dataset_dir=dataset_dir,
        output_dir=ns_output,
        request=request,
        steps=steps,
    )

    from nerfstudio.scripts.train import main as nerfstudio_train

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    nerfstudio_train(config)
    training_duration_s = time.monotonic() - started

    base_dir = config.get_base_dir().resolve()
    config_path = base_dir / "config.yml"
    checkpoint_dir = config.get_checkpoint_dir().resolve()
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    if not config_path.is_file() or not checkpoints:
        raise RuntimeError("Nerfstudio training completed without config/checkpoint artifacts")
    checkpoint_path = checkpoints[-1]

    preview_path = _render_checkpoint(config_path, artifact_root)
    if not preview_path.is_file() or preview_path.stat().st_size <= 0:
        raise RuntimeError("Nerfstudio qualification preview is missing or empty")

    device_index = int(torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device_index)
    runtime_evidence = {
        "nerfstudio_root": str(nerfstudio_root),
        "nerfstudio_version": version,
        "nerfstudio_revision": nerfstudio_revision,
        "tiny_cuda_nn_root": str(tcnn_root),
        "tiny_cuda_nn_revision": tcnn_revision,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "gpu_name": str(torch.cuda.get_device_name(device_index)),
        "compute_capability": list(torch.cuda.get_device_capability(device_index)),
        "total_vram_mib": float(props.total_memory / (1024 ** 2)),
        # This is execution evidence only; E4 owns final performance metrics.
        "training_completed_without_oom": True,
        "preview_path": str(preview_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "qualification_training_steps": steps,
    }
    (artifact_root / "e2_runtime_evidence.json").write_text(
        json.dumps(runtime_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifact = NerfSceneArtifact(
        request_id=str(request["request_id"]),
        backend_id="nerfstudio-nerfacto",
        backend_version=version,
        backend_revision=nerfstudio_revision,
        backend_provenance=nerf_backend_provenance(
            NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL
        ),
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
        pose_convention=NERF_POSE_CONVENTION,
        reconstruction_frame_id=dataset.reconstruction_frame_id,
        metric_geometry_status=dataset.metric_geometry_status,
        unit_to_metre=dataset.unit_to_metre,
        relative_scale_uncertainty=dataset.relative_scale_uncertainty,
        canonical_geometry_authoritative=False,
        certified_metrology=False,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    try:
        _run(args.request.resolve(), args.result.resolve())
    except Exception as exc:
        print(f"EIA Nerfstudio worker error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
