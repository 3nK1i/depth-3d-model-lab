#!/usr/bin/env python3
"""Run real isolated Nerfstudio/Nerfacto qualification for Phase 8.7E2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    MetricGeometryStatus,
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_PYTHON,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfWorkerExecutionError,
    NerfWorkerProcessSpec,
    NerfWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    probe_nerf_environment,
    run_nerf_external_worker,
    write_neural_scene_dataset_manifest,
)

DEFAULT_OUTPUT = Path("/tmp/phase8_7e2_nerf_qualification")
RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB = 5500.0



def _default_env_python() -> Path:
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "bin" and len(executable.parents) >= 3:
        envs_dir = executable.parents[2]
        if envs_dir.name == "envs":
            return envs_dir / "depth3d-nerf-nerfstudio" / "bin" / "python"
    return Path.home() / "anaconda3" / "envs" / "depth3d-nerf-nerfstudio" / "bin" / "python"


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
        check=False,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip().lower()


def _write_ppm(path: Path, index: int, width: int = 64, height: int = 48) -> None:
    """Write a deterministic textured RGB view without adding image dependencies."""

    yy, xx = np.mgrid[0:height, 0:width]
    phase = index * 0.55
    r = 80 + 65 * (np.sin(xx / 8.0 + phase) * 0.5 + 0.5)
    g = 55 + 100 * (np.cos(yy / 7.0 - phase) * 0.5 + 0.5)
    radial = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
    b = 190 - np.minimum(140, radial * 4) + index * 3
    # A fixed bright square gives the field a stable high-frequency target.
    square = (np.abs(xx - width / 2) < 9) & (np.abs(yy - height / 2) < 7)
    r[square] = 230
    g[square] = 180 - index * 5
    b[square] = 60 + index * 8
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes())


def _look_at_opencv_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _dataset(output: Path) -> NeuralSceneDataset:
    images = output / "source-images"
    images.mkdir(parents=True, exist_ok=True)
    width, height = 64, 48
    frames = []
    for index in range(6):
        angle = (2.0 * math.pi * index) / 6.0
        eye = np.asarray([1.5 * math.sin(angle), 0.12 * math.sin(angle * 2.0), 1.5 * math.cos(angle)])
        image = (images / f"frame_{index:03d}.ppm").resolve()
        _write_ppm(image, index, width=width, height=height)
        frames.append(
            NeuralSceneFrame(
                frame_id=f"nerf-e2-{index:03d}",
                image_path=str(image),
                width=width,
                height=height,
                fx=55.0,
                fy=55.0,
                cx=width / 2.0,
                cy=height / 2.0,
                T_reconstruction_from_camera=_look_at_opencv_c2w(eye, np.zeros(3)),
                pose_layer="optimized_keyframe",
                camera_profile_id="phase8-7e2-camera",
                timestamp_ns=index * 33_333_333,
            )
        )
    return NeuralSceneDataset(
        session_id="phase8-7e2-real-backend-qualification",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id="reconstruction",
        camera_profile_id="phase8-7e2-camera",
        scale_profile_id="arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
    )


@contextmanager
def _isolated_environment(python_executable: Path):
    python = python_executable.expanduser().resolve()
    prefix = python.parent.parent
    if python.parent.name != "bin":
        raise RuntimeError(f"cannot infer isolated environment prefix from {python}")
    keys = (
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "CUDA_HOME",
        "CUDACXX",
        "PATH",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PYTHONNOUSERSITE",
    )
    previous = {key: os.environ.get(key) for key in keys}

    def prepend(key: str, values: list[Path]) -> None:
        parts = [str(value) for value in values if value.is_dir()]
        old = previous.get(key) or ""
        if old:
            parts.append(old)
        os.environ[key] = os.pathsep.join(parts)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)
    lib_dirs = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", lib_dirs)
    prepend("LIBRARY_PATH", lib_dirs)
    try:
        yield prefix
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _version_base(value: str) -> str:
    return str(value).split("+")[0]


def _preflight_reasons(*, probe, nerfstudio_root: Path, tcnn_root: Path, hardware_profile: str) -> list[str]:
    reasons: list[str] = []
    if not probe.interpreter_is_isolated:
        reasons.append("interpreter_not_isolated")
    if not probe.backend_ready:
        reasons.append("backend_environment_not_ready")
    if probe.nerfstudio_version != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
        reasons.append("nerfstudio_version_mismatch")
    if _version_base(probe.torch_version) != NERFSTUDIO_E2_TORCH:
        reasons.append("torch_version_mismatch")
    if not probe.torch_cuda_version.startswith(NERFSTUDIO_E2_CUDA):
        reasons.append("torch_cuda_version_mismatch")
    if not _git_revision(nerfstudio_root).startswith(NERFSTUDIO_RELEASE_REVISION_PREFIX):
        reasons.append("nerfstudio_revision_mismatch")
    if _git_revision(tcnn_root) != TINY_CUDA_NN_E2_REVISION:
        reasons.append("tiny_cuda_nn_revision_mismatch")
    if hardware_profile == "rtx_2060_6gb":
        if not probe.cuda_device_names or "RTX 2060" not in probe.cuda_device_names[0]:
            reasons.append("rtx2060_gpu_name_mismatch")
        if not probe.cuda_compute_capabilities or probe.cuda_compute_capabilities[0] != NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY:
            reasons.append("rtx2060_compute_capability_mismatch")
        # ``torch.cuda.get_device_properties().total_memory`` reports CUDA-visible
        # memory, which can be lower than the board's nominal 6 GiB capacity.
        # Keep the measured value as provenance and only reject materially smaller
        # devices here; the real E2 training run remains the authoritative fit/OOM
        # gate for this backend.
        if not probe.cuda_total_vram_mib or probe.cuda_total_vram_mib[0] < RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB:
            reasons.append("rtx2060_cuda_visible_vram_below_profile")
    return reasons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nerf-python", type=Path, default=_default_env_python())
    parser.add_argument("--nerfstudio-root", type=Path, default=Path.home() / "Desktop/depth3d-nerf-backends/nerfstudio")
    parser.add_argument("--tiny-cuda-nn-root", type=Path, default=Path.home() / "Desktop/depth3d-nerf-backends/tiny-cuda-nn")
    parser.add_argument("--profile", choices=[p.value for p in NerfOptimizationProfile], default="efficient")
    parser.add_argument("--training-steps", type=int, default=16)
    parser.add_argument("--hardware-profile", choices=["rtx_2060_6gb"], default="rtx_2060_6gb")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"

    python = args.nerf_python.expanduser().resolve()
    nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    tcnn_root = args.tiny_cuda_nn_root.expanduser().resolve()

    summary: dict = {
        "phase": "8.7E2",
        "pass": False,
        "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
        "method_name": "nerfacto",
        "hardware_profile": args.hardware_profile,
        "qualification_scope": "real_backend_execution_only",
        "quality_benchmark_deferred_to": "8.7E3",
        "performance_hardening_deferred_to": "8.7E4",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "scene_role": "appearance_only",
        "reconstruction_ui_enabled": False,
    }

    try:
        with _isolated_environment(python):
            probe = probe_nerf_environment(python, timeout_s=60.0)
            summary["environment_probe"] = probe.as_dict()
            reasons = _preflight_reasons(
                probe=probe,
                nerfstudio_root=nerfstudio_root,
                tcnn_root=tcnn_root,
                hardware_profile=args.hardware_profile,
            )
            summary["preflight_reasons"] = reasons
            if reasons:
                raise RuntimeError("preflight failed: " + ",".join(reasons))

            dataset = _dataset(output)
            manifest = write_neural_scene_dataset_manifest(dataset, (output / "dataset.json").resolve())
            request = NerfWorkerRequest.from_dataset(
                request_id="phase8-7e2-real-nerfacto",
                backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
                profile=NerfOptimizationProfile(args.profile),
                dataset=dataset,
                dataset_manifest_path=manifest,
                output_dir=(output / "worker-output").resolve(),
                maximum_input_frames=16,
                qualification_training_steps=args.training_steps,
            )
            worker = (ROOT / "scripts/nerf_workers/nerfstudio_worker.py").resolve()
            spec = NerfWorkerProcessSpec(
                python_executable=str(python),
                worker_script=str(worker),
                timeout_s=args.timeout_s,
                require_isolated_interpreter=True,
            )
            old_ns = os.environ.get("DEPTH3D_NERFSTUDIO_ROOT")
            old_tcnn = os.environ.get("DEPTH3D_TCNN_ROOT")
            os.environ["DEPTH3D_NERFSTUDIO_ROOT"] = str(nerfstudio_root)
            os.environ["DEPTH3D_TCNN_ROOT"] = str(tcnn_root)
            try:
                artifact = run_nerf_external_worker(spec, request)
            finally:
                if old_ns is None:
                    os.environ.pop("DEPTH3D_NERFSTUDIO_ROOT", None)
                else:
                    os.environ["DEPTH3D_NERFSTUDIO_ROOT"] = old_ns
                if old_tcnn is None:
                    os.environ.pop("DEPTH3D_TCNN_ROOT", None)
                else:
                    os.environ["DEPTH3D_TCNN_ROOT"] = old_tcnn

            artifact_dir = Path(artifact.artifact_dir)
            preview = artifact_dir / "preview.png"
            runtime_evidence = artifact_dir / "e2_runtime_evidence.json"
            evidence = json.loads(runtime_evidence.read_text(encoding="utf-8"))
            if not preview.is_file() or preview.stat().st_size <= 0:
                raise RuntimeError("real Nerfacto preview artifact is missing")
            if not Path(artifact.config_path).is_file() or not Path(artifact.checkpoint_path).is_file():
                raise RuntimeError("real Nerfacto config/checkpoint artifact is missing")
            if evidence.get("tiny_cuda_nn_revision") != TINY_CUDA_NN_E2_REVISION:
                raise RuntimeError("runtime evidence tiny-cuda-nn revision mismatch")
            if not bool(evidence.get("training_completed_without_oom", False)):
                raise RuntimeError("runtime evidence did not confirm OOM-free bounded training")

            summary.update(
                {
                    "pass": True,
                    "real_backend_qualified": True,
                    "training_completed_without_oom": True,
                    "artifact": artifact.as_dict(),
                    "runtime_evidence": evidence,
                    "preview_path": str(preview),
                    "config_path": artifact.config_path,
                    "checkpoint_path": artifact.checkpoint_path,
                }
            )
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        summary.update(
            {
                "pass": False,
                "real_backend_qualified": False,
                "failure": text,
                "oom_detected": "out of memory" in text.lower() or "cuda oom" in text.lower(),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Phase 8.7E2 real Nerfstudio/Nerfacto qualification: FAIL")
        print(text)
        print(f"summary={summary_path}")
        return 1

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact = summary["artifact"]
    runtime = summary["runtime_evidence"]
    print("Phase 8.7E2 real Nerfstudio/Nerfacto qualification: PASS")
    print(
        f"backend=qualified method=nerfacto steps={artifact['training_steps']} "
        f"device={artifact['device']}"
    )
    print(
        f"nerfstudio={runtime['nerfstudio_version']}@{runtime['nerfstudio_revision'][:12]} "
        f"tiny_cuda_nn={runtime['tiny_cuda_nn_revision'][:12]}"
    )
    print(
        f"torch={runtime['torch_version']} cuda={runtime['torch_cuda_version']} "
        f"gpu={runtime['gpu_name']} vram_mib={runtime['total_vram_mib']:.1f}"
    )
    print("training_completed_without_oom=True render_capable=True")
    print(f"preview={summary['preview_path']}")
    print(f"checkpoint={summary['checkpoint_path']}")
    print("scene_role=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print("quality_benchmark_deferred_to=8.7E3 performance_hardening_deferred_to=8.7E4")
    print("reconstruction_ui_enabled=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
