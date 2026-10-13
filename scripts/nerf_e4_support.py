#!/usr/bin/env python3
"""Dependency-light orchestration helpers for Phase 8.7E4 NeRF tools."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
from typing import Iterator

from core.reconstruction import (
    NERFSTUDIO_E2_CUDA,
    NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY,
    NERFSTUDIO_E2_TORCH,
    NERFSTUDIO_RELEASE_REVISION_PREFIX,
    NERFSTUDIO_RELEASE_TAG,
    TINY_CUDA_NN_E2_REVISION,
    NERF_E4_RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB,
    NerfHardwareProfile,
    probe_nerf_environment,
)


def default_env_python(name: str = "eia-nerf-nerfstudio") -> Path:
    return Path.home() / "anaconda3" / "envs" / name / "bin" / "python"


def git_revision(path: Path) -> str:
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


def version_base(value: str) -> str:
    return str(value).split("+")[0]


@contextmanager
def isolated_environment(python: Path) -> Iterator[Path]:
    prefix = python.resolve().parent.parent
    keys = (
        "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "CUDA_HOME", "CUDACXX",
        "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONNOUSERSITE",
    )
    previous = {key: os.environ.get(key) for key in keys}

    def prepend(key: str, paths: list[Path]) -> None:
        values = [str(path) for path in paths if path.exists()]
        old = os.environ.get(key, "")
        if old:
            values.append(old)
        if values:
            os.environ[key] = os.pathsep.join(values)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    prepend("PATH", [prefix / "bin"])
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)
    libraries = [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"]
    prepend("LD_LIBRARY_PATH", libraries)
    prepend("LIBRARY_PATH", libraries)
    try:
        yield prefix
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_e3_split(path: Path, dataset_fingerprint: str) -> tuple[tuple[str, ...], tuple[str, ...], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("phase") != "8.7E3":
        raise RuntimeError("split manifest is not a Phase 8.7E3 preparation manifest")
    if str(payload.get("dataset_fingerprint_sha256", "")) != dataset_fingerprint:
        raise RuntimeError("split manifest dataset fingerprint mismatch")
    training = tuple(str(value) for value in payload.get("training_frame_ids", ()))
    validation = tuple(str(value) for value in payload.get("validation_frame_ids", ()))
    if not training or not validation:
        raise RuntimeError("split manifest does not contain train/held-out views")
    return training, validation, payload


def preflight(
    *,
    python: Path,
    nerfstudio_root: Path,
    tcnn_root: Path,
    hardware_profile: NerfHardwareProfile,
) -> dict:
    probe = probe_nerf_environment(python, timeout_s=60.0)
    reasons: list[str] = []
    if not probe.interpreter_is_isolated:
        reasons.append("interpreter_not_isolated")
    if not probe.backend_ready:
        reasons.append("backend_environment_not_ready")
    if probe.nerfstudio_version != NERFSTUDIO_RELEASE_TAG.removeprefix("v"):
        reasons.append("nerfstudio_version_mismatch")
    if version_base(probe.torch_version) != NERFSTUDIO_E2_TORCH:
        reasons.append("torch_version_mismatch")
    if not probe.torch_cuda_version.startswith(NERFSTUDIO_E2_CUDA):
        reasons.append("torch_cuda_version_mismatch")
    if not git_revision(nerfstudio_root).startswith(NERFSTUDIO_RELEASE_REVISION_PREFIX):
        reasons.append("nerfstudio_revision_mismatch")
    if git_revision(tcnn_root) != TINY_CUDA_NN_E2_REVISION:
        reasons.append("tiny_cuda_nn_revision_mismatch")

    if hardware_profile is NerfHardwareProfile.RTX_2060_6GB:
        if not probe.cuda_device_names or "RTX 2060" not in probe.cuda_device_names[0]:
            reasons.append("rtx2060_gpu_name_mismatch")
        if (
            not probe.cuda_compute_capabilities
            or probe.cuda_compute_capabilities[0] != NERFSTUDIO_E2_RTX2060_COMPUTE_CAPABILITY
        ):
            reasons.append("rtx2060_compute_capability_mismatch")
        if (
            not probe.cuda_total_vram_mib
            or probe.cuda_total_vram_mib[0] < NERF_E4_RTX2060_MIN_CUDA_VISIBLE_VRAM_MIB
        ):
            reasons.append("rtx2060_cuda_visible_vram_below_profile")
    elif hardware_profile is NerfHardwareProfile.RTX_5090_32GB:
        # This profile is readiness-only unless the runner is actually executed on
        # an RTX 5090.  No synthetic performance numbers are generated.
        if probe.cuda_device_names and "RTX 5090" not in probe.cuda_device_names[0]:
            reasons.append("rtx5090_not_present_readiness_only")
    return {"probe": probe.as_dict(), "reasons": reasons}


def worker_environment(*, nerfstudio_root: Path, tcnn_root: Path, allow_faults: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["DEPTH3D_NERFSTUDIO_ROOT"] = str(nerfstudio_root)
    env["DEPTH3D_TCNN_ROOT"] = str(tcnn_root)
    if allow_faults:
        env["DEPTH3D_NERF_E4_TEST_FAULTS"] = "1"
    else:
        env.pop("DEPTH3D_NERF_E4_TEST_FAULTS", None)
    return env


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
