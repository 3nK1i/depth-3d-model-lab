"""Phase 9.6E3 NeRF build/train job preparation.

The production ``depth`` process prepares exact canonical RGB/camera evidence and
launches the already-qualified Nerfstudio runtime as the *direct* E1 process.
Nerfstudio/tiny-cuda-nn are therefore never imported into ``depth`` and E1's
process-group cancellation reaches the heavyweight trainer directly.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from uuid import uuid4

from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
)
from core.reconstruction import (
    NerfBackendKind,
    NerfOptimizationProfile,
    NerfWorkerRequest,
    default_phase9_nerf_runtime_spec,
    materialize_nerf_input_from_da3_persistent_scene,
)

NERF_BUILD_PROTOCOL = "eia.digital-twin.nerf-build.v1"
NERF_BUILD_PHASE = "9.6E3"


@dataclass(frozen=True, slots=True)
class NerfBuildPreset:
    key: str
    label: str
    profile: NerfOptimizationProfile
    training_steps: int


NERF_BUILD_PRESETS: tuple[NerfBuildPreset, ...] = (
    NerfBuildPreset("efficient", "Efficient • 64 steps", NerfOptimizationProfile.EFFICIENT, 64),
    NerfBuildPreset("balanced", "Balanced • 128 steps", NerfOptimizationProfile.BALANCED, 128),
    NerfBuildPreset("max_qualified", "Max qualified • 256 steps", NerfOptimizationProfile.HIGH_QUALITY, 256),
)
NERF_DEFAULT_BUILD_PRESET = "efficient"


def nerf_build_preset(key: str) -> NerfBuildPreset:
    target = str(key).strip()
    for preset in NERF_BUILD_PRESETS:
        if preset.key == target:
            return preset
    raise ValueError(f"Unknown NeRF build preset: {target or '<empty>'}")


def _isolated_environment_overrides(python_executable: Path, *, nerfstudio_root: Path, tcnn_root: Path) -> tuple[tuple[str, str], ...]:
    prefix = python_executable.parent.parent
    if python_executable.parent.name != "bin":
        raise ValueError(f"Cannot infer isolated NeRF conda prefix from {python_executable}.")

    def prepend(existing: str, paths: list[Path]) -> str:
        parts = [str(path) for path in paths if path.is_dir()]
        if existing:
            parts.append(existing)
        return os.pathsep.join(parts)

    overrides: list[tuple[str, str]] = [
        ("CONDA_PREFIX", str(prefix)),
        ("CONDA_DEFAULT_ENV", prefix.name),
        ("PYTHONNOUSERSITE", "1"),
        ("PYTHONPATH", ""),
        ("DEPTH3D_NERFSTUDIO_ROOT", str(nerfstudio_root)),
        ("DEPTH3D_TCNN_ROOT", str(tcnn_root)),
        ("PATH", prepend(os.environ.get("PATH", ""), [prefix / "bin"])),
        ("LD_LIBRARY_PATH", prepend(os.environ.get("LD_LIBRARY_PATH", ""), [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"])),
        ("LIBRARY_PATH", prepend(os.environ.get("LIBRARY_PATH", ""), [prefix / "lib", prefix / "lib64", prefix / "targets" / "x86_64-linux" / "lib"])),
    ]
    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        overrides.extend((("CUDA_HOME", str(prefix)), ("CUDACXX", str(nvcc))))
    return tuple(overrides)


def build_nerf_training_job(
    *,
    scene,
    scene_manifest: str | Path,
    output_root: str | Path,
    preset_key: str = NERF_DEFAULT_BUILD_PRESET,
) -> tuple[DigitalTwinProcessingJobSpec, DigitalTwinProcessingCommand]:
    """Prepare exact canonical evidence and return one isolated E1 NeRF job."""

    if scene is None or not bool(getattr(scene, "canonical_geometry_authoritative", False)):
        raise ValueError("NeRF build requires an authoritative persistent canonical scene.")
    manifest = Path(scene_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise ValueError(f"Persistent scene manifest is unavailable: {manifest}")
    preset = nerf_build_preset(preset_key)
    job_id = str(uuid4())
    root = (Path(output_root).expanduser().resolve() / f"nerf-build-{job_id}").resolve()
    root.mkdir(parents=True, exist_ok=False)

    bridge = materialize_nerf_input_from_da3_persistent_scene(scene, output_dir=root / "canonical-input")
    runtime = default_phase9_nerf_runtime_spec()
    python = Path(runtime.python_executable).expanduser().resolve()
    worker = (Path(__file__).resolve().parents[1] / "scripts" / "nerf_workers" / "phase9_6e3_nerf_build_worker.py").resolve()
    if not python.is_file():
        raise ValueError(f"Qualified external NeRF interpreter is unavailable: {python}")
    if not worker.is_file():
        raise ValueError(f"Phase 9.6E3 NeRF build worker is unavailable: {worker}")

    request = NerfWorkerRequest.from_dataset(
        request_id=f"phase9-6e3-{job_id}",
        backend_kind=NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL,
        profile=preset.profile,
        dataset=bridge.dataset,
        dataset_manifest_path=bridge.manifest_path,
        output_dir=root / "training",
        maximum_input_frames=max(3, bridge.frame_count),
        qualification_training_steps=preset.training_steps,
    )
    rebuild = bool(getattr(scene, "nerf_artifact", None) is not None)
    request_path = root / "phase9_6e3_build_request.json"
    request_payload = {
        "phase": NERF_BUILD_PHASE,
        "protocol": NERF_BUILD_PROTOCOL,
        "job_id": job_id,
        "scene_id": str(scene.scene_id),
        "source_scene_manifest": str(manifest),
        "dataset_manifest": bridge.manifest_path,
        "dataset_fingerprint_sha256": bridge.dataset_fingerprint_sha256,
        "frame_count": bridge.frame_count,
        "preset_key": preset.key,
        "preset_label": preset.label,
        "training_steps": preset.training_steps,
        "rebuild": rebuild,
        "nerf_worker_request": request.as_dict(),
        "canonical_geometry_authoritative": False,
        "metric_scale_authoritative": False,
        "measurement_authoritative": False,
        "qc_authoritative": False,
        "certified_metrology": False,
    }
    request_path.write_text(json.dumps(request_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    title = "Rebuild NeRF" if rebuild else "Build & Train NeRF"
    spec = DigitalTwinProcessingJobSpec(
        job_id=job_id,
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title=title,
        backend_label="Nerfstudio Nerfacto",
        runtime_label="external eia-nerf-nerfstudio",
        output_dir=root,
        expected_protocol=NERF_BUILD_PROTOCOL,
        scene_id=str(scene.scene_id),
        source_label=manifest.name,
        profile_label=preset.label,
    )
    command = DigitalTwinProcessingCommand(
        program=python,
        arguments=(str(worker), "--request", str(request_path)),
        working_directory=Path(__file__).resolve().parents[1],
        environment_overrides=_isolated_environment_overrides(
            python,
            nerfstudio_root=Path(runtime.nerfstudio_root).resolve(),
            tcnn_root=Path(runtime.tiny_cuda_nn_root).resolve(),
        ),
    )
    return spec, command


__all__ = [
    "NERF_BUILD_PROTOCOL",
    "NERF_BUILD_PHASE",
    "NERF_BUILD_PRESETS",
    "NERF_DEFAULT_BUILD_PRESET",
    "NerfBuildPreset",
    "build_nerf_training_job",
    "nerf_build_preset",
]
