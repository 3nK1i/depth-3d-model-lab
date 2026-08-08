#!/usr/bin/env python3
"""Run real isolated Graphdeco/gsplat qualification for Phase 8.7D2."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    GAUSSIAN_QUALIFIED_CUDA,
    GAUSSIAN_QUALIFIED_PYTHON,
    GAUSSIAN_QUALIFIED_TORCH,
    GRAPHDECO_QUALIFIED_REVISION,
    GSPLAT_QUALIFIED_VERSION,
    GaussianBackendKind,
    GaussianBackendQualificationStatus,
    GaussianOptimizationProfile,
    GaussianWorkerProcessSpec,
    GaussianWorkerQualificationEvidence,
    GaussianWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    assess_gaussian_backend_qualification,
    build_gaussian_seed,
    generate_quality_aware_surface_mesh,
    graphdeco_checkout_revision,
    neural_scene_dataset_fingerprint,
    not_ready_gaussian_backend_qualification,
    probe_gaussian_environment,
    probe_gaussian_runtime_evidence,
    run_gaussian_external_worker,
    write_gaussian_backend_qualification_json,
    write_gaussian_seed_npz,
    write_neural_scene_dataset_manifest,
)
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera

DEFAULT_OUTPUT = Path("/tmp/phase8_7d2_gaussian_qualification")


def _default_env_python(env_name: str) -> Path:
    executable = Path(sys.executable).resolve()
    # Production conda path is normally .../envs/depth/bin/python.
    if len(executable.parents) >= 3 and executable.parent.name == "bin":
        envs_dir = executable.parents[2]
        if envs_dir.name == "envs":
            return envs_dir / env_name / "bin" / "python"
    return Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python"


def _write_ppm(path: Path, value: int) -> None:
    width, height = 32, 24
    rgb = bytes([value, min(255, value + 20), max(0, value - 20)])
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb * (width * height))


def _dataset(surface, output: Path) -> NeuralSceneDataset:
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(4):
        image = images / f"frame_{index:03d}.ppm"
        _write_ppm(image, 70 + index * 20)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = (index - 1.5) * 0.08
        frames.append(
            NeuralSceneFrame(
                frame_id=f"gaussian-qualification-{index}",
                image_path=str(image.resolve()),
                width=32,
                height=24,
                fx=28.0,
                fy=28.0,
                cx=16.0,
                cy=12.0,
                T_reconstruction_from_camera=pose,
                pose_layer=("optimized_keyframe" if index in {0, 3} else "optimized_propagated"),
                camera_profile_id=surface.camera_profile_id,
                timestamp_ns=index * 33_333_333,
            )
        )
    return NeuralSceneDataset(
        session_id="phase8-7d2-real-backend-qualification",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        camera_profile_id=surface.camera_profile_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        scale_status=surface.scale_status,
        world_alignment_status=surface.world_alignment_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        frames=tuple(frames),
    )


def _version_base(value: str) -> str:
    return str(value).split("+")[0]


@contextmanager
def _isolated_backend_environment(python_executable: Path):
    """Temporarily expose the selected external conda prefix to child processes.

    Calling an environment's Python by absolute path selects its site-packages,
    but it does not activate that conda environment.  CUDA extensions may still
    need that prefix on PATH/CUDA_HOME/loader paths (especially gsplat JIT).
    These overrides are scoped to one backend qualification and are restored
    before returning to the production `depth` process.
    """

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

    def _prepend(key: str, values: list[Path]) -> None:
        existing = previous.get(key) or ""
        parts = [str(value) for value in values if value.is_dir()]
        if existing:
            parts.append(existing)
        os.environ[key] = os.pathsep.join(parts)

    os.environ["CONDA_PREFIX"] = str(prefix)
    os.environ["CONDA_DEFAULT_ENV"] = prefix.name
    os.environ["PYTHONNOUSERSITE"] = "1"
    _prepend("PATH", [prefix / "bin"])

    nvcc = prefix / "bin" / "nvcc"
    if nvcc.is_file():
        os.environ["CUDA_HOME"] = str(prefix)
        os.environ["CUDACXX"] = str(nvcc)

    cuda_lib_dirs = [
        prefix / "lib",
        prefix / "lib64",
        prefix / "targets" / "x86_64-linux" / "lib",
    ]
    _prepend("LD_LIBRARY_PATH", cuda_lib_dirs)
    _prepend("LIBRARY_PATH", cuda_lib_dirs)

    try:
        yield prefix
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _runtime_preflight_reasons(backend_kind, probe, runtime, graphdeco_revision: str) -> list[str]:
    reasons: list[str] = []
    if not probe.interpreter_is_isolated:
        reasons.append("interpreter_not_isolated")
    if not probe.backend_ready:
        reasons.append("backend_environment_not_ready")
    if not runtime.python_version.startswith(GAUSSIAN_QUALIFIED_PYTHON + "."):
        reasons.append("python_version_mismatch")
    if _version_base(runtime.torch_version) != GAUSSIAN_QUALIFIED_TORCH:
        reasons.append("torch_version_mismatch")
    if not runtime.torch_cuda_version.startswith(GAUSSIAN_QUALIFIED_CUDA):
        reasons.append("torch_cuda_version_mismatch")
    if not runtime.cuda_available or not runtime.device_capabilities:
        reasons.append("cuda_device_unavailable")
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        if _version_base(probe.gsplat_version) != GSPLAT_QUALIFIED_VERSION:
            reasons.append("gsplat_version_mismatch")
    elif graphdeco_revision.lower() != GRAPHDECO_QUALIFIED_REVISION:
        reasons.append("graphdeco_revision_mismatch")
    return reasons


def _qualify_one(
    *,
    backend_kind: GaussianBackendKind,
    python_executable: Path,
    graphdeco_root: Path | None,
    dataset: NeuralSceneDataset,
    manifest_path: Path,
    seed_path: Path,
    output: Path,
    profile: GaussianOptimizationProfile,
    timeout_s: float,
):
    with _isolated_backend_environment(python_executable):
        probe = probe_gaussian_environment(
            python_executable,
            backend_kind=backend_kind,
            graphdeco_root=graphdeco_root,
            timeout_s=60.0,
        )
        runtime = probe_gaussian_runtime_evidence(python_executable, timeout_s=60.0)
        revision = ""
        if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL and graphdeco_root is not None:
            revision = graphdeco_checkout_revision(graphdeco_root)

        preflight = _runtime_preflight_reasons(backend_kind, probe, runtime, revision)
        report_path = output / f"{backend_kind.value}_qualification.json"
        if preflight:
            qualification = not_ready_gaussian_backend_qualification(
                backend_kind=backend_kind,
                environment_probe=probe,
                runtime_evidence=runtime,
                reason=",".join(preflight),
            )
            write_gaussian_backend_qualification_json(qualification, report_path)
            return qualification

        request = GaussianWorkerRequest.from_dataset(
            request_id=f"phase8-7d2-{backend_kind.value}",
            backend_kind=backend_kind,
            profile=profile,
            dataset=dataset,
            dataset_manifest_path=manifest_path,
            seed_path=seed_path,
            output_dir=output / "worker-output",
            maximum_input_frames=16,
        )
        worker_script = (
            ROOT / "scripts" / "gaussian_workers" /
            ("gsplat_worker.py" if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL else "graphdeco_worker.py")
        ).resolve()
        spec = GaussianWorkerProcessSpec(
            python_executable=str(python_executable.resolve()),
            worker_script=str(worker_script),
            timeout_s=timeout_s,
            require_isolated_interpreter=True,
        )

        old_graphdeco_root = os.environ.get("EIA_GRAPHDECO_ROOT")
        if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
            assert graphdeco_root is not None
            os.environ["EIA_GRAPHDECO_ROOT"] = str(graphdeco_root.resolve())
        try:
            artifact = run_gaussian_external_worker(spec, request)
        finally:
            if backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL:
                if old_graphdeco_root is None:
                    os.environ.pop("EIA_GRAPHDECO_ROOT", None)
                else:
                    os.environ["EIA_GRAPHDECO_ROOT"] = old_graphdeco_root

        metrics_path = Path(artifact.artifact_dir) / "qualification_metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"real Gaussian worker did not create {metrics_path}")
        worker_evidence = GaussianWorkerQualificationEvidence.from_file(metrics_path)
        qualification = assess_gaussian_backend_qualification(
            backend_kind=backend_kind,
            environment_probe=probe,
            runtime_evidence=runtime,
            artifact=artifact,
            worker_evidence=worker_evidence,
            expected_dataset_fingerprint_sha256=neural_scene_dataset_fingerprint(dataset),
            graphdeco_revision=revision,
        )
        write_gaussian_backend_qualification_json(qualification, report_path)
        return qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gsplat", "graphdeco", "both"), default="both")
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("eia-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("eia-gaussian-graphdeco"))
    parser.add_argument(
        "--graphdeco-root",
        type=Path,
        default=Path.home() / "Desktop" / "eia-gaussian-backends" / "gaussian-splatting",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--profile",
        choices=("efficient", "balanced", "high_quality"),
        default="efficient",
    )
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    camera = _camera(96, 72)
    dense, quality, _scale = _arbitrary_surface(96, 72, camera)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    canonical_positions = surface.mesh.vertices_reconstruction.copy()
    seed = build_gaussian_seed(surface, max_gaussians=512, minimum_vertex_quality=0.75)
    dataset = _dataset(surface, output)
    manifest_path = write_neural_scene_dataset_manifest(dataset, output / "dataset.json")
    seed_path = write_gaussian_seed_npz(seed, output / "gaussian_seed.npz")
    canonical_surface_unchanged = bool(np.array_equal(surface.mesh.vertices_reconstruction, canonical_positions))

    selected: list[tuple[str, GaussianBackendKind, Path, Path | None]] = []
    if args.backend in {"gsplat", "both"}:
        selected.append(("gsplat", GaussianBackendKind.GSPLAT_EXTERNAL, args.gsplat_python.expanduser().resolve(), None))
    if args.backend in {"graphdeco", "both"}:
        selected.append((
            "graphdeco",
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
            args.graphdeco_python.expanduser().resolve(),
            args.graphdeco_root.expanduser().resolve(),
        ))

    profile = GaussianOptimizationProfile(args.profile)
    qualifications = {}
    for name, backend_kind, python_executable, graphdeco_root in selected:
        try:
            qualification = _qualify_one(
                backend_kind=backend_kind,
                python_executable=python_executable,
                graphdeco_root=graphdeco_root,
                dataset=dataset,
                manifest_path=manifest_path,
                seed_path=seed_path,
                output=output,
                profile=profile,
                timeout_s=float(args.timeout_s),
            )
        except Exception as exc:  # Preserve a concise top-level target-hardware report.
            qualifications[name] = {
                "status": "worker_error",
                "qualified": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        qualifications[name] = qualification.as_dict()

    all_qualified = bool(qualifications) and all(
        bool(payload.get("qualified", False)) for payload in qualifications.values()
    )
    app_yaml = (ROOT / "config" / "app.yaml").read_text(encoding="utf-8").lower()
    reconstruction_ui_enabled = "reconstruction_mode_enabled: true" in app_yaml
    summary = {
        "phase": "8.7D2",
        "pass": all_qualified and canonical_surface_unchanged and not reconstruction_ui_enabled,
        "selected_backend": args.backend,
        "profile": args.profile,
        "dataset_fingerprint_sha256": neural_scene_dataset_fingerprint(dataset),
        "dataset_frames": dataset.frame_count,
        "seed_gaussians": seed.gaussian_count,
        "canonical_surface_unchanged": canonical_surface_unchanged,
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "pose_convention": dataset.pose_convention,
        "metric_geometry_status": dataset.metric_geometry_status.value,
        "unit_to_metre": dataset.unit_to_metre,
        "reconstruction_ui_enabled": reconstruction_ui_enabled,
        "qualifications": qualifications,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not summary["pass"]:
        print("Phase 8.7D2 real Gaussian backend qualification: NOT QUALIFIED")
        for name, payload in qualifications.items():
            print(f"{name}={payload.get('status')} qualified={payload.get('qualified', False)}")
            if payload.get("error"):
                print(f"{name}_error={payload['error']}")
            reasons = payload.get("reasons")
            if reasons:
                print(f"{name}_reasons={','.join(str(value) for value in reasons)}")
        print(f"summary={summary_path}")
        return 1

    print("Phase 8.7D2 real Gaussian backend qualification: PASS")
    for name, payload in qualifications.items():
        artifact = payload.get("artifact") or {}
        evidence = payload.get("worker_evidence") or {}
        print(
            f"{name}=qualified steps={artifact.get('training_steps')} "
            f"gaussians={artifact.get('gaussian_count')} device={artifact.get('device')}"
        )
        print(f"{name}_preview={evidence.get('preview_path')}")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"pose_convention={dataset.pose_convention}")
    print(f"reconstruction_ui_enabled={reconstruction_ui_enabled}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
