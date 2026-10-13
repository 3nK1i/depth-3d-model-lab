#!/usr/bin/env python3
"""Run Phase 8.7D3 held-out quality and optional Step 3 performance benchmarking.

Without ``--measure-performance`` this preserves the committed Step 2 behavior.
Step 3 adds phase-specific VRAM, backend-native render throughput, and measured
hardware provenance without changing canonical geometry authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    GRAPHDECO_QUALIFIED_REVISION,
    GSPLAT_QUALIFIED_VERSION,
    GaussianBackendKind,
    GaussianBenchmarkBackendResult,
    GaussianBenchmarkHardwareProfile,
    GaussianBenchmarkRunState,
    GaussianHardwareEvidence,
    GaussianHeldOutQualityEvaluation,
    GaussianLifecycleEvidence,
    GaussianOptimizationProfile,
    GaussianPerformanceMetrics,
    RealSceneGaussianBenchmarkReport,
    RealSceneGaussianBenchmarkSpec,
    gaussian_benchmark_hardware_requirement,
    graphdeco_checkout_revision,
    probe_gaussian_environment,
    read_gaussian_held_out_quality_evaluation,
    read_gaussian_held_out_render_manifest,
    read_neural_scene_dataset_manifest,
    write_real_scene_gaussian_benchmark_report,
    write_real_scene_gaussian_benchmark_spec,
)
from scripts.run_gaussian_backend_qualification import _isolated_backend_environment
from scripts.gaussian_backend_lifecycle import run_supervised_process

DEFAULT_OUTPUT = Path("/tmp/phase8_7d3_real_scene_benchmark")
PROFILE_STEPS = {"efficient": 128, "balanced": 256, "high_quality": 512}


def _default_env_python(env_name: str) -> Path:
    executable = Path(sys.executable).resolve()
    if len(executable.parents) >= 3 and executable.parent.name == "bin":
        envs_dir = executable.parents[2]
        if envs_dir.name == "envs":
            return envs_dir / env_name / "bin" / "python"
    return Path.home() / "anaconda3" / "envs" / env_name / "bin" / "python"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_process(
    command: list[str],
    *,
    timeout_s: float,
    env: dict[str, str] | None = None,
) -> None:
    outcome = run_supervised_process(command, timeout_s=timeout_s, env=env)
    if outcome.stdout:
        print(outcome.stdout, end="" if outcome.stdout.endswith("\n") else "\n")
    if outcome.timed_out:
        tail = "\n".join(outcome.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command timed out after {timeout_s:.3f}s and process group was cleaned: "
            f"{' '.join(command)}\n{tail}"
        )
    if outcome.exit_code != 0:
        tail = "\n".join(outcome.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"command failed with exit {outcome.exit_code}: {' '.join(command)}\n{tail}"
        )


def _preflight_backend(
    backend_kind: GaussianBackendKind,
    python_executable: Path,
    graphdeco_root: Path | None,
) -> None:
    python = python_executable.expanduser().resolve()
    if not python.is_file():
        raise RuntimeError(f"isolated backend interpreter missing: {python}")
    if python == Path(sys.executable).resolve():
        raise RuntimeError("Gaussian benchmark backend must not use the production interpreter")
    with _isolated_backend_environment(python):
        probe = probe_gaussian_environment(
            python,
            backend_kind=backend_kind,
            graphdeco_root=graphdeco_root,
            timeout_s=60.0,
        )
    if not probe.backend_ready or not probe.interpreter_is_isolated:
        raise RuntimeError(f"Gaussian backend environment is not ready: {probe.as_dict()}")
    if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL:
        if str(probe.gsplat_version).split("+")[0] != GSPLAT_QUALIFIED_VERSION:
            raise RuntimeError("gsplat version differs from the D2-qualified runtime")
    else:
        assert graphdeco_root is not None
        revision = graphdeco_checkout_revision(graphdeco_root)
        if revision != GRAPHDECO_QUALIFIED_REVISION:
            raise RuntimeError("Graphdeco revision differs from the D2-qualified runtime")


def _backend_request(
    *,
    spec: RealSceneGaussianBenchmarkSpec,
    backend_kind: GaussianBackendKind,
    dataset_manifest: Path,
    seed_path: Path,
    output_dir: Path,
    training_steps: int,
    measure_performance: bool,
    render_repeats: int,
) -> dict:
    return {
        "schema_version": 1,
        "backend_kind": backend_kind.value,
        "benchmark_fingerprint_sha256": spec.benchmark_fingerprint_sha256,
        "dataset_fingerprint_sha256": spec.dataset_fingerprint_sha256,
        "profile": spec.optimization_profile.value,
        "hardware_profile": spec.hardware_profile.value,
        "dataset_manifest_path": str(dataset_manifest.resolve()),
        "seed_path": str(seed_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "training_frame_ids": list(spec.training_frame_ids),
        "validation_frame_ids": list(spec.validation_frame_ids),
        "training_steps": int(training_steps),
        "measure_performance": bool(measure_performance),
        "render_repeats": int(render_repeats),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }


def _backend_result_from_payload(
    *,
    spec: RealSceneGaussianBenchmarkSpec,
    backend_kind: GaussianBackendKind,
    payload: dict,
    quality: GaussianHeldOutQualityEvaluation,
    require_performance: bool,
) -> GaussianBenchmarkBackendResult:
    if payload.get("backend_kind") != backend_kind.value:
        raise RuntimeError("backend result kind mismatch")
    if payload.get("benchmark_fingerprint_sha256") != spec.benchmark_fingerprint_sha256:
        raise RuntimeError("backend result benchmark fingerprint mismatch")
    if payload.get("dataset_fingerprint_sha256") != spec.dataset_fingerprint_sha256:
        raise RuntimeError("backend result dataset fingerprint mismatch")
    if quality.backend_kind is not backend_kind:
        raise RuntimeError("quality evaluation backend mismatch")

    performance_payload = payload.get("performance") or {}
    hardware_payload = payload.get("hardware") or {}
    if require_performance:
        required_perf = {
            "peak_training_vram_mib",
            "peak_render_vram_mib",
            "peak_vram_mib",
            "training_duration_s",
            "render_duration_s",
            "render_frame_count",
            "render_repeats",
            "render_fps",
            "measurement_scope",
            "render_measurement_mode",
        }
        missing = sorted(required_perf - set(performance_payload))
        if missing:
            raise RuntimeError(f"backend performance payload is incomplete: {missing}")
        if performance_payload.get("measurement_scope") != "backend_process_tree_nvidia_smi":
            raise RuntimeError("unexpected GPU measurement scope")
        if float(performance_payload["peak_training_vram_mib"]) <= 0.0:
            raise RuntimeError("training VRAM measurement must be positive for a real CUDA benchmark")
        if float(performance_payload["peak_render_vram_mib"]) <= 0.0:
            raise RuntimeError("render VRAM measurement must be positive for a real CUDA benchmark")
        if float(performance_payload["render_fps"]) <= 0.0:
            raise RuntimeError("render FPS must be positive for a real CUDA benchmark")
        required_hardware = {
            "gpu_name", "gpu_uuid", "total_vram_mib", "driver_version",
            "cuda_runtime_version", "compute_capability",
        }
        missing_hardware = sorted(required_hardware - set(hardware_payload))
        if missing_hardware:
            raise RuntimeError(f"backend hardware payload is incomplete: {missing_hardware}")
        requirement = gaussian_benchmark_hardware_requirement(spec.hardware_profile)
        profile_compatible = requirement.matches(
            gpu_name=str(hardware_payload["gpu_name"]),
            total_vram_mib=float(hardware_payload["total_vram_mib"]),
        )
        if not profile_compatible:
            raise RuntimeError(
                f"measured GPU does not match requested hardware profile {spec.hardware_profile.value}: "
                f"{hardware_payload['gpu_name']} / {hardware_payload['total_vram_mib']} MiB"
            )
        performance = GaussianPerformanceMetrics(
            peak_vram_mib=float(performance_payload["peak_vram_mib"]),
            peak_training_vram_mib=float(performance_payload["peak_training_vram_mib"]),
            peak_render_vram_mib=float(performance_payload["peak_render_vram_mib"]),
            training_duration_s=float(performance_payload["training_duration_s"]),
            render_duration_s=float(performance_payload["render_duration_s"]),
            render_frame_count=int(performance_payload["render_frame_count"]),
            render_repeats=int(performance_payload["render_repeats"]),
            render_fps=float(performance_payload["render_fps"]),
            measurement_scope=str(performance_payload["measurement_scope"]),
            render_measurement_mode=str(performance_payload["render_measurement_mode"]),
        )
        hardware = GaussianHardwareEvidence(
            gpu_name=str(hardware_payload["gpu_name"]),
            gpu_uuid=str(hardware_payload["gpu_uuid"]),
            total_vram_mib=float(hardware_payload["total_vram_mib"]),
            driver_version=str(hardware_payload["driver_version"]),
            cuda_runtime_version=str(hardware_payload["cuda_runtime_version"]),
            compute_capability=tuple(int(value) for value in hardware_payload["compute_capability"]),
            profile_compatible=profile_compatible,
        )
        if not performance.detailed_measured or not hardware.measured:
            raise RuntimeError("detailed performance/hardware evidence is incomplete")
    else:
        performance = GaussianPerformanceMetrics(
            training_duration_s=float(payload["training_duration_s"]),
        )
        hardware = GaussianHardwareEvidence()

    return GaussianBenchmarkBackendResult(
        backend_kind=backend_kind,
        benchmark_fingerprint_sha256=spec.benchmark_fingerprint_sha256,
        dataset_fingerprint_sha256=spec.dataset_fingerprint_sha256,
        optimization_profile=spec.optimization_profile,
        hardware_profile=spec.hardware_profile,
        training_view_count=len(spec.training_frame_ids),
        validation_view_count=len(spec.validation_frame_ids),
        quality=quality.aggregate,
        performance=performance,
        hardware=hardware,
        lifecycle=GaussianLifecycleEvidence(
            state=GaussianBenchmarkRunState.COMPLETED,
            worker_exit_code=0,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--validation-frame-id", action="append", required=True)
    parser.add_argument("--benchmark-id", default="phase8-7d3-real-scene")
    parser.add_argument("--source-label", default="real-scene")
    parser.add_argument("--backend", choices=("gsplat", "graphdeco", "both"), default="both")
    parser.add_argument("--profile", choices=tuple(PROFILE_STEPS), default="efficient")
    parser.add_argument("--training-steps", type=int, default=0)
    parser.add_argument(
        "--measure-performance",
        action="store_true",
        help="Collect D3 Step 3 phase-specific VRAM, timing, FPS, and hardware provenance.",
    )
    parser.add_argument(
        "--render-repeats",
        type=int,
        default=3,
        help="Number of backend-native held-out render repetitions for Step 3.",
    )
    parser.add_argument(
        "--hardware-profile",
        choices=("rtx_2060_6gb", "rtx_5090_32gb", "custom"),
        default="rtx_2060_6gb",
    )
    parser.add_argument("--hardware-label", default="NVIDIA GeForce RTX 2060 6 GB")
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("eia-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("eia-gaussian-graphdeco"))
    parser.add_argument("--metrics-python", type=Path, default=_default_env_python("eia-gaussian-metrics"))
    parser.add_argument(
        "--graphdeco-root",
        type=Path,
        default=Path.home() / "Desktop" / "eia-gaussian-backends" / "gaussian-splatting",
    )
    parser.add_argument("--metrics-device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    dataset_manifest = args.dataset_manifest.expanduser().resolve()
    seed_path = args.seed.expanduser().resolve()
    if not seed_path.is_file():
        raise RuntimeError(f"Gaussian seed does not exist: {seed_path}")
    dataset = read_neural_scene_dataset_manifest(dataset_manifest)
    validation_ids = tuple(dict.fromkeys(str(value) for value in args.validation_frame_id))
    dataset_ids = tuple(frame.frame_id for frame in dataset.frames)
    unknown = sorted(set(validation_ids) - set(dataset_ids))
    if unknown:
        raise RuntimeError(f"unknown validation frame IDs: {unknown}")
    training_ids = tuple(value for value in dataset_ids if value not in set(validation_ids))

    selected: list[tuple[str, GaussianBackendKind, Path, Path | None]] = []
    if args.backend in {"gsplat", "both"}:
        selected.append(("gsplat", GaussianBackendKind.GSPLAT_EXTERNAL, args.gsplat_python, None))
    if args.backend in {"graphdeco", "both"}:
        selected.append((
            "graphdeco",
            GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
            args.graphdeco_python,
            args.graphdeco_root,
        ))

    backends = tuple(value[1] for value in selected)
    profile = GaussianOptimizationProfile(args.profile)
    hardware_profile = GaussianBenchmarkHardwareProfile(args.hardware_profile)
    spec = RealSceneGaussianBenchmarkSpec(
        benchmark_id=args.benchmark_id,
        source_label=args.source_label,
        dataset=dataset,
        training_frame_ids=training_ids,
        validation_frame_ids=validation_ids,
        backends=backends,
        optimization_profile=profile,
        hardware_profile=hardware_profile,
        hardware_label=args.hardware_label,
    )
    steps = int(args.training_steps) if int(args.training_steps) > 0 else PROFILE_STEPS[args.profile]
    if steps < 1 or steps > 100_000:
        raise RuntimeError("training_steps must lie in [1, 100000]")
    if args.render_repeats < 1 or args.render_repeats > 100:
        raise RuntimeError("render_repeats must lie in [1, 100]")

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_real_scene_gaussian_benchmark_spec(spec, output / "benchmark_spec.json")

    metrics_python = args.metrics_python.expanduser().resolve()
    if not metrics_python.is_file() or metrics_python == Path(sys.executable).resolve():
        raise RuntimeError("isolated metrics interpreter is missing or not isolated")

    results: list[GaussianBenchmarkBackendResult] = []
    evidence_paths: dict[str, dict[str, str]] = {}
    for name, backend_kind, python_value, root_value in selected:
        backend_python = python_value.expanduser().resolve()
        graphdeco_root = root_value.expanduser().resolve() if root_value is not None else None
        _preflight_backend(backend_kind, backend_python, graphdeco_root)
        backend_dir = output / name
        backend_dir.mkdir(parents=True, exist_ok=True)
        request_path = _write_json(
            backend_dir / "request.json",
            _backend_request(
                spec=spec,
                backend_kind=backend_kind,
                dataset_manifest=dataset_manifest,
                seed_path=seed_path,
                output_dir=backend_dir / "worker-output",
                training_steps=steps,
                measure_performance=bool(args.measure_performance),
                render_repeats=int(args.render_repeats),
            ),
        )
        backend_result_path = backend_dir / "backend_result.json"
        worker_name = "gsplat_worker.py" if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL else "graphdeco_worker.py"
        worker_script = ROOT / "scripts" / "gaussian_benchmark_workers" / worker_name
        with _isolated_backend_environment(backend_python):
            env = os.environ.copy()
            if graphdeco_root is not None:
                env["DEPTH3D_GRAPHDECO_ROOT"] = str(graphdeco_root)
            _run_process(
                [str(backend_python), str(worker_script), "--request", str(request_path), "--result", str(backend_result_path)],
                timeout_s=float(args.timeout_s),
                env=env,
            )
        backend_payload = json.loads(backend_result_path.read_text(encoding="utf-8"))
        render_manifest_path = Path(str(backend_payload["render_manifest_path"])).resolve()
        manifest = read_gaussian_held_out_render_manifest(render_manifest_path)
        if manifest.backend_kind is not backend_kind:
            raise RuntimeError("held-out render manifest backend mismatch")
        quality_path = backend_dir / "quality_evaluation.json"
        metrics_worker = ROOT / "scripts" / "gaussian_benchmark_workers" / "metrics_worker.py"
        _run_process(
            [
                str(metrics_python), str(metrics_worker),
                "--render-manifest", str(render_manifest_path),
                "--result", str(quality_path),
                "--device", args.metrics_device,
            ],
            timeout_s=900.0,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        quality = read_gaussian_held_out_quality_evaluation(quality_path)
        result = _backend_result_from_payload(
            spec=spec,
            backend_kind=backend_kind,
            payload=backend_payload,
            quality=quality,
            require_performance=bool(args.measure_performance),
        )
        results.append(result)
        evidence_paths[name] = {
            "backend_result": str(backend_result_path),
            "render_manifest": str(render_manifest_path),
            "quality_evaluation": str(quality_path),
            "scene_path": str(backend_payload["scene_path"]),
        }

    report = RealSceneGaussianBenchmarkReport(spec=spec, results=tuple(results))
    report_path = write_real_scene_gaussian_benchmark_report(report, output / "benchmark_report.json")
    quality_ready = len(results) == len(backends) and all(result.quality.measured for result in results)

    if args.measure_performance:
        performance_ready = all(result.performance.detailed_measured for result in results)
        hardware_ready = all(
            result.hardware.measured and result.hardware.profile_compatible is True
            for result in results
        )
        hardware_consistent = len({
            (result.hardware.gpu_uuid, result.hardware.driver_version, result.hardware.cuda_runtime_version)
            for result in results
        }) == 1
        step_pass = bool(
            quality_ready and performance_ready and hardware_ready and hardware_consistent
            and report.comparison_ready
        )
        requirement = gaussian_benchmark_hardware_requirement(spec.hardware_profile)
        summary = {
            "phase": "8.7D3",
            "step": 3,
            "pass": step_pass,
            "quality_comparison_ready": quality_ready,
            "performance_metrics_complete": performance_ready,
            "hardware_evidence_complete": hardware_ready,
            "hardware_consistent_across_backends": hardware_consistent,
            "comparison_ready": report.comparison_ready,
            "measured_hardware_profile": spec.hardware_profile.value,
            "requested_hardware_requirement": requirement.as_dict(),
            "rtx_5090_readiness_profile": gaussian_benchmark_hardware_requirement(
                GaussianBenchmarkHardwareProfile.RTX_5090_32GB
            ).as_dict(),
            "benchmark_fingerprint_sha256": spec.benchmark_fingerprint_sha256,
            "dataset_fingerprint_sha256": spec.dataset_fingerprint_sha256,
            "training_frame_ids": list(spec.training_frame_ids),
            "validation_frame_ids": list(spec.validation_frame_ids),
            "training_steps": steps,
            "render_repeats": int(args.render_repeats),
            "evidence_paths": evidence_paths,
            "report": report.as_dict(),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }
        summary_path = _write_json(output / "step3_summary.json", summary)
        if not step_pass:
            print("Phase 8.7D3 Step 3 Gaussian performance benchmark: FAIL")
            print(f"summary={summary_path}")
            return 1
        print("Phase 8.7D3 Step 3 Gaussian performance benchmark: PASS")
        for result in results:
            perf = result.performance
            print(
                f"{result.backend_kind.value}: train={perf.training_duration_s:.3f}s "
                f"train_vram={perf.peak_training_vram_mib:.1f}MiB "
                f"render_vram={perf.peak_render_vram_mib:.1f}MiB "
                f"render_fps={perf.render_fps:.2f} "
                f"gpu={result.hardware.gpu_name}"
            )
        print("quality_comparison_ready=True performance_metrics_complete=True comparison_ready=True")
        print("canonical_geometry_authoritative=False certified_metrology=False")
        print(f"report={report_path}")
        print(f"summary={summary_path}")
        return 0

    summary = {
        "phase": "8.7D3",
        "step": 2,
        "pass": quality_ready,
        "quality_comparison_ready": quality_ready,
        "comparison_ready": report.comparison_ready,
        "performance_metrics_complete": False,
        "benchmark_fingerprint_sha256": spec.benchmark_fingerprint_sha256,
        "dataset_fingerprint_sha256": spec.dataset_fingerprint_sha256,
        "training_frame_ids": list(spec.training_frame_ids),
        "validation_frame_ids": list(spec.validation_frame_ids),
        "training_steps": steps,
        "evidence_paths": evidence_paths,
        "report": report.as_dict(),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    summary_path = _write_json(output / "step2_summary.json", summary)
    if not quality_ready:
        print("Phase 8.7D3 Step 2 held-out quality benchmark: FAIL")
        print(f"summary={summary_path}")
        return 1
    print("Phase 8.7D3 Step 2 held-out quality benchmark: PASS")
    for result in results:
        assert result.quality.psnr_db is not None
        assert result.quality.ssim is not None
        assert result.quality.lpips is not None
        print(
            f"{result.backend_kind.value}: psnr={result.quality.psnr_db:.4f} "
            f"ssim={result.quality.ssim:.6f} lpips={result.quality.lpips:.6f}"
        )
    print("quality_comparison_ready=True comparison_ready=False")
    print("performance_metrics_complete=False  # VRAM/FPS belong to D3 Step 3")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"report={report_path}")
    print(f"summary={summary_path}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
