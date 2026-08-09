#!/usr/bin/env python3
"""Run Phase 8.7E4 real NeRF performance and artifact hardening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NERFSTUDIO_METHOD,
    NerfArtifactIntegrityEvidence,
    NerfBackendKind,
    NerfHardwareEvidence,
    NerfHardwareProfile,
    NerfOptimizationProfile,
    NerfPerformanceHardeningReport,
    NerfPerformanceMetrics,
    neural_scene_dataset_fingerprint,
    nerf_rtx5090_readiness_profile,
    read_neural_scene_dataset_manifest,
)
from scripts.nerf_backend_lifecycle import run_supervised_process  # noqa: E402
from scripts.nerf_e4_support import (  # noqa: E402
    default_env_python,
    isolated_environment,
    load_e3_split,
    preflight,
    worker_environment,
    write_json,
)

DEFAULT_OUTPUT = Path("/tmp/phase8_7e4_nerf_performance")


def _hardware(payload: dict) -> NerfHardwareEvidence:
    return NerfHardwareEvidence(
        profile=NerfHardwareProfile(str(payload["profile"])),
        gpu_name=str(payload["gpu_name"]),
        gpu_uuid=str(payload["gpu_uuid"]),
        compute_capability=tuple(int(v) for v in payload["compute_capability"]),
        total_vram_mib=float(payload["total_vram_mib"]),
        driver_version=str(payload["driver_version"]),
        torch_cuda_version=str(payload["torch_cuda_version"]),
        measured=bool(payload.get("measured", True)),
    )


def _performance(payload: dict) -> NerfPerformanceMetrics:
    return NerfPerformanceMetrics(
        training_steps=int(payload["training_steps"]),
        training_duration_s=float(payload["training_duration_s"]),
        training_peak_vram_mib=float(payload["training_peak_vram_mib"]),
        checkpoint_reload_s=float(payload["checkpoint_reload_s"]),
        render_latency_ms=tuple(float(v) for v in payload["render_latency_ms"]),
        render_peak_vram_mib=float(payload["render_peak_vram_mib"]),
        render_width=int(payload["render_width"]),
        render_height=int(payload["render_height"]),
        rendering_definition=str(payload.get("rendering_definition", "native_model_full_image_cuda_latency_excludes_file_io")),
    )


def _artifact(payload: dict) -> NerfArtifactIntegrityEvidence:
    return NerfArtifactIntegrityEvidence(
        config_path=str(payload["config_path"]),
        checkpoint_path=str(payload["checkpoint_path"]),
        config_size_bytes=int(payload["config_size_bytes"]),
        checkpoint_size_bytes=int(payload["checkpoint_size_bytes"]),
        config_sha256=str(payload["config_sha256"]),
        checkpoint_sha256=str(payload["checkpoint_sha256"]),
        checkpoint_reload_s=float(payload["checkpoint_reload_s"]),
        render_probe_passed=bool(payload["render_probe_passed"]),
        missing_artifact_rejected=bool(payload["missing_artifact_rejected"]),
        malformed_checkpoint_rejected=bool(payload["malformed_checkpoint_rejected"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--source-label", default="phase8-7e3-real-scene")
    parser.add_argument("--profile", choices=[value.value for value in NerfOptimizationProfile], default="efficient")
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--render-repeats", type=int, default=8)
    parser.add_argument(
        "--hardware-profile",
        choices=[NerfHardwareProfile.RTX_2060_6GB.value, NerfHardwareProfile.CUSTOM.value],
        default=NerfHardwareProfile.RTX_2060_6GB.value,
    )
    parser.add_argument("--nerf-python", type=Path, default=default_env_python())
    parser.add_argument("--nerfstudio-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/nerfstudio")
    parser.add_argument("--tiny-cuda-nn-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/tiny-cuda-nn")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    if not 128 <= args.training_steps <= 5000:
        raise RuntimeError("training-steps must lie in [128, 5000]")
    if not 3 <= args.render_repeats <= 100:
        raise RuntimeError("render-repeats must lie in [3, 100]")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"

    python = args.nerf_python.expanduser().resolve()
    nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    tcnn_root = args.tiny_cuda_nn_root.expanduser().resolve()
    hardware_profile = NerfHardwareProfile(args.hardware_profile)
    profile = NerfOptimizationProfile(args.profile)

    summary: dict = {
        "phase": "8.7E4",
        "pass": False,
        "scope": "nerf_backend_performance_and_artifact_hardening",
        "source_label": args.source_label,
        "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
        "method_name": NERFSTUDIO_METHOD,
        "hardware_profile": hardware_profile.value,
        "optimization_profile": profile.value,
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
        "whole_engine_hardening_deferred_to": "8.7G",
        "whole_application_gpu_profiling_deferred_to": "8.9D",
        "rtx5090_readiness": nerf_rtx5090_readiness_profile(),
    }

    try:
        dataset_path = args.dataset_manifest.expanduser().resolve()
        dataset = read_neural_scene_dataset_manifest(dataset_path)
        dataset_fingerprint = neural_scene_dataset_fingerprint(dataset)
        training_ids, validation_ids, split_payload = load_e3_split(
            args.split_manifest.expanduser().resolve(), dataset_fingerprint
        )
        summary["dataset_fingerprint_sha256"] = dataset_fingerprint
        summary["training_frame_count"] = len(training_ids)
        summary["validation_frame_count"] = len(validation_ids)
        summary["split_preparation"] = {
            "phase": split_payload.get("phase"),
            "heldout_policy": split_payload.get("heldout_policy"),
            "selected_frame_count": split_payload.get("selected_frame_count"),
        }

        with isolated_environment(python):
            pf = preflight(
                python=python,
                nerfstudio_root=nerfstudio_root,
                tcnn_root=tcnn_root,
                hardware_profile=hardware_profile,
            )
            summary["environment_probe"] = pf["probe"]
            summary["preflight_reasons"] = pf["reasons"]
            if pf["reasons"]:
                raise RuntimeError("preflight failed: " + ",".join(pf["reasons"]))

            request = {
                "phase": "8.7E4",
                "request_id": "phase8-7e4-real-nerfacto-performance",
                "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
                "method_name": NERFSTUDIO_METHOD,
                "dataset_manifest_path": str(dataset_path),
                "dataset_fingerprint_sha256": dataset_fingerprint,
                "training_frame_ids": list(training_ids),
                "validation_frame_ids": list(validation_ids),
                "profile": profile.value,
                "hardware_profile": hardware_profile.value,
                "training_steps": int(args.training_steps),
                "render_repeats": int(args.render_repeats),
                "output_dir": str(output / "worker-output"),
                "lifecycle_fault_mode": "none",
            }
            request_path = write_json(output / "worker_request.json", request)
            result_path = output / "worker_result.json"
            worker = ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py"
            outcome = run_supervised_process(
                [str(python), str(worker), "--request", str(request_path), "--result", str(result_path)],
                timeout_s=float(args.timeout_s),
                env=worker_environment(nerfstudio_root=nerfstudio_root, tcnn_root=tcnn_root),
            )
            if outcome.exit_code != 0 or outcome.cancelled or outcome.timed_out:
                raise RuntimeError(f"E4 worker failed: {outcome.stdout.strip()}")
            if not outcome.process_group_clean:
                raise RuntimeError("E4 performance worker left a process group behind")
            if not result_path.is_file():
                raise RuntimeError("E4 performance worker did not produce result JSON")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not bool(result.get("complete")):
                raise RuntimeError("E4 worker result is incomplete")

        hardware = _hardware(result["hardware"])
        performance = _performance(result["performance"])
        artifact_integrity = _artifact(result["artifact_integrity"])
        report = NerfPerformanceHardeningReport(
            hardware=hardware,
            profile=profile,
            performance=performance,
            artifact_integrity=artifact_integrity,
            real_measurements_complete=True,
        )
        if not report.passed:
            raise RuntimeError("E4 performance/artifact report did not pass")

        report_path = write_json(output / "performance_hardening_report.json", report.as_dict())
        rtx5090_readiness_path = write_json(
            output / "rtx5090_readiness_report.json",
            summary["rtx5090_readiness"],
        )
        summary.update(
            {
                "pass": True,
                "performance_metrics_complete": True,
                "artifact_hardening_complete": True,
                "report": report.as_dict(),
                "report_path": str(report_path),
                "worker_process_group_clean": True,
                "runtime_evidence_path": result.get("runtime_evidence_path"),
                "rtx5090_readiness_report_path": str(rtx5090_readiness_path),
            }
        )
    except Exception as exc:
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(summary_path, summary)
        print("Phase 8.7E4 NeRF performance + artifact hardening: FAIL")
        print(summary["failure"])
        print(f"summary={summary_path}")
        return 1

    write_json(summary_path, summary)
    perf = summary["report"]["performance"]
    artifact = summary["report"]["artifact_integrity"]
    hardware = summary["report"]["hardware"]
    print("Phase 8.7E4 NeRF performance + artifact hardening: PASS")
    print(
        f"gpu={hardware['gpu_name']} vram_mib={hardware['total_vram_mib']:.1f} "
        f"steps={perf['training_steps']} training_s={perf['training_duration_s']:.3f} "
        f"training_peak_vram_mib={perf['training_peak_vram_mib']:.1f}"
    )
    print(
        f"render={perf['render_width']}x{perf['render_height']} "
        f"mean_ms={perf['mean_render_latency_ms']:.3f} p95_ms={perf['p95_render_latency_ms']:.3f} "
        f"fps={perf['render_fps']:.2f} render_peak_vram_mib={perf['render_peak_vram_mib']:.1f}"
    )
    print(
        f"checkpoint_reload_s={artifact['checkpoint_reload_s']:.3f} "
        "missing_artifact_rejected=True malformed_checkpoint_rejected=True render_probe_passed=True"
    )
    readiness = summary["rtx5090_readiness"]
    print(
        "rtx5090_profile_defined=True "
        f"current_e2_runtime_blackwell_qualified={readiness['current_runtime_blackwell_qualified']} "
        "real_5090_performance_measurements=None"
    )
    print(
        "blackwell_runtime_requires_cuda>=12.8_and_tcnn_sm120=True "
        "no_fake_rtx5090_numbers=True"
    )
    print("backend_process_scope_only=True whole_engine_hardening_deferred_to=8.7G")
    print("whole_application_gpu_profiling_deferred_to=8.9D")
    print("canonical_geometry_authoritative=False certified_metrology=False reconstruction_ui_enabled=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
