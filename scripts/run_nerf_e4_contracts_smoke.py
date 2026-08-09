#!/usr/bin/env python3
"""Dependency-light Phase 8.7E4 NeRF hardening contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NerfArtifactIntegrityEvidence,
    NerfBackendRobustnessEvidence,
    NerfBackendRobustnessReport,
    NerfBackendRobustnessScenario,
    NerfBackendRunState,
    NerfHardwareEvidence,
    NerfHardwareProfile,
    NerfLifecycleEvidence,
    NerfOptimizationProfile,
    NerfPerformanceHardeningReport,
    NerfPerformanceMetrics,
    nerf_rtx5090_readiness_profile,
    validate_nerf_artifact_files,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase8_7e4_contracts_") as tmp:
        root = Path(tmp)
        config = root / "config.yml"
        checkpoint = root / "step-000000999.ckpt"
        config.write_text("method: nerfacto\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint-placeholder")
        validate_nerf_artifact_files(config_path=config, checkpoint_path=checkpoint, artifact_root=root)

        hardware = NerfHardwareEvidence(
            profile=NerfHardwareProfile.RTX_2060_6GB,
            gpu_name="NVIDIA GeForce RTX 2060",
            gpu_uuid="GPU-contract-smoke",
            compute_capability=(7, 5),
            total_vram_mib=5744.5,
            driver_version="contract",
            torch_cuda_version="11.8",
        )
        perf = NerfPerformanceMetrics(
            training_steps=1000,
            training_duration_s=25.0,
            training_peak_vram_mib=1800.0,
            checkpoint_reload_s=2.0,
            render_latency_ms=(120.0, 110.0, 115.0, 118.0),
            render_peak_vram_mib=1600.0,
            render_width=540,
            render_height=720,
        )
        artifact = NerfArtifactIntegrityEvidence(
            config_path=str(config),
            checkpoint_path=str(checkpoint),
            config_size_bytes=config.stat().st_size,
            checkpoint_size_bytes=checkpoint.stat().st_size,
            config_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
            checkpoint_reload_s=2.0,
            render_probe_passed=True,
            missing_artifact_rejected=True,
            malformed_checkpoint_rejected=True,
        )
        performance_report = NerfPerformanceHardeningReport(
            hardware=hardware,
            profile=NerfOptimizationProfile.EFFICIENT,
            performance=perf,
            artifact_integrity=artifact,
            real_measurements_complete=True,
        )

        evidence = []
        for scenario in NerfBackendRobustnessScenario:
            expected = (
                NerfBackendRunState.CANCELLED
                if scenario is NerfBackendRobustnessScenario.CANCELLATION
                else NerfBackendRunState.TIMED_OUT
                if scenario is NerfBackendRobustnessScenario.TIMEOUT
                else NerfBackendRunState.CRASHED
            )
            evidence.append(
                NerfBackendRobustnessEvidence(
                    scenario=scenario,
                    fault_lifecycle=NerfLifecycleEvidence(
                        state=expected,
                        worker_exit_code=-15,
                        cancellation_requested=(scenario is NerfBackendRobustnessScenario.CANCELLATION),
                    ),
                    process_group_clean=True,
                    partial_artifacts_detected=(scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT),
                    partial_artifacts_cleaned=(scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT),
                    recovery_lifecycle=NerfLifecycleEvidence(
                        state=NerfBackendRunState.RECOVERED,
                        worker_exit_code=0,
                        recovery_attempted=True,
                        recovery_succeeded=True,
                    ),
                    recovery_result_valid=True,
                )
            )
        robustness = NerfBackendRobustnessReport(evidence=tuple(evidence))
        readiness = nerf_rtx5090_readiness_profile()
        if readiness["current_runtime_blackwell_qualified"]:
            raise RuntimeError("E4 must not claim the CUDA 11.8 / SM75 E2 runtime is Blackwell-qualified.")
        if readiness["real_performance_measurements"] is not None or not readiness["no_fake_numbers"]:
            raise RuntimeError("E4 RTX 5090 readiness must not contain fabricated performance measurements.")

        payload = {
            "phase": "8.7E4",
            "performance_report_passed": performance_report.passed,
            "robustness_report_passed": robustness.passed,
            "rendering_definition": perf.rendering_definition,
            "rtx5090_readiness": readiness,
            "rtx5090_readiness_only": True,
            "rtx5090_fake_numbers": False,
            "whole_engine_hardening_deferred_to": "8.7G",
            "whole_application_gpu_profiling_deferred_to": "8.9D",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        }
        summary = Path("/tmp/phase8_7e4_nerf_contracts/summary.json")
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not performance_report.passed or not robustness.passed:
        print("Phase 8.7E4 NeRF hardening contracts: FAIL")
        return 1
    print("Phase 8.7E4 NeRF hardening contracts: PASS")
    print("performance=training_time/VRAM/native_render_latency/FPS/checkpoint_reload")
    print("robustness=cancellation/timeout/crash/partial_artifact+recovery")
    print("rtx5090_profile_defined=True current_e2_runtime_blackwell_qualified=False")
    print("blackwell_min_cuda=12.8 tcnn_target=SM120 fake_performance_numbers=False")
    print("backend_process_scope_only=True whole_engine_hardening_deferred_to=8.7G")
    print("whole_application_gpu_profiling_deferred_to=8.9D")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
