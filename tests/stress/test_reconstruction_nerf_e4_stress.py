from __future__ import annotations

from core.reconstruction import (
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
)


def _robustness_report() -> NerfBackendRobustnessReport:
    values = []
    for scenario in NerfBackendRobustnessScenario:
        expected = (
            NerfBackendRunState.CANCELLED if scenario is NerfBackendRobustnessScenario.CANCELLATION
            else NerfBackendRunState.TIMED_OUT if scenario is NerfBackendRobustnessScenario.TIMEOUT
            else NerfBackendRunState.CRASHED
        )
        values.append(
            NerfBackendRobustnessEvidence(
                scenario=scenario,
                fault_lifecycle=NerfLifecycleEvidence(state=expected, worker_exit_code=-15),
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
    return NerfBackendRobustnessReport(evidence=tuple(values))


def test_e4_contract_serialization_is_stable_across_repeated_construction(tmp_path) -> None:
    config = tmp_path / "config.yml"
    checkpoint = tmp_path / "x.ckpt"
    config.write_text("x: 1\n")
    checkpoint.write_bytes(b"x")
    values = set()
    for _ in range(100):
        hardware = NerfHardwareEvidence(
            profile=NerfHardwareProfile.RTX_2060_6GB,
            gpu_name="RTX 2060", gpu_uuid="GPU-x", compute_capability=(7, 5),
            total_vram_mib=5744.5, driver_version="driver", torch_cuda_version="11.8",
        )
        perf = NerfPerformanceMetrics(
            training_steps=1000, training_duration_s=20.0, training_peak_vram_mib=1200.0,
            checkpoint_reload_s=2.0, render_latency_ms=(100.0, 110.0, 105.0),
            render_peak_vram_mib=1000.0, render_width=720, render_height=540,
        )
        artifact = NerfArtifactIntegrityEvidence(
            config_path=str(config), checkpoint_path=str(checkpoint),
            config_size_bytes=5, checkpoint_size_bytes=1,
            config_sha256="a" * 64, checkpoint_sha256="b" * 64,
            checkpoint_reload_s=2.0, render_probe_passed=True,
            missing_artifact_rejected=True, malformed_checkpoint_rejected=True,
        )
        report = NerfPerformanceHardeningReport(
            hardware=hardware, profile=NerfOptimizationProfile.EFFICIENT,
            performance=perf, artifact_integrity=artifact, real_measurements_complete=True,
        )
        values.add(repr(report.as_dict()))
    assert len(values) == 1


def test_e4_robustness_report_stays_green_across_repeated_evaluation() -> None:
    report = _robustness_report()
    for _ in range(500):
        assert report.passed
        assert all(item.passed for item in report.evidence)
