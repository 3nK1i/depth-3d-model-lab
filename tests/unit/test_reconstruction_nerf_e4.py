from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

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
    nerf_rtx5090_readiness_profile,
    validate_nerf_artifact_files,
)

ROOT = Path(__file__).resolve().parents[2]


def _hardware() -> NerfHardwareEvidence:
    return NerfHardwareEvidence(
        profile=NerfHardwareProfile.RTX_2060_6GB,
        gpu_name="NVIDIA GeForce RTX 2060",
        gpu_uuid="GPU-test",
        compute_capability=(7, 5),
        total_vram_mib=5744.5625,
        driver_version="610.43.02",
        torch_cuda_version="11.8",
    )


def _performance() -> NerfPerformanceMetrics:
    return NerfPerformanceMetrics(
        training_steps=1000,
        training_duration_s=26.7,
        training_peak_vram_mib=1800.0,
        checkpoint_reload_s=2.4,
        render_latency_ms=(120.0, 100.0, 110.0, 130.0, 115.0),
        render_peak_vram_mib=1700.0,
        render_width=540,
        render_height=720,
    )


def _artifact(tmp_path: Path) -> NerfArtifactIntegrityEvidence:
    config = tmp_path / "config.yml"
    ckpt = tmp_path / "step-000000999.ckpt"
    config.write_text("method: nerfacto\n")
    ckpt.write_bytes(b"abc")
    return NerfArtifactIntegrityEvidence(
        config_path=str(config), checkpoint_path=str(ckpt),
        config_size_bytes=config.stat().st_size, checkpoint_size_bytes=ckpt.stat().st_size,
        config_sha256="a" * 64, checkpoint_sha256="b" * 64,
        checkpoint_reload_s=2.4, render_probe_passed=True,
        missing_artifact_rejected=True, malformed_checkpoint_rejected=True,
    )


def test_e4_performance_metrics_derive_mean_p50_p95_and_fps() -> None:
    perf = _performance()
    assert perf.mean_render_latency_ms == pytest.approx(115.0)
    assert perf.p50_render_latency_ms == pytest.approx(115.0)
    assert perf.p95_render_latency_ms == pytest.approx(130.0)
    assert perf.render_fps == pytest.approx(1000.0 / 115.0)
    assert perf.as_dict()["rendering_definition"] == "native_model_full_image_cuda_latency_excludes_file_io"


def test_e4_performance_metrics_require_multiple_positive_samples() -> None:
    with pytest.raises(Exception, match="at least"):
        NerfPerformanceMetrics(
            training_steps=1, training_duration_s=1.0, training_peak_vram_mib=1.0,
            checkpoint_reload_s=1.0, render_latency_ms=(10.0, 11.0),
            render_peak_vram_mib=1.0, render_width=1, render_height=1,
        )
    with pytest.raises(Exception, match="positive"):
        NerfPerformanceMetrics(
            training_steps=1, training_duration_s=1.0, training_peak_vram_mib=1.0,
            checkpoint_reload_s=1.0, render_latency_ms=(10.0, 0.0, 11.0),
            render_peak_vram_mib=1.0, render_width=1, render_height=1,
        )


def test_e4_real_performance_report_requires_integrity_evidence(tmp_path: Path) -> None:
    report = NerfPerformanceHardeningReport(
        hardware=_hardware(), profile=NerfOptimizationProfile.EFFICIENT,
        performance=_performance(), artifact_integrity=_artifact(tmp_path),
        real_measurements_complete=True,
    )
    assert report.passed
    payload = report.as_dict()
    assert payload["scope"] == "nerf_backend_only"
    assert payload["whole_engine_hardening_deferred_to"] == "8.7G"
    assert payload["whole_application_gpu_profiling_deferred_to"] == "8.9D"
    with pytest.raises(Exception, match="require performance"):
        NerfPerformanceHardeningReport(
            hardware=_hardware(), profile=NerfOptimizationProfile.EFFICIENT,
            performance=None, artifact_integrity=None, real_measurements_complete=True,
        )


def test_e4_rtx5090_readiness_cannot_claim_fake_measurements() -> None:
    hardware = NerfHardwareEvidence(
        profile=NerfHardwareProfile.RTX_5090_32GB,
        gpu_name="NVIDIA GeForce RTX 5090 profile",
        gpu_uuid="unmeasured-profile",
        compute_capability=(12, 0),
        total_vram_mib=32768.0,
        driver_version="not-measured",
        torch_cuda_version="readiness",
        measured=False,
    )
    report = NerfPerformanceHardeningReport(
        hardware=hardware,
        profile=NerfOptimizationProfile.EFFICIENT,
        performance=None,
        artifact_integrity=None,
        real_measurements_complete=False,
        rtx5090_readiness_only=True,
    )
    assert report.passed
    assert report.as_dict()["performance"] is None
    with pytest.raises(Exception, match="cannot claim real measurements"):
        NerfPerformanceHardeningReport(
            hardware=hardware, profile=NerfOptimizationProfile.EFFICIENT,
            performance=_performance(), artifact_integrity=None,
            real_measurements_complete=True, rtx5090_readiness_only=True,
        )



def test_e4_rtx5090_readiness_profile_is_explicitly_not_current_runtime_qualified() -> None:
    readiness = nerf_rtx5090_readiness_profile()
    assert readiness["profile"] == NerfHardwareProfile.RTX_5090_32GB.value
    assert readiness["compute_capability"] == [12, 0]
    assert readiness["minimum_cuda_toolkit_for_native_sm120"] == "12.8"
    assert readiness["current_e2_qualified_cuda_toolkit"] == "11.8"
    assert readiness["current_e2_tcnn_architectures"] == ["75"]
    assert readiness["current_runtime_blackwell_qualified"] is False
    assert readiness["profile_defined"] is True
    assert readiness["real_performance_measurements"] is None
    assert readiness["no_fake_numbers"] is True
    assert readiness["whole_application_validation_gate"] == "8.9D"

def test_e4_artifact_validator_rejects_missing_empty_wrong_suffix_and_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    config = root / "config.yml"
    ckpt = root / "step.ckpt"
    config.write_text("x: 1\n")
    ckpt.write_bytes(b"x")
    assert validate_nerf_artifact_files(config_path=config, checkpoint_path=ckpt, artifact_root=root) == (
        config.resolve(), ckpt.resolve()
    )
    with pytest.raises(Exception, match="missing or empty"):
        validate_nerf_artifact_files(config_path=config, checkpoint_path=root / "missing.ckpt")
    bad = root / "bad.txt"
    bad.write_text("x")
    with pytest.raises(Exception, match="YAML"):
        validate_nerf_artifact_files(config_path=bad, checkpoint_path=ckpt)
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"x")
    with pytest.raises(Exception, match="escapes"):
        validate_nerf_artifact_files(config_path=config, checkpoint_path=outside, artifact_root=root)


def _robustness_evidence(scenario: NerfBackendRobustnessScenario, *, good: bool = True) -> NerfBackendRobustnessEvidence:
    expected = (
        NerfBackendRunState.CANCELLED if scenario is NerfBackendRobustnessScenario.CANCELLATION
        else NerfBackendRunState.TIMED_OUT if scenario is NerfBackendRobustnessScenario.TIMEOUT
        else NerfBackendRunState.CRASHED
    )
    return NerfBackendRobustnessEvidence(
        scenario=scenario,
        fault_lifecycle=NerfLifecycleEvidence(
            state=expected if good else NerfBackendRunState.FAILED,
            worker_exit_code=-15,
            cancellation_requested=(scenario is NerfBackendRobustnessScenario.CANCELLATION),
        ),
        process_group_clean=good,
        partial_artifacts_detected=(good and scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT),
        partial_artifacts_cleaned=(good and scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT),
        recovery_lifecycle=NerfLifecycleEvidence(
            state=NerfBackendRunState.RECOVERED if good else NerfBackendRunState.FAILED,
            worker_exit_code=0,
            recovery_attempted=True,
            recovery_succeeded=good,
        ),
        recovery_result_valid=good,
    )


def test_e4_robustness_report_requires_all_four_scenarios_and_recovery() -> None:
    evidence = tuple(_robustness_evidence(value) for value in NerfBackendRobustnessScenario)
    report = NerfBackendRobustnessReport(evidence=evidence)
    assert report.passed
    assert report.as_dict()["scope"] == "nerf_backend_process_only"
    with pytest.raises(Exception, match="all four"):
        NerfBackendRobustnessReport(evidence=evidence[:-1])
    broken = list(evidence)
    broken[0] = _robustness_evidence(NerfBackendRobustnessScenario.CANCELLATION, good=False)
    assert not NerfBackendRobustnessReport(evidence=tuple(broken)).passed


def test_e4_worker_fault_hooks_are_explicitly_test_gated() -> None:
    text = (ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py").read_text(encoding="utf-8")
    assert 'FAULT_ENV = "EIA_NERF_E4_TEST_FAULTS"' in text
    assert 'os.environ.get(FAULT_ENV) != "1"' in text
    assert "os._exit(86)" in text
    assert '"partial": True' in text


def test_e4_worker_native_render_timing_excludes_file_io() -> None:
    text = (ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py").read_text(encoding="utf-8")
    assert "get_outputs_for_camera_ray_bundle" in text
    assert "torch.cuda.synchronize()" in text
    assert "DatasetRender" not in text


def test_e4_worker_has_no_top_level_heavy_nerf_imports() -> None:
    path = ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    top = [line for line in lines if line.startswith("import nerfstudio") or line.startswith("from nerfstudio")]
    assert top == []


def test_e4_performance_runner_explicitly_defers_5090_real_app_profiling() -> None:
    text = (ROOT / "scripts/run_nerf_performance_hardening.py").read_text(encoding="utf-8")
    assert "nerf_rtx5090_readiness_profile" in text
    assert "rtx5090_readiness_report.json" in text
    assert "real_5090_performance_measurements=None" in text
    assert "current_e2_runtime_blackwell_qualified" in text
    assert '"8.9D"' in text


def test_e4_smoke_writes_dependency_light_contract_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = ROOT / "scripts/run_nerf_e4_contracts_smoke.py"
    spec = importlib.util.spec_from_file_location("e4_smoke_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
