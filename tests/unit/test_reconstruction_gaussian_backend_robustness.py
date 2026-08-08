from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from core.reconstruction import (
    GaussianBackendKind,
    GaussianBackendRobustnessEvidence,
    GaussianBackendRobustnessReport,
    GaussianBackendRobustnessScenario,
    GaussianBenchmarkRunState,
    GaussianLifecycleEvidence,
)
from scripts.gaussian_backend_lifecycle import run_supervised_process
from scripts.gaussian_benchmark_workers.common import (
    LIFECYCLE_FAULT_ENV,
    apply_lifecycle_fault_injection,
)


def _recovered() -> GaussianLifecycleEvidence:
    return GaussianLifecycleEvidence(
        state=GaussianBenchmarkRunState.RECOVERED,
        worker_exit_code=0,
        recovery_attempted=True,
        recovery_succeeded=True,
    )


def _evidence(scenario: GaussianBackendRobustnessScenario) -> GaussianBackendRobustnessEvidence:
    expected = {
        GaussianBackendRobustnessScenario.CANCELLATION: GaussianLifecycleEvidence(
            state=GaussianBenchmarkRunState.CANCELLED,
            worker_exit_code=-15,
            cancellation_requested=True,
        ),
        GaussianBackendRobustnessScenario.TIMEOUT: GaussianLifecycleEvidence(
            state=GaussianBenchmarkRunState.TIMED_OUT,
            worker_exit_code=-15,
            error="timeout",
        ),
        GaussianBackendRobustnessScenario.CRASH: GaussianLifecycleEvidence(
            state=GaussianBenchmarkRunState.CRASHED,
            worker_exit_code=1,
            error="crash",
        ),
        GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT: GaussianLifecycleEvidence(
            state=GaussianBenchmarkRunState.CRASHED,
            worker_exit_code=1,
            error="partial crash",
        ),
    }[scenario]
    return GaussianBackendRobustnessEvidence(
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        scenario=scenario,
        fault_lifecycle=expected,
        process_group_clean=True,
        partial_artifacts_detected=(scenario is GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT),
        partial_artifacts_cleaned=(scenario is GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT),
        recovery_lifecycle=_recovered(),
        recovery_result_valid=True,
    )


def test_robustness_evidence_requires_expected_fault_and_real_recovery() -> None:
    for scenario in GaussianBackendRobustnessScenario:
        assert _evidence(scenario).passed


def test_partial_artifact_scenario_requires_detection_and_cleanup() -> None:
    with pytest.raises(Exception):
        GaussianBackendRobustnessEvidence(
            backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
            scenario=GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT,
            fault_lifecycle=GaussianLifecycleEvidence(
                state=GaussianBenchmarkRunState.CRASHED,
                worker_exit_code=1,
                error="partial crash",
            ),
            process_group_clean=True,
            recovery_lifecycle=_recovered(),
            recovery_result_valid=True,
        )


def test_report_requires_every_step4_scenario_per_selected_backend() -> None:
    complete = tuple(_evidence(scenario) for scenario in GaussianBackendRobustnessScenario)
    assert GaussianBackendRobustnessReport(evidence=complete).passed
    assert not GaussianBackendRobustnessReport(evidence=complete[:-1]).passed


def test_fault_injection_is_disabled_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(LIFECYCLE_FAULT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="not enabled"):
        apply_lifecycle_fault_injection(
            request={
                "lifecycle_fault_mode": "crash",
                "lifecycle_fault_marker_path": str(tmp_path / "marker.json"),
            },
            result_path=tmp_path / "result.json",
        )


def test_partial_fault_writes_only_invalid_partial_result_when_test_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LIFECYCLE_FAULT_ENV, "1")
    result = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="partial-artifact"):
        apply_lifecycle_fault_injection(
            request={
                "lifecycle_fault_mode": "partial_artifact_crash",
                "lifecycle_fault_marker_path": str(tmp_path / "marker.json"),
                "backend_kind": "gsplat_external",
                "benchmark_fingerprint_sha256": "a" * 64,
            },
            result_path=result,
        )
    payload = json.loads(result.read_text())
    assert payload["partial"] is True
    assert "scene_path" not in payload
    assert "render_manifest_path" not in payload


def test_supervisor_cancels_entire_process_group() -> None:
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "time.sleep(60)"
    )
    outcome = run_supervised_process(
        [sys.executable, "-c", code],
        timeout_s=5.0,
        cancel_after_s=0.15,
        terminate_grace_s=0.1,
    )
    assert outcome.cancelled
    assert not outcome.timed_out
    assert outcome.process_group_clean


def test_supervisor_times_out_entire_process_group() -> None:
    outcome = run_supervised_process(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        timeout_s=0.15,
        terminate_grace_s=0.1,
    )
    assert outcome.timed_out
    assert not outcome.cancelled
    assert outcome.process_group_clean
