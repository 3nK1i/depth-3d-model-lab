from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
from uuid import uuid4

import pytest

from core.reconstruction import (
    ENGINE_HARDENING_PHASE,
    ENGINE_HARDENING_SCHEMA_VERSION,
    DeterministicMockStageExecutor,
    ReconstructionArtifactAuthority,
    ReconstructionArtifactKind,
    ReconstructionEngineProcessOutcome,
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRecoveryReport,
    ReconstructionEngineRunState,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineArtifact,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    ReconstructionStageRunState,
    ReconstructionValidationError,
    build_reconstruction_pipeline_plan,
    recovery_report,
    run_supervised_engine_process,
)

ROOT = Path(__file__).resolve().parents[2]
FAULT_WORKER = (ROOT / "scripts/reconstruction_engine_fault_worker.py").resolve()


def _spec(tmp_path: Path, name: str, *, mode: str, cancel_after_s=None, timeout_s=2.0):
    owned_root = (tmp_path / "owned").resolve()
    workspace = (owned_root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(
            sys.executable,
            str(FAULT_WORKER),
            "--workspace",
            str(workspace),
            "--mode",
            mode,
            "--child-sleep-s",
            "5",
        ),
        owned_root=str(owned_root),
        workspace=str(workspace),
        timeout_s=timeout_s,
        cancel_after_s=cancel_after_s,
        terminate_grace_s=0.1,
        cwd=str(ROOT),
    )


def _provider(provider_id: str, role: ReconstructionProviderRole):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="g1-test",
    )


def _canonical_pipeline():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-g1-unit",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    return request, providers, build_reconstruction_pipeline_plan(request, providers)


def test_g1_schema_and_phase_are_explicit() -> None:
    assert ENGINE_HARDENING_SCHEMA_VERSION == 1
    assert ENGINE_HARDENING_PHASE == "8.7G1"


def test_process_spec_normalizes_paths_and_environment(tmp_path: Path) -> None:
    spec = ReconstructionEngineProcessSpec(
        command=(sys.executable, "-c", "print('ok')"),
        owned_root=str((tmp_path / "root").resolve()),
        workspace=str((tmp_path / "root/run").resolve()),
        timeout_s=2,
        environment=(("B", "2"), ("A", "1")),
    )
    assert Path(spec.workspace).is_absolute()
    assert spec.environment_dict == {"B": "2", "A": "1"}
    assert spec.as_dict()["environment_keys"] == ["A", "B"]


def test_process_spec_rejects_workspace_outside_owned_root(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError, match="strict child"):
        ReconstructionEngineProcessSpec(
            command=(sys.executable, "-c", "pass"),
            owned_root=str((tmp_path / "a").resolve()),
            workspace=str((tmp_path / "b").resolve()),
            timeout_s=1,
        )


def test_process_spec_rejects_bad_timing(tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    with pytest.raises(ReconstructionValidationError):
        ReconstructionEngineProcessSpec(
            command=(sys.executable, "-c", "pass"), owned_root=str(root), workspace=str(root / "run"), timeout_s=0
        )
    with pytest.raises(ReconstructionValidationError):
        ReconstructionEngineProcessSpec(
            command=(sys.executable, "-c", "pass"), owned_root=str(root), workspace=str(root / "run"), timeout_s=1, cancel_after_s=1
        )


def test_completed_engine_attempt_preserves_workspace_and_reaps_child(tmp_path: Path) -> None:
    outcome = run_supervised_engine_process(_spec(tmp_path, "complete", mode="complete"))
    assert outcome.state is ReconstructionEngineRunState.COMPLETED
    assert outcome.process_group_clean is True
    assert outcome.workspace_exists_after is True
    assert outcome.failed_workspace_removed is False
    assert outcome.safe_for_recovery is True
    assert (Path(outcome.workspace) / "engine_result.json").is_file()
    assert not (Path(outcome.workspace) / "partial_artifact.bin").exists()


def test_cancelled_engine_attempt_removes_partial_workspace_and_group(tmp_path: Path) -> None:
    outcome = run_supervised_engine_process(
        _spec(tmp_path, "cancel", mode="hang", cancel_after_s=0.12, timeout_s=2.0)
    )
    assert outcome.state is ReconstructionEngineRunState.CANCELLED
    assert outcome.process_group_clean is True
    assert outcome.failed_workspace_removed is True
    assert not Path(outcome.workspace).exists()
    assert outcome.safe_for_recovery is True


def test_timed_out_engine_attempt_removes_partial_workspace_and_group(tmp_path: Path) -> None:
    outcome = run_supervised_engine_process(_spec(tmp_path, "timeout", mode="hang", timeout_s=0.15))
    assert outcome.state is ReconstructionEngineRunState.TIMED_OUT
    assert outcome.process_group_clean is True
    assert outcome.failed_workspace_removed is True
    assert not Path(outcome.workspace).exists()


def test_crashed_parent_with_live_descendant_is_reaped_and_cleaned(tmp_path: Path) -> None:
    outcome = run_supervised_engine_process(_spec(tmp_path, "crash", mode="crash", timeout_s=2.0))
    assert outcome.state is ReconstructionEngineRunState.CRASHED
    assert outcome.exit_code == 17
    assert outcome.process_group_clean is True
    assert outcome.failed_workspace_removed is True
    assert not Path(outcome.workspace).exists()


def test_launch_failure_is_contained_and_workspace_removed(tmp_path: Path) -> None:
    root = (tmp_path / "owned").resolve()
    workspace = (root / "launch-fail").resolve()
    outcome = run_supervised_engine_process(
        ReconstructionEngineProcessSpec(
            command=(str((tmp_path / "does-not-exist").resolve()),),
            owned_root=str(root),
            workspace=str(workspace),
            timeout_s=1.0,
        )
    )
    assert outcome.state is ReconstructionEngineRunState.CRASHED
    assert outcome.exit_code is None
    assert outcome.process_group_clean is True
    assert outcome.failed_workspace_removed is True
    assert "launch_failed" in outcome.reason


def test_nonempty_existing_workspace_is_never_destroyed(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "existing", mode="complete")
    workspace = Path(spec.workspace)
    workspace.mkdir(parents=True)
    sentinel = workspace / "user-data.txt"
    sentinel.write_text("keep")
    with pytest.raises(ReconstructionValidationError, match="refusing destructive cleanup"):
        run_supervised_engine_process(spec)
    assert sentinel.read_text() == "keep"


def test_recovery_report_requires_a_real_fault(tmp_path: Path) -> None:
    completed = run_supervised_engine_process(_spec(tmp_path, "a", mode="complete"))
    completed2 = run_supervised_engine_process(_spec(tmp_path, "b", mode="complete"))
    with pytest.raises(ReconstructionValidationError, match="fault attempt"):
        ReconstructionEngineRecoveryReport(scenario="bad", fault=completed, recovery=completed2)


def test_recovery_report_passes_after_cancel_then_clean_run(tmp_path: Path) -> None:
    fault = run_supervised_engine_process(
        _spec(tmp_path, "fault", mode="hang", cancel_after_s=0.1, timeout_s=2)
    )
    recovery = run_supervised_engine_process(_spec(tmp_path, "recovery", mode="complete"))
    report = recovery_report(scenario="cancel", fault=fault, recovery=recovery)
    assert report.passed is True
    payload = report.as_dict()
    assert payload["checkpoint_resume_performed"] is False
    assert payload["checkpoint_resume_deferred_to"] == "8.7G2"
    assert payload["endurance_hardening_deferred_to"] == "8.7G3"
    assert payload["full_regression_deferred_to"] == "8.7G4"


def test_stage_context_carries_the_same_control_instance() -> None:
    request, providers, plan = _canonical_pipeline()
    seen = []

    class RecordingExecutor(DeterministicMockStageExecutor):
        def execute(self, context):
            seen.append(context.control)
            return super().execute(context)

    control = ReconstructionOperationControl()
    executors = {p.provider_id: RecordingExecutor(p.provider_id) for p in providers}
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(control=control)
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert seen and all(item is control for item in seen)


def test_provider_can_cooperatively_cancel_mid_stage_via_context_checkpoint() -> None:
    request, providers, plan = _canonical_pipeline()
    control = ReconstructionOperationControl()
    started = threading.Event()

    class SlowIngestion:
        def execute(self, context):
            started.set()
            for _ in range(1000):
                context.checkpoint()
                time.sleep(0.002)
            raise AssertionError("provider should have observed cancellation")

    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    executors["ingestion"] = SlowIngestion()
    result_holder = []

    thread = threading.Thread(
        target=lambda: result_holder.append(
            ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(control=control)
        )
    )
    thread.start()
    assert started.wait(timeout=1.0)
    control.request_cancel()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    result = result_holder[0]
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    assert result.stages[-1].state is ReconstructionStageRunState.CANCELLED
    assert result.stages[-1].reason == "provider_observed_cancellation"


def test_post_provider_cancel_prevents_accepting_artifact() -> None:
    request, providers, plan = _canonical_pipeline()
    control = ReconstructionOperationControl()

    class CancellingIngestion:
        def execute(self, context):
            contract = context.planned_stage.contract
            assert context.control is control
            control.request_cancel()
            return ReconstructionPipelineArtifact(
                artifact_id="would-have-completed",
                session_id=context.request.session_id,
                stage=contract.stage,
                kind=contract.output_kind,
                authority=contract.output_authority,
                provider_id=context.provider.provider_id,
            )

    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    executors["ingestion"] = CancellingIngestion()
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run(control=control)
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    assert result.stages[0].artifact is None
    assert result.stages[0].state is ReconstructionStageRunState.CANCELLED


def test_context_without_control_remains_backward_compatible() -> None:
    request, providers, plan = _canonical_pipeline()
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors={p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers},
    ).run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED
    assert result.canonical_geometry_authoritative is True
    assert result.certified_metrology is False


def test_outcome_json_is_explicit_about_recovery_safety(tmp_path: Path) -> None:
    outcome = run_supervised_engine_process(_spec(tmp_path, "json", mode="complete"))
    payload = outcome.as_dict()
    assert payload["phase"] == "8.7G1"
    assert payload["safe_for_recovery"] is True
    assert payload["state"] == "completed"
