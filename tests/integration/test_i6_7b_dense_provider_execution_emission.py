from __future__ import annotations

from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.mvs_provider import MvsProductionProviderAdapter, MvsProviderRunContext
from core.reconstruction.reconstruction_progressive_dense_emission import (
    ProgressiveDenseExecutionEnvelope,
    execute_dense_provider_with_progressive_emission,
)
from tests.i4_3b_helpers import FakeWorkerRunner, explicit_selection, good_environment, good_resources, make_sfm_result


class CollectingSnapshotSink:
    def __init__(self) -> None:
        self.snapshots = []

    def publish(self, snapshot) -> None:
        self.snapshots.append(snapshot)


class TerminalCompleteWorkerRunner(FakeWorkerRunner):
    """Mirror the production worker's terminal state callback after stereo fusion."""

    def run(
        self,
        *,
        command,
        result_path,
        state_path,
        dense_workspace,
        total_views,
        timeout_s,
        poll_s,
        progress_callback,
    ):
        def terminal_progress_callback(stage, depth_count, state):
            progress_callback(stage, depth_count, state)
            if stage == "stereo_fusion":
                progress_callback("complete", depth_count, {"stage": "complete", "completed": True})

        return super().run(
            command=command,
            result_path=result_path,
            state_path=state_path,
            dense_workspace=dense_workspace,
            total_views=total_views,
            timeout_s=timeout_s,
            poll_s=poll_s,
            progress_callback=terminal_progress_callback,
        )


def test_i67b_real_mvs_adapter_execution_emits_stage_and_actual_geometry_preview(tmp_path: Path) -> None:
    py = tmp_path / "python3.11"
    worker = tmp_path / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    sfm = make_sfm_result(tmp_path, view_count=3)
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="sfm-frame-1",
        output_dir=tmp_path / "mvs-output",
        pycolmap_python=py,
        worker_script=worker,
    )
    request, decision = explicit_selection(3)
    sink = CollectingSnapshotSink()

    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
            context,
            progress_sink=progress_sink,
            environment_probe=good_environment(str(py)),
            resource_probe=good_resources(context.output_dir),
            worker_runner=FakeWorkerRunner(ply_points=12, vis_points=12),
        ),
        request=request,
        decision=decision,
        source_sfm_result=sfm,
        reconstruction_frame_id="sfm-frame-1",
        snapshot_sink=sink,
        dense_run_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
    )

    assert isinstance(envelope, ProgressiveDenseExecutionEnvelope)
    assert envelope.result.status is DenseProviderRunStatus.COMPLETED
    assert envelope.result.final_preview_revision > 0
    assert envelope.report.geometry_snapshot_emitted is True
    assert envelope.report.emission_failure_count == 0
    assert len(sink.snapshots) >= 3
    geometry = [item for item in sink.snapshots if item.geometry is not None]
    assert len(geometry) == 1
    assert geometry[0].statistics.current_dense_points == 12
    assert len(geometry[0].geometry.point_samples) == 12
    assert geometry[0].final_dense_result_admitted is False
    assert Path(envelope.result.canonical_geometry_ref).is_file()


def test_i67b_execution_wrapper_rejects_adapter_descriptor_mismatch(tmp_path: Path) -> None:
    py = tmp_path / "python3.11"
    worker = tmp_path / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    sfm = make_sfm_result(tmp_path, view_count=3)
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="sfm-frame-1",
        output_dir=tmp_path / "mvs-output",
        pycolmap_python=py,
        worker_script=worker,
    )
    request, decision = explicit_selection(3)

    from dataclasses import replace
    from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
    bad = replace(decision, selected_implementation=DenseProviderImplementation.DA3)
    try:
        execute_dense_provider_with_progressive_emission(
            adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
                context,
                progress_sink=progress_sink,
                environment_probe=good_environment(str(py)),
                resource_probe=good_resources(context.output_dir),
                worker_runner=FakeWorkerRunner(),
            ),
            request=request,
            decision=bad,
            source_sfm_result=sfm,
            reconstruction_frame_id="sfm-frame-1",
            snapshot_sink=CollectingSnapshotSink(),
        )
    except Exception as exc:
        assert "implementation" in str(exc) or "provider" in str(exc)
    else:
        raise AssertionError("descriptor mismatch must fail closed")


def test_i67b_real_da3_adapter_execution_emits_npz_geometry_preview(tmp_path: Path) -> None:
    from core.reconstruction.da3_dense_provider import DA3ProductionProviderAdapter
    from tests.i4_3c_helpers import FakeDA3Runner, explicit_da3_selection, make_context

    context, env = make_context(tmp_path, view_count=3)
    request, decision = explicit_da3_selection(3)
    sink = CollectingSnapshotSink()

    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: DA3ProductionProviderAdapter(
            context,
            progress_sink=progress_sink,
            environment_probe=env,
            worker_runner=FakeDA3Runner(),
        ),
        request=request,
        decision=decision,
        source_sfm_result=context.sfm_result,
        reconstruction_frame_id=context.reconstruction_frame_id,
        snapshot_sink=sink,
        dense_run_id="bbbbbbbb-cccc-4ddd-8eee-ffffffffffff",
    )

    assert envelope.result.status in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}
    assert envelope.report.geometry_snapshot_emitted is True
    geometry = [item for item in sink.snapshots if item.geometry is not None]
    assert geometry
    final = geometry[-1]
    assert final.statistics.current_dense_points > 0
    assert len(final.geometry.point_samples) <= 4096
    assert final.final_dense_result_admitted is False
    assert final.metric_scale_claimed is False


def test_i67b_real_mvs_terminal_complete_progress_remains_monotonic(tmp_path: Path) -> None:
    py = tmp_path / "python3.11"
    worker = tmp_path / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    sfm = make_sfm_result(tmp_path, view_count=3)
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="sfm-frame-1",
        output_dir=tmp_path / "mvs-output",
        pycolmap_python=py,
        worker_script=worker,
    )
    request, decision = explicit_selection(3)
    sink = CollectingSnapshotSink()

    envelope = execute_dense_provider_with_progressive_emission(
        adapter_factory=lambda progress_sink: MvsProductionProviderAdapter(
            context,
            progress_sink=progress_sink,
            environment_probe=good_environment(str(py)),
            resource_probe=good_resources(context.output_dir),
            worker_runner=TerminalCompleteWorkerRunner(ply_points=12, vis_points=12),
        ),
        request=request,
        decision=decision,
        source_sfm_result=sfm,
        reconstruction_frame_id="sfm-frame-1",
        snapshot_sink=sink,
        dense_run_id="cccccccc-dddd-4eee-8fff-000000000000",
    )

    assert envelope.result.status is DenseProviderRunStatus.COMPLETED
    assert envelope.report.emission_failure_count == 0
    assert envelope.report.geometry_snapshot_emitted is True
