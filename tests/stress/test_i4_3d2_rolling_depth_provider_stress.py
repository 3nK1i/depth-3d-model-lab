import json
from pathlib import Path

from core.reconstruction.rolling_depth_dense_provider import RollingDepthProductionProviderAdapter
from tests.i4_3d2_helpers import (
    FakeRollingDepthSequenceBuilder,
    FakeRollingDepthWorkerRunner,
    explicit_rolling_selection,
    make_rolling_context,
)


def test_i43d2_repeated_runs_are_deterministic_in_order_alignment_and_authority(tmp_path: Path) -> None:
    records = []
    for run_index in range(3):
        root = tmp_path / f"run-{run_index}"
        context, env = make_rolling_context(root, view_count=4)
        runner = FakeRollingDepthWorkerRunner(context.sfm_result)
        adapter = RollingDepthProductionProviderAdapter(
            context=context, environment=env, runner=runner, sequence_builder=FakeRollingDepthSequenceBuilder()
        )
        request, decision = explicit_rolling_selection(4)
        result = adapter.execute(request, decision)
        summary = json.loads((context.output_dir / "i4_3d2_rolling_depth_execution_summary.json").read_text(encoding="utf-8"))
        records.append((
            summary["temporal_sequence"]["ordered_source_sequence_indices"],
            round(summary["alignment"]["scale_to_reconstruction"], 6),
            round(summary["alignment"]["shift_to_reconstruction"], 6),
            result.camera_pose_authority_owner,
            result.permanent_global_view_cap,
        ))
    assert records[0] == records[1] == records[2]


def test_i43d2_longer_sequence_has_no_hidden_global_view_cap(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=12)
    runner = FakeRollingDepthWorkerRunner(context.sfm_result)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env, runner=runner, sequence_builder=FakeRollingDepthSequenceBuilder()
    )
    request, decision = explicit_rolling_selection(12)
    result = adapter.execute(request, decision)
    assert result.source_view_count == 12
    assert result.selected_view_count == 12
    assert result.permanent_global_view_cap is None
    assert len(runner.last_manifest["sequence"]) == 12
