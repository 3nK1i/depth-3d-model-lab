from __future__ import annotations

from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.video_depth_anything_dense_provider import VideoDepthAnythingProductionProviderAdapter
from tests.i4_3d1_helpers import FakeVDAWorkerRunner, explicit_vda_selection, make_vda_context


def test_i43d1_repeated_temporal_provider_runs_are_deterministic_in_order_and_authority(tmp_path: Path) -> None:
    summaries = []
    for attempt in range(5):
        root = tmp_path / f"attempt-{attempt}"
        context, environment = make_vda_context(root, view_count=5)
        runner = FakeVDAWorkerRunner()
        adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=environment, runner=runner)
        request, decision = explicit_vda_selection(5)
        result = adapter.execute(request, decision)
        assert result.status is DenseProviderRunStatus.COMPLETED
        assert result.source_view_count == result.selected_view_count == 5
        assert result.camera_pose_authority_owner == "pycolmap_sfm"
        assert result.metric_scale_claimed is False
        assert result.provider_promotion_performed is False
        summaries.append(tuple(item["source_sequence_index"] for item in runner.last_manifest["sequence"]))
    assert summaries == [(0, 1, 2, 3, 4)] * 5


def test_i43d1_longer_sequence_has_no_hidden_global_view_cap(tmp_path: Path) -> None:
    context, environment = make_vda_context(tmp_path, view_count=17)
    runner = FakeVDAWorkerRunner()
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=environment, runner=runner)
    request, decision = explicit_vda_selection(17)
    result = adapter.execute(request, decision)
    assert result.source_view_count == 17
    assert result.selected_view_count == 17
    assert result.permanent_global_view_cap is None
    assert len(runner.last_manifest["sequence"]) == 17
