from __future__ import annotations

from pathlib import Path

from core.reconstruction.depth_anything_v2_dense_provider import (
    DepthAnythingV2ProductionProviderAdapter,
)
from tests.i4_3e5_helpers import (
    FakeDepthAnythingV2WorkerRunner,
    explicit_depth_anything_v2_selection,
    make_depth_anything_v2_context,
)


def test_i43e5_repeated_all_view_runs_have_no_global_view_cap(tmp_path: Path) -> None:
    for views in (2, 5, 9):
        context, env = make_depth_anything_v2_context(
            tmp_path / f"views-{views}", view_count=views
        )
        adapter = DepthAnythingV2ProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
        )
        request, decision = explicit_depth_anything_v2_selection(views)
        result = adapter.execute(request, decision)
        assert result.source_view_count == views
        assert result.selected_view_count == views
        assert result.permanent_global_view_cap is None
        assert result.adaptive_view_selection is False


def test_i43e5_repeated_runs_never_promote_metric_or_measurement_authority(tmp_path: Path) -> None:
    for index in range(3):
        context, env = make_depth_anything_v2_context(
            tmp_path / f"run-{index}", view_count=3
        )
        adapter = DepthAnythingV2ProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
        )
        request, decision = explicit_depth_anything_v2_selection(3)
        result = adapter.execute(request, decision)
        assert result.metric_scale_claimed is False
        assert result.measurement_authoritative is False
        assert result.certified_metrology is False
        assert result.surface_mesh_generated is False
        assert result.provider_promotion_performed is False
