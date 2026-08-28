from __future__ import annotations

from pathlib import Path

from core.reconstruction.marigold_dense_provider import MarigoldProductionProviderAdapter
from tests.i4_3e6_helpers import FakeMarigoldWorkerRunner, explicit_marigold_selection, make_marigold_context


def test_i43e6_repeated_all_view_runs_have_no_global_view_cap(tmp_path: Path) -> None:
    for views in (2, 5, 9):
        context, env = make_marigold_context(tmp_path / f"views-{views}", view_count=views)
        adapter = MarigoldProductionProviderAdapter(context=context, environment=env, runner=FakeMarigoldWorkerRunner(context.sfm_result))
        request, decision = explicit_marigold_selection(views)
        result = adapter.execute(request, decision)
        assert result.source_view_count == views
        assert result.selected_view_count == views
        assert result.permanent_global_view_cap is None
        assert result.adaptive_view_selection is False


def test_i43e6_repeated_depth_and_disparity_runs_never_promote_authority(tmp_path: Path) -> None:
    for index, prediction_type in enumerate(("depth", "disparity", "depth")):
        context, env = make_marigold_context(tmp_path / f"run-{index}", view_count=3)
        adapter = MarigoldProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeMarigoldWorkerRunner(context.sfm_result, prediction_type=prediction_type, uncertainty=(index % 2 == 1)),
        )
        request, decision = explicit_marigold_selection(3)
        result = adapter.execute(request, decision)
        assert result.metric_scale_claimed is False
        assert result.measurement_authoritative is False
        assert result.certified_metrology is False
        assert result.surface_mesh_generated is False
        assert result.provider_promotion_performed is False
