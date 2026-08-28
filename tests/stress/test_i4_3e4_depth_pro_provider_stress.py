from __future__ import annotations

from pathlib import Path

from core.reconstruction.depth_pro_dense_provider import DepthProProductionProviderAdapter
from tests.i4_3e4_helpers import FakeDepthProWorkerRunner, explicit_depth_pro_selection, make_depth_pro_context


def test_i43e4_repeated_all_view_runs_have_no_global_view_cap(tmp_path: Path) -> None:
    for views in (2, 5, 9):
        context, env = make_depth_pro_context(tmp_path / f"views-{views}", view_count=views)
        adapter = DepthProProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeDepthProWorkerRunner(),
        )
        request, decision = explicit_depth_pro_selection(views)
        result = adapter.execute(request, decision)
        assert result.source_view_count == views
        assert result.selected_view_count == views
        assert result.permanent_global_view_cap is None
        assert result.adaptive_view_selection is False


def test_i43e4_repeated_runs_never_promote_metric_or_measurement_authority(tmp_path: Path) -> None:
    for index in range(3):
        context, env = make_depth_pro_context(tmp_path / f"run-{index}", view_count=3)
        adapter = DepthProProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeDepthProWorkerRunner(depth_value=1.0 + index * 0.05),
        )
        request, decision = explicit_depth_pro_selection(3)
        result = adapter.execute(request, decision)
        assert result.metric_scale_claimed is False
        assert result.measurement_authoritative is False
        assert result.certified_metrology is False
        assert result.surface_mesh_generated is False
        assert result.provider_promotion_performed is False
