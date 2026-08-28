from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.unidepth_v2_dense_provider import UniDepthV2ProductionProviderAdapter
from tests.i4_3e2_helpers import FakeUniDepthWorkerRunner, explicit_unidepth_selection, make_unidepth_context


@pytest.mark.parametrize("view_count", [2, 3, 6, 9])
def test_i43e2_repeated_registered_view_counts_have_no_global_cap(tmp_path: Path, view_count: int) -> None:
    context, env = make_unidepth_context(tmp_path, view_count=view_count)
    adapter = UniDepthV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeUniDepthWorkerRunner(),
    )
    request, decision = explicit_unidepth_selection(view_count)
    result = adapter.execute(request, decision)
    assert result.source_view_count == view_count
    assert result.selected_view_count == view_count
    assert result.permanent_global_view_cap is None
    assert result.metric_scale_claimed is False


def test_i43e2_repeated_runs_never_promote_native_metric_or_direct_geometry_authority(tmp_path: Path) -> None:
    for index in range(5):
        run_root = tmp_path / f"run-{index}"
        context, env = make_unidepth_context(run_root, view_count=3)
        adapter = UniDepthV2ProductionProviderAdapter(
            context=context,
            environment=env,
            runner=FakeUniDepthWorkerRunner(uncertainty_value=0.1 + index * 0.2),
        )
        request, decision = explicit_unidepth_selection(3)
        result = adapter.execute(request, decision)
        assert result.provider_promotion_performed is False
        assert result.metric_scale_claimed is False
        assert result.measurement_authoritative is False
        assert result.certified_metrology is False
