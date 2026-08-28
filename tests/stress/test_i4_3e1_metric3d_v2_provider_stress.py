from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.metric3d_v2_dense_provider import Metric3DV2ProductionProviderAdapter
from tests.i4_3e1_helpers import FakeMetric3DWorkerRunner, explicit_metric3d_selection, make_metric3d_context


@pytest.mark.parametrize("view_count", [2, 3, 6, 9])
def test_i43e1_repeated_registered_view_counts_have_no_global_cap(tmp_path: Path, view_count: int) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=view_count)
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(view_count)
    result = adapter.execute(request, decision)
    assert result.source_view_count == view_count
    assert result.selected_view_count == view_count
    assert result.permanent_global_view_cap is None
    assert result.metric_scale_claimed is False


def test_i43e1_repeated_runs_do_not_promote_provider_or_measurement_authority(tmp_path: Path) -> None:
    for index in range(5):
        run_root = tmp_path / f"run-{index}"
        context, env = make_metric3d_context(run_root, view_count=3)
        adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner(confidence_value=0.5 + index * 0.1))
        request, decision = explicit_metric3d_selection(3)
        result = adapter.execute(request, decision)
        assert result.provider_promotion_performed is False
        assert result.metric_scale_claimed is False
        assert result.measurement_authoritative is False
        assert result.certified_metrology is False
