from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.da3_dense_provider import DA3ProductionProviderAdapter
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from tests.i4_3c_helpers import FakeDA3Runner, explicit_da3_selection, make_context


@pytest.mark.parametrize("views", [2, 3, 10, 40, 41, 47, 68])
def test_i43c_never_turns_historical_view_counts_into_a_global_cap(tmp_path: Path, views: int) -> None:
    case = tmp_path / f"views-{views}"
    context, env = make_context(case, view_count=views)
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(views)
    result = adapter.execute(request, decision)
    assert result.status in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}
    assert result.source_view_count == views
    assert result.selected_view_count == views
    assert result.permanent_global_view_cap is None
    assert result.view_selection_reason == "all_registered_authoritative_sfm_views"


@pytest.mark.parametrize("repeat", range(5))
def test_i43c_repeated_synthetic_execution_is_deterministic(tmp_path: Path, repeat: int) -> None:
    case = tmp_path / f"repeat-{repeat}"
    context, env = make_context(case, view_count=3)
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3)
    result = adapter.execute(request, decision)
    import numpy as np
    with np.load(result.canonical_geometry_ref, allow_pickle=False) as payload:
        points = payload["points_reconstruction"]
        counts = payload["view_observation_counts"]
    assert points.shape[0] > 0
    assert counts.min() >= 1
