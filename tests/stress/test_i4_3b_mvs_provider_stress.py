from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction.mvs_provider import MvsProductionConfig, MvsProductionProviderAdapter, MvsProviderRunContext
from tests.i4_3b_helpers import FakeWorkerRunner, explicit_selection, good_environment, good_resources, make_sfm_result


@pytest.mark.parametrize("views", [2, 3, 10, 40, 41, 47, 68, 96, 128])
def test_i43b_never_turns_historical_view_counts_into_a_global_cap(views: int) -> None:
    resolved = MvsProductionConfig(source_neighbors=10).resolved(selected_view_count=views)
    assert resolved["selected_view_count"] == views
    assert resolved["permanent_global_view_cap"] is None
    assert resolved["source_neighbors"] == min(10, views - 1)


@pytest.mark.parametrize("views", [2, 11, 47, 68])
def test_i43b_adapter_preserves_all_registered_views_across_scales(tmp_path: Path, views: int) -> None:
    case = tmp_path / f"views-{views}"
    sfm = make_sfm_result(case, view_count=views)
    py = case / "python3.11"
    worker = case / "worker.py"
    py.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")
    context = MvsProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id="sfm-frame",
        output_dir=case / "provider-output",
        pycolmap_python=py,
        worker_script=worker,
    )
    adapter = MvsProductionProviderAdapter(
        context,
        environment_probe=good_environment(str(py)),
        resource_probe=good_resources(context.output_dir),
        worker_runner=FakeWorkerRunner(),
    )
    request, decision = explicit_selection(views, frame_id="sfm-frame")
    result = adapter.execute(request, decision)
    assert result.source_view_count == views
    assert result.selected_view_count == views
    assert result.adaptive_view_selection is False
    assert result.permanent_global_view_cap is None
