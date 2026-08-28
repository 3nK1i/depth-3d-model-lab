from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.reconstruction.metric3d_v2_dense_provider import Metric3DV2ProductionProviderAdapter
from tests.i4_3e1_helpers import FakeMetric3DWorkerRunner, explicit_metric3d_selection, make_metric3d_context


def test_i43e1_metric3d_execution_persists_canonical_candidate_and_provenance(tmp_path: Path) -> None:
    context, env = make_metric3d_context(tmp_path, view_count=4)
    before = {
        item.keyframe_id: (Path(context.sfm_result.workspace.workspace_root) / item.staged_image_path).read_bytes()
        for item in context.sfm_result.workspace.images
    }
    adapter = Metric3DV2ProductionProviderAdapter(context=context, environment=env, runner=FakeMetric3DWorkerRunner())
    request, decision = explicit_metric3d_selection(4)
    result = adapter.execute(request, decision)

    canonical = Path(result.canonical_geometry_ref)
    assert canonical.is_file()
    with np.load(canonical, allow_pickle=False) as archive:
        assert archive["points_reconstruction"].ndim == 2
        assert archive["points_reconstruction"].shape[1] == 3
        assert len(archive["points_reconstruction"]) > 0
        assert np.isfinite(archive["mean_confidence"]).all()

    summary_path = context.output_dir / "i4_3e1_metric3d_v2_execution_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["provider"]["family"] == "single_image_depth"
    assert summary["execution"]["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert summary["authority"]["provider_promoted"] is False
    assert summary["config"]["permanent_global_view_cap"] is None
    assert summary["config"]["adaptive_view_selection_deferred_to"] == "I4.3F"

    for item in context.sfm_result.workspace.images:
        path = Path(context.sfm_result.workspace.workspace_root) / item.staged_image_path
        assert path.read_bytes() == before[item.keyframe_id]
