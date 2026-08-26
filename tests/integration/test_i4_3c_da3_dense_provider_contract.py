from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.da3_dense_provider import DA3ProductionProviderAdapter
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from tests.i4_3c_helpers import CollectingProgressSink, FakeDA3Runner, explicit_da3_selection, make_context


def test_i43c_end_to_end_summary_keeps_da3_pose_supporting_only(tmp_path: Path) -> None:
    context, env = make_context(tmp_path, view_count=4)
    sink = CollectingProgressSink()
    adapter = DA3ProductionProviderAdapter(
        context,
        environment_probe=env,
        worker_runner=FakeDA3Runner(output_shape=(24, 32)),
        progress_sink=sink,
    )
    request, decision = explicit_da3_selection(4)
    result = adapter.execute(request, decision)
    assert result.status is DenseProviderRunStatus.COMPLETED
    summary = json.loads((context.output_dir / "i4_3c_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "pass"
    assert summary["provider"]["family"] == "multi_view_ai_depth"
    assert summary["provider"]["implementation_contract"]["frame_binding"] == "align_to_authoritative_sfm"
    assert summary["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert summary["authority"]["da3_predicted_extrinsics_validated"] is True
    assert summary["authority"]["da3_predicted_extrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["canonical_projection_pose_source"] == "pycolmap_sfm"
    assert summary["authority"]["sfm_rerun"] is False
    assert summary["authority"]["bundle_adjustment_rerun"] is False
    assert summary["authority"]["provider_promotion_performed"] is False
    assert summary["integrity"]["authoritative_sfm_unchanged"] is True
    assert summary["integrity"]["registered_images_unchanged"] is True
    assert summary["config"]["permanent_global_view_cap"] is None
    assert summary["execution_result"]["selected_view_count"] == 4
    assert summary["fusion"]["statistics"]["frames_integrated"] == 4
    assert all(item["da3_predicted_pose_consumed_for_canonical_geometry"] is False for item in summary["alignment"]["records"])
    assert len(sink.events) >= 8


def test_i43c_canonical_npz_records_only_authoritative_sfm_pose_layer(tmp_path: Path) -> None:
    context, env = make_context(tmp_path, view_count=3)
    adapter = DA3ProductionProviderAdapter(context, environment_probe=env, worker_runner=FakeDA3Runner())
    request, decision = explicit_da3_selection(3)
    result = adapter.execute(request, decision)
    import numpy as np
    with np.load(result.canonical_geometry_ref, allow_pickle=False) as payload:
        assert set(payload["source_pose_layers"].tolist()) == {"offline_sfm_bundle_adjusted"}
        assert len(payload["source_frame_ids"]) == 3
        assert payload["points_reconstruction"].shape[1] == 3
        assert payload["colors_rgb"].shape[1] == 3
