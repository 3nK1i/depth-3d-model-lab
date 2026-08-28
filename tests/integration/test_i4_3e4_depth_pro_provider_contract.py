from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.depth_pro_dense_provider import DepthProProductionProviderAdapter
from tests.i4_3e4_helpers import (
    CollectingProgressSink,
    FakeDepthProWorkerRunner,
    explicit_depth_pro_selection,
    make_depth_pro_context,
)


def test_i43e4_end_to_end_fake_worker_preserves_canonical_authority(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=4)
    runner = FakeDepthProWorkerRunner()
    sink = CollectingProgressSink()
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_depth_pro_selection(4)
    result = adapter.execute(request, decision)
    assert result.canonical_frame_id == "offline-sfm-reconstruction"
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert result.measurement_authoritative is False
    assert result.certified_metrology is False
    assert result.surface_mesh_generated is False
    assert result.permanent_global_view_cap is None
    assert Path(result.canonical_geometry_ref).is_file()
    assert runner.last_manifest["all_registered_views_selected"] is True
    assert runner.last_manifest["authority"]["model_predicted_intrinsics_are_canonical_authority"] is False
    assert runner.last_manifest["authority"]["model_metric_depth_establishes_metric_scale"] is False
    assert sink.events[-1].progress_fraction == 1.0


def test_i43e4_summary_keeps_depth_pro_intrinsics_and_metric_claims_as_evidence_only(tmp_path: Path) -> None:
    context, env = make_depth_pro_context(tmp_path, view_count=3)
    adapter = DepthProProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthProWorkerRunner(intrinsics_source_native="estimated"),
    )
    request, decision = explicit_depth_pro_selection(3)
    adapter.execute(request, decision)
    summary = json.loads(
        (context.output_dir / "i4_3e4_depth_pro_execution_summary.json").read_text(encoding="utf-8")
    )
    assert summary["authority"]["native_metric_depth_promoted_to_canonical_metric_scale"] is False
    assert summary["authority"]["model_predicted_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["integrity"]["registered_image_hashes_preserved"] is True
