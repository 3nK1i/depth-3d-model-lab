from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.depth_anything_v2_dense_provider import (
    DepthAnythingV2ProductionProviderAdapter,
)
from tests.i4_3e5_helpers import (
    CollectingProgressSink,
    FakeDepthAnythingV2WorkerRunner,
    explicit_depth_anything_v2_selection,
    make_depth_anything_v2_context,
)


def test_i43e5_end_to_end_fake_worker_preserves_canonical_authority(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(tmp_path, view_count=4)
    runner = FakeDepthAnythingV2WorkerRunner(context.sfm_result)
    sink = CollectingProgressSink()
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=runner,
        progress_sink=sink,
    )
    request, decision = explicit_depth_anything_v2_selection(4)
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
    assert runner.last_manifest["authority"]["estimated_model_intrinsics_are_canonical_authority"] is False
    assert runner.last_manifest["authority"]["adapter_projected_points_are_canonical_geometry"] is False
    assert runner.last_manifest["authority"]["relative_depth_establishes_metric_scale"] is False
    assert sink.events[-1].progress_fraction == 1.0


def test_i43e5_summary_keeps_relative_depth_intrinsics_and_adapter_points_as_evidence_only(tmp_path: Path) -> None:
    context, env = make_depth_anything_v2_context(tmp_path, view_count=3)
    adapter = DepthAnythingV2ProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeDepthAnythingV2WorkerRunner(context.sfm_result),
    )
    request, decision = explicit_depth_anything_v2_selection(3)
    adapter.execute(request, decision)
    summary = json.loads(
        (context.output_dir / "i4_3e5_depth_anything_v2_execution_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["alignment"]["method"] == "affine_inverse_depth_to_authoritative_sfm"
    assert summary["authority"]["relative_depth_promoted_to_metric_scale"] is False
    assert summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["adapter_projected_points_consumed_for_canonical_geometry"] is False
    assert summary["integrity"]["registered_image_hashes_preserved"] is True
