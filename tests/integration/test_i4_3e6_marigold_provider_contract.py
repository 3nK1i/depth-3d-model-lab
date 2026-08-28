from __future__ import annotations

import json
from pathlib import Path

from core.reconstruction.marigold_dense_provider import MarigoldProductionProviderAdapter
from tests.i4_3e6_helpers import (
    CollectingProgressSink,
    FakeMarigoldWorkerRunner,
    explicit_marigold_selection,
    make_marigold_context,
)


def test_i43e6_end_to_end_fake_worker_preserves_canonical_authority(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=4)
    runner = FakeMarigoldWorkerRunner(context.sfm_result, prediction_type="depth", uncertainty=True)
    sink = CollectingProgressSink()
    adapter = MarigoldProductionProviderAdapter(context=context, environment=env, runner=runner, progress_sink=sink)
    request, decision = explicit_marigold_selection(4)
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


def test_i43e6_summary_records_dynamic_affine_domain_and_optional_uncertainty(tmp_path: Path) -> None:
    context, env = make_marigold_context(tmp_path, view_count=3)
    adapter = MarigoldProductionProviderAdapter(
        context=context,
        environment=env,
        runner=FakeMarigoldWorkerRunner(context.sfm_result, prediction_type="disparity", uncertainty=True),
    )
    request, decision = explicit_marigold_selection(3)
    adapter.execute(request, decision)
    summary = json.loads((context.output_dir / "i4_3e6_marigold_execution_summary.json").read_text(encoding="utf-8"))
    assert summary["alignment"]["method"] == "affine_domain_according_to_prediction_type_to_authoritative_sfm"
    accepted = [item for item in summary["alignment"]["records"] if item["status"] == "accepted"]
    assert accepted and all(item["method"] == "affine_inverse_depth_to_authoritative_sfm" for item in accepted)
    assert all(item["native_uncertainty_available"] is True for item in accepted)
    assert summary["authority"]["relative_depth_promoted_to_metric_scale"] is False
    assert summary["authority"]["estimated_model_intrinsics_consumed_for_canonical_projection"] is False
    assert summary["authority"]["adapter_projected_points_consumed_for_canonical_geometry"] is False
    assert summary["integrity"]["registered_image_hashes_preserved"] is True
