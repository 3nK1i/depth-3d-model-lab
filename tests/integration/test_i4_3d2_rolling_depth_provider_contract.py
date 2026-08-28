from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderFamily, DenseProviderImplementation
from core.reconstruction.rolling_depth_dense_provider import RollingDepthProductionProviderAdapter
from tests.i4_3d2_helpers import (
    FakeRollingDepthSequenceBuilder,
    FakeRollingDepthWorkerRunner,
    explicit_rolling_selection,
    make_rolling_context,
)


def test_i43d2_contract_binds_whole_sequence_without_replacing_sfm_authority(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=4)
    runner = FakeRollingDepthWorkerRunner(context.sfm_result)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env, runner=runner, sequence_builder=FakeRollingDepthSequenceBuilder()
    )
    request, decision = explicit_rolling_selection(4)
    result = adapter.execute(request, decision)
    assert result.implementation is DenseProviderImplementation.ROLLING_DEPTH
    assert result.family is DenseProviderFamily.TEMPORAL_DEPTH
    assert result.selected_view_count == 4
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.metric_scale_claimed is False
    assert runner.last_manifest["authority"]["per_frame_alignment"] is False
    assert runner.last_manifest["authority"]["sequence_alignment"] == "one_global_affine_to_authoritative_sfm"


def test_i43d2_manifest_keeps_engineering_resource_policy_visible(tmp_path: Path) -> None:
    context, env = make_rolling_context(tmp_path, view_count=3)
    runner = FakeRollingDepthWorkerRunner(context.sfm_result)
    adapter = RollingDepthProductionProviderAdapter(
        context=context, environment=env, runner=runner, sequence_builder=FakeRollingDepthSequenceBuilder()
    )
    request, decision = explicit_rolling_selection(3)
    adapter.execute(request, decision)
    manifest = runner.last_manifest
    assert manifest["all_registered_views_selected"] is True
    assert manifest["permanent_global_view_cap"] is None
    assert manifest["config"]["processing_resolution"] == 512
    assert manifest["config"]["max_vae_batch_size"] == 1
    assert manifest["config"]["unload_snippet"] is True
