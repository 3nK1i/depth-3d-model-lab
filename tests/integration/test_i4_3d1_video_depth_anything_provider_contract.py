from __future__ import annotations

from pathlib import Path

from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.video_depth_anything_dense_provider import (
    I43D1_PROVIDER_ID,
    VideoDepthAnythingProductionProviderAdapter,
    video_depth_anything_dense_provider_descriptor,
)
from tests.i4_3d1_helpers import FakeVDAWorkerRunner, explicit_vda_selection, good_vda_environment, make_vda_context


def test_i43d1_contract_binds_existing_streaming_adapter_without_changing_sfm_authority(tmp_path: Path) -> None:
    context, environment = make_vda_context(tmp_path, view_count=4)
    adapter = VideoDepthAnythingProductionProviderAdapter(
        context=context,
        environment=environment,
        runner=FakeVDAWorkerRunner(),
    )
    request, decision = explicit_vda_selection(4)
    result = adapter.execute(request, decision)
    assert result.provider_id == I43D1_PROVIDER_ID
    assert result.implementation is DenseProviderImplementation.VIDEO_DEPTH_ANYTHING
    assert result.camera_pose_authority_owner == "pycolmap_sfm"
    assert result.single_authoritative_sfm_frame_preserved is True
    assert result.permanent_global_view_cap is None
    assert result.production_filter_policy_changed is False
    assert result.provider_promotion_performed is False


def test_i43d1_descriptor_retains_i43a_temporal_family_contract(tmp_path: Path) -> None:
    descriptor = video_depth_anything_dense_provider_descriptor(good_vda_environment(tmp_path))
    contract = descriptor.implementation_contract.as_dict()
    assert contract["integration_stage"] == "I4.3D"
    assert contract["family"] == "temporal_depth"
    assert contract["native_output"] == "depth_evidence"
    assert contract["frame_binding"] == "align_to_authoritative_sfm"
    assert contract["existing_adapter_expected"] is True
