from copy import deepcopy
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.temporal_provider_closure import build_i4_3d3_temporal_provider_closure
from tests.i4_3d3_helpers import d1_snapshot, d2_snapshot


def test_i43d3_closes_temporal_architecture_without_promotion() -> None:
    closure = build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1_snapshot(), i4_3d2_snapshot=d2_snapshot())
    assert closure["pass"] is True
    assert closure["decision"]["i4_3d_temporal_provider_architecture_closed"] is True
    assert closure["decision"]["i4_3e_authorized"] is True
    assert closure["decision"]["provider_promoted"] is False
    assert closure["decision"]["auto_policy_implemented"] is False
    assert closure["authority"]["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert closure["authority"]["metric_scale_claimed"] is False


def test_i43d3_preserves_distinct_temporal_semantics() -> None:
    closure = build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1_snapshot(), i4_3d2_snapshot=d2_snapshot())
    assert closure["providers"]["video_depth_anything"]["temporal_mode"] == "official_streaming_cache"
    assert closure["providers"]["rolling_depth"]["temporal_mode"] == "multi_dilation_snippet_coalignment"
    assert closure["providers"]["rolling_depth"]["per_frame_alignment"] is False
    assert closure["providers"]["rolling_depth"]["sequence_affine_establishes_physical_metric_scale"] is False


def test_i43d3_rejects_provider_promotion_drift() -> None:
    d2 = deepcopy(d2_snapshot())
    d2["authority"]["provider_promotion_performed"] = True
    with pytest.raises(ReconstructionValidationError, match="provider_promotion_performed=False"):
        build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1_snapshot(), i4_3d2_snapshot=d2)


def test_i43d3_rejects_per_frame_rolling_alignment() -> None:
    d2 = deepcopy(d2_snapshot())
    d2["architecture"]["per_frame_alignment"] = True
    with pytest.raises(ReconstructionValidationError, match="per_frame_alignment=False"):
        build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1_snapshot(), i4_3d2_snapshot=d2)


def test_i43d3_rejects_vda_metric_promotion() -> None:
    d1 = deepcopy(d1_snapshot())
    d1["authority"]["model_metric_depth_promoted_to_canonical_metric_scale"] = True
    with pytest.raises(ReconstructionValidationError, match="model_metric_depth_promoted_to_canonical_metric_scale=False"):
        build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1, i4_3d2_snapshot=d2_snapshot())


def test_i43d3_rejects_unresolved_d2_closure_marker() -> None:
    d2 = deepcopy(d2_snapshot())
    d2["qualification"]["temporal_provider_closure_pending_i4_3d3"] = False
    with pytest.raises(ReconstructionValidationError, match="temporal_provider_closure_pending_i4_3d3=True"):
        build_i4_3d3_temporal_provider_closure(i4_3d1_snapshot=d1_snapshot(), i4_3d2_snapshot=d2)
