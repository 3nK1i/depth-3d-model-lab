from pathlib import Path

from core.reconstruction.temporal_provider_qualification import build_i4_3g3a_preflight_snapshot
from tests.i4_3g3a_helpers import all_bindings, passing_g1_snapshot, passing_g2_closure, runtime_probe


def test_g3a_snapshot_is_directly_g1_run_schema_ready_for_later_source_binding() -> None:
    snapshot = build_i4_3g3a_preflight_snapshot(
        i4_3g1_snapshot=passing_g1_snapshot(),
        i4_3g2_closure=passing_g2_closure(),
        scenario_bindings=all_bindings(),
        video_depth_anything_runtime=runtime_probe("video_depth_anything"),
        rolling_depth_runtime=runtime_probe("rolling_depth"),
    )
    assert [item["scenario_id"] for item in snapshot["scenario_bindings"]] == [
        "g3_temporal_static_sweep", "g3_temporal_motion", "g3_temporal_low_texture"
    ]
    for binding in snapshot["scenario_bindings"]:
        identity = binding["source_identity"]
        assert Path(identity["manifest_path"]).is_absolute()
        assert len(identity["manifest_sha256"]) == 64
        assert identity["asset_set_sha256"] == identity["source_file_sha256"]
        assert binding["semantic_binding_inferred"] is False
        assert binding["temporal_contract"]["matched_view_contiguous_source_order_required"] is True
