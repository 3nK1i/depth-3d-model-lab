from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.reconstruction.mvs_da3_fresh_scenario_binding import build_fresh_scenario_binding_snapshot
from core.reconstruction.mvs_da3_qualification import build_g2a_preflight_snapshot
from tests.i4_3g2a3_helpers import make_multiview_fixture
from tests.i4_3g2a_helpers import passing_i4_3g1_snapshot


def _opaque_images(root: Path, count: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"{index:03d}.png").write_bytes(f"opaque-{root.name}-{index}".encode())
    return root


def _i42f_chain(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen_i4_2a.json"
    frozen.write_text(json.dumps({"upstream_house_mvs_closure": {
        "classification": "MVS clearly better than DA3, but imperfect",
        "path": "/tmp/original-house-closure.json",
        "sha256": "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50",
        "verdict": "mvs_better_but_imperfect",
    }}) + "\n")
    frozen_sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    closure = tmp_path / "i42f.json"
    closure.write_text(json.dumps({
        "phase": "I4.2F", "status": "pass", "pass": True,
        "decision": {"i4_2_closed": True, "i4_2f_closed": True, "i4_3_provider_architecture_authorized": True},
        "authority": {
            "authoritative_sparse_modified": False, "sfm_rerun": False, "bundle_adjustment_rerun": False,
            "pose_refinement": False, "patch_match_rerun": False, "stereo_fusion_rerun": False,
            "production_filter_policy_changed": False, "dense_provider_promoted": False,
            "metric_scale_claimed": False, "measurement_authoritative": False,
            "certified_metrology": False, "surface_mesh_generated": False,
        },
        "frozen_summaries": {"i4_2a_summary": {"frozen_copy": str(frozen), "frozen_sha256": frozen_sha}},
    }) + "\n")
    return closure


def test_ready_g2a3_snapshot_is_the_only_new_input_needed_to_authorize_g2b(tmp_path: Path) -> None:
    room = make_multiview_fixture(tmp_path / "room")
    fresh = build_fresh_scenario_binding_snapshot(
        complete_room_dir=room,
        output_dir=tmp_path / "fresh",
        confirm_complete_room=True,
        confirm_low_texture=True,
        confirm_weak_overlap=True,
    )
    temple = _opaque_images(tmp_path / "temple", 47)
    house = _opaque_images(tmp_path / "house", 68)
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=_i42f_chain(tmp_path),
        house_dir=house,
        house_historical_closure=None,
        fresh_binding_snapshot=fresh,
    )
    assert snapshot["status"] == "ready"
    assert snapshot["binding_summary"]["fresh_scenario_bindings_ready"] == 4
    assert snapshot["binding_summary"]["historical_binding_verified"] is True
    assert snapshot["binding_summary"]["all_g2_sources_bound"] is True
    assert snapshot["decision"]["i4_3g2b_real_execution_authorized"] is True
    assert snapshot["scope"]["real_provider_execution_started"] is False
    by_id = {item["scenario_id"]: item for item in snapshot["scenario_bindings"]}
    for scenario_id in ("g2_complete_room_mvs_da3", "g2_low_texture_mvs_da3", "g2_weak_overlap_mvs_da3"):
        assert by_id[scenario_id]["binding_method"] == "frozen_i4_3g2a3_fresh_scenario_binding"
        assert by_id[scenario_id]["semantic_binding_inferred"] is False
