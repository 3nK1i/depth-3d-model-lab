from __future__ import annotations

from pathlib import Path
import hashlib
import json

from core.reconstruction.mvs_da3_qualification import build_g2a_preflight_snapshot
from tests.i4_3g2a_helpers import passing_i4_3g1_snapshot


def _images(root: Path, count: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (root / f"{idx:03d}.png").write_bytes(f"{root.name}-{idx}".encode())
    return root



def _i42f_chain(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen_i4_2a.json"
    frozen.write_text(json.dumps({"upstream_house_mvs_closure": {
        "classification": "MVS clearly better than DA3, but imperfect",
        "path": "/tmp/historical.json",
        "sha256": "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50",
        "verdict": "mvs_better_but_imperfect"}}) + "\n")
    sha = hashlib.sha256(frozen.read_bytes()).hexdigest()
    closure = tmp_path / "i42f.json"
    closure.write_text(json.dumps({
        "phase":"I4.2F", "status":"pass", "pass":True,
        "decision":{"i4_2_closed":True,"i4_2f_closed":True,"i4_3_provider_architecture_authorized":True},
        "authority":{"authoritative_sparse_modified":False,"sfm_rerun":False,"bundle_adjustment_rerun":False,"pose_refinement":False,"patch_match_rerun":False,"stereo_fusion_rerun":False,"production_filter_policy_changed":False,"dense_provider_promoted":False,"metric_scale_claimed":False,"measurement_authoritative":False,"certified_metrology":False,"surface_mesh_generated":False},
        "frozen_summaries":{"i4_2a_summary":{"frozen_copy":str(frozen),"frozen_sha256":sha}}}) + "\n")
    return closure


def test_g2a_is_discovery_and_binding_only(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47)
    house = _images(tmp_path / "house", 68)
    closure = _i42f_chain(tmp_path)
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(), temple_ring_dir=temple,
        i4_2f_closure=closure, house_dir=house, house_historical_closure=None,
        discovery_roots=[tmp_path],
    )
    scope = snapshot["scope"]
    assert scope["source_evidence_binding_preflight_only"] is True
    assert scope["mvs_started"] is False
    assert scope["da3_started"] is False
    assert scope["sfm_started"] is False
    assert scope["g1_methodology_modified"] is False
    authority = snapshot["authority"]
    assert authority["camera_pose_authority_owner"] == "pycolmap_sfm"
    assert all(authority[key] is False for key in (
        "authoritative_sfm_modified", "provider_selected", "provider_promoted", "auto_policy_implemented",
        "hybrid_authorized", "metric_scale_claimed", "measurement_authoritative", "certified_metrology",
        "surface_mesh_generated",
    ))
