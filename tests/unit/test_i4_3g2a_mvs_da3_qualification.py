from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_qualification import (
    build_g2a_preflight_snapshot,
    build_source_identity,
    discover_image_directories,
    validate_i4_3g1_for_g2,
)
from tests.i4_3g2a_helpers import passing_i4_3g1_snapshot


def _images(root: Path, count: int, *, prefix: str = "img") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (root / f"{prefix}{idx:03d}.jpg").write_bytes(f"image-{idx}".encode())
    return root



def _i42f_chain(tmp_path: Path) -> Path:
    frozen_dir = tmp_path / "i42f" / "evidence" / "frozen_summaries"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    i42a = frozen_dir / "i4_2a_summary.json"
    i42a.write_text(json.dumps({
        "upstream_house_mvs_closure": {
            "classification": "MVS clearly better than DA3, but imperfect",
            "path": "/tmp/historical.json",
            "sha256": "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50",
            "verdict": "mvs_better_but_imperfect",
        }
    }) + "\n")
    import hashlib
    frozen_sha = hashlib.sha256(i42a.read_bytes()).hexdigest()
    source_dir = tmp_path / "i42b" / "evidence"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_i42a = source_dir / "i4_2a_summary.json"
    source_i42a.write_bytes(i42a.read_bytes())
    closure = tmp_path / "i42f" / "i4_2f_closure.json"
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
        "frozen_summaries": {"i4_2a_summary": {
            "frozen_copy": str(i42a), "frozen_sha256": frozen_sha,
            "source": str(source_i42a), "source_sha256": frozen_sha,
        }},
    }) + "\n")
    return closure



def _recovered_house_chain(tmp_path: Path) -> dict[str, Path]:
    import hashlib

    root = tmp_path / "recovered_chain"
    frozen_dir = root / "i42f" / "evidence" / "frozen_summaries"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    recovered = root / "persistent" / "i4_1_9a_a3_1_closure_recovered.json"
    recovered.parent.mkdir(parents=True, exist_ok=True)
    recovered.write_text(json.dumps({
        "authority": {
            "bundle_adjustment_rerun": False,
            "camera_geometry_modified": False,
            "certified_metrology": False,
            "da3_rerun": False,
            "dense_provider_promoted": False,
            "house_k_txt_consumed": False,
            "measurement_authoritative": False,
            "metric_scale_claimed": False,
            "mvs_rerun": False,
            "patch_match_rerun": False,
            "pose_refinement": False,
            "sfm_rerun": False,
            "single_authoritative_sfm_frame": True,
            "stereo_fusion_rerun": False,
            "surface_mesh_generated": False,
        },
        "manual_review": {
            "classification": "MVS clearly better than DA3, but imperfect",
            "confirmed": True,
            "required": True,
            "verdict": "mvs_better_but_imperfect",
        },
        "pass": True,
        "phase": "I4.1.9A-A3.1",
        "schema_version": 1,
        "source_a3": {
            "controlled_comparison_revalidated": True,
            "independent_auto_fit_forbidden": True,
            "phase": "I4.1.9A-A3",
            "same_authoritative_sfm_frame": True,
            "same_house_input": True,
            "shared_camera_parameters": True,
        },
        "status": "pass",
        "recovery_provenance": {
            "artifact_role": "recovered_semantic_copy_of_previously_passed_I4.1.9A_A3.1_closure",
            "source": "user-provided terminal transcript captured in this conversation on 2026-08-24",
            "original_closure_sha256_recorded_by_I4.2A": "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50",
            "original_bytes_available": False,
        },
    }) + "\n")

    recovered_sha = hashlib.sha256(recovered.read_bytes()).hexdigest()
    i42a = frozen_dir / "i4_2a_summary.json"
    i42a.write_text(json.dumps({
        "phase": "I4.2A",
        "status": "pass",
        "pass": True,
        "upstream_house_mvs_closure": {
            "classification": "MVS clearly better than DA3, but imperfect",
            "path": str(recovered),
            "sha256": recovered_sha,
            "verdict": "mvs_better_but_imperfect",
        },
    }) + "\n")
    i42a_sha = hashlib.sha256(i42a.read_bytes()).hexdigest()

    manifest = root / "i42f" / "i4_2f_evidence_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "summaries": {
            "i4_2a_summary": {
                "frozen_copy": str(i42a),
                "frozen_sha256": i42a_sha,
                "source": str(i42a),
                "source_sha256": i42a_sha,
            }
        },
    }) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    closure = root / "i42f" / "i4_2f_closure.json"
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
        "evidence_manifest": {"path": str(manifest), "sha256": manifest_sha},
    }) + "\n")
    return {"closure": closure, "manifest": manifest, "i42a": i42a, "recovered": recovered}


def _rehash_recovered_house_chain(chain: dict[str, Path]) -> None:
    import hashlib

    recovered = chain["recovered"]
    i42a = chain["i42a"]
    manifest = chain["manifest"]
    closure = chain["closure"]

    i42a_payload = json.loads(i42a.read_text())
    i42a_payload["upstream_house_mvs_closure"]["sha256"] = hashlib.sha256(recovered.read_bytes()).hexdigest()
    i42a.write_text(json.dumps(i42a_payload) + "\n")
    i42a_sha = hashlib.sha256(i42a.read_bytes()).hexdigest()

    manifest_payload = json.loads(manifest.read_text())
    record = manifest_payload["summaries"]["i4_2a_summary"]
    record["frozen_sha256"] = i42a_sha
    record["source_sha256"] = i42a_sha
    manifest.write_text(json.dumps(manifest_payload) + "\n")
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    closure_payload = json.loads(closure.read_text())
    closure_payload["evidence_manifest"]["sha256"] = manifest_sha
    closure.write_text(json.dumps(closure_payload) + "\n")


def _run_recovered_house_binding(tmp_path: Path, chain: dict[str, Path]) -> dict[str, object]:
    temple = _images(tmp_path / "temple_recovered", 47, prefix="templeR")
    house = _images(tmp_path / "house_recovered", 68)
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=chain["closure"],
        house_dir=house,
        house_historical_closure=None,
    )
    return next(
        item for item in snapshot["scenario_bindings"]
        if item["scenario_id"] == "g2_house_mvs_da3_historical"
    )

def test_g1_exact_contract_is_accepted() -> None:
    validate_i4_3g1_for_g2(passing_i4_3g1_snapshot())


@pytest.mark.parametrize("path,value", [
    (("coverage", "metric_count"), 20),
    (("coverage", "g2_scenarios"), 4),
    (("decision", "i4_3g2_mvs_da3_qualification_authorized"), False),
    (("authority", "auto_policy_implemented"), True),
])
def test_g1_drift_fails_closed(path: tuple[str, str], value: object) -> None:
    payload = json.loads(json.dumps(passing_i4_3g1_snapshot()))
    payload[path[0]][path[1]] = value
    with pytest.raises(ReconstructionValidationError):
        validate_i4_3g1_for_g2(payload)


def test_source_identity_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    root = _images(tmp_path / "images", 4)
    first = build_source_identity(root)
    second = build_source_identity(root)
    assert first.asset_set_sha256 == second.asset_set_sha256
    assert first.image_count == 4
    (root / "img002.jpg").write_bytes(b"mutated")
    third = build_source_identity(root)
    assert third.asset_set_sha256 != first.asset_set_sha256


def test_source_identity_rejects_too_few_images(tmp_path: Path) -> None:
    root = _images(tmp_path / "images", 2)
    with pytest.raises(ReconstructionValidationError):
        build_source_identity(root)


def test_discovery_does_not_infer_scenario_semantics(tmp_path: Path) -> None:
    _images(tmp_path / "candidate", 5)
    found = discover_image_directories([tmp_path])
    assert found and found[0]["image_count"] == 5
    assert found[0]["semantic_binding_inferred"] is False


def test_preflight_without_explicit_fresh_sources_is_safe_and_not_authorized(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47, prefix="templeR")
    house = _images(tmp_path / "house", 68)
    i42f = _i42f_chain(tmp_path)
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=i42f,
        house_dir=house,
        house_historical_closure=None,
        discovery_roots=[tmp_path],
    )
    assert snapshot["pass"] is True
    assert snapshot["status"] == "needs_binding"
    assert snapshot["decision"]["i4_3g2b_real_execution_authorized"] is False
    assert snapshot["scope"]["real_provider_execution_started"] is False
    assert snapshot["authority"]["provider_selected"] is False
    assert snapshot["authority"]["permanent_global_view_cap"] is None
    assert snapshot["binding_summary"]["fresh_scenario_bindings_ready"] == 1
    assert snapshot["binding_summary"]["historical_binding_verified"] is True



def test_historical_house_chain_falls_back_to_persistent_i42a_source(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47, prefix="templeR")
    house = _images(tmp_path / "house", 68)
    i42f = _i42f_chain(tmp_path)
    closure = json.loads(i42f.read_text())
    frozen_path = Path(closure["frozen_summaries"]["i4_2a_summary"]["frozen_copy"])
    frozen_path.unlink()

    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=i42f,
        house_dir=house,
        house_historical_closure=None,
    )
    binding = next(
        item for item in snapshot["scenario_bindings"]
        if item["scenario_id"] == "g2_house_mvs_da3_historical"
    )
    assert snapshot["binding_summary"]["historical_binding_verified"] is True
    assert binding["historical_evidence_verified"] is True
    assert binding["historical_evidence"]["verification_method"] == "persistent_i4_2f_closure_source_i4_2a_chain"



def test_historical_house_chain_resolves_via_sha_pinned_i42f_manifest(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47, prefix="templeR")
    house = _images(tmp_path / "house", 68)
    i42f = _i42f_chain(tmp_path)
    closure = json.loads(i42f.read_text())

    # Reproduce the persistent workstation layout that exposed the bug:
    # the closure record itself does not carry frozen_summaries, but it points
    # to a content-addressed evidence manifest that does.
    summaries = closure.pop("frozen_summaries")
    manifest = tmp_path / "i42f" / "i4_2f_evidence_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "summaries": summaries}) + "\n")
    import hashlib
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    closure["evidence_manifest"] = {"path": str(manifest), "sha256": manifest_sha}
    i42f.write_text(json.dumps(closure) + "\n")

    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=i42f,
        house_dir=house,
        house_historical_closure=None,
    )
    binding = next(
        item for item in snapshot["scenario_bindings"]
        if item["scenario_id"] == "g2_house_mvs_da3_historical"
    )
    assert snapshot["binding_summary"]["historical_binding_verified"] is True
    assert binding["historical_evidence_verified"] is True
    evidence = binding["historical_evidence"]
    assert evidence["verification_method"] == "persistent_i4_2f_manifest_frozen_i4_2a_chain"
    assert evidence["i4_2f_evidence_manifest"] == str(manifest.resolve())
    assert evidence["i4_2f_evidence_manifest_sha256"] == manifest_sha


def test_historical_house_chain_rejects_tampered_i42f_manifest(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47, prefix="templeR")
    house = _images(tmp_path / "house", 68)
    i42f = _i42f_chain(tmp_path)
    closure = json.loads(i42f.read_text())
    summaries = closure.pop("frozen_summaries")
    manifest = tmp_path / "i42f" / "i4_2f_evidence_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "summaries": summaries}) + "\n")
    import hashlib
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    closure["evidence_manifest"] = {"path": str(manifest), "sha256": manifest_sha}
    i42f.write_text(json.dumps(closure) + "\n")
    manifest.write_text(json.dumps({"schema_version": 1, "summaries": {}}) + "\n")

    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(),
        temple_ring_dir=temple,
        i4_2f_closure=i42f,
        house_dir=house,
        house_historical_closure=None,
    )
    binding = next(
        item for item in snapshot["scenario_bindings"]
        if item["scenario_id"] == "g2_house_mvs_da3_historical"
    )
    assert snapshot["binding_summary"]["historical_binding_verified"] is False
    assert binding["historical_evidence_verified"] is False
    assert binding["status"] == "needs_historical_evidence_binding"


def test_historical_house_recovered_semantic_chain_is_verified(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is True
    assert binding["status"] == "verified_historical_read_only"
    evidence = binding["historical_evidence"]
    assert evidence["verification_method"] == "persistent_recovered_house_closure_semantic_chain"
    assert evidence["historical_house_original_closure_sha256"] == "0fbc46280ec38e571a92c15f48eda9f6103735e6a8915c80798b58a7f2330b50"
    assert evidence["historical_house_recovered_closure"] == str(chain["recovered"].resolve())
    assert evidence["historical_house_recovered_closure_sha256"] != evidence["historical_house_original_closure_sha256"]
    assert evidence["original_bytes_available"] is False


def test_historical_house_recovered_chain_rejects_wrong_recovered_sha(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    chain["recovered"].write_text(chain["recovered"].read_text() + " ")
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False


def test_historical_house_recovered_chain_rejects_wrong_original_recorded_sha(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    recovered = json.loads(chain["recovered"].read_text())
    recovered["recovery_provenance"]["original_closure_sha256_recorded_by_I4.2A"] = "1" * 64
    chain["recovered"].write_text(json.dumps(recovered) + "\n")
    _rehash_recovered_house_chain(chain)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False


def test_historical_house_recovered_chain_rejects_wrong_verdict(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    recovered = json.loads(chain["recovered"].read_text())
    recovered["manual_review"]["verdict"] = "da3_better"
    chain["recovered"].write_text(json.dumps(recovered) + "\n")
    i42a = json.loads(chain["i42a"].read_text())
    i42a["upstream_house_mvs_closure"]["verdict"] = "da3_better"
    chain["i42a"].write_text(json.dumps(i42a) + "\n")
    _rehash_recovered_house_chain(chain)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False


def test_historical_house_recovered_chain_rejects_missing_recovery_provenance(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    recovered = json.loads(chain["recovered"].read_text())
    recovered.pop("recovery_provenance")
    chain["recovered"].write_text(json.dumps(recovered) + "\n")
    _rehash_recovered_house_chain(chain)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False


def test_historical_house_recovered_chain_rejects_original_bytes_available_true(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    recovered = json.loads(chain["recovered"].read_text())
    recovered["recovery_provenance"]["original_bytes_available"] = True
    chain["recovered"].write_text(json.dumps(recovered) + "\n")
    _rehash_recovered_house_chain(chain)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False


def test_historical_house_recovered_chain_rejects_arbitrary_replacement_json(tmp_path: Path) -> None:
    chain = _recovered_house_chain(tmp_path)
    chain["recovered"].write_text(json.dumps({
        "phase": "I4.1.9A-A3.1",
        "status": "pass",
        "pass": True,
        "schema_version": 1,
        "manual_review": {
            "confirmed": True,
            "classification": "MVS clearly better than DA3, but imperfect",
            "verdict": "mvs_better_but_imperfect",
        },
    }) + "\n")
    _rehash_recovered_house_chain(chain)
    binding = _run_recovered_house_binding(tmp_path, chain)
    assert binding["historical_evidence_verified"] is False

def test_explicit_source_binding_does_not_claim_semantics_were_inferred(tmp_path: Path) -> None:
    temple = _images(tmp_path / "temple", 47, prefix="templeR")
    house = _images(tmp_path / "house", 68)
    complete = _images(tmp_path / "complete", 8)
    low = _images(tmp_path / "low", 7)
    weak = _images(tmp_path / "weak", 6)
    i42f = _i42f_chain(tmp_path)
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=passing_i4_3g1_snapshot(), temple_ring_dir=temple, i4_2f_closure=i42f,
        house_dir=house, house_historical_closure=None,
        complete_room_dir=complete, low_texture_dir=low, weak_overlap_dir=weak,
    )
    bindings = {item["scenario_id"]: item for item in snapshot["scenario_bindings"]}
    for sid in ("g2_complete_room_mvs_da3", "g2_low_texture_mvs_da3", "g2_weak_overlap_mvs_da3"):
        assert bindings[sid]["status"] == "bound"
        assert bindings[sid]["semantic_binding_inferred"] is False
    assert snapshot["decision"]["i4_3g2b_real_execution_authorized"] is True
