from __future__ import annotations
import json
import struct
import pytest

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_mvs_robustness import (
    I56B_PHASE,
    I56BReplayPolicy,
    analyze_i5_6b_provider_summary,
    build_i5_6b_mvs_robustness_replay,
    compare_i5_6b_scenarios,
    read_i5_6b_visibility_support,
    recover_i5_6b_interrupted_native_scenario,
    validate_i5_6a_for_i5_6b,
    validate_i5_6b_existing_scenario_record,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from tests.i5_6b_helpers import admission, receipt, recovery_fixture, scenario_record


def binding(name="x"):
    return I51EvidenceBinding(f"/{name}.json", "1" * 64)


def test_policy_preserves_production_mvs_and_no_retuning():
    p = I56BReplayPolicy()
    assert p.geometric_consistency is True
    assert p.sequential_execution is True
    assert p.permanent_global_view_cap is None
    assert p.result_driven_retuning is False


def test_validate_admission_requires_exact_authorized_matrix():
    out = validate_i5_6a_for_i5_6b(admission())
    assert set(out["scenario_matrix"]) == set(out["execution_policy"]["scenario_ids"])
    bad = admission(); bad["execution_policy"]["geometric_consistency"] = False
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6a_for_i5_6b(bad)


def test_validate_admission_accepts_formal_sort_keys_json_round_trip():
    serialized = json.dumps(admission(), sort_keys=True)
    loaded = json.loads(serialized)
    assert tuple(loaded["scenario_matrix"]) != tuple(loaded["execution_policy"]["scenario_ids"])
    out = validate_i5_6a_for_i5_6b(loaded)
    assert out["execution_policy"]["scenario_ids"] == admission()["execution_policy"]["scenario_ids"]
    assert set(out["scenario_matrix"]) == set(out["execution_policy"]["scenario_ids"])


def test_visibility_support_parser(tmp_path):
    p = tmp_path / "fused.ply.vis"
    data = bytearray(struct.pack("<Q", 3))
    for ids in ([0, 1], [2], [0, 2, 4]):
        data += struct.pack("<I", len(ids))
        data += struct.pack("<" + "I" * len(ids), *ids)
    p.write_bytes(bytes(data))
    s = read_i5_6b_visibility_support(p)
    assert s["point_count"] == 3
    assert s["support_min"] == 1
    assert s["support_max"] == 3
    assert s["unique_contributing_image_indices"] == [0, 1, 2, 4]
    assert s["support_mean"] == pytest.approx(2.0)


def test_comparison_is_directional_only_and_no_absolute_geometry():
    records = {sid: scenario_record(sid, points=1000) for sid in admission()["scenario_matrix"]}
    records["castle_symmetry"] = scenario_record("castle_symmetry", points=500, support=1.5)
    c = compare_i5_6b_scenarios(records)
    sym = c["castle_symmetry"]
    assert sym["fused_point_ratio_to_control"] == pytest.approx(0.5)
    assert sym["directional_native_support_reduction_observed"] is True
    assert sym["cross_run_absolute_geometry_comparison"] is False
    assert sym["classification_threshold_used"] is None


def test_build_replay_pass_when_all_six_structurally_complete():
    a = admission()
    records = {sid: scenario_record(sid) for sid in a["scenario_matrix"]}
    result = build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=a, i5_6a_binding=binding("a"),
        replay_receipt=receipt(), replay_receipt_binding=binding("r"),
        runtime_preflight={"mvs_runtime_ready": True, "cuda_devices": 1},
        scenario_records=records,
    )
    assert result["phase"] == I56B_PHASE
    assert result["quality_status"] == "pass"
    assert result["decision"]["i5_6c_authorized"] is True
    assert result["scope"]["sfm_rerun"] is False


def test_build_replay_degraded_when_stress_rejected_but_control_passes():
    a = admission(); records = {sid: scenario_record(sid) for sid in a["scenario_matrix"]}
    records["castle_low_overlap"] = scenario_record("castle_low_overlap", status="rejected")
    result = build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=a, i5_6a_binding=binding("a"), replay_receipt=receipt(),
        replay_receipt_binding=binding("r"), runtime_preflight={"mvs_runtime_ready": True, "cuda_devices": 1},
        scenario_records=records,
    )
    assert result["quality_status"] == "degraded"
    assert result["pass"] is True


def _provider_summary(vis_path, *, passed=True):
    authority = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "single_authoritative_sfm_frame_preserved": True,
        "sfm_rerun": False,
        "bundle_adjustment_rerun": False,
        "pose_refinement": False,
        "production_filter_policy_changed": False,
        "provider_promotion_performed": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
    }
    worker = {
        "status": "pass" if passed else "error",
        "error": "" if passed else "RuntimeError: synthetic PatchMatch failure",
        "heavy_mvs_started": True,
        "image_undistortion_started": True,
        "patch_match_started": True,
        "stereo_fusion_started": passed,
        "process_peak_rss_gib": 1.25,
        "resource_snapshots": {
            "after_patch_match": {"gpus": [{"used_vram_gib": 2.0}]},
        },
        "stage_seconds": {"patch_match": 3.0},
    }
    if passed:
        worker["patch_match_outputs"] = {
            "geometric_depth_maps": 3,
            "geometric_normal_maps": 3,
            "expected_reference_views": 3,
        }
        worker["fusion"] = {
            "ply_vertex_count": 3,
            "native_visibility_point_count": 3,
            "native_visibility": str(vis_path),
        }
    return {
        "schema_version": 1,
        "phase": "I4.3B",
        "status": "pass" if passed else "fail",
        "pass": passed,
        "resolved_config": {
            "selected_view_count": 3,
            "geometric_consistency": True,
            "quality_thresholds_relaxed": False,
            "permanent_global_view_cap": None,
        },
        "authority": authority,
        "integrity": {
            "authoritative_sparse_unchanged": True,
            "registered_images_unchanged": True,
        },
        "worker": worker,
        "execution_result": {"failure_reason": "" if passed else "synthetic PatchMatch failure"},
    }


def test_provider_summary_extracts_native_visibility_and_resources(tmp_path):
    p = tmp_path / "fused.ply.vis"
    data = bytearray(struct.pack("<Q", 3))
    for ids in ([0, 1], [1, 2], [0, 2, 3]):
        data += struct.pack("<I", len(ids))
        data += struct.pack("<" + "I" * len(ids), *ids)
    p.write_bytes(bytes(data))
    out = analyze_i5_6b_provider_summary(_provider_summary(p), wall_time_s=12.5)
    assert out["runtime_status"] == "pass"
    assert out["native_metrics"]["fused_point_count"] == 3
    assert out["native_metrics"]["visibility_support_mean"] == pytest.approx(7 / 3)
    assert out["native_metrics"]["unique_contributing_image_count"] == 4
    assert out["native_metrics"]["process_peak_rss_gib"] == pytest.approx(1.25)
    assert out["native_metrics"]["peak_vram_mib"] == pytest.approx(2048.0)


def test_provider_summary_preserves_structured_worker_rejection(tmp_path):
    out = analyze_i5_6b_provider_summary(_provider_summary(tmp_path / "unused.vis", passed=False), wall_time_s=4.0)
    assert out["runtime_status"] == "rejected"
    assert out["native_metrics"]["fused_point_count"] is None
    assert "synthetic PatchMatch failure" in out["failure"]["reason"]
    assert out["failure"]["patch_match_started"] is True


def test_recovery_reuses_complete_native_artifacts_without_rerun(tmp_path):
    admission_node,plan,request,state,fused,vis,depth,normal=recovery_fixture(tmp_path)
    record,summary=recover_i5_6b_interrupted_native_scenario(
        admission_node=admission_node,adaptive_plan=plan,worker_request=request,worker_state=state,
        fused_ply=fused,native_visibility=vis,depth_maps_dir=depth,normal_maps_dir=normal,
        artifact_bindings={"fused_ply":{"path":str(fused),"sha256":"1"*64}},resource_probe_after={"disk_free_gib":30.0})
    assert record["runtime_status"] == "pass"
    assert record["recovered_from_existing_native_artifacts"] is True
    assert record["provider_adapter_returned"] is False
    assert record["native_metrics"]["geometric_depth_maps"] == 19
    assert record["native_metrics"]["geometric_normal_maps"] == 19
    assert record["native_metrics"]["fused_point_count"] == 3
    assert record["native_metrics"]["visibility_point_count"] == 3
    assert record["native_metrics"]["wall_time_s"] is None
    assert record["recovery_provenance"]["mvs_rerun"] is False
    assert summary["status"] == "native_mvs_recovered_after_interrupted_finalization"


def test_recovery_rejects_authoritative_input_drift(tmp_path):
    admission_node,plan,request,state,fused,vis,depth,normal=recovery_fixture(tmp_path)
    first=next(iter(request["image_names"]))
    (tmp_path/'images'/first).write_bytes(b'drift')
    with pytest.raises(ReconstructionValidationError):
        recover_i5_6b_interrupted_native_scenario(
            admission_node=admission_node,adaptive_plan=plan,worker_request=request,worker_state=state,
            fused_ply=fused,native_visibility=vis,depth_maps_dir=depth,normal_maps_dir=normal,artifact_bindings={})


def test_recovery_rejects_missing_patchmatch_reference(tmp_path):
    admission_node,plan,request,state,fused,vis,depth,normal=recovery_fixture(tmp_path)
    next(depth.glob('*.geometric.bin')).unlink()
    with pytest.raises(ReconstructionValidationError):
        recover_i5_6b_interrupted_native_scenario(
            admission_node=admission_node,adaptive_plan=plan,worker_request=request,worker_state=state,
            fused_ply=fused,native_visibility=vis,depth_maps_dir=depth,normal_maps_dir=normal,artifact_bindings={})


def test_existing_record_validation_preserves_authority():
    out=validate_i5_6b_existing_scenario_record(scenario_record('castle_control'),scenario_id='castle_control')
    assert out['runtime_status']=='pass'
    bad=scenario_record('castle_control'); bad['authority']['authoritative_sfm_modified']=True
    with pytest.raises(ReconstructionValidationError):
        validate_i5_6b_existing_scenario_record(bad,scenario_id='castle_control')
