from __future__ import annotations


def admission() -> dict:
    scenarios = {}
    specs = {
        "castle_control": (19, 19),
        "castle_low_overlap": (7, 6),
        "castle_weak_parallax": (7, 7),
        "castle_overexposure": (19, 19),
        "castle_symmetry": (19, 12),
        "castle_dynamic_proxy": (19, 19),
    }
    for sid, (n, r) in specs.items():
        scenarios[sid] = {
            "scenario_id": sid,
            "input_images": n,
            "registered_images": r,
            "quality_status": "accepted",
            "mvs_input_integrity_verified": True,
            "report_binding": {"path": f"/{sid}.json", "sha256": "a" * 64},
        }
    return {
        "schema_version": 1,
        "phase": "I5.6A",
        "status": "mvs_robustness_scenarios_admitted",
        "pass": True,
        "scope": {
            "scenario_admission_only": True,
            "mvs_execution": False,
            "sfm_rerun": False,
            "dense_provider_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
        },
        "execution_policy": {
            "scenario_ids": list(specs),
            "provider_id": "colmap-mvs-external",
            "implementation": "colmap_mvs",
            "production_adapter": "MvsProductionProviderAdapter",
            "adaptive_preparation": "prepare_adaptive_provider_execution",
            "adaptive_selection": True,
            "engineer_override": None,
            "source_neighbors": 10,
            "patch_match_cache_gib": 2.0,
            "stereo_fusion_cache_gib": 2.0,
            "max_image_size": -1,
            "geometric_consistency": True,
            "sequential_execution": True,
            "permanent_global_view_cap": None,
            "result_driven_retuning": False,
        },
        "scenario_matrix": scenarios,
        "exclusions": {
            "castle_blur": {"classification": "blocked_upstream", "mvs_failure": False},
            "castle_underexposure": {"classification": "blocked_upstream", "mvs_failure": False},
            "castle_clutter": {"classification": "not_formally_admitted", "mvs_failure": False},
            "kitti_forward_motion": {"classification": "blocked_upstream", "mvs_failure": False},
        },
        "historical_positive_anchor": {
            "scenario_id": "historical_temple_ring_positive_anchor",
            "reference_views": 47,
            "evidence_only": True,
            "rerun_authorized": False,
        },
        "classification_policy": {
            "classification_threshold_used": None,
            "global_scalar_score_created": False,
            "retry_until_pass": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected_by_auto": False,
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "decision": {
            "i5_6a_complete": True,
            "i5_6b_authorized": True,
            "i5_6b_started": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "mp_sfm_authorized": False,
            "threshold_relaxation_authorized": False,
            "production_default_modified": False,
        },
    }


def receipt() -> dict:
    return {
        "schema_version": 1,
        "phase": "I5.6B_REPLAY_RECEIPT",
        "status": "pass",
        "pass": True,
        "scope": {
            "real_mvs_executed_by_test_receipt": False,
            "sfm_rerun": False,
            "thresholds_modified": False,
            "production_default_modified": False,
        },
        "counts": {
            "focused_tests": 5,
            "predecessor_total_tests": 10,
            "combined_exercised_tests": 15,
        },
    }


def scenario_record(sid: str, *, status: str = "pass", selected: int = 10, points: int = 1000, support: float = 2.5) -> dict:
    source = {
        "castle_control": 19,
        "castle_low_overlap": 6,
        "castle_weak_parallax": 7,
        "castle_overexposure": 19,
        "castle_symmetry": 12,
        "castle_dynamic_proxy": 19,
    }[sid]
    passed = status == "pass"
    return {
        "scenario_id": sid,
        "runtime_status": status,
        "provider_execution_completed": passed,
        "source_registered_views": source,
        "selected_views": min(selected, source),
        "selection_fraction": min(selected, source) / source,
        "adaptive_selection": True,
        "plan_reason_codes": ["all_views_fit_budget"],
        "native_metrics": {
            "geometric_depth_maps": min(selected, source) if passed else 0,
            "geometric_normal_maps": min(selected, source) if passed else 0,
            "fused_point_count": points if passed else None,
            "visibility_point_count": points if passed else None,
            "visibility_support_mean": support if passed else None,
            "wall_time_s": 10.0 if passed else 5.0,
            "process_peak_rss_gib": 1.0,
            "peak_vram_mib": 1024.0,
        },
        "failure": None if passed else {"reason": "synthetic worker failure", "worker_status": "fail"},
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "permanent_global_view_cap": None,
        },
    }


def recovery_fixture(tmp_path):
    import hashlib
    import struct

    def sha(path):
        h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

    images=tmp_path/'images'; sparse=tmp_path/'sparse'; dense=tmp_path/'dense'
    depth=dense/'stereo'/'depth_maps'; normal=dense/'stereo'/'normal_maps'
    for d in (images,sparse,depth,normal): d.mkdir(parents=True,exist_ok=True)
    names=[f'kf_{i:06d}.png' for i in range(19)]
    image_sha={}
    for i,name in enumerate(names):
        p=images/name; p.write_bytes(f'image-{i}'.encode()); image_sha[name]=sha(p)
        (depth/f'{name}.geometric.bin').write_bytes(b'depth'+bytes([i]))
        (normal/f'{name}.geometric.bin').write_bytes(b'normal'+bytes([i]))
    sparse_sha={}
    for name in ('cameras.bin','images.bin','points3D.bin'):
        p=sparse/name; p.write_bytes(name.encode()); sparse_sha[name]=sha(p)
    fused=dense/'fused.ply'
    fused.write_bytes(b'ply\nformat binary_little_endian 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+b'xyz')
    vis=tmp_path/'dense'/'fused.ply.vis'
    data=bytearray(struct.pack('<Q',3))
    for ids in ([0,1],[1,2],[0,2,3]):
        data+=struct.pack('<I',len(ids)); data+=struct.pack('<'+'I'*len(ids),*ids)
    vis.write_bytes(data)
    frame='i5_6b:castle_dynamic_proxy:pycolmap_sfm:synthetic'
    plan={
      'schema_version':1,'provider':{'provider_id':'colmap-mvs-external'},'reconstruction_frame_id':frame,
      'source_view_count':19,'selected_view_count':19,
      'selection_plan':{'reason_codes':['all_registered_views_fit_provider_budget']},
      'selection_request':{'available_view_count':19,'authoritative_sfm_available':True,'authoritative_sfm_frame_id':frame},
      'selection_decision':{'selected_provider_id':'colmap-mvs-external','selected_implementation':'colmap_mvs','automatic':False,'engineer_override_applied':False},
    }
    request={
      'schema_version':1,'phase':'I4.3B','action':'colmap_mvs_dense_provider',
      'input_images':str(images),'input_sparse':str(sparse),'selected_model_id':0,
      'image_names':names,'image_sha256':image_sha,'input_sparse_sha256':sparse_sha,
      'expected_registered_views':19,'reconstruction_frame_id':frame,
      'output_workspace':str(dense),'fused_ply':str(fused),
      'config':{
        'source_neighbors':10,'source_selection':'__auto__, 10','all_registered_views_selected':True,
        'selected_view_count':19,'patch_match_cache_gib':2.0,'stereo_fusion_cache_gib':2.0,
        'gpu_index':'0','max_image_size':-1,'geometric_consistency':True,
        'quality_thresholds_relaxed':False,'permanent_global_view_cap':None,
      },
      'authority':{
        'camera_pose_authority_owner':'pycolmap_sfm','single_authoritative_sfm_frame':True,
        'sfm_rerun':False,'bundle_adjustment_rerun':False,'pose_refinement':False,
        'metric_scale_claimed':False,'measurement_authoritative':False,'certified_metrology':False,
        'surface_mesh_generated':False,'production_filter_policy_changed':False,
        'provider_promotion_performed':False,'permanent_global_view_cap':None,
      },
    }
    state={
      'schema_version':1,'phase':'I4.3B','stage':'stereo_fusion','stage_started':True,'stage_completed':False,
      'heavy_mvs_started':True,'image_undistortion_started':True,'patch_match_started':True,'stereo_fusion_started':True,
      'resources':{'process_peak_rss_gib':3.5,'gpus':[{'used_vram_gib':0.5}]},
    }
    admission_node=admission()['scenario_matrix']['castle_dynamic_proxy']
    return admission_node,plan,request,state,fused,vis,depth,normal
