from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4
import hashlib

import numpy as np

from core.contracts import ImageSize
from core.reconstruction.contracts import InputSourceKind, InputSourceMetadata
from core.reconstruction.sfm_contracts import (
    OfflineSfmResult, ReconstructionGauge, SfmCameraEstimate, SfmStageStatistics,
    SfmWorkspaceImage, SfmWorkspaceManifest,
)
from core.reconstruction.sfm_pycolmap import PYCOLMAP_BACKEND_ID


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_real_sfm_report(tmp_path: Path, *, label: str, input_count: int, registered_count: int, quality: str = "accepted"):
    root = tmp_path / label
    images = root / "images"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(f"{label}-{name}".encode())
    source = InputSourceMetadata(
        source_id=f"{label}-source", kind=InputSourceKind.IMAGE_SEQUENCE,
        display_name="Castle-P19 synthetic", source_uri=str(tmp_path / "source"),
        expected_image_size=ImageSize(640, 480),
    )
    workspace_images=[]
    cameras=[]
    for i in range(input_count):
        name=f"{i:04d}.png"
        p=images/name
        p.write_bytes(f"{label}-image-{i}".encode())
        k=f"kf-{i}"
        workspace_images.append(SfmWorkspaceImage(
            order=i,keyframe_id=k,frame_id=f"frame-{i}",source_sequence_index=i,
            timestamp_ns=i*1_000_000,image_size=ImageSize(640,480),
            source_locator=str(p),staged_image_path=f"images/{name}",staged_sha256=sha(p),
        ))
        if i < registered_count:
            T=np.eye(4,dtype=np.float64); T[0,3]=i*0.1
            cameras.append(SfmCameraEstimate(
                estimate_id=f"est-{i}",camera_id="cam-0",keyframe_id=k,frame_id=f"frame-{i}",
                backend_image_id=i+1,T_reconstruction_from_camera=T,
            ))
    sid=str(uuid4())
    ws=SfmWorkspaceManifest(
        session_id=sid,source=source,workspace_root=str(root),images_directory="images",
        database_path="database.db",sparse_directory="sparse",checkpoints_directory="checkpoints",
        reports_directory="reports",logs_directory="logs",images=tuple(workspace_images),
    )
    sfm=OfflineSfmResult(
        session_id=sid,backend_id=PYCOLMAP_BACKEND_ID,backend_version="4.1.1",
        gauge=ReconstructionGauge(),cameras=tuple(cameras),landmarks=(),
        statistics=SfmStageStatistics(
            input_keyframes=input_count,registered_keyframes=registered_count,
            landmarks=0,observations=0,elapsed_ns=1,
            mean_reprojection_error_px=0.5,
        ),workspace=ws,metadata={"selected_model_id":0,"real_sfm":True,"quality_assessment":{"status":quality}},
    )
    return {
        "phase":"8.3D","real_sfm_executed":True,
        "registration_ratio":registered_count/input_count,
        "quality_assessment":{"status":quality,"mean_track_length":3.0},
        "sfm_result":sfm.as_dict(),
    }


def scenario(input_count:int, registered_count:int, quality="accepted"):
    return {
        "input_images":input_count,"registered_images":registered_count,
        "registration_ratio":registered_count/input_count,"quality_status":quality,
        "statistics":{
            "input_keyframes":input_count,"registered_keyframes":registered_count,
            "landmarks":0,"observations":0,
            "elapsed_ns":1,"mean_reprojection_error_px":0.5,
        },
        "authority":{"camera_pose_authority_owner":"pycolmap_sfm","per_scenario_gauge":"arbitrary"},
    }


def i55c_closure():
    return {
        "schema_version":1,"phase":"I5.5C","status":"dynamic_scene_handling_robustness_qualified_and_closed",
        "pass":True,"quality_status":"pass",
        "scope":{"evidence_only_qualification_and_closure":True,"sfm_rerun":False,"dense_provider_rerun":False,
                 "thresholds_modified":False,"production_default_modified":False},
        "decision":{"i5_5_complete":True,"i5_6_authorized":True,"i5_6_started":False,"provider_specific_robustness_started":False,
                    "auto_policy_implemented":False,"hybrid_authorized":False},
        "authority":{"camera_pose_authority_owner":"pycolmap_sfm","permanent_global_view_cap":None,
                     "metric_scale_claimed":False,"measurement_authoritative":False,"certified_metrology":False},
    }


def parent_replays():
    source_sha="a"*64
    i52={"phase":"I5.2B","status":"low_overlap_weak_parallax_replay_evidence_complete","pass":True,
         "provenance":{"frozen_source_asset_set_sha256":source_sha},
         "scenarios":{"control":scenario(19,19),"low_overlap":scenario(7,6),"weak_parallax":scenario(7,7)}}
    blur=scenario(19,0,"rejected"); blur.update({"sfm_executed":False,"pre_sfm_ingestion_rejected":True,"rejection_stage":"ingestion"})
    under=dict(blur)
    i53={"phase":"I5.3B","status":"blur_exposure_clutter_symmetry_replay_evidence_complete","pass":True,
         "provenance":{"frozen_source_asset_set_sha256":source_sha},
         "image_domain_revalidation":{"clutter":{"semantic_stress_revalidated":False}},
         "scenarios":{"blur":blur,"underexposure":under,"overexposure":scenario(19,19),"clutter":scenario(19,19),"symmetry":scenario(19,12)}}
    i55={"phase":"I5.5B","status":"current_frozen_foundation_dynamic_scene_replayed","pass":True,
         "provenance":{"frozen_source_asset_set_sha256":source_sha},
         "scenarios":{"dynamic_proxy":scenario(19,19)}}
    i54={"schema_version":1,"phase":"I5.4C","status":"forward_motion_kitti_robustness_qualified_and_closed","pass":True,"quality_status":"rejected",
         "qualification":{"operating_envelope":"outside_current_production_sparse_sfm_envelope_on_admitted_kitti_style_forward_motion_case",
                          "primary_bottleneck":"incremental_mapping_no_sparse_model","registered_images":0,"input_images":60}}
    return i52,i53,i54,i55
