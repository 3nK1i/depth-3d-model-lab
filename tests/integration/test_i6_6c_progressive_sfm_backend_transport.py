from __future__ import annotations

import json
from pathlib import Path
import sys

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    PYCOLMAP_REQUIRED_APIS,
    PycolmapEnvironmentProbe,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    materialize_sfm_workspace,
)
import core.reconstruction.sfm_pycolmap as sfm_pycolmap
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def _ready_probe(executable: str) -> PycolmapEnvironmentProbe:
    return PycolmapEnvironmentProbe(
        python_executable=executable,
        available=True,
        pycolmap_version="4.1.1",
        distribution="pycolmap-cuda12",
        distribution_version="4.1.1",
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_REQUIRED_APIS),
    )


def _request(tmp_path: Path) -> OfflineSfmRequest:
    ingestion = make_sfm_ingestion_result(tmp_path / "sequence", count=3)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    return OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=SfmMatchingStrategy.SEQUENTIAL,
            use_gpu=True,
        ),
    )


def _result_payload(request: OfflineSfmRequest) -> dict:
    worker_request = sfm_pycolmap._worker_request_payload(request)
    names = [Path(item.staged_image_path).name for item in request.workspace.images]
    registered = [
        {
            "image_id": index,
            "camera_id": 7,
            "name": name,
            "camera_from_reconstruction": [
                [1.0, 0.0, 0.0, float(index)],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
        }
        for index, name in enumerate(names, start=1)
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "session_id": request.session_id,
        "request_fingerprint": worker_request["request_fingerprint"],
        "resume_from_stage": "none",
        "resolved_configuration": worker_request["configuration"],
        "worker_elapsed_s": 0.1,
        "backend": {
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "1",
            "pycolmap_version": "4.1.1",
            "python_executable": sys.executable,
            "cuda_devices": 1,
            "use_gpu": True,
        },
        "selected_model_id": 0,
        "camera_models": [
            {
                "camera_id": 7,
                "model_name": "SIMPLE_RADIAL",
                "width": 320,
                "height": 240,
                "params": [280.0, 160.0, 120.0, 0.01],
            }
        ],
        "models": [
            {
                "model_id": 0,
                "registered_images": len(registered),
                "points3D": 1,
                "observations": 2,
                "mean_reprojection_error_px": 0.7,
            }
        ],
        "registered_images": registered,
        "points3D": [
            {
                "point3D_id": 11,
                "xyz": [1.0, 2.0, 3.0],
                "color": [10, 20, 30],
                "error": 0.7,
                "track": [
                    {"image_id": 1, "point2D_idx": 4, "xy": [10.0, 20.0]},
                    {"image_id": 2, "point2D_idx": 5, "xy": [12.0, 20.5]},
                ],
            }
        ],
        "statistics": {
            "registered_keyframes": len(registered),
            "landmarks": 1,
            "observations": 2,
            "mean_reprojection_error_px": 0.7,
        },
        "warnings": [],
    }


def test_external_backend_validates_and_forwards_progressive_snapshots(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result_payload = _result_payload(request)
    fake_worker = tmp_path / "progressive_worker.py"
    fake_worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
request=json.loads(Path(a.request).read_text())
def snap(revision, session_id=None, stage='mapping'):
    return {
        'schema_version':1,
        'phase':'I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT',
        'session_id': session_id or request['session_id'],
        'revision':revision,
        'timestamp_utc':f'2026-09-17T12:00:0{revision}+00:00',
        'elapsed_ns':revision*1000000,
        'stage':stage,
        'stage_detail':'transport-integration',
        'backend_id':'pycolmap-offline-sfm',
        'backend_version':'1',
        'request_fingerprint_sha256':request['request_fingerprint'],
        'statistics':{'input_views':3,'registered_cameras':3,'sparse_landmarks':1,'observations':2,'mean_reprojection_error_px':0.7},
        'progress_fraction':0.94 if stage=='finalizing' else 0.7,
        'candidate_model_id':'', 'candidate_model_count':None,
        'geometry':None, 'warnings':[],
        'authority':{
            'snapshot_authority':'provisional_visualization_only',
            'camera_pose_authority_owner':'PyCOLMAP',
            'pose_convention':'T_reconstruction_from_camera',
            'scale_status':'arbitrary','alignment_status':'arbitrary',
            'provisional':True,'final_result_admitted':False,
            'persistent_canonical_scene':False,'canonical_geometry_authoritative':False,
            'metric_scale_claimed':False,'measurement_authoritative':False,'certified_metrology':False,
        },
    }
print('EIA_SFM_PROGRESSIVE '+json.dumps(snap(1)), flush=True)
print('EIA_SFM_PROGRESSIVE '+json.dumps(snap(2, stage='finalizing')), flush=True)
Path(a.result).write_text(json.dumps(RESULT))
""".replace("RESULT", repr(result_payload)),
        encoding="utf-8",
    )
    snapshots = []
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda _event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        probe_function=lambda executable: _ready_probe(executable),
        progressive_snapshot_callback=snapshots.append,
    )

    result = backend.run_sparse_reconstruction(request, context)

    assert result.statistics.registered_keyframes == 3
    assert [item.revision for item in snapshots] == [1, 2]
    assert snapshots[-1].stage.value == "finalizing"
    assert all(item.final_result_admitted is False for item in snapshots)


def test_external_backend_rejects_cross_session_progressive_snapshot_fail_soft(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result_payload = _result_payload(request)
    fake_worker = tmp_path / "cross_session_worker.py"
    fake_worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
request=json.loads(Path(a.request).read_text())
payload={
 'schema_version':1,'phase':'I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT',
 'session_id':'aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee','revision':1,
 'timestamp_utc':'2026-09-17T12:00:01+00:00','elapsed_ns':1000000,
 'stage':'mapping','stage_detail':'wrong-session','backend_id':'pycolmap-offline-sfm','backend_version':'1',
 'request_fingerprint_sha256':request['request_fingerprint'],
 'statistics':{'input_views':3,'registered_cameras':3,'sparse_landmarks':1,'observations':2,'mean_reprojection_error_px':0.7},
 'progress_fraction':0.7,'candidate_model_id':'','candidate_model_count':None,'geometry':None,'warnings':[],
 'authority':{'snapshot_authority':'provisional_visualization_only','camera_pose_authority_owner':'PyCOLMAP','pose_convention':'T_reconstruction_from_camera','scale_status':'arbitrary','alignment_status':'arbitrary','provisional':True,'final_result_admitted':False,'persistent_canonical_scene':False,'canonical_geometry_authoritative':False,'metric_scale_claimed':False,'measurement_authoritative':False,'certified_metrology':False},
}
print('EIA_SFM_PROGRESSIVE '+json.dumps(payload), flush=True)
Path(a.result).write_text(json.dumps(RESULT))
""".replace("RESULT", repr(result_payload)),
        encoding="utf-8",
    )
    snapshots = []
    events = []
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=events.append,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        probe_function=lambda executable: _ready_probe(executable),
        progressive_snapshot_callback=snapshots.append,
    )
    result = backend.run_sparse_reconstruction(request, context)
    assert result.statistics.registered_keyframes == 3
    assert snapshots == []
    assert any("cross-session" in event.message for event in events)


def test_external_backend_rejects_cross_request_progressive_snapshot_fail_soft(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result_payload = _result_payload(request)
    fake_worker = tmp_path / "cross_request_worker.py"
    fake_worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
request=json.loads(Path(a.request).read_text())
payload={
 'schema_version':1,'phase':'I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT',
 'session_id':request['session_id'],'revision':1,
 'timestamp_utc':'2026-09-17T12:00:01+00:00','elapsed_ns':1000000,
 'stage':'mapping','stage_detail':'wrong-request','backend_id':'pycolmap-offline-sfm','backend_version':'1',
 'request_fingerprint_sha256':'e'*64,
 'statistics':{'input_views':3,'registered_cameras':3,'sparse_landmarks':1,'observations':2,'mean_reprojection_error_px':0.7},
 'progress_fraction':0.7,'candidate_model_id':'','candidate_model_count':None,'geometry':None,'warnings':[],
 'authority':{'snapshot_authority':'provisional_visualization_only','camera_pose_authority_owner':'PyCOLMAP','pose_convention':'T_reconstruction_from_camera','scale_status':'arbitrary','alignment_status':'arbitrary','provisional':True,'final_result_admitted':False,'persistent_canonical_scene':False,'canonical_geometry_authoritative':False,'metric_scale_claimed':False,'measurement_authoritative':False,'certified_metrology':False},
}
print('EIA_SFM_PROGRESSIVE '+json.dumps(payload), flush=True)
Path(a.result).write_text(json.dumps(RESULT))
""".replace("RESULT", repr(result_payload)),
        encoding="utf-8",
    )
    snapshots = []
    events = []
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=events.append,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        probe_function=lambda executable: _ready_probe(executable),
        progressive_snapshot_callback=snapshots.append,
    )
    result = backend.run_sparse_reconstruction(request, context)
    assert result.statistics.registered_keyframes == 3
    assert snapshots == []
    assert any("mismatched-request" in event.message for event in events)
