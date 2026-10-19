from __future__ import annotations

import json
from pathlib import Path
import sys

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    PYCOLMAP_REQUIRED_APIS,
    PycolmapEnvironmentProbe,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
)
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result
from core.reconstruction import OfflineSfmRequest, materialize_sfm_workspace


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


def test_external_backend_accepts_multimodel_status_seed_without_progress_restart(tmp_path) -> None:
    ingestion = make_sfm_ingestion_result(tmp_path / "source", count=4)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
    )

    worker = tmp_path / "fake_multimodel_worker.py"
    worker.write_text(
        r'''
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
r=json.loads(Path(a.request).read_text())
def emit(t, stage, msg, progress=None):
    print('DEPTH3D_SFM_EVENT '+json.dumps({'schema_version':1,'event_type':t,'stage':stage,'message':msg,'progress_fraction':progress,'metadata':{}}), flush=True)
emit('progress','mapping','Starting mapping',0.64)
emit('progress','mapping','Initial image pair registered',0.68)
emit('progress','mapping','Registered image',0.74)
emit('status','mapping','Additional sparse-model initial image pair registered',None)
emit('progress','mapping','Registered image',0.80)
names=r['image_names']
registered=[{'image_id':i,'camera_id':1,'name':name,'camera_from_reconstruction':[[1,0,0,float(i)],[0,1,0,0],[0,0,1,0]]} for i,name in enumerate(names, start=1)]
payload={
 'schema_version':1,'status':'ok','session_id':r['session_id'],
 'request_fingerprint':r['request_fingerprint'],'resume_from_stage':'none','worker_elapsed_s':0.01,
 'backend':{'backend_id':'pycolmap-offline-sfm','backend_version':'1','pycolmap_version':'4.1.1','python_executable':sys.executable,'cuda_devices':1,'use_gpu':True},
 'resolved_configuration':r['configuration'],'selected_model_id':0,
 'models':[{'model_id':0,'registered_images':len(names),'points3D':1,'observations':2,'mean_reprojection_error_px':0.5}],
 'camera_models':[{'camera_id':1,'model_name':'SIMPLE_RADIAL','width':320,'height':240,'params':[280,160,120,0]}],
 'registered_images':registered,
 'points3D':[{'point3D_id':1,'xyz':[0,0,2],'color':[100,120,140],'track':[{'image_id':1,'point2D_idx':0,'xy':[20,20]},{'image_id':2,'point2D_idx':0,'xy':[22,20]}]}],
 'statistics':{'registered_keyframes':len(names),'landmarks':1,'observations':2,'mean_reprojection_error_px':0.5},'warnings':[]}
Path(a.result).write_text(json.dumps(payload))
''',
        encoding="utf-8",
    )

    events = []
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=worker,
        probe_function=lambda executable: _ready_probe(executable),
    )
    result = backend.run_sparse_reconstruction(
        request,
        BackendOperationContext(
            session_id=SFM_SESSION_ID,
            stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
            profile=ReconstructionQualityProfile.EFFICIENT,
            control=ReconstructionOperationControl(),
            emit_event=events.append,
        ),
    )

    assert result.statistics.registered_keyframes == len(workspace.images)
    progress = [event.progress_fraction for event in events if event.progress_fraction is not None]
    assert progress == sorted(progress)
    assert any(
        event.as_dict()["metadata"].get("worker_stage") == "mapping"
        and "Additional sparse-model" in event.message
        and event.progress_fraction is None
        for event in events
    )
