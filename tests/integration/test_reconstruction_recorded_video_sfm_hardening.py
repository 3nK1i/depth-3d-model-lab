from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import pytest

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    OfflineSfmRequest,
    PYCOLMAP_REQUIRED_APIS,
    PycolmapEnvironmentProbe,
    RecordedVideoReader,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    materialize_sfm_workspace,
)
from tests.reconstruction_ingestion_helpers import textured_frame
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID


def test_recorded_video_materialization_reaches_external_sfm_contract(tmp_path) -> None:
    video_path = tmp_path / "recorded.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 240),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable in this environment.")
    for index in range(6):
        writer.write(textured_frame(index, width=320, height=240))
    writer.release()

    ingestion = ReconstructionIngestionPipeline(
        session_id=SFM_SESSION_ID,
        source=RecordedVideoReader(video_path),
        sampling_policy=FrameSamplingPolicy(mode=FrameSamplingMode.EVERY_FRAME),
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=6,
        ),
        quality_policy=FrameQualityPolicy(
            minimum_blur_variance=0.0,
            minimum_contrast_stddev=0.0,
            minimum_mean_luma=0.0,
            maximum_mean_luma=255.0,
            maximum_clipped_fraction=1.0,
            minimum_quality_score=0.0,
        ),
        cache_policy=IngestionCachePolicy(
            maximum_cpu_bytes=32 * 1024 * 1024,
            maximum_frames=1,
            gpu_staging_bytes=0,
        ),
    ).run()
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
    )

    worker = tmp_path / "fake_video_worker.py"
    worker.write_text(
        """
import argparse, json, sys
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
r=json.loads(Path(a.request).read_text())
names=r['image_names']
registered=[]
for i,name in enumerate(names, start=1):
    registered.append({'image_id':i,'camera_id':1,'name':name,'camera_from_reconstruction':[[1,0,0,float(i)],[0,1,0,0],[0,0,1,0]]})
payload={
 'schema_version':1,'status':'ok','session_id':r['session_id'],
 'request_fingerprint':r['request_fingerprint'],'resume_from_stage':'none',
 'worker_elapsed_s':0.01,
 'backend':{'pycolmap_version':'4.1.1','python_executable':sys.executable,'cuda_devices':1,'use_gpu':True},
 'resolved_configuration':r['configuration'],'selected_model_id':0,
 'models':[{'model_id':0,'registered_images':len(names),'points3D':1,'observations':2,'mean_reprojection_error_px':0.5}],
 'camera_models':[{'camera_id':1,'model_name':'SIMPLE_RADIAL','width':320,'height':240,'params':[280,160,120,0]}],
 'registered_images':registered,
 'points3D':[{'point3D_id':1,'xyz':[0,0,2],'color':[100,120,140],'track':[{'image_id':1,'point2D_idx':0,'xy':[20,20]},{'image_id':2,'point2D_idx':0,'xy':[22,20]}]}],
 'statistics':{'mean_reprojection_error_px':0.5},'warnings':[]}
Path(a.result).write_text(json.dumps(payload))
""",
        encoding="utf-8",
    )
    probe = PycolmapEnvironmentProbe(
        python_executable=sys.executable,
        available=True,
        pycolmap_version="4.1.1",
        distribution="pycolmap-cuda12",
        distribution_version="4.1.1",
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_REQUIRED_APIS),
    )
    result = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=worker,
        probe_function=lambda executable: probe,
    ).run_sparse_reconstruction(
        request,
        BackendOperationContext(
            session_id=SFM_SESSION_ID,
            stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
            profile=ReconstructionQualityProfile.EFFICIENT,
            control=ReconstructionOperationControl(),
            emit_event=lambda event: None,
        ),
    )
    payload = result.as_dict()
    assert result.statistics.registered_keyframes == len(workspace.images) == 6
    assert payload["workspace"]["source"]["kind"] == "recorded_video"
    assert payload["metadata"]["worker_request_fingerprint"]
    assert payload["metadata"]["resume_policy"] == "reuse_completed_stages"
