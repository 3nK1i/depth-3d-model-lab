from __future__ import annotations

import json
from pathlib import Path
import sys

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    OfflineSfmRequest,
    PYCOLMAP_REQUIRED_APIS,
    PycolmapEnvironmentProbe,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    materialize_sfm_workspace,
)
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def test_workspace_external_worker_conversion_lifecycle(tmp_path) -> None:
    ingestion = make_sfm_ingestion_result(tmp_path / "sequence", count=4)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    request = OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
    )
    names = [Path(item.staged_image_path).name for item in workspace.images]
    payload = {
        "schema_version": 1,
        "status": "ok",
        "session_id": request.session_id,
        "backend": {"pycolmap_version": "4.1.1"},
        "selected_model_id": 0,
        "camera_models": [
            {
                "camera_id": 1,
                "model_name": "SIMPLE_RADIAL",
                "width": 320,
                "height": 240,
                "params": [280.0, 160.0, 120.0, 0.0],
            }
        ],
        "models": [{"model_id": 0}],
        "registered_images": [
            {
                "image_id": index + 1,
                "camera_id": 1,
                "name": name,
                "camera_from_reconstruction": [
                    [1.0, 0.0, 0.0, float(index)],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
            }
            for index, name in enumerate(names)
        ],
        "points3D": [
            {
                "point3D_id": 1,
                "xyz": [0.0, 0.0, 2.0],
                "color": [100, 120, 140],
                "track": [
                    {"image_id": 1, "point2D_idx": 0, "xy": [20.0, 20.0]},
                    {"image_id": 2, "point2D_idx": 0, "xy": [22.0, 20.0]},
                ],
            }
        ],
        "statistics": {"mean_reprojection_error_px": 0.4},
        "warnings": [],
    }
    worker = tmp_path / "worker.py"
    worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
print('EIA_SFM_EVENT '+json.dumps({'schema_version':1,'event_type':'progress','stage':'mapping','message':'mapping','progress_fraction':0.7,'metadata':{}}), flush=True)
Path(a.result).write_text(json.dumps(PAYLOAD))
""".replace("PAYLOAD", repr(payload)),
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
    events = []
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
            emit_event=events.append,
        ),
    )
    report = Path(workspace.workspace_root) / workspace.reports_directory / "real_sfm_contract_report.json"
    report.write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n")
    restored = json.loads(report.read_text())
    assert restored["metadata"]["real_sfm"] is True
    assert restored["gauge"]["scale_status"] == "arbitrary"
    assert restored["gauge"]["unit_to_metre"] is None
    assert restored["statistics"]["registered_keyframes"] == 4
    assert restored["camera_models"][0]["backend_model_name"] == "SIMPLE_RADIAL"
    assert events[-1].progress_fraction == 1.0
