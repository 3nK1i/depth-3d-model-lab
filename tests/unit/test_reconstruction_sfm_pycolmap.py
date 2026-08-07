from __future__ import annotations

import json
from pathlib import Path
import sys
from threading import Thread
from time import sleep

import numpy as np
import pytest

from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    PYCOLMAP_REQUIRED_APIS,
    PycolmapEnvironmentProbe,
    ReconstructionCancelledError,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    ReconstructionValidationError,
    SfmBackendUnavailableError,
    SfmMatchingStrategy,
    SfmScaleStatus,
    SfmWorkerExecutionError,
    materialize_sfm_workspace,
)
import core.reconstruction.sfm_pycolmap as sfm_pycolmap
from tests.reconstruction_sfm_helpers import SFM_SESSION_ID, make_sfm_ingestion_result


def _ready_probe(executable: str, *, version: str = "4.1.1", cuda_devices: int = 1):
    return PycolmapEnvironmentProbe(
        python_executable=executable,
        available=True,
        pycolmap_version=version,
        distribution="pycolmap-cuda12",
        distribution_version=version,
        cuda_devices=cuda_devices,
        apis=tuple((name, True) for name in PYCOLMAP_REQUIRED_APIS),
    )


def _request(tmp_path: Path, *, count: int = 3, use_gpu: bool = True) -> OfflineSfmRequest:
    ingestion = make_sfm_ingestion_result(tmp_path / "sequence", count=count)
    workspace = materialize_sfm_workspace(ingestion, tmp_path / "workspace")
    return OfflineSfmRequest(
        session_id=SFM_SESSION_ID,
        profile=ReconstructionQualityProfile.EFFICIENT,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=SfmMatchingStrategy.SEQUENTIAL,
            use_gpu=use_gpu,
        ),
    )


def _worker_payload(request: OfflineSfmRequest) -> dict:
    names = [Path(item.staged_image_path).name for item in request.workspace.images]
    registered = []
    for index, name in enumerate(names, start=1):
        registered.append(
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
        )
    points = [
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
    ]
    return {
        "schema_version": 1,
        "status": "ok",
        "session_id": request.session_id,
        "backend": {
            "backend_id": "pycolmap-offline-sfm",
            "backend_version": "1",
            "pycolmap_version": "4.1.1",
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
        "points3D": points,
        "statistics": {
            "registered_keyframes": len(registered),
            "landmarks": 1,
            "observations": 2,
            "mean_reprojection_error_px": 0.7,
        },
        "warnings": [],
    }


def test_worker_payload_conversion_preserves_arbitrary_gauge_and_inverts_colmap_pose(tmp_path) -> None:
    request = _request(tmp_path)
    result = sfm_pycolmap._convert_worker_payload(
        request,
        _worker_payload(request),
        elapsed_ns=1234,
    )
    assert result.gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert result.gauge.unit_to_metre is None
    assert result.statistics.registered_keyframes == 3
    assert result.statistics.landmarks == 1
    assert len(result.camera_models) == 1
    assert result.camera_models[0].backend_model_name == "SIMPLE_RADIAL"
    assert result.camera_models[0].parameters == (280.0, 160.0, 120.0, 0.01)
    assert result.metadata[0][0] if isinstance(result.metadata, tuple) else True
    # COLMAP camera-from-reconstruction translation +1 on X becomes camera center -1.
    assert np.allclose(result.cameras[0].camera_center_reconstruction, [-1.0, 0.0, 0.0])
    assert result.landmarks[0].position_reconstruction.tolist() == [1.0, 2.0, 3.0]


def test_external_backend_runs_dependency_neutral_worker_process(tmp_path) -> None:
    request = _request(tmp_path)
    payload = _worker_payload(request)
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
request=json.loads(Path(a.request).read_text())
print('EIA_SFM_EVENT '+json.dumps({'schema_version':1,'event_type':'progress','stage':'matching','message':'fake matching','progress_fraction':0.5,'metadata':{}}), flush=True)
payload=PAYLOAD
Path(a.result).write_text(json.dumps(payload))
""".replace("PAYLOAD", repr(payload)),
        encoding="utf-8",
    )
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
    )
    result = backend.run_sparse_reconstruction(request, context)
    assert result.statistics.registered_keyframes == 3
    assert result.metadata
    assert any(event.progress_fraction == pytest.approx(0.5) for event in events if event.progress_fraction is not None)
    request_payload = json.loads(
        (Path(request.workspace.workspace_root) / "checkpoints" / "pycolmap_worker_request.json").read_text()
    )
    assert request_payload["configuration"]["maximum_image_size"] == 1600
    assert request_payload["configuration"]["camera_mode"] == "single"


def test_external_backend_rejects_wrong_pycolmap_version(tmp_path) -> None:
    request = _request(tmp_path)
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=tmp_path / "worker.py",
        probe_function=lambda executable: _ready_probe(executable, version="4.0.1"),
    )
    (tmp_path / "worker.py").write_text("pass\n", encoding="utf-8")
    with pytest.raises(SfmBackendUnavailableError):
        backend.run_sparse_reconstruction(request, context)


def test_external_backend_terminates_worker_on_cancellation(tmp_path) -> None:
    request = _request(tmp_path)
    fake_worker = tmp_path / "slow_worker.py"
    fake_worker.write_text(
        """
import argparse, time
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); p.parse_args()
print('EIA_SFM_EVENT {"schema_version":1,"event_type":"status","stage":"startup","message":"sleeping","progress_fraction":0.1,"metadata":{}}', flush=True)
time.sleep(10)
""",
        encoding="utf-8",
    )
    control = ReconstructionOperationControl()
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=control,
        emit_event=lambda event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        timeout_s=5.0,
        probe_function=lambda executable: _ready_probe(executable),
    )
    canceller = Thread(target=lambda: (sleep(0.15), control.request_cancel()), daemon=True)
    canceller.start()
    with pytest.raises(ReconstructionCancelledError):
        backend.run_sparse_reconstruction(request, context)
    canceller.join(timeout=1.0)


def test_external_backend_surfaces_worker_failure(tmp_path) -> None:
    request = _request(tmp_path)
    fake_worker = tmp_path / "failing_worker.py"
    fake_worker.write_text(
        """
import argparse, sys
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); p.parse_args()
print('synthetic worker failure', file=sys.stderr)
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        probe_function=lambda executable: _ready_probe(executable),
    )
    with pytest.raises(SfmWorkerExecutionError, match="synthetic worker failure"):
        backend.run_sparse_reconstruction(request, context)


def test_external_backend_requires_at_least_three_materialized_keyframes(tmp_path) -> None:
    request = _request(tmp_path, count=2)
    with pytest.raises(ReconstructionValidationError, match="at least three materialized keyframes"):
        sfm_pycolmap._worker_request_payload(request)



def test_external_backend_prefers_structured_worker_failure_payload(tmp_path) -> None:
    request = _request(tmp_path)
    fake_worker = tmp_path / "structured_failure_worker.py"
    fake_worker.write_text(
        """
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); a=p.parse_args()
Path(a.result).write_text(json.dumps({'schema_version':1,'status':'error','error':'structured failure detail'}))
raise SystemExit(2)
""",
        encoding="utf-8",
    )
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        probe_function=lambda executable: _ready_probe(executable),
    )
    with pytest.raises(SfmWorkerExecutionError, match="structured failure detail"):
        backend.run_sparse_reconstruction(request, context)


def test_external_backend_enforces_worker_timeout(tmp_path) -> None:
    request = _request(tmp_path)
    fake_worker = tmp_path / "timeout_worker.py"
    fake_worker.write_text(
        """
import argparse, time
p=argparse.ArgumentParser(); p.add_argument('--request'); p.add_argument('--result'); p.parse_args()
time.sleep(10)
""",
        encoding="utf-8",
    )
    context = BackendOperationContext(
        session_id=SFM_SESSION_ID,
        stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
        profile=ReconstructionQualityProfile.EFFICIENT,
        control=ReconstructionOperationControl(),
        emit_event=lambda event: None,
    )
    backend = ExternalPycolmapSfmBackend(
        sys.executable,
        worker_script=fake_worker,
        timeout_s=0.1,
        probe_function=lambda executable: _ready_probe(executable),
    )
    with pytest.raises(SfmWorkerExecutionError, match="exceeded timeout"):
        backend.run_sparse_reconstruction(request, context)
