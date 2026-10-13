from __future__ import annotations

import json
from pathlib import Path
import sys
import textwrap

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionRunState,
    OfflineReconstructionSourceKind,
    ReconstructionController,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar

SESSION_ID = "00000000-0000-4000-8000-0000000066c0"
REQUEST_SHA = "c" * 64


@pytest.fixture
def application_controller(qtbot, monkeypatch):
    monkeypatch.setattr("app.application_controller.CameraEnumerator.start", lambda self: None)
    source_bar = SourceBar()
    original_view = OriginalSourceView()
    qtbot.addWidget(source_bar)
    qtbot.addWidget(original_view)
    controller = ApplicationController(source_bar, original_view)
    yield controller
    controller.shutdown()


def _snapshot_payload(revision: int, *, session_id: str = SESSION_ID, stage: str = "mapping") -> dict:
    return {
        "schema_version": 1,
        "phase": "I6.6A_PROGRESSIVE_SFM_SNAPSHOT_AND_AUTHORITY_CONTRACT",
        "session_id": session_id,
        "revision": revision,
        "timestamp_utc": f"2026-09-17T12:00:{revision:02d}+00:00",
        "elapsed_ns": revision * 1_000_000,
        "stage": stage,
        "stage_detail": "qt-transport",
        "backend_id": "pycolmap-offline-sfm",
        "backend_version": "1",
        "request_fingerprint_sha256": REQUEST_SHA,
        "statistics": {
            "input_views": 24,
            "registered_cameras": min(24, revision + 2),
            "sparse_landmarks": 100,
            "observations": 300,
            "mean_reprojection_error_px": 0.6,
        },
        "progress_fraction": 0.94 if stage == "finalizing" else 0.7,
        "candidate_model_id": "",
        "candidate_model_count": None,
        "geometry": None,
        "warnings": [],
        "authority": {
            "snapshot_authority": "provisional_visualization_only",
            "camera_pose_authority_owner": "PyCOLMAP",
            "pose_convention": "T_reconstruction_from_camera",
            "scale_status": "arbitrary",
            "alignment_status": "arbitrary",
            "provisional": True,
            "final_result_admitted": False,
            "persistent_canonical_scene": False,
            "canonical_geometry_authoritative": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
        },
    }


def _mock_outer_worker(path: Path) -> None:
    first = json.dumps(_snapshot_payload(1)).replace("'", "\\'")
    second = json.dumps(_snapshot_payload(2)).replace("'", "\\'")
    final = json.dumps(_snapshot_payload(3, stage="finalizing")).replace("'", "\\'")
    path.write_text(
        textwrap.dedent(
            f"""
            import argparse, json, time
            from pathlib import Path
            p=argparse.ArgumentParser(add_help=False)
            p.add_argument('--session-id', required=True)
            p.add_argument('--output-dir', required=True)
            args, _ = p.parse_known_args()
            out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
            def emit(raw):
                payload=json.loads(raw); payload['session_id']=args.session_id
                print('DEPTH3D_SFM_PROGRESSIVE '+json.dumps(payload), flush=True)
            emit({first!r}); time.sleep(0.08)
            emit({second!r}); time.sleep(0.08)
            emit({final!r})
            result={{
                'phase':'8.8A3','pass':True,'status':'completed','session_id':args.session_id,
                'production_input_assembly':True,'tests_fixture_used':False,
                'execution_boundary':{{'pycolmap_external_process':True,'production_depth_imports_pycolmap':False}},
                'canonical_geometry_authoritative':False,'persistent_scene_ready':False,
                'certified_metrology':False,'reconstruction_ui_enabled':False,
            }}
            (out/'offline_result.json').write_text(json.dumps(result), encoding='utf-8')
            """
        ).strip() + "\n",
        encoding="utf-8",
    )


def test_qprocess_progressive_transport_is_session_bound_and_cleared_by_terminal_result(
    application_controller, tmp_path: Path, qtbot
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    worker = tmp_path / "mock_progressive_outer_worker.py"
    _mock_outer_worker(worker)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    observed = []
    controller.progressive_sparse_snapshot_changed.connect(observed.append)
    configuration = OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=tmp_path / "offline-output",
        sfm_python=Path(sys.executable),
    )
    try:
        controller.start_offline(configuration, session_id=SESSION_ID)
        qtbot.waitUntil(
            lambda: controller.offline_state in {
                OfflineReconstructionRunState.COMPLETED,
                OfflineReconstructionRunState.FAILED,
            },
            timeout=5_000,
        )
        assert controller.offline_state is OfflineReconstructionRunState.COMPLETED
        revisions = [item.revision for item in observed if item is not None]
        assert revisions == [1, 2, 3]
        assert observed[-1] is None
        assert controller.progressive_sparse_snapshot is None
        stats = controller.progressive_sfm_transport_stats
        assert stats["delivered"] == 3
        assert stats["rejected"] == 0
    finally:
        controller.shutdown()
        qtbot.wait(20)


def test_outer_controller_rejects_cross_session_snapshot(application_controller) -> None:
    controller = ReconstructionController(application_controller)
    observed = []
    logs = []
    controller.progressive_sparse_snapshot_changed.connect(observed.append)
    controller.offline_log_received.connect(logs.append)
    controller.offline_controller._session_id = SESSION_ID
    controller.offline_controller._progressive_sfm_buffer.reset(SESSION_ID)
    controller.offline_controller._set_state(OfflineReconstructionRunState.RUNNING)

    from core.reconstruction.reconstruction_progressive_sfm_contract import ProgressiveSfmSnapshot

    wrong = ProgressiveSfmSnapshot.from_payload(
        _snapshot_payload(
            1,
            session_id="aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee",
        )
    )
    controller._on_progressive_sparse_snapshot_received(wrong)
    assert observed == []
    assert controller.progressive_sparse_snapshot is None
    assert any("cross-session" in item for item in logs)
    controller.shutdown()
