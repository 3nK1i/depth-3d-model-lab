from __future__ import annotations

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


def _mock_worker(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse, json
            from pathlib import Path
            p=argparse.ArgumentParser(add_help=False)
            p.add_argument('--session-id', required=True)
            p.add_argument('--output-dir', required=True)
            args, _ = p.parse_known_args()
            out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
            print(json.dumps({'protocol':'eia.reconstruction.offline.v1','event':'progress','stage':'source_ingestion','message':'mock ingestion','progress_fraction':0.4}), flush=True)
            payload={
                'phase':'8.8A3','pass':True,'status':'completed','session_id':args.session_id,
                'production_input_assembly':True,'tests_fixture_used':False,
                'execution_boundary':{'pycolmap_external_process':True,'production_depth_imports_pycolmap':False},
                'canonical_geometry_authoritative':False,'persistent_scene_ready':False,
                'certified_metrology':False,'reconstruction_ui_enabled':False,
            }
            (out/'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')
            print(json.dumps({'protocol':'eia.reconstruction.offline.v1','event':'completed','stage':'camera_pose','message':'mock complete','progress_fraction':1.0}), flush=True)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_qprocess_offline_bridge_reports_events_result_and_completion(
    application_controller, tmp_path: Path, qtbot
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    worker = tmp_path / "mock_offline_worker.py"
    _mock_worker(worker)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    states: list[OfflineReconstructionRunState] = []
    events: list[dict] = []
    results: list[tuple[dict, str]] = []
    finished: list[dict] = []
    controller.offline_state_changed.connect(states.append)
    controller.offline_event_received.connect(events.append)
    controller.offline_result_ready.connect(lambda payload, out: results.append((payload, out)))
    controller.offline_run_finished.connect(finished.append)

    configuration = OfflineReconstructionConfiguration(
        source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
        inputs=(images,),
        output_dir=tmp_path / "offline-output",
        sfm_python=Path(sys.executable),
    )
    session_id = "00000000-0000-4000-8000-0000000088a3"

    try:
        returned = controller.start_offline(configuration, session_id=session_id)
        assert returned == session_id
        qtbot.waitUntil(
            lambda: controller.offline_state in {
                OfflineReconstructionRunState.COMPLETED,
                OfflineReconstructionRunState.FAILED,
            },
            timeout=5_000,
        )
        assert controller.offline_state is OfflineReconstructionRunState.COMPLETED
        assert OfflineReconstructionRunState.STARTING in states
        assert OfflineReconstructionRunState.RUNNING in states
        assert OfflineReconstructionRunState.COMPLETED in states
        assert [item["event"] for item in events] == ["progress", "completed"]
        assert results and results[0][0]["session_id"] == session_id
        assert results[0][0]["tests_fixture_used"] is False
        assert finished and finished[0]["state"] is OfflineReconstructionRunState.COMPLETED
        assert controller.offline_is_running is False
    finally:
        controller.shutdown()
        qtbot.wait(30)
