from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import ReconstructionController
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from app.reconstruction_one_start_preflight import OneStartBlockerCode
from core.reconstruction import ReconstructionQualityProfile
from core.reconstruction.errors import ReconstructionValidationError


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


def _images(root: Path, count: int = 5) -> Path:
    path = root / "images"
    path.mkdir()
    for index in range(count):
        (path / f"frame-{index:03d}.jpg").write_bytes(b"fixture")
    return path


def _mock_worker(path: Path, *, delay_seconds: float = 0.0) -> None:
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
            print(json.dumps({{'protocol':'depth3d.reconstruction.offline.v1','event':'progress','stage':'source_ingestion','message':'mock ingestion','progress_fraction':0.2}}), flush=True)
            time.sleep({float(delay_seconds)!r})
            payload={{
                'phase':'I6.2B-test','pass':True,'status':'completed','session_id':args.session_id,
                'production_input_assembly':True,'tests_fixture_used':False,
                'execution_boundary':{{'pycolmap_external_process':True,'production_depth_imports_pycolmap':False}},
                'canonical_geometry_authoritative':False,'persistent_scene_ready':False,
                'certified_metrology':False,'reconstruction_ui_enabled':True,
            }}
            (out/'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')
            print(json.dumps({{'protocol':'depth3d.reconstruction.offline.v1','event':'completed','stage':'camera_pose','message':'mock complete','progress_fraction':1.0}}), flush=True)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_i6_2b_preflight_and_launch_share_exact_session_output_and_source_identity(
    application_controller, tmp_path: Path, qtbot
) -> None:
    worker = tmp_path / "worker.py"
    _mock_worker(worker, delay_seconds=0.15)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    images = _images(tmp_path)
    output_root = tmp_path / "outputs"
    session_id = "00000000-0000-4000-8000-0000000062b1"
    guard_calls: list[str] = []
    controller.configure_start_guards(offline=lambda: guard_calls.append("guard") or None)
    controller.select_image_sequence(images)

    try:
        returned = controller.start_selected_offline(
            sfm_python=Path(sys.executable),
            output_root=output_root,
            profile=ReconstructionQualityProfile.EFFICIENT,
            session_id=session_id,
        )
        assert returned == session_id
        assert guard_calls == ["guard"]

        preflight = controller.last_offline_one_start_preflight
        assert preflight is not None and preflight.ready
        assert preflight.session_id == session_id
        assert preflight.source_path == str(images.resolve())
        assert preflight.proposed_output_dir == str((output_root / session_id).resolve())
        assert controller.offline_session_id == session_id
        assert controller.offline_output_dir == (output_root / session_id).resolve()
        assert controller.session_model.session_id == session_id
        assert controller.offline_workspace_snapshot.session_id == session_id

        with pytest.raises(ReconstructionValidationError, match="source cannot change"):
            controller.clear_source_selection()

        qtbot.waitUntil(lambda: not controller.is_active, timeout=5_000)
    finally:
        controller.shutdown()
        qtbot.wait(20)


def test_i6_2b_static_preflight_blocker_prevents_resource_guard_and_worker_launch(
    application_controller, tmp_path: Path, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    images = _images(tmp_path)
    output_root = tmp_path / "outputs"
    session_id = "00000000-0000-4000-8000-0000000062b2"
    stale = output_root / session_id
    stale.mkdir(parents=True)
    (stale / "stale.json").write_text("{}", encoding="utf-8")
    guard_calls: list[str] = []
    worker_calls: list[str] = []
    controller.configure_start_guards(offline=lambda: guard_calls.append("guard") or None)
    monkeypatch.setattr(
        controller.offline_controller,
        "start",
        lambda *_args, **_kwargs: worker_calls.append("worker") or session_id,
    )
    controller.select_image_sequence(images)

    try:
        with pytest.raises(ReconstructionValidationError, match="not empty"):
            controller.start_selected_offline(
                sfm_python=Path(sys.executable),
                output_root=output_root,
                session_id=session_id,
            )
        assert guard_calls == []
        assert worker_calls == []
        preflight = controller.last_offline_one_start_preflight
        assert preflight is not None and not preflight.ready
        assert OneStartBlockerCode.OUTPUT_DIR_NOT_EMPTY in preflight.blocker_codes
    finally:
        controller.shutdown()


def test_i6_2b_application_guard_conflict_is_preserved_as_preflight_evidence(
    application_controller, tmp_path: Path, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    images = _images(tmp_path)
    output_root = tmp_path / "outputs"
    session_id = "00000000-0000-4000-8000-0000000062b3"
    guard_calls: list[str] = []
    worker_calls: list[str] = []
    controller.configure_start_guards(
        offline=lambda: guard_calls.append("guard") or "Benchmark is running."
    )
    monkeypatch.setattr(
        controller.offline_controller,
        "start",
        lambda *_args, **_kwargs: worker_calls.append("worker") or session_id,
    )
    controller.select_image_sequence(images)

    try:
        with pytest.raises(ReconstructionValidationError, match="Benchmark is running"):
            controller.start_selected_offline(
                sfm_python=Path(sys.executable),
                output_root=output_root,
                session_id=session_id,
            )
        assert guard_calls == ["guard"]
        assert worker_calls == []
        preflight = controller.last_offline_one_start_preflight
        assert preflight is not None and not preflight.ready
        assert OneStartBlockerCode.APPLICATION_RESOURCE_CONFLICT in preflight.blocker_codes
    finally:
        controller.shutdown()


def test_i6_2b_commit_rechecks_source_after_resource_guard_before_worker_launch(
    application_controller, tmp_path: Path, monkeypatch
) -> None:
    controller = ReconstructionController(application_controller)
    images = _images(tmp_path)
    output_root = tmp_path / "outputs"
    session_id = "00000000-0000-4000-8000-0000000062b4"
    worker_calls: list[str] = []

    def mutate_source_during_guard() -> None:
        for image in images.glob("*.jpg"):
            image.unlink()
        return None

    controller.configure_start_guards(offline=mutate_source_during_guard)
    monkeypatch.setattr(
        controller.offline_controller,
        "start",
        lambda *_args, **_kwargs: worker_calls.append("worker") or session_id,
    )
    controller.select_image_sequence(images)

    try:
        with pytest.raises(ReconstructionValidationError, match="at least 3"):
            controller.start_selected_offline(
                sfm_python=Path(sys.executable),
                output_root=output_root,
                session_id=session_id,
            )
        assert worker_calls == []
        preflight = controller.last_offline_one_start_preflight
        assert preflight is not None and not preflight.ready
        assert OneStartBlockerCode.IMAGE_COUNT_TOO_SMALL in preflight.blocker_codes
    finally:
        controller.shutdown()
