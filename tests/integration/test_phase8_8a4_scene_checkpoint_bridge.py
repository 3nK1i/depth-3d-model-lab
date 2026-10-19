from __future__ import annotations

from pathlib import Path
import sys
import textwrap
from uuid import uuid4

import pytest

from app.application_controller import ApplicationController
from app.reconstruction_controller import (
    OfflineReconstructionConfiguration,
    OfflineReconstructionSourceKind,
    ReconstructionController,
    ReconstructionSceneAvailability,
)
from app.widgets.original_view import OriginalSourceView
from app.widgets.source_bar import SourceBar
from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionOperationControl,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderDescriptor,
    ReconstructionProviderExecution,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    build_reconstruction_pipeline_plan,
)


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


def _mock_a3_worker(path: Path) -> None:
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
            payload={
                'phase':'8.8A3','pass':True,'status':'completed','session_id':args.session_id,
                'camera_pose_available':True,'canonical_geometry_authoritative':False,
                'persistent_scene_ready':False,'certified_metrology':False,
                'reconstruction_ui_enabled':False,
            }
            (out/'offline_result.json').write_text(json.dumps(payload), encoding='utf-8')
            print(json.dumps({'protocol':'depth3d.reconstruction.offline.v1','event':'completed','stage':'camera_pose','message':'mock complete','progress_fraction':1.0}), flush=True)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _provider(provider_id: str, role: ReconstructionProviderRole):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        adapter_version="phase8.8a4-integration",
        priority=10,
    )


def _integrated_pipeline():
    providers = (
        _provider("ingestion", ReconstructionProviderRole.INGESTION),
        _provider("pose", ReconstructionProviderRole.CAMERA_POSE),
        _provider("depth", ReconstructionProviderRole.DEPTH),
        _provider("dense", ReconstructionProviderRole.DENSE_GEOMETRY),
        _provider("mesh", ReconstructionProviderRole.SURFACE_MESH),
        _provider("semantics", ReconstructionProviderRole.SEMANTICS),
        _provider("inspection", ReconstructionProviderRole.INSPECTION_QC),
    )
    request = ReconstructionPipelineRequest(
        request_id="phase8-8a4-integration",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers}
    return providers, request, plan, executors


def test_a3_partial_to_scene_checkpoint_resume_application_bridge(
    application_controller, tmp_path: Path, qtbot
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    worker = tmp_path / "mock_a3_worker.py"
    _mock_a3_worker(worker)
    controller = ReconstructionController(
        application_controller,
        parent=application_controller,
        offline_worker_script=worker,
    )
    try:
        session_id = controller.start_offline(
            OfflineReconstructionConfiguration(
                source_kind=OfflineReconstructionSourceKind.IMAGE_DIRECTORY,
                inputs=(images,),
                output_dir=(tmp_path / "offline").resolve(),
                sfm_python=Path(sys.executable),
                max_keyframes=3,
                use_gpu=False,
            )
        )
        qtbot.waitUntil(lambda: not controller.offline_is_running, timeout=5000)
        qtbot.wait(50)
        assert controller.persistence_controller.offline_result is not None
        assert controller.persistence_controller.offline_result["session_id"] == session_id
        assert controller.scene_availability is ReconstructionSceneAvailability.PARTIAL
        assert controller.persistent_scene is None

        providers, request, plan, executors = _integrated_pipeline()
        result = ReconstructionPipelineOrchestrator(
            plan=plan,
            providers=providers,
            executors=executors,
        ).run()
        scene = controller.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
        assert scene.session_id == request.session_id
        assert controller.scene_availability is ReconstructionSceneAvailability.READY
        assert controller.persistent_scene_manifest is not None

        control = ReconstructionOperationControl()
        committed = []

        def hook(stages):
            committed.append(stages)
            if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
                control.request_cancel()

        cancelled = ReconstructionPipelineOrchestrator(
            plan=plan,
            providers=providers,
            executors={p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers},
        ).run(control=control, stage_commit_hook=hook)
        assert cancelled.status is ReconstructionPipelineRunStatus.CANCELLED
        checkpoint = controller.create_engine_checkpoint(
            plan=plan,
            providers=providers,
            stage_results=committed[-1],
            output_dir=(tmp_path / "checkpoint").resolve(),
        )
        preparation = controller.prepare_engine_resume(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs,
        )
        assert preparation.compatibility.compatible
        assert preparation.reused_stage_count == 3
        assert preparation.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY

        resumed_executors = {
            p.provider_id: DeterministicMockStageExecutor(p.provider_id) for p in providers
        }
        resumed = ReconstructionPipelineOrchestrator(
            plan=plan,
            providers=providers,
            executors=resumed_executors,
        ).run(resume_stage_results=preparation.resume_stage_results)
        assert resumed.status is ReconstructionPipelineRunStatus.COMPLETED
        assert resumed_executors["ingestion"].calls == []
        assert resumed_executors["pose"].calls == []
        assert resumed_executors["depth"].calls == []
        assert resumed_executors["dense"].calls == ["dense_geometry"]
    finally:
        controller.shutdown()
        qtbot.wait(25)
