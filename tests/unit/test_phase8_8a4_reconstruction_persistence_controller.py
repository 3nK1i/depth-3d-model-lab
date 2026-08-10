from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.reconstruction_persistence_controller import (
    ReconstructionPersistenceController,
    ReconstructionSceneAvailability,
)
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
    ReconstructionResumeCompatibilityStatus,
    build_reconstruction_pipeline_plan,
)
from core.reconstruction.errors import IncompatibleCheckpointError, ReconstructionValidationError


def _provider(provider_id: str, role: ReconstructionProviderRole, *, version: str = "1"):
    return ReconstructionProviderDescriptor(
        provider_id=provider_id,
        provider_version=version,
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.8a4-test",
    )


def _semantic_fixture(session_id: str | None = None):
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
        request_id="phase8-8a4-unit",
        session_id=session_id or str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.SEMANTIC_INSPECTION,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
    }
    return providers, request, plan, executors


def _checkpoint_prefix(plan, providers, executors):
    control = ReconstructionOperationControl()
    committed = []

    def hook(stages):
        committed.append(stages)
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run(control=control, stage_commit_hook=hook)
    assert result.status is ReconstructionPipelineRunStatus.CANCELLED
    return committed[-1]


def test_a3_real_shape_is_preserved_as_partial_not_persistent(tmp_path: Path) -> None:
    bridge = ReconstructionPersistenceController()
    session_id = str(uuid4())
    payload = {
        "phase": "8.8A3",
        "pass": True,
        "status": "completed",
        "session_id": session_id,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": False,
        "persistent_scene_ready": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }
    bridge.accept_offline_result(payload, tmp_path.resolve())

    assert bridge.scene_availability is ReconstructionSceneAvailability.PARTIAL
    assert bridge.scene is None
    assert bridge.offline_result is not None
    assert bridge.offline_result["session_id"] == session_id
    assert bridge.offline_result_path == (tmp_path / "offline_result.json").resolve()


def test_offline_result_cannot_fake_persistent_scene_without_canonical_authority(tmp_path: Path) -> None:
    bridge = ReconstructionPersistenceController()
    with pytest.raises(ReconstructionValidationError, match="canonical geometry authority"):
        bridge.accept_offline_result(
            {
                "status": "completed",
                "session_id": str(uuid4()),
                "persistent_scene_ready": True,
                "canonical_geometry_authoritative": False,
            },
            tmp_path.resolve(),
        )


def test_integrated_pipeline_result_adopts_and_roundtrips_persistent_scene(tmp_path: Path) -> None:
    providers, request, plan, executors = _semantic_fixture()
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    assert result.status is ReconstructionPipelineRunStatus.COMPLETED

    bridge = ReconstructionPersistenceController()
    scene = bridge.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
    assert scene.session_id == request.session_id
    assert scene.canonical_geometry_authoritative
    assert scene.certified_metrology is False
    assert scene.reconstruction_ui_enabled is False
    assert bridge.scene_availability is ReconstructionSceneAvailability.READY

    reloaded = ReconstructionPersistenceController().load_scene(
        (tmp_path / "scene").resolve(),
        expected_session_id=request.session_id,
    )
    assert reloaded.scene_id == scene.scene_id
    assert reloaded.scene_fingerprint_sha256 == scene.scene_fingerprint_sha256


def test_scene_load_rejects_expected_session_mismatch(tmp_path: Path) -> None:
    providers, _request, plan, executors = _semantic_fixture()
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    bridge = ReconstructionPersistenceController()
    bridge.adopt_pipeline_result(result, (tmp_path / "scene").resolve())

    other = ReconstructionPersistenceController()
    with pytest.raises(ReconstructionValidationError, match="session identity"):
        other.load_scene(
            (tmp_path / "scene").resolve(),
            expected_session_id=str(uuid4()),
        )


def test_checkpoint_create_load_and_prepare_compatible_resume(tmp_path: Path) -> None:
    providers, request, plan, executors = _semantic_fixture()
    prefix = _checkpoint_prefix(plan, providers, executors)
    bridge = ReconstructionPersistenceController()
    checkpoint = bridge.create_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    assert checkpoint.session_id == request.session_id
    assert checkpoint.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY

    loaded_bridge = ReconstructionPersistenceController()
    loaded = loaded_bridge.load_checkpoint((tmp_path / "checkpoint").resolve())
    assert loaded.checkpoint_id == checkpoint.checkpoint_id
    preparation = loaded_bridge.prepare_resume(
        plan=plan,
        providers=providers,
        available_payload_refs=loaded.payload_refs,
    )
    assert preparation.compatibility.status is ReconstructionResumeCompatibilityStatus.COMPATIBLE
    assert preparation.reused_stage_count == 3
    assert preparation.next_stage is ReconstructionPipelineStage.DENSE_GEOMETRY
    assert preparation.resume_stage_results == loaded.stage_results


def test_resume_compatibility_fails_closed_when_payload_is_missing(tmp_path: Path) -> None:
    providers, _request, plan, executors = _semantic_fixture()
    prefix = _checkpoint_prefix(plan, providers, executors)
    bridge = ReconstructionPersistenceController()
    checkpoint = bridge.create_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    compatibility = bridge.check_resume_compatibility(
        plan=plan,
        providers=providers,
        available_payload_refs=checkpoint.payload_refs[:-1],
    )
    assert compatibility.status is ReconstructionResumeCompatibilityStatus.INCOMPATIBLE
    with pytest.raises(IncompatibleCheckpointError):
        bridge.prepare_resume(
            plan=plan,
            providers=providers,
            available_payload_refs=checkpoint.payload_refs[:-1],
        )


def test_scene_and_checkpoint_manifests_fail_closed_on_tampering(tmp_path: Path) -> None:
    providers, _request, plan, executors = _semantic_fixture()
    result = ReconstructionPipelineOrchestrator(plan=plan, providers=providers, executors=executors).run()
    bridge = ReconstructionPersistenceController()
    bridge.adopt_pipeline_result(result, (tmp_path / "scene").resolve())
    scene_manifest = tmp_path / "scene" / "persistent_reconstruction_scene.json"
    scene_payload = json.loads(scene_manifest.read_text(encoding="utf-8"))
    scene_payload["canonical_geometry_artifact_id"] = "tampered"
    scene_manifest.write_text(json.dumps(scene_payload) + "\n", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError):
        ReconstructionPersistenceController().load_scene(scene_manifest.resolve())

    providers, _request, plan, executors = _semantic_fixture()
    prefix = _checkpoint_prefix(plan, providers, executors)
    checkpoint_bridge = ReconstructionPersistenceController()
    checkpoint_bridge.create_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=prefix,
        output_dir=(tmp_path / "checkpoint").resolve(),
    )
    checkpoint_manifest = tmp_path / "checkpoint" / "reconstruction_engine_checkpoint.json"
    checkpoint_manifest.write_text(checkpoint_manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="SHA-256"):
        ReconstructionPersistenceController().load_checkpoint(checkpoint_manifest.resolve())


def test_a4_application_bridge_has_no_heavy_backend_or_tests_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    texts = [
        (root / "app/reconstruction_persistence_controller.py").read_text(encoding="utf-8").lower(),
        (root / "app/reconstruction_controller.py").read_text(encoding="utf-8").lower(),
    ]
    for text in texts:
        for forbidden in (
            "import pycolmap",
            "from pycolmap",
            "import gsplat",
            "from gsplat",
            "import nerfstudio",
            "from nerfstudio",
            "import depth_anything_3",
            "from depth_anything_3",
            "import stella_vslam",
            "from stella_vslam",
            "from tests",
            "import tests",
        ):
            assert forbidden not in text
