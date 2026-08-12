#!/usr/bin/env python3
"""Phase 8.9C persistence/reload/reproducibility qualification smoke.

This smoke starts no heavy/native reconstruction backend. It exercises the
already-qualified neutral persistence/checkpoint contracts with deterministic
core executors and real on-disk artifact references. It does not introduce a
new persistence format or modify authority semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DeterministicMockStageExecutor,
    PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
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
    check_reconstruction_engine_resume_compatibility,
    create_reconstruction_engine_checkpoint,
    persistent_scene_from_pipeline_result,
    read_persistent_reconstruction_scene,
    read_reconstruction_engine_checkpoint,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
    write_reconstruction_engine_checkpoint,
)
from core.reconstruction.errors import ReconstructionValidationError


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/phase8_9c_persistence_reproducibility"),
    )
    return parser.parse_args()


def _provider(role: ReconstructionProviderRole) -> ReconstructionProviderDescriptor:
    return ReconstructionProviderDescriptor(
        provider_id=f"phase8-9c-{role.value}",
        provider_version="1",
        roles=(role,),
        execution=ReconstructionProviderExecution.DEVELOPMENT_MOCK,
        priority=10,
        adapter_version="phase8.9c-qualification",
    )


def _build_plan():
    providers = tuple(_provider(role) for role in ReconstructionProviderRole)
    request = ReconstructionPipelineRequest(
        request_id="phase8-9c-persistence-smoke",
        session_id=str(uuid4()),
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(
            ReconstructionPipelineGoal.SEMANTIC_INSPECTION,
            ReconstructionPipelineGoal.GAUSSIAN_APPEARANCE,
            ReconstructionPipelineGoal.NERF_APPEARANCE,
        ),
        allow_optional_degradation=True,
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    executors = {
        provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
        for provider in providers
    }
    return providers, request, plan, executors


def _appearance_snapshots(root: Path, request_id: str) -> dict[str, dict[str, object]]:
    gaussian_dir = (root / "artifacts" / "gaussian").resolve()
    nerf_dir = (root / "artifacts" / "nerf").resolve()
    gaussian_dir.mkdir(parents=True, exist_ok=True)
    nerf_dir.mkdir(parents=True, exist_ok=True)
    gaussian_scene = gaussian_dir / "scene.npz"
    nerf_config = nerf_dir / "config.yml"
    nerf_checkpoint = nerf_dir / "model.ckpt"
    gaussian_scene.write_bytes(b"phase8-9c-gaussian-artifact")
    nerf_config.write_text("phase8-9c: nerfacto\n", encoding="utf-8")
    nerf_checkpoint.write_bytes(b"phase8-9c-nerf-checkpoint")
    return {
        "gaussian_scene": {
            "request_id": request_id,
            "scene_role": "appearance_only",
            "scene_path": str(gaussian_scene),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        "nerf_scene": {
            "request_id": request_id,
            "scene_role": "appearance_only",
            "config_path": str(nerf_config),
            "checkpoint_path": str(nerf_checkpoint),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    }


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    providers, request, plan, executors = _build_plan()
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    if result.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError(f"Unexpected qualification pipeline status: {result.status.value}")

    snapshots = _appearance_snapshots(output, request.request_id)
    scene = persistent_scene_from_pipeline_result(
        result,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T12:00:00+00:00",
    )
    first_manifest = write_persistent_reconstruction_scene(scene, (output / "scene-a").resolve())
    second_manifest = write_persistent_reconstruction_scene(scene, (output / "scene-b").resolve())
    first = read_persistent_reconstruction_scene(first_manifest)
    second = read_persistent_reconstruction_scene(second_manifest)
    refs = verify_persistent_scene_file_references(second)

    byte_reproducible = first_manifest.read_bytes() == second_manifest.read_bytes()
    checksum_reproducible = (
        first_manifest.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM
    ).read_text(encoding="utf-8") == (
        second_manifest.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM
    ).read_text(encoding="utf-8")
    identity_reproducible = bool(
        first.scene_id == second.scene_id == scene.scene_id
        and first.scene_fingerprint_sha256
        == second.scene_fingerprint_sha256
        == scene.scene_fingerprint_sha256
    )

    referenced_tamper_rejected = False
    if refs:
        artifact_path = Path(refs[0].path)
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(bytes((value ^ 0x01) for value in original))
        try:
            verify_persistent_scene_file_references(second)
        except ReconstructionValidationError:
            referenced_tamper_rejected = True
        finally:
            artifact_path.write_bytes(original)
    verify_persistent_scene_file_references(second)

    manifest_tamper_rejected = False
    original_manifest = second_manifest.read_bytes()
    second_manifest.write_bytes(original_manifest + b" ")
    try:
        read_persistent_reconstruction_scene(second_manifest)
    except ReconstructionValidationError:
        manifest_tamper_rejected = True
    finally:
        second_manifest.write_bytes(original_manifest)
    read_persistent_reconstruction_scene(second_manifest)

    control = ReconstructionOperationControl()
    committed: list[tuple] = []

    def hook(stages):
        committed.append(tuple(stages))
        if stages[-1].stage is ReconstructionPipelineStage.DEPTH_EVIDENCE:
            control.request_cancel()

    cancelled = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors={
            provider.provider_id: DeterministicMockStageExecutor(provider.provider_id)
            for provider in providers
        },
    ).run(control=control, stage_commit_hook=hook)
    if cancelled.status is not ReconstructionPipelineRunStatus.CANCELLED or not committed:
        raise RuntimeError("Could not create deterministic checkpoint prefix.")
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=committed[-1],
    )
    checkpoint_manifest = write_reconstruction_engine_checkpoint(
        checkpoint, (output / "checkpoint").resolve()
    )
    loaded_checkpoint = read_reconstruction_engine_checkpoint(checkpoint_manifest)
    compatibility = check_reconstruction_engine_resume_compatibility(
        loaded_checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=loaded_checkpoint.payload_refs,
    )

    drifted = list(providers)
    drifted[-1] = replace(
        drifted[-1], provider_version=f"{drifted[-1].provider_version}-drift"
    )
    provider_drift = check_reconstruction_engine_resume_compatibility(
        loaded_checkpoint,
        plan=plan,
        providers=tuple(drifted),
        available_payload_refs=loaded_checkpoint.payload_refs,
    )
    missing_payload = check_reconstruction_engine_resume_compatibility(
        loaded_checkpoint,
        plan=plan,
        providers=providers,
        available_payload_refs=loaded_checkpoint.payload_refs[:-1],
    )

    passed = bool(
        byte_reproducible
        and checksum_reproducible
        and identity_reproducible
        and len(refs) >= 3
        and referenced_tamper_rejected
        and manifest_tamper_rejected
        and scene.canonical_geometry_authoritative
        and scene.certified_metrology is False
        and scene.gaussian_artifact is not None
        and scene.nerf_artifact is not None
        and scene.gaussian_artifact.authority.value == "appearance_only"
        and scene.nerf_artifact.authority.value == "appearance_only"
        and compatibility.compatible
        and not provider_drift.compatible
        and not missing_payload.compatible
        and not errors
    )

    summary = {
        "phase": "8.9C",
        "pass": passed,
        "production_code_changed_by_8_9c": False,
        "heavy_backends_started": False,
        "new_persistence_format_created": False,
        "scene": {
            "scene_id": scene.scene_id,
            "fingerprint": scene.scene_fingerprint_sha256,
            "byte_reproducible": byte_reproducible,
            "checksum_reproducible": checksum_reproducible,
            "identity_reproducible": identity_reproducible,
            "referenced_file_count": len(refs),
            "referenced_tamper_rejected": referenced_tamper_rejected,
            "manifest_tamper_rejected": manifest_tamper_rejected,
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "gaussian_authority": scene.gaussian_artifact.authority.value if scene.gaussian_artifact else "missing",
            "nerf_authority": scene.nerf_artifact.authority.value if scene.nerf_artifact else "missing",
            "certified_metrology": scene.certified_metrology,
        },
        "checkpoint": {
            "checkpoint_id": loaded_checkpoint.checkpoint_id,
            "fingerprint": loaded_checkpoint.checkpoint_fingerprint_sha256,
            "compatible": compatibility.compatible,
            "provider_drift_rejected": not provider_drift.compatible,
            "missing_payload_rejected": not missing_payload.compatible,
            "reused_stage_count": compatibility.reused_stage_count,
            "next_stage": compatibility.next_stage.value if compatibility.next_stage else "",
        },
        "errors": errors,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("Phase 8.9C persistence/reproducibility smoke: " + ("PASS" if passed else "FAIL"))
    print(
        f"scene_id={scene.scene_id} refs={len(refs)} "
        f"byte_reproducible={byte_reproducible} tamper_rejected={referenced_tamper_rejected}"
    )
    print(
        f"checkpoint={loaded_checkpoint.checkpoint_id} compatible={compatibility.compatible} "
        f"provider_drift_rejected={not provider_drift.compatible} "
        f"missing_payload_rejected={not missing_payload.compatible}"
    )
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
