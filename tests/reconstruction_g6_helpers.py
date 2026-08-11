from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.reconstruction_persistence_controller import ReconstructionPersistenceController
from core.reconstruction import (
    DeterministicMockStageExecutor,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRunStatus,
    create_reconstruction_engine_checkpoint,
    write_reconstruction_engine_checkpoint,
)
from tests.reconstruction_g3_helpers import checkpoint_prefix, semantic_fixture


G6_SESSION_ID = "77777777-7777-4777-8777-777777777777"


@dataclass(frozen=True, slots=True)
class ReconstructionG6EvidenceFixture:
    source_dir: Path
    scene_dir: Path
    checkpoint_manifest: Path
    providers: tuple
    plan: object
    checkpoint: object
    scene_session_id: str


def write_ready_image_sequence(root: Path) -> Path:
    source = (root / "review-sequence").resolve()
    source.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        (source / f"frame-{index:02d}.png").write_bytes(b"phase8.8g6-review")
    return source


def write_full_review_evidence(
    root: Path,
    *,
    session_id: str = G6_SESSION_ID,
) -> ReconstructionG6EvidenceFixture:
    """Write same-session persistent-scene + checkpoint fixtures using existing contracts."""

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    source_dir = write_ready_image_sequence(root)
    scene_dir = (root / "scene").resolve()
    checkpoint_dir = (root / "checkpoint").resolve()
    scene_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    providers, _request, plan, executors = semantic_fixture(session_id=session_id)
    result = ReconstructionPipelineOrchestrator(
        plan=plan,
        providers=providers,
        executors=executors,
    ).run()
    if result.status is not ReconstructionPipelineRunStatus.COMPLETED:
        raise RuntimeError(f"G6 scene fixture did not complete: {result.status.value}")

    persistence = ReconstructionPersistenceController()
    try:
        scene = persistence.adopt_pipeline_result(result, scene_dir)
    finally:
        persistence.shutdown()

    prefix_executors = {
        item.provider_id: DeterministicMockStageExecutor(item.provider_id)
        for item in providers
    }
    prefix = checkpoint_prefix(plan, providers, prefix_executors)
    checkpoint = create_reconstruction_engine_checkpoint(
        plan=plan,
        providers=providers,
        stage_results=prefix,
    )
    checkpoint_manifest = write_reconstruction_engine_checkpoint(checkpoint, checkpoint_dir)

    return ReconstructionG6EvidenceFixture(
        source_dir=source_dir,
        scene_dir=scene_dir,
        checkpoint_manifest=checkpoint_manifest,
        providers=providers,
        plan=plan,
        checkpoint=checkpoint,
        scene_session_id=scene.session_id,
    )
