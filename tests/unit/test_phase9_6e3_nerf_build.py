from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app import reconstruction_nerf_build as build
from app.reconstruction_processing_jobs import DigitalTwinProcessingKind
from core.reconstruction import (
    NerfOptimizationProfile,
    persist_nerf_representation,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.errors import ReconstructionValidationError
from tests.phase9_4b_helpers import make_phase9_4b_da3_scene
from tests.phase9_5d_helpers import make_phase9_5d_source


def test_e3_presets_are_bounded_by_existing_qualified_training_contract() -> None:
    assert [item.training_steps for item in build.NERF_BUILD_PRESETS] == [64, 128, 256]
    assert build.nerf_build_preset("efficient").profile is NerfOptimizationProfile.EFFICIENT
    assert build.nerf_build_preset("max_qualified").profile is NerfOptimizationProfile.HIGH_QUALITY
    with pytest.raises(ValueError, match="Unknown NeRF build preset"):
        build.nerf_build_preset("production_magic")


def test_e3_build_job_materializes_exact_da3_source_and_uses_direct_external_program(tmp_path: Path, monkeypatch) -> None:
    scene, _paths, _poses, _intrinsics = make_phase9_4b_da3_scene(tmp_path / "fixture")
    manifest = write_persistent_reconstruction_scene(scene, tmp_path / "persistent")
    ns = tmp_path / "nerfstudio"; ns.mkdir()
    tc = tmp_path / "tiny-cuda-nn"; tc.mkdir()
    monkeypatch.setattr(
        build,
        "default_phase9_nerf_runtime_spec",
        lambda: SimpleNamespace(
            python_executable=sys.executable,
            nerfstudio_root=str(ns),
            tiny_cuda_nn_root=str(tc),
        ),
    )
    spec, command = build.build_nerf_training_job(
        scene=scene,
        scene_manifest=manifest,
        output_root=tmp_path / "jobs",
        preset_key="balanced",
    )
    assert spec.kind is DigitalTwinProcessingKind.NERF_APPEARANCE
    assert spec.expected_protocol == build.NERF_BUILD_PROTOCOL
    assert spec.scene_id == scene.scene_id
    assert spec.profile_label == "Balanced • 128 steps"
    assert command.program == Path(sys.executable).resolve()
    assert command.arguments[0].endswith("phase9_6e3_nerf_build_worker.py")
    request = spec.output_dir / "phase9_6e3_build_request.json"
    assert request.is_file()
    assert (spec.output_dir / "canonical-input" / "dataset.json").is_file()


def test_e3_rebuild_replaces_only_existing_nerf_provider_provenance(tmp_path: Path) -> None:
    scene, artifact, preview, runtime, optimization, summary, console = make_phase9_5d_source(tmp_path / "fixture")
    first = persist_nerf_representation(
        base_scene=scene,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=optimization,
        source_optimization_summary_path=summary,
        source_worker_console_path=console,
        output_dir=tmp_path / "first",
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
    )
    with pytest.raises(ReconstructionValidationError, match="refuses to overwrite an existing persisted NeRF appearance artifact"):
        persist_nerf_representation(
            base_scene=first.scene,
            nerf_artifact=artifact,
            preview_path=preview,
            runtime_evidence_path=runtime,
            source_optimization_manifest_path=optimization,
            output_dir=tmp_path / "reject",
            expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        )
    rebuilt = persist_nerf_representation(
        base_scene=first.scene,
        nerf_artifact=artifact,
        preview_path=preview,
        runtime_evidence_path=runtime,
        source_optimization_manifest_path=optimization,
        output_dir=tmp_path / "rebuilt",
        expected_dataset_fingerprint_sha256=artifact.dataset_fingerprint_sha256,
        replace_existing=True,
    )
    assert rebuilt.scene.canonical_geometry_artifact_id == first.scene.canonical_geometry_artifact_id
    assert rebuilt.scene.nerf_artifact is not None
    assert sum(1 for item in rebuilt.scene.provider_provenance if item.provider_id == rebuilt.scene.nerf_artifact.provider_id) == 1
