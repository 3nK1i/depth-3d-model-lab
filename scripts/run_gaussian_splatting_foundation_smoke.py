#!/usr/bin/env python3
"""Phase 8.7D1 deterministic Gaussian dataset/seed/external-worker smoke."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DenseSceneProfile,
    GaussianBackendKind,
    GaussianOptimizationProfile,
    GaussianOptimizationState,
    GaussianUsageScope,
    GaussianWorkerProcessSpec,
    GaussianWorkerRequest,
    NeuralSceneDataset,
    NeuralSceneFrame,
    build_gaussian_seed,
    gaussian_backend_provenance,
    neural_scene_dataset_fingerprint,
    run_gaussian_external_worker,
    write_gaussian_seed_npz,
    write_neural_scene_dataset_manifest,
)
from core.reconstruction import generate_quality_aware_surface_mesh
from scripts.run_surface_mesh_smoke import _arbitrary_surface, _camera

OUTPUT = Path("/tmp/phase8_7d1_gaussian_foundation")


def _write_ppm(path: Path, value: int) -> None:
    width, height = 32, 24
    rgb = bytes([value, min(255, value + 20), max(0, value - 20)])
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + rgb * (width * height))


def _dataset(surface) -> NeuralSceneDataset:
    images = OUTPUT / "images"
    images.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(4):
        image = images / f"frame_{index:03d}.ppm"
        _write_ppm(image, 70 + index * 20)
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = (index - 1.5) * 0.08
        frames.append(
            NeuralSceneFrame(
                frame_id=f"gaussian-{index}",
                image_path=str(image.resolve()),
                width=32,
                height=24,
                fx=28.0,
                fy=28.0,
                cx=16.0,
                cy=12.0,
                T_reconstruction_from_camera=pose,
                pose_layer=("optimized_keyframe" if index in {0, 3} else "optimized_propagated"),
                camera_profile_id=surface.camera_profile_id,
                timestamp_ns=index * 33_333_333,
            )
        )
    return NeuralSceneDataset(
        session_id="phase8-7d1-foundation-smoke",
        scene_profile=DenseSceneProfile.OBJECT_INSPECTION,
        reconstruction_frame_id=surface.reconstruction_frame_id,
        camera_profile_id=surface.camera_profile_id,
        scale_profile_id=surface.scale_profile_id,
        metric_geometry_status=surface.metric_geometry_status,
        scale_status=surface.scale_status,
        world_alignment_status=surface.world_alignment_status,
        unit_to_metre=surface.unit_to_metre,
        relative_scale_uncertainty=surface.relative_scale_uncertainty,
        frames=tuple(frames),
    )


def main() -> int:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    camera = _camera(96, 72)
    dense, quality, _scale = _arbitrary_surface(96, 72, camera)
    surface = generate_quality_aware_surface_mesh(dense, quality)
    original_positions = surface.mesh.vertices_reconstruction.copy()
    seed = build_gaussian_seed(surface, max_gaussians=512, minimum_vertex_quality=0.75)
    selected_positions = surface.mesh.vertices_reconstruction[seed.source_vertex_indices]
    positions_preserved = bool(np.array_equal(seed.positions_reconstruction, selected_positions.astype(np.float32)))
    canonical_surface_unchanged = bool(np.array_equal(surface.mesh.vertices_reconstruction, original_positions))
    dataset = _dataset(surface)
    manifest_path = write_neural_scene_dataset_manifest(dataset, OUTPUT / "dataset.json")
    seed_path = write_gaussian_seed_npz(seed, OUTPUT / "gaussian_seed.npz")
    request = GaussianWorkerRequest.from_dataset(
        request_id="phase8-7d1-mock-request",
        backend_kind=GaussianBackendKind.DEVELOPMENT_MOCK,
        profile=GaussianOptimizationProfile.EFFICIENT,
        dataset=dataset,
        dataset_manifest_path=manifest_path,
        seed_path=seed_path,
        output_dir=OUTPUT / "worker-output",
        maximum_input_frames=32,
    )
    spec = GaussianWorkerProcessSpec(
        python_executable=str(Path(sys.executable).resolve()),
        worker_script=str((ROOT / "scripts/run_gaussian_worker_mock.py").resolve()),
        timeout_s=30.0,
        require_isolated_interpreter=False,
    )
    artifact = run_gaussian_external_worker(spec, request)
    graphdeco_provenance = gaussian_backend_provenance(
        GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL
    )
    gsplat_provenance = gaussian_backend_provenance(GaussianBackendKind.GSPLAT_EXTERNAL)
    graphdeco_research_only_enforced = bool(
        graphdeco_provenance.usage_scope is GaussianUsageScope.RESEARCH_EVALUATION_ONLY
        and not graphdeco_provenance.commercial_use_permitted_by_backend_license
        and graphdeco_provenance.explicit_commercial_consent_required
    )
    gsplat_permissive_path_available = bool(
        gsplat_provenance.usage_scope is GaussianUsageScope.PERMISSIVE_GENERAL_USE
        and gsplat_provenance.commercial_use_permitted_by_backend_license
        and not gsplat_provenance.explicit_commercial_consent_required
    )
    app_yaml = (ROOT / "config/app.yaml").read_text(encoding="utf-8")
    reconstruction_ui_enabled = "reconstruction_mode_enabled: true" in app_yaml.lower()
    passed = all(
        [
            positions_preserved,
            canonical_surface_unchanged,
            dataset.frame_count == 4,
            artifact.optimization_state is GaussianOptimizationState.SEED_ONLY,
            artifact.render_capable is False,
            artifact.canonical_geometry_authoritative is False,
            artifact.certified_metrology is False,
            artifact.dataset_fingerprint_sha256 == neural_scene_dataset_fingerprint(dataset),
            artifact.backend_provenance.backend_kind is GaussianBackendKind.DEVELOPMENT_MOCK,
            graphdeco_research_only_enforced,
            gsplat_permissive_path_available,
            not reconstruction_ui_enabled,
        ]
    )
    summary = {
        "phase": "8.7D1",
        "pass": passed,
        "dataset_frames": dataset.frame_count,
        "dataset_fingerprint_sha256": neural_scene_dataset_fingerprint(dataset),
        "seed_gaussian_count": seed.gaussian_count,
        "seed_positions_preserved": positions_preserved,
        "canonical_surface_unchanged": canonical_surface_unchanged,
        "external_worker_executed": True,
        "development_worker": True,
        "development_backend_provenance": artifact.backend_provenance.as_dict(),
        "graphdeco_reference_backend": graphdeco_provenance.as_dict(),
        "gsplat_backend": gsplat_provenance.as_dict(),
        "graphdeco_research_only_enforced": graphdeco_research_only_enforced,
        "gsplat_permissive_path_available": gsplat_permissive_path_available,
        "dual_backend_contract_ready": True,
        "optimization_state": artifact.optimization_state.value,
        "render_capable": artifact.render_capable,
        "canonical_geometry_authoritative": artifact.canonical_geometry_authoritative,
        "scene_role": artifact.scene_role.value,
        "pose_convention": artifact.pose_convention,
        "metric_geometry_status": artifact.metric_geometry_status.value,
        "unit_to_metre": artifact.unit_to_metre,
        "certified_metrology": artifact.certified_metrology,
        "reconstruction_ui_enabled": reconstruction_ui_enabled,
        "scene_path": artifact.scene_path,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        print("Phase 8.7D1 Gaussian splatting foundation smoke: FAIL")
        return 1
    print("Phase 8.7D1 Gaussian splatting foundation smoke: PASS")
    print(f"frames={dataset.frame_count} seed_gaussians={seed.gaussian_count} positions_preserved={positions_preserved}")
    print("worker=external-development-mock optimization_state=seed_only render_capable=False")
    print("dual_backend_contract=graphdeco_reference+gsplat")
    print("graphdeco_usage_scope=research_evaluation_only commercial_use_permitted=False")
    print("gsplat_usage_scope=permissive_general_use commercial_use_permitted=True")
    print("canonical_geometry_authoritative=False scene_role=appearance_only")
    print(f"pose_convention={artifact.pose_convention}")
    print(f"gauge={artifact.metric_geometry_status.value} unit_to_metre={artifact.unit_to_metre}")
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print(f"summary={OUTPUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
