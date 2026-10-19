#!/usr/bin/env python3
"""Phase 8.9F3 real Gaussian qualification from real F2 scene evidence.

This release-validation runner composes existing Phase 8 contracts only.  It
replays materialized F2 DA3 evidence through the already-qualified canonical
geometry pipeline in memory, materializes the existing backend-neutral neural
scene dataset, seeds Gaussians from the canonical surface, and invokes the
already-qualified D3 gsplat held-out/performance benchmark in an isolated
interpreter.

Gaussian remains appearance-only.  No metric scale or metrology claim is made.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    BoundEvidenceProviderExecutor,
    CanonicalDenseGeometryProviderExecutor,
    DenseFusionPolicy,
    DenseGeometryQualityPolicy,
    DenseSceneProfile,
    PipelineIngestionEvidence,
    PipelinePayloadStore,
    ReconstructionArtifactKind,
    ReconstructionPath,
    ReconstructionPipelineGoal,
    ReconstructionPipelineOrchestrator,
    ReconstructionPipelineRequest,
    ReconstructionPipelineRunStatus,
    ReconstructionPipelineStage,
    ReconstructionProviderRole,
    ReconstructionQualityProfile,
    SurfaceMeshProviderExecutor,
    SurfaceMeshResult,
    arbitrary_scale_reference,
    bound_evidence_provider_descriptor,
    build_gaussian_seed,
    build_reconstruction_pipeline_plan,
    canonical_geometry_provider_descriptors,
    da3_camera_profile_from_result,
    da3_supporting_evidence_from_result,
    materialize_appearance_dataset,
    new_reconstruction_session_id,
    read_da3_result_manifest,
    validate_da3_npz_result,
    write_gaussian_seed_npz,
)
from core.reconstruction.da3_provider import read_da3_input_manifest

PHASE = "8.9F3"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "a0366d4e889a227bcdbf19ab60c929814eb96c63"
EXPECTED_GSPLAT_VERSION = "1.5.3"
EXPECTED_GSPLAT_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
DEFAULT_F2_ROOT = Path("/tmp/phase8_9f2_real_geometry")
DEFAULT_OUTPUT = Path("/tmp/phase8_9f3_real_gaussian")
DEFAULT_GSPLAT_PYTHON = Path.home() / "anaconda3/envs/depth3d-gaussian-gsplat/bin/python"
DEFAULT_METRICS_PYTHON = Path.home() / "anaconda3/envs/depth3d-gaussian-metrics/bin/python"
DEFAULT_GSPLAT_ROOT = Path.home() / "Desktop/depth3d-gaussian-backends/gsplat"
DEFAULT_TRAINING_STEPS = 64
DEFAULT_RENDER_REPEATS = 3

F3_REPO_FILES = {
    "docs/phase_history/PHASE8_9F3_VALIDATION.md",
    "scripts/run_phase8_9f3_real_gaussian_campaign.py",
    "scripts/run_phase8_9f3_tests.sh",
    "tests/integration/test_phase8_9f3_real_gaussian_contract.py",
    "tests/unit/test_phase8_9f3_real_gaussian_campaign.py",
}

def _new_validation_session_id() -> str:
    """Create a contract-valid session identifier for the F3 validation bridge."""
    return new_reconstruction_session_id()


HEAVY_MODULES = (
    "gsplat",
    "diff_gaussian_rasterization",
    "simple_knn",
    "nerfstudio",
    "tinycudann",
    "depth_anything_3",
    "pycolmap",
)


def _git(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git command failed")
    return completed.stdout.strip()


def _repository_state(expected_head: str) -> dict[str, Any]:
    branch = _git(["branch", "--show-current"])
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/phase8-reconstruction"])
    status_lines = _git(["status", "--short", "--untracked-files=all"]).splitlines()
    unexpected: list[str] = []
    overlay: list[str] = []
    for line in status_lines:
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if path in F3_REPO_FILES:
            overlay.append(path)
        else:
            unexpected.append(line)
    passed = bool(
        branch == EXPECTED_BRANCH
        and head == origin
        and (not expected_head or head == expected_head)
        and not unexpected
    )
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": expected_head,
        "head_matches_origin": head == origin,
        "head_matches_expected": not expected_head or head == expected_head,
        "f3_overlay_files_present": sorted(overlay),
        "unexpected_worktree_changes": unexpected,
        "pass": passed,
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid or missing JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _run_streamed(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"stage failed with exit code {returncode}: {' '.join(command)}")


def _heavy_runtime_importability() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in HEAVY_MODULES}


def _validate_f2(f2_root: Path) -> dict[str, Any]:
    summary = _load_json(f2_root / "summary.json")
    evidence = summary.get("evidence") if isinstance(summary.get("evidence"), dict) else {}
    canonical = evidence.get("canonical") if isinstance(evidence.get("canonical"), dict) else {}
    authority = evidence.get("authority") if isinstance(evidence.get("authority"), dict) else {}
    truthful = evidence.get("truthful_chain") if isinstance(evidence.get("truthful_chain"), dict) else {}
    checks = {
        "phase": summary.get("phase") == "8.9F2",
        "pass": summary.get("pass") is True,
        "production_code_unchanged": summary.get("production_code_changed_by_8_9f2") is False,
        "canonical_authoritative": authority.get("canonical_geometry") == "authoritative",
        "canonical_arbitrary_scale": canonical.get("metric_geometry_status") == "arbitrary",
        "canonical_unit_absent": canonical.get("unit_to_metre") is None,
        "not_certified_metrology": authority.get("certified_metrology") is False,
        "sfm_not_injected_into_da3": truthful.get("literal_sfm_pose_to_da3_feed_supported") is False,
        "independent_pose_paths": truthful.get("sfm_and_da3_pose_paths_are_independent") is True,
    }
    required = (
        f2_root / "da3/input_manifest.json",
        f2_root / "da3/worker-output/result_manifest.json",
        f2_root / "canonical/da3_canonical_surface.obj",
    )
    checks["required_files"] = all(path.is_file() and path.stat().st_size > 0 for path in required)
    if not all(checks.values()):
        raise RuntimeError(f"F2 evidence is not release-qualified for F3: {checks}")
    return {"summary": summary, "checks": checks}


def _gsplat_checkout_state(root: Path) -> dict[str, Any]:
    if not root.is_dir() or not (root / ".git").exists():
        return {"path": str(root), "exists": False, "ready": False}
    revision = _git(["rev-parse", "HEAD"], cwd=root)
    status = _git(["status", "--short", "--untracked-files=all"], cwd=root).splitlines()
    return {
        "path": str(root.resolve()),
        "exists": True,
        "revision": revision,
        "expected_revision": EXPECTED_GSPLAT_REVISION,
        "revision_match": revision == EXPECTED_GSPLAT_REVISION,
        "git_status": status,
        "clean": not status,
        "ready": revision == EXPECTED_GSPLAT_REVISION and not status,
    }


def _build_real_bridge(f2_root: Path, output: Path) -> dict[str, Any]:
    """Replay F2 DA3 evidence through existing contracts and prepare D3 inputs."""
    f2_summary = _load_json(f2_root / "summary.json")
    expected_canonical = f2_summary["evidence"]["canonical"]
    input_manifest = read_da3_input_manifest(f2_root / "da3/input_manifest.json")
    result_manifest = read_da3_result_manifest(f2_root / "da3/worker-output/result_manifest.json")
    if input_manifest.request_id != result_manifest.request_id:
        raise RuntimeError("F2 DA3 input/result request IDs differ")
    arrays = validate_da3_npz_result(result_manifest)

    session_id = _new_validation_session_id()
    frame_ids = tuple(f"phase8-9f3-da3-{i}" for i in range(result_manifest.frame_count))
    timestamps = tuple(2_000_000_000 + i * 50_000_000 for i in range(result_manifest.frame_count))
    reconstruction_frame_id = "phase8-9f3-da3-reconstruction"
    camera = da3_camera_profile_from_result(result_manifest, reconstruction_frame_id=reconstruction_frame_id)

    rgbs: list[np.ndarray] = []
    for path_text in input_manifest.image_paths:
        image = cv2.imread(path_text, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to read real F2 DA3 image: {path_text}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (result_manifest.width, result_manifest.height),
            interpolation=cv2.INTER_AREA,
        )
        rgbs.append(image)

    scale = arbitrary_scale_reference(profile_id="phase8-9f3-da3-arbitrary")
    ingestion = PipelineIngestionEvidence(
        session_id=session_id,
        frame_ids=frame_ids,
        timestamps_ns=timestamps,
        rgb_frames=tuple(rgbs),
        camera_profile=camera,
        scale_profile=scale,
        reconstruction_frame_id=reconstruction_frame_id,
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        source_label="phase8-9f2-real-da3-evidence",
        metadata={
            "phase8_9f2_summary": str((f2_root / "summary.json").resolve()),
            "da3_source_revision": result_manifest.source_revision,
            "da3_model_id": result_manifest.model_id,
            "da3_model_revision": result_manifest.model_revision,
        },
    )
    trajectory, depth = da3_supporting_evidence_from_result(
        result_manifest,
        session_id=session_id,
        frame_ids=frame_ids,
    )

    store = PipelinePayloadStore()
    ingestion_ref = store.put(ingestion, payload_ref="memory://phase8-9f3/ingestion")
    pose_ref = store.put(trajectory, payload_ref="memory://phase8-9f3/poses")
    depth_ref = store.put(depth, payload_ref="memory://phase8-9f3/depth")
    source_provider = bound_evidence_provider_descriptor(
        provider_id="phase8-9f3-real-input-evidence",
        roles=(ReconstructionProviderRole.INGESTION,),
        metadata={"source": "8.9F2 real DA3 evidence"},
    )
    da3_provider = bound_evidence_provider_descriptor(
        provider_id="phase8-9f3-da3-qualified-evidence",
        provider_version=result_manifest.source_revision[:12],
        roles=(ReconstructionProviderRole.CAMERA_POSE, ReconstructionProviderRole.DEPTH),
        metadata={
            "upstream_provider_id": "depth-anything-3-external",
            "replayed_qualified_evidence": True,
            "canonical_geometry_authoritative": False,
        },
    )
    geometry = canonical_geometry_provider_descriptors()
    providers = (source_provider, da3_provider, geometry[0], geometry[1])
    source_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=source_provider.provider_id,
        payload_ref_by_stage={ReconstructionPipelineStage.SOURCE_INGESTION: ingestion_ref},
    )
    da3_executor = BoundEvidenceProviderExecutor(
        store=store,
        provider_id=da3_provider.provider_id,
        payload_ref_by_stage={
            ReconstructionPipelineStage.CAMERA_POSE: pose_ref,
            ReconstructionPipelineStage.DEPTH_EVIDENCE: depth_ref,
        },
    )
    median_depth = float(np.median(arrays["depth"]))
    voxel = max(median_depth * 0.02, 1e-4)
    dense_executor = CanonicalDenseGeometryProviderExecutor(
        store=store,
        policy=DenseFusionPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            voxel_size_reconstruction_units=voxel,
            pixel_stride=4,
            minimum_view_observations=1,
            minimum_confidence=0.05,
            max_voxels=200_000,
        ),
        quality_policy=DenseGeometryQualityPolicy(
            scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
            target_view_observations=3,
            minimum_view_observations=1,
            minimum_confidence=0.05,
            desired_view_angle_span_degrees=12.0,
            maximum_depth_relative_residual=0.10,
            high_score_threshold=0.72,
            medium_score_threshold=0.40,
            minimum_input_depth_coverage=0.35,
        ),
    )
    surface_executor = SurfaceMeshProviderExecutor(store=store)
    executors = {
        source_provider.provider_id: source_executor,
        da3_provider.provider_id: da3_executor,
        geometry[0].provider_id: dense_executor,
        geometry[1].provider_id: surface_executor,
    }
    request = ReconstructionPipelineRequest(
        request_id="phase8-9f3-real-gaussian-bridge",
        session_id=session_id,
        path=ReconstructionPath.OFFLINE,
        quality_profile=ReconstructionQualityProfile.EFFICIENT,
        goals=(ReconstructionPipelineGoal.CANONICAL_GEOMETRY,),
    )
    plan = build_reconstruction_pipeline_plan(request, providers)
    pipeline = ReconstructionPipelineOrchestrator(
        plan=plan, providers=providers, executors=executors
    ).run()
    if pipeline.status is ReconstructionPipelineRunStatus.FAILED:
        reasons = [item.reason for item in pipeline.stages if item.reason]
        raise RuntimeError("F3 validation bridge canonical replay failed: " + " | ".join(reasons))
    artifacts = {
        item.artifact.kind: item.artifact
        for item in pipeline.stages
        if item.artifact is not None
    }
    surface = store.get(
        artifacts[ReconstructionArtifactKind.SURFACE_MESH].payload_ref,
        SurfaceMeshResult,
    )
    expected_vertices = int(expected_canonical.get("surface_vertex_count", 0))
    expected_faces = int(expected_canonical.get("surface_face_count", 0))
    if surface.mesh.vertex_count != expected_vertices or surface.mesh.face_count != expected_faces:
        raise RuntimeError(
            "F3 replayed canonical surface differs from qualified F2 evidence: "
            f"{surface.mesh.vertex_count}/{surface.mesh.face_count} != "
            f"{expected_vertices}/{expected_faces}"
        )
    if surface.metric_geometry_status.value != "arbitrary" or surface.unit_to_metre is not None:
        raise RuntimeError("F3 bridge must preserve arbitrary reconstruction scale")

    bundle = materialize_appearance_dataset(
        ingestion,
        trajectory,
        output_dir=(output / "real-neural-scene").resolve(),
    )
    seed = build_gaussian_seed(surface, max_gaussians=2048, minimum_vertex_quality=0.0)
    seed_path = write_gaussian_seed_npz(seed, (output / "gaussian-seed.npz").resolve())
    validation_id = bundle.dataset.frames[len(bundle.dataset.frames) // 2].frame_id
    bridge = {
        "phase": PHASE,
        "source_phase": "8.9F2",
        "validation_only_bridge": True,
        "production_architecture_change_required": False,
        "frame_count": bundle.dataset.frame_count,
        "training_frame_ids": [
            frame.frame_id for frame in bundle.dataset.frames if frame.frame_id != validation_id
        ],
        "validation_frame_id": validation_id,
        "dataset_manifest": bundle.manifest_path,
        "dataset_fingerprint_sha256": bundle.dataset_fingerprint_sha256,
        "gaussian_seed": str(seed_path.resolve()),
        "gaussian_seed_count": seed.gaussian_count,
        "canonical_surface_vertex_count": surface.mesh.vertex_count,
        "canonical_surface_face_count": surface.mesh.face_count,
        "metric_geometry_status": surface.metric_geometry_status.value,
        "unit_to_metre": surface.unit_to_metre,
        "canonical_geometry_authoritative": True,
        "gaussian_scene_role": "appearance_only",
        "certified_metrology": False,
        "sfm_pose_to_da3_injection": False,
        "sfm_fused_into_gaussian_in_f3": False,
    }
    (output / "bridge_summary.json").write_text(
        json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bridge


def _benchmark_command(args: argparse.Namespace, output: Path, bridge: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_gaussian_real_scene_benchmark.py"),
        "--dataset-manifest", str(bridge["dataset_manifest"]),
        "--seed", str(bridge["gaussian_seed"]),
        "--validation-frame-id", str(bridge["validation_frame_id"]),
        "--benchmark-id", "phase8-9f3-real-gaussian",
        "--source-label", "phase8-9f2-real-da3-canonical",
        "--backend", "gsplat",
        "--profile", "efficient",
        "--training-steps", str(args.training_steps),
        "--measure-performance",
        "--render-repeats", str(args.render_repeats),
        "--hardware-profile", "rtx_2060_6gb",
        "--hardware-label", "NVIDIA GeForce RTX 2060 6 GB",
        "--gsplat-python", str(args.gsplat_python),
        "--metrics-python", str(args.metrics_python),
        "--metrics-device", "cpu",
        "--output-dir", str((output / "benchmark").resolve()),
        "--timeout-s", str(args.timeout_s),
    ]


def _make_comparison(render_manifest_path: Path, output: Path) -> str:
    try:
        from PIL import Image, ImageDraw
        manifest = _load_json(render_manifest_path)
        renders = manifest.get("renders") if isinstance(manifest.get("renders"), list) else []
        if len(renders) != 1 or not isinstance(renders[0], dict):
            return ""
        item = renders[0]
        ref_path = Path(str(item["reference_image_path"]))
        render_path = Path(str(item["rendered_image_path"]))
        if not ref_path.is_file() or not render_path.is_file():
            return ""
        with Image.open(ref_path) as image:
            reference = image.convert("RGB").copy()
        with Image.open(render_path) as image:
            rendered = image.convert("RGB").copy()
        height = max(reference.height, rendered.height)
        panels = []
        for label, image in (("Held-out reference", reference), ("gsplat render", rendered)):
            scale = height / image.height
            resized = image.resize((max(1, int(round(image.width * scale))), height))
            panel = Image.new("RGB", (resized.width, height + 34), "white")
            panel.paste(resized, (0, 34))
            ImageDraw.Draw(panel).text((8, 8), label, fill="black")
            panels.append(panel)
        canvas = Image.new("RGB", (sum(panel.width for panel in panels), height + 34), "white")
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width
        path = output / "held_out_comparison.png"
        canvas.save(path)
        return str(path.resolve())
    except Exception:
        return ""


def _qualify(output: Path, f2_root: Path, checkout: dict[str, Any]) -> dict[str, Any]:
    f2 = _validate_f2(f2_root)
    bridge = _load_json(output / "bridge_summary.json")
    step3 = _load_json(output / "benchmark/step3_summary.json")
    report = step3.get("report") if isinstance(step3.get("report"), dict) else {}
    results = report.get("results") if isinstance(report.get("results"), list) else []
    if len(results) != 1 or not isinstance(results[0], dict):
        raise RuntimeError("F3 requires exactly one gsplat benchmark result")
    result = results[0]
    quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
    perf = result.get("performance") if isinstance(result.get("performance"), dict) else {}
    hardware = result.get("hardware") if isinstance(result.get("hardware"), dict) else {}
    lifecycle = result.get("lifecycle") if isinstance(result.get("lifecycle"), dict) else {}
    paths = step3.get("evidence_paths") if isinstance(step3.get("evidence_paths"), dict) else {}
    gsplat_paths = paths.get("gsplat") if isinstance(paths.get("gsplat"), dict) else {}
    scene_path = Path(str(gsplat_paths.get("scene_path", "")))
    render_manifest_path = Path(str(gsplat_paths.get("render_manifest", "")))
    backend_result_path = Path(str(gsplat_paths.get("backend_result", "")))
    quality_path = Path(str(gsplat_paths.get("quality_evaluation", "")))
    render_manifest = _load_json(render_manifest_path)
    renders = render_manifest.get("renders") if isinstance(render_manifest.get("renders"), list) else []
    rendered_files = [
        Path(str(item.get("rendered_image_path", "")))
        for item in renders if isinstance(item, dict)
    ]

    def finite_number(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    checks = {
        "f2_pass": f2["summary"].get("pass") is True,
        "bridge_real_frames": int(bridge.get("frame_count", 0)) == 3,
        "bridge_seed_nonempty": int(bridge.get("gaussian_seed_count", 0)) > 0,
        "bridge_matches_f2_vertices": int(bridge.get("canonical_surface_vertex_count", -1)) == int(f2["summary"]["evidence"]["canonical"]["surface_vertex_count"]),
        "bridge_matches_f2_faces": int(bridge.get("canonical_surface_face_count", -1)) == int(f2["summary"]["evidence"]["canonical"]["surface_face_count"]),
        "bridge_arbitrary_scale": bridge.get("metric_geometry_status") == "arbitrary" and bridge.get("unit_to_metre") is None,
        "benchmark_pass": step3.get("pass") is True,
        "comparison_ready": step3.get("comparison_ready") is True,
        "backend": result.get("backend_kind") == "gsplat_external",
        "lifecycle_completed": lifecycle.get("state") == "completed" and lifecycle.get("worker_exit_code") == 0,
        "quality_measured": quality.get("measured") is True,
        "quality_finite": all(finite_number(quality.get(name)) for name in ("psnr_db", "ssim", "lpips")),
        "performance_measured": perf.get("detailed_measured") is True,
        "training_vram_positive": float(perf.get("peak_training_vram_mib", 0.0) or 0.0) > 0.0,
        "render_vram_positive": float(perf.get("peak_render_vram_mib", 0.0) or 0.0) > 0.0,
        "render_fps_positive": float(perf.get("render_fps", 0.0) or 0.0) > 0.0,
        "hardware_measured": hardware.get("measured") is True and hardware.get("profile_compatible") is True,
        "physical_gpu": "RTX 2060" in str(hardware.get("gpu_name", "")),
        "scene_exists": scene_path.is_file() and scene_path.stat().st_size > 0,
        "render_manifest_exists": render_manifest_path.is_file() and render_manifest_path.stat().st_size > 0,
        "backend_result_exists": backend_result_path.is_file() and backend_result_path.stat().st_size > 0,
        "quality_evidence_exists": quality_path.is_file() and quality_path.stat().st_size > 0,
        "held_out_render_exists": bool(rendered_files) and all(path.is_file() and path.stat().st_size > 0 for path in rendered_files),
        "appearance_only": result.get("canonical_geometry_authoritative") is False,
        "not_certified_metrology": result.get("certified_metrology") is False,
        "gsplat_checkout_ready": checkout.get("ready") is True,
    }
    importability = _heavy_runtime_importability()
    isolation = {
        "heavy_module_importability_in_depth": importability,
        "production_heavy_runtime_leak": any(importability.values()),
    }
    checks["production_isolation"] = not isolation["production_heavy_runtime_leak"]
    comparison = _make_comparison(render_manifest_path, output)
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "f2": {
            "summary": str((f2_root / "summary.json").resolve()),
            "canonical_vertices": f2["summary"]["evidence"]["canonical"]["surface_vertex_count"],
            "canonical_faces": f2["summary"]["evidence"]["canonical"]["surface_face_count"],
            "metric_geometry_status": f2["summary"]["evidence"]["canonical"]["metric_geometry_status"],
            "unit_to_metre": f2["summary"]["evidence"]["canonical"]["unit_to_metre"],
        },
        "bridge": bridge,
        "gaussian": {
            "backend_kind": result.get("backend_kind"),
            "scene_role": "appearance_only",
            "scene_path": str(scene_path),
            "render_manifest": str(render_manifest_path),
            "held_out_comparison": comparison,
            "training_steps": step3.get("training_steps"),
            "training_view_count": result.get("training_view_count"),
            "validation_view_count": result.get("validation_view_count"),
            "quality": quality,
            "performance": perf,
            "hardware": hardware,
            "lifecycle": lifecycle,
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        "gsplat_checkout": checkout,
        "isolation": isolation,
        "authority": {
            "canonical_geometry": "authoritative",
            "gaussian": "appearance_only",
            "nerf": "not_executed_in_f3",
            "certified_metrology": False,
        },
        "truthful_chain": {
            "source": "real_phase8_9f2_da3_canonical_evidence",
            "validation_only_bridge": True,
            "sfm_pose_to_da3_injection": False,
            "sfm_fused_into_gaussian_in_f3": False,
            "gaussian_seed_source": "replayed_F2_canonical_surface_via_existing_contract",
            "statement": (
                "F3 consumed real F2 DA3/canonical evidence through a validation-only bridge. "
                "The existing canonical pipeline was replayed without changing production code, the resulting "
                "surface was required to match F2 counts, and gsplat trained/rendered externally from that "
                "canonical seed plus the same real DA3 RGB/pose evidence. Gaussian remains appearance-only."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f2-root", type=Path, default=DEFAULT_F2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--gsplat-python", type=Path, default=DEFAULT_GSPLAT_PYTHON)
    parser.add_argument("--metrics-python", type=Path, default=DEFAULT_METRICS_PYTHON)
    parser.add_argument("--gsplat-root", type=Path, default=DEFAULT_GSPLAT_ROOT)
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--render-repeats", type=int, default=DEFAULT_RENDER_REPEATS)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    f2_root = args.f2_root.expanduser().resolve()
    args.gsplat_python = args.gsplat_python.expanduser().resolve()
    args.metrics_python = args.metrics_python.expanduser().resolve()
    args.gsplat_root = args.gsplat_root.expanduser().resolve()
    if args.training_steps < 16 or args.training_steps > 512:
        raise SystemExit("--training-steps must lie in [16, 512] for bounded F3 qualification")
    if args.render_repeats < 1 or args.render_repeats > 20:
        raise SystemExit("--render-repeats must lie in [1, 20]")

    repo = _repository_state(str(args.expected_head).strip())
    if not repo["pass"]:
        output.mkdir(parents=True, exist_ok=True)
        payload = {"phase": PHASE, "pass": False, "repository": repo, "blockers": ["repository_baseline_not_exact"]}
        (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase {PHASE} real Gaussian campaign: BLOCKED")
        print("blocker=repository_baseline_not_exact")
        return 2

    _validate_f2(f2_root)
    checkout = _gsplat_checkout_state(args.gsplat_root)
    if not checkout.get("ready"):
        raise SystemExit(f"qualified gsplat checkout is not ready: {checkout}")
    for path, label in ((args.gsplat_python, "gsplat"), (args.metrics_python, "metrics")):
        if not path.is_file():
            raise SystemExit(f"isolated {label} Python is missing: {path}")
        if path.resolve() == Path(sys.executable).resolve():
            raise SystemExit(f"isolated {label} Python must not be the production interpreter")

    if not args.validate_only:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        print(f"[{PHASE}] preparing validation-only F2 → Gaussian bridge", flush=True)
        bridge = _build_real_bridge(f2_root, output)
        command = _benchmark_command(args, output, bridge)
        print(f"[{PHASE}] starting real isolated gsplat benchmark: {' '.join(command)}", flush=True)
        _run_streamed(command, log_path=output / "logs/gsplat_benchmark.log")

    evidence = _qualify(output, f2_root, checkout)
    payload = {
        "phase": PHASE,
        "pass": bool(evidence["pass"]),
        "repository": repo,
        "production_code_changed_by_8_9f3": False,
        "new_gaussian_algorithm_created": False,
        "real_execution": {
            "f2_reused": True,
            "canonical_contract_replayed_for_validation_bridge": not args.validate_only,
            "gsplat_training_and_render": not args.validate_only,
            "graphdeco_started": False,
            "nerf_started": False,
        },
        "evidence": evidence,
        "thresholds": {
            "training_steps": args.training_steps,
            "render_repeats": args.render_repeats,
            "quality_metric_floor_invented": False,
        },
        "next_step_if_pass": "8.9F4_real_nerf_nerfacto_qualification",
        "blockers": [] if evidence["pass"] else ["real_gaussian_evidence_failed_qualification"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gaussian = evidence["gaussian"]
    quality = gaussian["quality"]
    perf = gaussian["performance"]
    print()
    print(f"Phase {PHASE} real Gaussian qualification: {'PASS' if payload['pass'] else 'FAIL'}")
    print(
        f"backend={gaussian['backend_kind']} training_views={gaussian['training_view_count']} "
        f"validation_views={gaussian['validation_view_count']} steps={gaussian['training_steps']}"
    )
    print(
        f"quality_psnr={float(quality['psnr_db']):.4f} quality_ssim={float(quality['ssim']):.6f} "
        f"quality_lpips={float(quality['lpips']):.6f}"
    )
    print(
        f"train_vram_mib={float(perf['peak_training_vram_mib']):.1f} "
        f"render_vram_mib={float(perf['peak_render_vram_mib']):.1f} "
        f"render_fps={float(perf['render_fps']):.2f} gpu={gaussian['hardware'].get('gpu_name')}"
    )
    print("gaussian_authority=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print("sfm_pose_to_da3_injection=False sfm_fused_into_gaussian_in_f3=False")
    print(f"comparison={gaussian['held_out_comparison']}")
    print(f"summary={summary_path}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
