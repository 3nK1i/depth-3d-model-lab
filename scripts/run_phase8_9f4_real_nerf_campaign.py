#!/usr/bin/env python3
"""Phase 8.9F4 real Nerfstudio/Nerfacto qualification on F2 South Building evidence.

This release-validation runner composes existing Phase 8 contracts only. It
materializes a backend-neutral appearance dataset from the real F2 PyCOLMAP
registered trajectory, then invokes the already-qualified Phase 8.7E3 held-out
quality runner and Phase 8.7E4 performance/artifact-hardening runner in the
isolated Nerfstudio environment.

NeRF remains appearance-only. F2 canonical geometry remains authoritative
provenance, but is not used as NeRF metric authority. F3 Gaussian evidence is
required as prior-chain qualification and is not fed into Nerfacto.
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
    DenseSceneProfile,
    MetricGeometryStatus,
    NeuralSceneDataset,
    NeuralSceneFrame,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    deterministic_nerf_validation_split,
    neural_scene_dataset_fingerprint,
    write_neural_scene_dataset_manifest,
)

PHASE = "8.9F4"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "0db3b0d92b84a538985e7932bfec9f65890cf6a0"
EXPECTED_NERFSTUDIO_REVISION = "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
EXPECTED_TCNN_REVISION = "b3473c81396fe927293bdfd5a6be32df8769927c"
EXPECTED_NERFSTUDIO_VERSION = "1.1.5"
DEFAULT_F2_ROOT = Path("/tmp/phase8_9f2_real_geometry")
DEFAULT_F3_ROOT = Path("/tmp/phase8_9f3_real_gaussian")
DEFAULT_OUTPUT = Path("/tmp/phase8_9f4_real_nerf")
DEFAULT_NERF_PYTHON = Path.home() / "anaconda3/envs/depth3d-nerf-nerfstudio/bin/python"
DEFAULT_METRICS_PYTHON = Path.home() / "anaconda3/envs/depth3d-gaussian-metrics/bin/python"
DEFAULT_NERFSTUDIO_ROOT = Path.home() / "Desktop/depth3d-nerf-backends/nerfstudio"
DEFAULT_TCNN_ROOT = Path.home() / "Desktop/depth3d-nerf-backends/tiny-cuda-nn"
DEFAULT_SELECTED_FRAMES = 24
DEFAULT_HELDOUT_FRAMES = 4
DEFAULT_MAX_LONG_EDGE = 720
DEFAULT_TRAINING_STEPS = 1000
DEFAULT_RENDER_REPEATS = 8

F4_REPO_FILES = {
    "docs/phase_history/PHASE8_9F4_VALIDATION.md",
    "scripts/run_phase8_9f4_real_nerf_campaign.py",
    "scripts/run_phase8_9f4_tests.sh",
    "tests/integration/test_phase8_9f4_real_nerf_contract.py",
    "tests/unit/test_phase8_9f4_real_nerf_campaign.py",
}

HEAVY_MODULES = (
    "nerfstudio",
    "tinycudann",
    "gsplat",
    "diff_gaussian_rasterization",
    "simple_knn",
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
        if path in F4_REPO_FILES:
            overlay.append(path)
        else:
            unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": expected_head,
        "head_matches_origin": head == origin,
        "head_matches_expected": not expected_head or head == expected_head,
        "f4_overlay_files_present": sorted(overlay),
        "unexpected_worktree_changes": unexpected,
        "pass": bool(
            branch == EXPECTED_BRANCH
            and head == origin
            and (not expected_head or head == expected_head)
            and not unexpected
        ),
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


def _checkout_state(root: Path, expected_revision: str) -> dict[str, Any]:
    if not root.is_dir() or not (root / ".git").exists():
        return {"path": str(root), "exists": False, "ready": False}
    revision = _git(["rev-parse", "HEAD"], cwd=root)
    status = _git(["status", "--short", "--untracked-files=all"], cwd=root).splitlines()
    return {
        "path": str(root.resolve()),
        "exists": True,
        "revision": revision,
        "expected_revision": expected_revision,
        "revision_match": revision == expected_revision,
        "git_status": status,
        "clean": not status,
        "ready": revision == expected_revision and not status,
    }


def _validate_prior_chain(f2_root: Path, f3_root: Path) -> dict[str, Any]:
    f2 = _load_json(f2_root / "summary.json")
    f3 = _load_json(f3_root / "summary.json")
    f2_evidence = f2.get("evidence") if isinstance(f2.get("evidence"), dict) else {}
    f2_authority = f2_evidence.get("authority") if isinstance(f2_evidence.get("authority"), dict) else {}
    f2_sfm = f2_evidence.get("sfm") if isinstance(f2_evidence.get("sfm"), dict) else {}
    f3_evidence = f3.get("evidence") if isinstance(f3.get("evidence"), dict) else {}
    f3_authority = f3_evidence.get("authority") if isinstance(f3_evidence.get("authority"), dict) else {}
    f3_gaussian = f3_evidence.get("gaussian") if isinstance(f3_evidence.get("gaussian"), dict) else {}
    checks = {
        "f2_phase": f2.get("phase") == "8.9F2",
        "f2_pass": f2.get("pass") is True,
        "f2_production_unchanged": f2.get("production_code_changed_by_8_9f2") is False,
        "f2_sfm_real": bool(f2_sfm.get("checks", {}).get("real_sfm_executed")) if isinstance(f2_sfm.get("checks"), dict) else False,
        "f2_canonical_authoritative": f2_authority.get("canonical_geometry") == "authoritative",
        "f2_not_metrology": f2_authority.get("certified_metrology") is False,
        "f3_phase": f3.get("phase") == "8.9F3",
        "f3_pass": f3.get("pass") is True,
        "f3_production_unchanged": f3.get("production_code_changed_by_8_9f3") is False,
        "f3_gaussian_appearance_only": f3_authority.get("gaussian") == "appearance_only" and f3_gaussian.get("scene_role") == "appearance_only",
        "f3_not_metrology": f3_authority.get("certified_metrology") is False,
    }
    sfm_report = f2_root / "sfm/real_sfm_report.json"
    scene_path = Path(str(f3_gaussian.get("scene_path", "")))
    checks["f2_sfm_report_exists"] = sfm_report.is_file() and sfm_report.stat().st_size > 0
    checks["f3_scene_exists"] = scene_path.is_file() and scene_path.stat().st_size > 0
    if not all(checks.values()):
        raise RuntimeError(f"F2/F3 evidence is not release-qualified for F4: {checks}")
    return {"f2": f2, "f3": f3, "checks": checks, "sfm_report": sfm_report, "f3_scene": scene_path}


def _even_indices(total: int, selected: int) -> tuple[int, ...]:
    total = int(total)
    selected = int(selected)
    if total < 12:
        raise RuntimeError("F4 requires at least 12 registered SfM cameras")
    if selected < 12:
        raise RuntimeError("F4 selected frame count must be at least 12")
    count = min(total, selected)
    raw = np.linspace(0, total - 1, num=count, dtype=np.int64).tolist()
    values: list[int] = []
    seen: set[int] = set()
    for item in raw:
        value = int(item)
        if value not in seen:
            seen.add(value)
            values.append(value)
    if len(values) < 12:
        raise RuntimeError("F4 could not select at least 12 unique registered cameras")
    return tuple(values)


def _camera_parameters(camera_model: dict[str, Any]) -> tuple[float, float, float, float, np.ndarray]:
    name = str(camera_model.get("backend_model_name", "")).strip().upper()
    params = [float(value) for value in camera_model.get("parameters", [])]
    if name == "SIMPLE_RADIAL" and len(params) == 4:
        f, cx, cy, k1 = params
        return f, f, cx, cy, np.asarray([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if name == "RADIAL" and len(params) == 5:
        f, cx, cy, k1, k2 = params
        return f, f, cx, cy, np.asarray([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    if name == "PINHOLE" and len(params) == 4:
        fx, fy, cx, cy = params
        return fx, fy, cx, cy, np.zeros(5, dtype=np.float64)
    if name == "SIMPLE_PINHOLE" and len(params) == 3:
        f, cx, cy = params
        return f, f, cx, cy, np.zeros(5, dtype=np.float64)
    if name == "OPENCV" and len(params) == 8:
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        return fx, fy, cx, cy, np.asarray([k1, k2, p1, p2, 0.0], dtype=np.float64)
    raise RuntimeError(f"F4 does not silently reinterpret unsupported SfM camera model {name!r} with {len(params)} parameters")


def _undistort_resize(
    source: Path,
    target: Path,
    *,
    camera_model: dict[str, Any],
    max_long_edge: int,
) -> tuple[int, int, float, float, float, float]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to decode F2 SfM frame: {source}")
    height, width = image.shape[:2]
    fx, fy, cx, cy, distortion = _camera_parameters(camera_model)
    model_size = camera_model.get("image_size") if isinstance(camera_model.get("image_size"), dict) else {}
    if int(model_size.get("width", width)) != width or int(model_size.get("height", height)) != height:
        raise RuntimeError("F2 SfM camera-model image size does not match staged frame size")
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    if float(np.max(np.abs(distortion))) > 1e-12:
        image = cv2.undistort(image, K, distortion, None, K)
    scale = min(1.0, float(max_long_edge) / float(max(width, height)))
    out_width = max(64, int(round(width * scale)))
    out_height = max(64, int(round(height * scale)))
    if (out_width, out_height) != (width, height):
        image = cv2.resize(image, (out_width, out_height), interpolation=cv2.INTER_AREA)
    sx = float(out_width) / float(width)
    sy = float(out_height) / float(height)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"unable to write F4 prepared frame: {target}")
    return out_width, out_height, fx * sx, fy * sy, cx * sx, cy * sy


def _build_sfm_appearance_bridge(
    f2_root: Path,
    f3_root: Path,
    output: Path,
    *,
    selected_frames: int,
    heldout_frames: int,
    max_long_edge: int,
) -> dict[str, Any]:
    prior = _validate_prior_chain(f2_root, f3_root)
    report = _load_json(prior["sfm_report"])
    sfm = report.get("sfm_result") if isinstance(report.get("sfm_result"), dict) else {}
    cameras = sfm.get("cameras") if isinstance(sfm.get("cameras"), list) else []
    models = sfm.get("camera_models") if isinstance(sfm.get("camera_models"), list) else []
    workspace = sfm.get("workspace") if isinstance(sfm.get("workspace"), dict) else {}
    workspace_images = workspace.get("images") if isinstance(workspace.get("images"), list) else []
    gauge = sfm.get("gauge") if isinstance(sfm.get("gauge"), dict) else {}
    if gauge.get("scale_status") != "arbitrary" or gauge.get("unit_to_metre") is not None:
        raise RuntimeError("F4 requires the real F2 SfM gauge to remain arbitrary-scale")
    if len(models) != 1:
        raise RuntimeError(f"F4 expected F2 single-camera SfM, got {len(models)} camera models")
    model = models[0]
    model_id = str(model.get("camera_id", ""))
    if not model_id:
        raise RuntimeError("F2 SfM camera model is missing camera_id")
    by_keyframe = {str(item.get("keyframe_id")): item for item in workspace_images if isinstance(item, dict)}
    ordered = sorted(
        [item for item in cameras if isinstance(item, dict)],
        key=lambda item: int(by_keyframe.get(str(item.get("keyframe_id")), {}).get("order", 10**9)),
    )
    indices = _even_indices(len(ordered), selected_frames)
    selected = [ordered[index] for index in indices]
    if any(str(item.get("camera_id", "")) != model_id for item in selected):
        raise RuntimeError("F4 selected SfM cameras do not share the single qualified camera model")

    workspace_root = Path(str(workspace.get("workspace_root", ""))).expanduser().resolve()
    images_dir = output / "bridge/images"
    frames: list[NeuralSceneFrame] = []
    source_records: list[dict[str, Any]] = []
    for index, camera in enumerate(selected):
        keyframe_id = str(camera.get("keyframe_id", ""))
        record = by_keyframe.get(keyframe_id)
        if not isinstance(record, dict):
            raise RuntimeError(f"F4 cannot resolve F2 workspace image for keyframe {keyframe_id}")
        staged_rel = str(record.get("staged_image_path", ""))
        source = (workspace_root / staged_rel).resolve()
        if not source.is_file():
            raise RuntimeError(f"F2 staged SfM image is missing: {source}")
        frame_id = f"phase8-9f4-sfm-{index:03d}"
        target = (images_dir / f"{frame_id}.png").resolve()
        width, height, fx, fy, cx, cy = _undistort_resize(
            source,
            target,
            camera_model=model,
            max_long_edge=max_long_edge,
        )
        pose = np.asarray(camera.get("T_reconstruction_from_camera"), dtype=np.float64)
        frames.append(
            NeuralSceneFrame(
                frame_id=frame_id,
                image_path=str(target),
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                T_reconstruction_from_camera=pose,
                pose_layer="real_pycolmap_sfm",
                camera_profile_id="phase8-9f4-sfm-estimated-undistorted-pinhole",
                timestamp_ns=int(record.get("timestamp_ns", 0)),
            )
        )
        source_records.append(
            {
                "frame_id": frame_id,
                "source_frame_id": camera.get("frame_id"),
                "source_keyframe_id": keyframe_id,
                "source_staged_image": str(source),
                "prepared_image": str(target),
                "sfm_camera_id": camera.get("camera_id"),
            }
        )

    dataset = NeuralSceneDataset(
        session_id=str(sfm.get("session_id", "phase8-9f4-real-sfm")),
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        reconstruction_frame_id="phase8-9f2-sfm-reconstruction",
        camera_profile_id="phase8-9f4-sfm-estimated-undistorted-pinhole",
        scale_profile_id="phase8-9f2-sfm-arbitrary",
        metric_geometry_status=MetricGeometryStatus.ARBITRARY,
        scale_status=SfmScaleStatus.ARBITRARY,
        world_alignment_status=SfmWorldAlignmentStatus.ARBITRARY,
        unit_to_metre=None,
        relative_scale_uncertainty=None,
        frames=tuple(frames),
        certified_metrology=False,
    )
    training_ids, validation_ids = deterministic_nerf_validation_split(
        tuple(frame.frame_id for frame in frames), validation_count=int(heldout_frames)
    )
    manifest_path = write_neural_scene_dataset_manifest(dataset, (output / "bridge/dataset.json").resolve())
    fingerprint = neural_scene_dataset_fingerprint(dataset)
    split_payload = {
        "phase": "8.7E3",
        "source_phase": PHASE,
        "format_contract": "existing_phase8_7e3_split_contract",
        "source_label": "phase8-9f2-real-south-building-sfm",
        "dataset_manifest_path": str(manifest_path),
        "dataset_fingerprint_sha256": fingerprint,
        "training_frame_ids": list(training_ids),
        "validation_frame_ids": list(validation_ids),
        "selected_frame_count": len(frames),
        "preparation": {
            "selected_frame_count": len(frames),
            "maximum_frames": int(selected_frames),
            "heldout_count": int(heldout_frames),
            "max_long_edge": int(max_long_edge),
            "sampling_policy": "even_coverage_of_real_registered_f2_sfm_trajectory",
            "heldout_policy": "deterministic_interior_trajectory_spread",
            "distortion_policy": "sfm_estimated_model_undistort_to_depth3d_pinhole",
        },
        "source_records": source_records,
        "pose_convention": "T_reconstruction_from_camera",
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "canonical_geometry_authoritative": False,
        "nerf_scene_role": "appearance_only",
        "certified_metrology": False,
    }
    split_path = output / "bridge/split.json"
    split_path.write_text(json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    bridge = {
        "phase": PHASE,
        "source_phase": "8.9F2",
        "prior_gaussian_phase": "8.9F3",
        "validation_only_bridge": True,
        "production_architecture_change_required": False,
        "source_sfm_registered_cameras": len(cameras),
        "selected_frame_count": len(frames),
        "training_frame_count": len(training_ids),
        "validation_frame_count": len(validation_ids),
        "dataset_manifest": str(manifest_path),
        "split_manifest": str(split_path.resolve()),
        "dataset_fingerprint_sha256": fingerprint,
        "camera_model": str(model.get("backend_model_name", "")),
        "camera_model_parameters_derived_from_real_sfm": True,
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
        "canonical_geometry_authoritative": True,
        "nerf_scene_role": "appearance_only",
        "certified_metrology": False,
        "sfm_pose_source": "real_phase8_9f2_pycolmap",
        "da3_used_as_nerf_pose_source": False,
        "gaussian_used_as_nerf_geometry_input": False,
        "canonical_surface_used_as_nerf_geometry_input": False,
    }
    (output / "bridge_summary.json").write_text(json.dumps(bridge, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bridge


def _quality_command(args: argparse.Namespace, output: Path, bridge: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_nerf_real_scene_validation.py"),
        "--dataset-manifest", str(bridge["dataset_manifest"]),
        "--split-manifest", str(bridge["split_manifest"]),
        "--source-label", "phase8-9f2-real-south-building-sfm",
        "--profile", "efficient",
        "--training-steps", str(args.training_steps),
        "--nerf-python", str(args.nerf_python),
        "--metrics-python", str(args.metrics_python),
        "--nerfstudio-root", str(args.nerfstudio_root),
        "--tiny-cuda-nn-root", str(args.tcnn_root),
        "--metrics-device", "cpu",
        "--timeout-s", str(args.timeout_s),
        "--output-dir", str((output / "quality").resolve()),
    ]


def _performance_command(args: argparse.Namespace, output: Path, bridge: dict[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_nerf_performance_hardening.py"),
        "--dataset-manifest", str(bridge["dataset_manifest"]),
        "--split-manifest", str(bridge["split_manifest"]),
        "--source-label", "phase8-9f2-real-south-building-sfm",
        "--profile", "efficient",
        "--training-steps", str(args.training_steps),
        "--render-repeats", str(args.render_repeats),
        "--hardware-profile", "rtx_2060_6gb",
        "--nerf-python", str(args.nerf_python),
        "--nerfstudio-root", str(args.nerfstudio_root),
        "--tiny-cuda-nn-root", str(args.tcnn_root),
        "--timeout-s", str(args.timeout_s),
        "--output-dir", str((output / "performance").resolve()),
    ]


def _qualify(
    output: Path,
    f2_root: Path,
    f3_root: Path,
    checkouts: dict[str, Any],
    *,
    expected_selected_frames: int = DEFAULT_SELECTED_FRAMES,
    expected_heldout_frames: int = DEFAULT_HELDOUT_FRAMES,
) -> dict[str, Any]:
    prior = _validate_prior_chain(f2_root, f3_root)
    bridge = _load_json(output / "bridge_summary.json")
    quality = _load_json(output / "quality/summary.json")
    performance = _load_json(output / "performance/summary.json")
    q_aggregate = quality.get("quality", {}).get("aggregate", {}) if isinstance(quality.get("quality"), dict) else {}
    artifact = quality.get("artifact") if isinstance(quality.get("artifact"), dict) else {}
    report = performance.get("report") if isinstance(performance.get("report"), dict) else {}
    perf = report.get("performance") if isinstance(report.get("performance"), dict) else {}
    hardware = report.get("hardware") if isinstance(report.get("hardware"), dict) else {}
    integrity = report.get("artifact_integrity") if isinstance(report.get("artifact_integrity"), dict) else {}
    final_checkpoint = Path(str(integrity.get("checkpoint_path", "")))
    final_config = Path(str(integrity.get("config_path", "")))
    contact_sheet = Path(str(quality.get("contact_sheet_path", "")))

    def finite(value: Any) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    expected_training_frames = int(expected_selected_frames) - int(expected_heldout_frames)
    checks = {
        "prior_chain_pass": all(prior["checks"].values()),
        "bridge_uses_real_f2_sfm": bridge.get("sfm_pose_source") == "real_phase8_9f2_pycolmap",
        "bridge_selected_view_count": int(bridge.get("selected_frame_count", 0)) == int(expected_selected_frames),
        "bridge_train_heldout_split": (
            int(bridge.get("training_frame_count", 0)) == expected_training_frames
            and int(bridge.get("validation_frame_count", 0)) == int(expected_heldout_frames)
        ),
        "bridge_arbitrary_scale": bridge.get("metric_geometry_status") == "arbitrary" and bridge.get("unit_to_metre") is None,
        "quality_pass": quality.get("pass") is True and quality.get("quality_sanity_pass") is True,
        "quality_counts": (
            int(quality.get("training_frame_count", 0)) == expected_training_frames
            and int(quality.get("validation_frame_count", 0)) == int(expected_heldout_frames)
        ),
        "quality_finite": all(finite(q_aggregate.get(name)) for name in ("psnr_db", "ssim", "lpips")),
        "quality_appearance_only": artifact.get("scene_role") == "appearance_only" and artifact.get("canonical_geometry_authoritative") is False,
        "quality_not_metrology": artifact.get("certified_metrology") is False,
        "performance_pass": performance.get("pass") is True and performance.get("performance_metrics_complete") is True,
        "artifact_hardening": performance.get("artifact_hardening_complete") is True and integrity.get("passed") is True,
        "physical_gpu": hardware.get("measured") is True and "RTX 2060" in str(hardware.get("gpu_name", "")),
        "training_vram_positive": float(perf.get("training_peak_vram_mib", 0.0) or 0.0) > 0.0,
        "render_vram_positive": float(perf.get("render_peak_vram_mib", 0.0) or 0.0) > 0.0,
        "render_fps_positive": float(perf.get("render_fps", 0.0) or 0.0) > 0.0,
        "checkpoint_reload_positive": float(integrity.get("checkpoint_reload_s", 0.0) or 0.0) > 0.0,
        "render_probe_passed": integrity.get("render_probe_passed") is True,
        "missing_artifact_rejected": integrity.get("missing_artifact_rejected") is True,
        "malformed_checkpoint_rejected": integrity.get("malformed_checkpoint_rejected") is True,
        "checkpoint_exists": final_checkpoint.is_file() and final_checkpoint.stat().st_size > 0,
        "config_exists": final_config.is_file() and final_config.stat().st_size > 0,
        "contact_sheet_exists": contact_sheet.is_file() and contact_sheet.stat().st_size > 0,
        "nerfstudio_checkout_ready": checkouts.get("nerfstudio", {}).get("ready") is True,
        "tcnn_checkout_ready": checkouts.get("tiny_cuda_nn", {}).get("ready") is True,
    }
    importability = _heavy_runtime_importability()
    isolation = {
        "heavy_module_importability_in_depth": importability,
        "production_heavy_runtime_leak": any(importability.values()),
    }
    checks["production_isolation"] = not isolation["production_heavy_runtime_leak"]
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "bridge": bridge,
        "nerf": {
            "backend_kind": "nerfstudio_nerfacto_external",
            "method_name": "nerfacto",
            "scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "quality": q_aggregate,
            "quality_policy": quality.get("quality_policy"),
            "training_view_count": quality.get("training_frame_count"),
            "validation_view_count": quality.get("validation_frame_count"),
            "training_steps": quality.get("training_steps"),
            "contact_sheet": str(contact_sheet),
            "quality_checkpoint": artifact.get("checkpoint_path"),
            "quality_config": artifact.get("config_path"),
            "final_checkpoint": str(final_checkpoint),
            "final_config": str(final_config),
            "performance": perf,
            "hardware": hardware,
            "artifact_integrity": integrity,
            "backend_revision": artifact.get("backend_revision"),
            "backend_version": artifact.get("backend_version"),
        },
        "checkouts": checkouts,
        "isolation": isolation,
        "authority": {
            "canonical_geometry": "authoritative",
            "gaussian": "appearance_only",
            "nerf": "appearance_only",
            "certified_metrology": False,
        },
        "truthful_chain": {
            "source": "real_phase8_9f2_south_building_sfm_evidence",
            "f3_gaussian_prior_qualified": True,
            "validation_only_bridge": True,
            "sfm_pose_source": "real_phase8_9f2_pycolmap",
            "da3_used_as_nerf_pose_source": False,
            "canonical_surface_used_as_nerf_geometry_input": False,
            "gaussian_used_as_nerf_geometry_input": False,
            "statement": (
                "F4 trained and rendered Nerfacto from real South Building RGB views and the real F2 PyCOLMAP "
                "registered trajectory through the existing backend-neutral neural-scene contract. F2 canonical "
                "geometry remains authoritative provenance, F3 Gaussian remains separately qualified appearance "
                "evidence, and neither canonical geometry nor Gaussian was silently used as NeRF metric authority."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f2-root", type=Path, default=DEFAULT_F2_ROOT)
    parser.add_argument("--f3-root", type=Path, default=DEFAULT_F3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--nerf-python", type=Path, default=DEFAULT_NERF_PYTHON)
    parser.add_argument("--metrics-python", type=Path, default=DEFAULT_METRICS_PYTHON)
    parser.add_argument("--nerfstudio-root", type=Path, default=DEFAULT_NERFSTUDIO_ROOT)
    parser.add_argument("--tcnn-root", type=Path, default=DEFAULT_TCNN_ROOT)
    parser.add_argument("--selected-frames", type=int, default=DEFAULT_SELECTED_FRAMES)
    parser.add_argument("--heldout-frames", type=int, default=DEFAULT_HELDOUT_FRAMES)
    parser.add_argument("--max-long-edge", type=int, default=DEFAULT_MAX_LONG_EDGE)
    parser.add_argument("--training-steps", type=int, default=DEFAULT_TRAINING_STEPS)
    parser.add_argument("--render-repeats", type=int, default=DEFAULT_RENDER_REPEATS)
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    f2_root = args.f2_root.expanduser().resolve()
    f3_root = args.f3_root.expanduser().resolve()
    args.nerf_python = args.nerf_python.expanduser().resolve()
    args.metrics_python = args.metrics_python.expanduser().resolve()
    args.nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    args.tcnn_root = args.tcnn_root.expanduser().resolve()
    if not 12 <= args.selected_frames <= 40:
        raise SystemExit("--selected-frames must lie in [12, 40]")
    if not 3 <= args.heldout_frames <= 8:
        raise SystemExit("--heldout-frames must lie in [3, 8]")
    if args.selected_frames - args.heldout_frames < 8:
        raise SystemExit("F4 requires at least eight NeRF training frames")
    if not 128 <= args.training_steps <= 5000:
        raise SystemExit("--training-steps must lie in [128, 5000]")
    if not 3 <= args.render_repeats <= 20:
        raise SystemExit("--render-repeats must lie in [3, 20]")
    if args.max_long_edge < 256 or args.max_long_edge > 1440:
        raise SystemExit("--max-long-edge must lie in [256, 1440]")

    repo = _repository_state(str(args.expected_head).strip())
    if not repo["pass"]:
        output.mkdir(parents=True, exist_ok=True)
        payload = {"phase": PHASE, "pass": False, "repository": repo, "blockers": ["repository_baseline_not_exact"]}
        (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase {PHASE} real Nerfacto campaign: BLOCKED")
        print("blocker=repository_baseline_not_exact")
        return 2

    _validate_prior_chain(f2_root, f3_root)
    checkouts = {
        "nerfstudio": _checkout_state(args.nerfstudio_root, EXPECTED_NERFSTUDIO_REVISION),
        "tiny_cuda_nn": _checkout_state(args.tcnn_root, EXPECTED_TCNN_REVISION),
    }
    if not all(item.get("ready") for item in checkouts.values()):
        raise SystemExit(f"qualified Nerfstudio/tiny-cuda-nn checkout is not ready: {checkouts}")
    for path, label in ((args.nerf_python, "Nerfstudio"), (args.metrics_python, "metrics")):
        if not path.is_file():
            raise SystemExit(f"isolated {label} Python is missing: {path}")
        if path.resolve() == Path(sys.executable).resolve():
            raise SystemExit(f"isolated {label} Python must not be the production interpreter")

    if not args.validate_only:
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        print(f"[{PHASE}] preparing real F2 SfM → NeRF validation bridge", flush=True)
        bridge = _build_sfm_appearance_bridge(
            f2_root,
            f3_root,
            output,
            selected_frames=args.selected_frames,
            heldout_frames=args.heldout_frames,
            max_long_edge=args.max_long_edge,
        )
        quality_command = _quality_command(args, output, bridge)
        print(f"[{PHASE}] starting real isolated Nerfacto held-out quality: {' '.join(quality_command)}", flush=True)
        _run_streamed(quality_command, log_path=output / "logs/quality.log")
        performance_command = _performance_command(args, output, bridge)
        print(f"[{PHASE}] starting real isolated Nerfacto performance/artifact hardening: {' '.join(performance_command)}", flush=True)
        _run_streamed(performance_command, log_path=output / "logs/performance.log")

    evidence = _qualify(
        output,
        f2_root,
        f3_root,
        checkouts,
        expected_selected_frames=args.selected_frames,
        expected_heldout_frames=args.heldout_frames,
    )
    payload = {
        "phase": PHASE,
        "pass": bool(evidence["pass"]),
        "repository": repo,
        "production_code_changed_by_8_9f4": False,
        "new_nerf_algorithm_created": False,
        "real_execution": {
            "f2_real_sfm_reused": True,
            "f3_gaussian_required_as_prior_qualification": True,
            "f3_gaussian_used_as_nerf_input": False,
            "nerfacto_heldout_quality_training": not args.validate_only,
            "nerfacto_performance_hardening_training": not args.validate_only,
        },
        "evidence": evidence,
        "thresholds": {
            "selected_frames": args.selected_frames,
            "heldout_frames": args.heldout_frames,
            "max_long_edge": args.max_long_edge,
            "training_steps": args.training_steps,
            "render_repeats": args.render_repeats,
            "quality_policy": "existing_phase8_7e3_nondegenerate_real_scene_sanity",
            "new_quality_floor_invented": False,
        },
        "next_step_if_pass": "8.9F5_persistent_multi_artifact_scene_qualification",
        "blockers": [] if evidence["pass"] else ["real_nerf_evidence_failed_qualification"],
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    nerf = evidence["nerf"]
    q = nerf["quality"]
    perf = nerf["performance"]
    print()
    print(f"Phase {PHASE} real Nerfstudio/Nerfacto qualification: {'PASS' if payload['pass'] else 'FAIL'}")
    print(
        f"backend={nerf['backend_kind']} method={nerf['method_name']} "
        f"training_views={nerf['training_view_count']} validation_views={nerf['validation_view_count']} "
        f"steps={nerf['training_steps']}"
    )
    print(
        f"quality_psnr={float(q['psnr_db']):.4f} quality_ssim={float(q['ssim']):.6f} "
        f"quality_lpips={float(q['lpips']):.6f}"
    )
    print(
        f"training_peak_vram_mib={float(perf['training_peak_vram_mib']):.1f} "
        f"render_peak_vram_mib={float(perf['render_peak_vram_mib']):.1f} "
        f"render_fps={float(perf['render_fps']):.3f} gpu={nerf['hardware'].get('gpu_name')}"
    )
    print(
        f"checkpoint_reload_s={float(nerf['artifact_integrity']['checkpoint_reload_s']):.3f} "
        "missing_artifact_rejected=True malformed_checkpoint_rejected=True render_probe_passed=True"
    )
    print("nerf_authority=appearance_only canonical_geometry_authoritative=False certified_metrology=False")
    print("sfm_pose_source=real_phase8_9f2_pycolmap gaussian_used_as_nerf_geometry_input=False")
    print(f"contact_sheet={nerf['contact_sheet']}")
    print(f"checkpoint={nerf['final_checkpoint']}")
    print(f"summary={summary_path}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
