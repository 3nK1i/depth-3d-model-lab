#!/usr/bin/env python3
"""Phase 8.9F2 real SfM + DA3 + canonical-geometry qualification.

This release-validation runner does not add or replace reconstruction algorithms.
It orchestrates already-qualified Phase 8 components against one real scene:

* real external PyCOLMAP sparse SfM,
* real isolated Depth Anything 3 inference, and
* the existing DA3 -> canonical geometry integration contract.

SfM and DA3 remain independent evidence paths.  The runner never claims that
PyCOLMAP poses are injected into DA3, and it never invents metric scale.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE = "8.9F2"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "57041c27a20c17a225b35ac50fed8528e8081b71"
EXPECTED_DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
EXPECTED_DA3_MODEL_ID = "depth-anything/DA3-SMALL"
DEFAULT_DATASET = Path("/home/eiatech/Desktop/phase8_3c_south_building/south-building/images")
DEFAULT_SFM_PYTHON = Path("/home/eiatech/anaconda3/envs/eia-pycolmap-probe/bin/python")
DEFAULT_MAX_KEYFRAMES = 80
DEFAULT_DA3_MAX_IMAGES = 3
DEFAULT_DA3_PROCESS_RES = 392
DEFAULT_MIN_REGISTRATION_RATIO = 0.75
DEFAULT_MIN_LANDMARKS = 1000

F2_REPO_FILES = {
    "docs/phase_history/PHASE8_9F2_VALIDATION.md",
    "scripts/run_phase8_9f2_real_geometry_campaign.py",
    "scripts/run_phase8_9f2_tests.sh",
    "tests/integration/test_phase8_9f2_real_geometry_contract.py",
    "tests/unit/test_phase8_9f2_real_geometry_campaign.py",
}

HEAVY_MODULES = (
    "pycolmap",
    "depth_anything_3",
    "gsplat",
    "nerfstudio",
    "tinycudann",
    "diff_gaussian_rasterization",
    "simple_knn",
)


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
        if path in F2_REPO_FILES:
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
        "f2_overlay_files_present": sorted(overlay),
        "unexpected_worktree_changes": unexpected,
        "pass": passed,
    }


def _dataset_images(dataset: Path) -> list[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if not dataset.is_dir():
        return []
    return sorted(p.resolve() for p in dataset.iterdir() if p.is_file() and p.suffix.lower() in supported)


def _select_da3_overlap_window(images: list[Path], count: int) -> list[Path]:
    """Select a contiguous mid-sequence window for multi-view surface overlap.

    The legacy DA3 qualification runner samples with linspace across its input
    directory.  Giving it a directory containing exactly this contiguous window
    keeps the real DA3 runtime untouched while ensuring the canonical mesher sees
    neighbouring views rather than first/middle/last scene-spanning views.
    """
    if count < 2:
        raise ValueError("DA3 overlap-window count must be at least 2")
    if len(images) < count:
        raise ValueError("dataset contains fewer images than the requested DA3 overlap window")
    start = max(0, (len(images) - count) // 2)
    return list(images[start : start + count])


def _materialize_da3_overlap_window(images: list[Path], output: Path, count: int) -> Path:
    selected = _select_da3_overlap_window(images, count)
    subset = output / "da3-input-overlap-window"
    if subset.exists():
        shutil.rmtree(subset)
    subset.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(selected):
        target = subset / f"{index:03d}_{source.name}"
        target.symlink_to(source)
    return subset


def _stage_command_sfm(args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_reconstruction_pycolmap_smoke.py"),
        "--images", str(args.dataset),
        "--sfm-python", str(args.sfm_python),
        "--output-dir", str(out / "sfm"),
        "--profile", "efficient",
        "--matching", "sequential",
        "--max-keyframes", str(args.max_keyframes),
        "--timeout-minutes", str(args.sfm_timeout_minutes),
    ]


def _stage_command_da3(args: argparse.Namespace, out: Path, *, image_dir: Path | None = None) -> list[str]:
    source_dir = image_dir if image_dir is not None else args.dataset
    return [
        sys.executable,
        str(ROOT / "scripts/run_da3_real_qualification.py"),
        "--image-dir", str(source_dir),
        "--output-dir", str(out / "da3"),
        "--max-images", str(args.da3_max_images),
        "--process-res", str(args.da3_process_res),
        "--model-id", EXPECTED_DA3_MODEL_ID,
    ]


def _stage_command_canonical(args: argparse.Namespace, out: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_da3_canonical_geometry_integration.py"),
        "--da3-input-manifest", str(out / "da3/input_manifest.json"),
        "--da3-result-manifest", str(out / "da3/worker-output/result_manifest.json"),
        "--output-dir", str(out / "canonical"),
        "--pixel-stride", str(args.pixel_stride),
    ]


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid or missing JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _heavy_runtime_importability() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in HEAVY_MODULES}


def _qualify_evidence(
    output: Path,
    dataset: Path,
    *,
    min_registration_ratio: float,
    min_landmarks: int,
) -> dict[str, Any]:
    images = _dataset_images(dataset)
    sfm = _load_json(output / "sfm/real_sfm_summary.json")
    da3 = _load_json(output / "da3/summary.json")
    da3_input = _load_json(output / "da3/input_manifest.json")
    da3_result = _load_json(output / "da3/worker-output/result_manifest.json")
    canonical = _load_json(output / "canonical/summary.json")

    selected = [Path(text).resolve() for text in da3_input.get("image_paths", [])]
    selected_in_dataset = bool(selected) and all(_path_is_within(path, dataset) for path in selected)
    request_ids_match = bool(da3_input.get("request_id")) and da3_input.get("request_id") == da3_result.get("request_id")

    sfm_gauge = sfm.get("gauge", {}) if isinstance(sfm.get("gauge"), dict) else {}
    sfm_checks = {
        "real_sfm_executed": sfm.get("real_sfm_executed") is True,
        "registration_ratio": float(sfm.get("registration_ratio", 0.0)) >= min_registration_ratio,
        "registered_keyframes": int(sfm.get("registered_keyframes", 0)) >= 3,
        "landmarks": int(sfm.get("landmarks", 0)) >= min_landmarks,
        "camera_models": int(sfm.get("camera_models", 0)) > 0,
        "arbitrary_scale": sfm_gauge.get("scale_status") == "arbitrary",
        "no_metric_conversion": sfm_gauge.get("unit_to_metre") is None,
    }

    da3_checks = {
        "pass": da3.get("pass") is True,
        "frame_count": int(da3.get("frame_count", 0)) >= 2,
        "relative_arbitrary_scale": da3.get("geometry_scale") == "relative_arbitrary_scale",
        "not_canonical_authority": da3.get("canonical_geometry_authoritative") is False,
        "not_certified_metrology": da3.get("certified_metrology") is False,
        "expected_model": da3.get("model_id") == EXPECTED_DA3_MODEL_ID,
        "expected_source_revision": da3.get("source_revision") == EXPECTED_DA3_SOURCE_REVISION,
        "input_result_request_ids_match": request_ids_match,
        "selected_images_within_dataset": selected_in_dataset,
        "result_manifest_exists": (output / "da3/worker-output/result_manifest.json").is_file(),
    }

    canonical_checks = {
        "pass": canonical.get("pass") is True,
        "pipeline_completed": canonical.get("pipeline_status") == "completed",
        "canonical_geometry_authoritative": canonical.get("canonical_geometry_authoritative") is True,
        "da3_supporting_evidence_not_authoritative": canonical.get("da3_supporting_evidence_authoritative") is False,
        "arbitrary_metric_status": canonical.get("metric_geometry_status") == "arbitrary",
        "unit_to_metre_absent": canonical.get("unit_to_metre") is None,
        "not_certified_metrology": canonical.get("certified_metrology") is False,
        "surface_vertices": int(canonical.get("surface_vertex_count", 0)) > 0,
        "surface_faces": int(canonical.get("surface_face_count", 0)) > 0,
        "surface_obj_exists": Path(str(canonical.get("surface_obj", ""))).is_file(),
    }

    heavy_importability = _heavy_runtime_importability()
    isolation_checks = {
        "production_heavy_runtime_importable": any(heavy_importability.values()),
        "production_heavy_runtime_leak": any(heavy_importability.values()),
    }

    evidence_pass = all(sfm_checks.values()) and all(da3_checks.values()) and all(canonical_checks.values())
    isolation_pass = not isolation_checks["production_heavy_runtime_leak"]
    return {
        "pass": evidence_pass and isolation_pass,
        "dataset": {
            "path": str(dataset.resolve()),
            "image_count": len(images),
            "same_real_scene_for_sfm_and_da3": selected_in_dataset,
            "da3_selection_strategy": "contiguous_mid_sequence_overlap_window",
            "da3_selected_images": [str(path) for path in selected],
        },
        "sfm": {
            "checks": sfm_checks,
            "input_keyframes": sfm.get("input_keyframes"),
            "registered_keyframes": sfm.get("registered_keyframes"),
            "registration_ratio": sfm.get("registration_ratio"),
            "landmarks": sfm.get("landmarks"),
            "observations": sfm.get("observations"),
            "mean_reprojection_error_px": sfm.get("mean_reprojection_error_px"),
            "gauge": sfm_gauge,
            "pycolmap_version": sfm.get("pycolmap_version"),
        },
        "da3": {
            "checks": da3_checks,
            "model_id": da3.get("model_id"),
            "model_revision": da3.get("model_revision"),
            "source_revision": da3.get("source_revision"),
            "frame_count": da3.get("frame_count"),
            "inference_duration_s": da3.get("inference_duration_s"),
            "peak_vram_mib": da3.get("peak_vram_mib"),
            "gpu_name": da3.get("gpu_name"),
            "pose_count": da3.get("pose_count"),
            "geometry_scale": da3.get("geometry_scale"),
        },
        "canonical": {
            "checks": canonical_checks,
            "artifact_id": canonical.get("canonical_geometry_artifact_id"),
            "surface_obj": canonical.get("surface_obj"),
            "surface_vertex_count": canonical.get("surface_vertex_count"),
            "surface_face_count": canonical.get("surface_face_count"),
            "metric_geometry_status": canonical.get("metric_geometry_status"),
            "unit_to_metre": canonical.get("unit_to_metre"),
        },
        "isolation": {
            "heavy_module_importability_in_depth": heavy_importability,
            **isolation_checks,
            "pass": isolation_pass,
        },
        "authority": {
            "sfm": "supporting_sparse_geometry_and_pose_evidence",
            "da3": "supporting_relative_geometry_until_canonicalized",
            "canonical_geometry": "authoritative",
            "gaussian": "not_executed_in_f2",
            "nerf": "not_executed_in_f2",
            "certified_metrology": False,
        },
        "truthful_chain": {
            "literal_sfm_pose_to_da3_feed_supported": False,
            "sfm_and_da3_pose_paths_are_independent": True,
            "canonical_surface_source": "real_da3_evidence_via_existing_canonical_integration_contract",
            "sfm_fused_into_canonical_surface_in_f2": False,
            "statement": (
                "PyCOLMAP SfM and DA3 were executed as separate real evidence paths against the same scene. "
                "DA3 inferred its own supporting camera evidence. The canonical surface was built from real DA3 "
                "evidence through the existing canonical geometry contract; SfM poses were not injected into DA3 "
                "and were not fused into the F2 canonical surface."
            ),
        },
    }


def _completed_stage_ready(output: Path, stage: str) -> bool:
    required: dict[str, Iterable[Path]] = {
        "sfm": (output / "sfm/real_sfm_summary.json", output / "sfm/real_sparse_arbitrary_scale.ply"),
        "da3": (output / "da3/summary.json", output / "da3/input_manifest.json", output / "da3/worker-output/result_manifest.json"),
        "canonical": (output / "canonical/summary.json", output / "canonical/da3_canonical_surface.obj"),
    }
    return all(path.is_file() and path.stat().st_size > 0 for path in required[stage])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_9f2_real_geometry"))
    parser.add_argument("--sfm-python", type=Path, default=DEFAULT_SFM_PYTHON)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    parser.add_argument("--max-keyframes", type=int, default=DEFAULT_MAX_KEYFRAMES)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--da3-max-images", type=int, default=DEFAULT_DA3_MAX_IMAGES)
    parser.add_argument("--da3-process-res", type=int, default=DEFAULT_DA3_PROCESS_RES)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-registration-ratio", type=float, default=DEFAULT_MIN_REGISTRATION_RATIO)
    parser.add_argument("--min-landmarks", type=int, default=DEFAULT_MIN_LANDMARKS)
    parser.add_argument("--reuse-completed", action="store_true", help="Reuse only non-empty stage evidence already present in --output-dir.")
    parser.add_argument("--validate-only", action="store_true", help="Do not execute backends; validate already-materialized F2 evidence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    args.dataset = dataset
    args.sfm_python = args.sfm_python.expanduser().resolve()

    if args.max_keyframes < 8:
        raise SystemExit("--max-keyframes must be at least 8 for F2 release qualification")
    if args.da3_max_images < 2:
        raise SystemExit("--da3-max-images must be at least 2")
    if args.da3_process_res < 224:
        raise SystemExit("--da3-process-res is below the qualified validation envelope")
    if not (0.0 < args.min_registration_ratio <= 1.0):
        raise SystemExit("--min-registration-ratio must be in (0, 1]")
    if args.min_landmarks <= 0:
        raise SystemExit("--min-landmarks must be positive")

    images = _dataset_images(dataset)
    if len(images) < max(8, args.da3_max_images):
        raise SystemExit(f"dataset is not ready: found {len(images)} supported images at {dataset}")

    repo_state = _repository_state(str(args.expected_head).strip())
    if not repo_state["pass"]:
        output.mkdir(parents=True, exist_ok=True)
        payload = {"phase": PHASE, "pass": False, "repository": repo_state, "blockers": ["repository_baseline_not_exact"]}
        (output / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase {PHASE} real geometry campaign: BLOCKED")
        print("blocker=repository_baseline_not_exact")
        return 2

    if not args.validate_only:
        if output.exists() and not args.reuse_completed:
            shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)
        da3_input_dir = _materialize_da3_overlap_window(images, output, args.da3_max_images)
        stages = [
            ("sfm", _stage_command_sfm(args, output)),
            ("da3", _stage_command_da3(args, output, image_dir=da3_input_dir)),
            ("canonical", _stage_command_canonical(args, output)),
        ]
        for name, command in stages:
            if args.reuse_completed and _completed_stage_ready(output, name):
                print(f"[{PHASE}] reusing completed {name} evidence", flush=True)
                continue
            print(f"[{PHASE}] starting {name}: {' '.join(command)}", flush=True)
            _run_streamed(command, log_path=output / "logs" / f"{name}.log")

    evidence = _qualify_evidence(
        output,
        dataset,
        min_registration_ratio=args.min_registration_ratio,
        min_landmarks=args.min_landmarks,
    )
    payload = {
        "phase": PHASE,
        "pass": bool(evidence["pass"]),
        "repository": repo_state,
        "real_execution": {
            "pycolmap_sfm": not args.validate_only,
            "da3_inference": not args.validate_only,
            "canonical_integration": not args.validate_only,
            "gaussian_started": False,
            "nerf_started": False,
        },
        "production_code_changed_by_8_9f2": False,
        "new_reconstruction_algorithm_created": False,
        "evidence": evidence,
        "thresholds": {
            "minimum_sfm_registration_ratio": args.min_registration_ratio,
            "minimum_sfm_landmarks": args.min_landmarks,
            "sfm_max_keyframes": args.max_keyframes,
            "da3_max_images": args.da3_max_images,
            "da3_process_res": args.da3_process_res,
        },
        "next_step_if_pass": "8.9F3_real_gaussian_qualification",
        "blockers": [] if evidence["pass"] else ["real_geometry_evidence_failed_qualification"],
    }
    summary = output / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sfm = evidence["sfm"]
    da3 = evidence["da3"]
    canonical = evidence["canonical"]
    print()
    print(f"Phase {PHASE} real SfM + DA3 + canonical geometry campaign: {'PASS' if payload['pass'] else 'FAIL'}")
    print(
        f"sfm_registered={sfm['registered_keyframes']}/{sfm['input_keyframes']} "
        f"ratio={float(sfm['registration_ratio'] or 0.0):.1%} landmarks={sfm['landmarks']}"
    )
    print(
        f"da3_frames={da3['frame_count']} model={da3['model_id']} gpu={da3['gpu_name']} "
        f"peak_vram_mib={da3['peak_vram_mib']}"
    )
    print(
        f"canonical_vertices={canonical['surface_vertex_count']} faces={canonical['surface_face_count']} "
        f"metric_status={canonical['metric_geometry_status']} unit_to_metre={canonical['unit_to_metre']}"
    )
    print("sfm_pose_to_da3_injection=False canonical_geometry_authoritative=True certified_metrology=False")
    print(f"summary={summary}")
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
