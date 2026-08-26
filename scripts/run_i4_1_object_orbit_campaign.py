#!/usr/bin/env python3
"""I4.1 object/connector orbit Dense RGB qualification campaign.

Runs or reuses the production I1 offline Dense RGB pipeline on a new static
object capture, performs the ordinary I2 baseline quality assessment, then
adds scale-invariant object-orbit coverage diagnostics.  The object must stay
static while the camera moves around it; a rotating object in a static camera
view violates the current global-SfM scene model.

I4.1 is qualification only.  It does not retune filtering, does not claim
metric scale, does not authorize measurement, and does not generate a mesh.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_orbit_qualification import (  # noqa: E402
    I4_1_SCHEMA_VERSION,
    build_i4_1_orbit_assessment,
    camera_poses_from_sfm_report,
    select_camera_poses,
)

_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DEFAULT_I3_CLOSURE = Path("/tmp/eia_i3_2_second_scene_promotion/i3_2_closure.json")
_DEFAULT_OUTPUT = Path("/tmp/eia_i4_object_orbit_campaign")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--video", type=Path)
    source.add_argument("--images", type=Path)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--reuse-i1-dir", type=Path, default=None, help="Reuse a completed I1 object-orbit baseline.")
    p.add_argument("--i3-closure", type=Path, default=_DEFAULT_I3_CLOSURE)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    p.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    p.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    p.add_argument(
        "--matching",
        choices=("sequential", "exhaustive"),
        default="exhaustive",
        help="I4 defaults to exhaustive because an object orbit revisits/opposes viewpoints.",
    )
    p.add_argument("--max-keyframes", type=int, default=40)
    p.add_argument("--max-dense-views", type=int, default=40)
    p.add_argument("--video-target-fps", type=float, default=2.0)
    p.add_argument("--image-sequence-fps", type=float, default=10.0)
    p.add_argument("--process-res", type=int, default=392)
    p.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    p.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    args = p.parse_args()
    if args.reuse_i1_dir is None and args.video is None and args.images is None:
        p.error("provide --video/--images or --reuse-i1-dir")
    if args.reuse_i1_dir is not None and (args.video is not None or args.images is not None):
        p.error("--reuse-i1-dir cannot be combined with --video/--images")
    if args.max_keyframes < 10 or args.max_dense_views < 10:
        p.error("I4.1 requires at least 10 keyframes/dense views")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I4.1 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I4.1 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I4.1 required NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _run(command: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("\nI4.1 command: " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
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
            handle.write(line)
            handle.flush()
        return int(process.wait())


def _source_args(args: argparse.Namespace) -> list[str]:
    if args.video is not None:
        return ["--video", str(args.video.expanduser().resolve()), "--video-target-fps", str(args.video_target_fps)]
    result = ["--images", str(args.images.expanduser().resolve()), "--image-sequence-fps", str(args.image_sequence_fps)]
    if args.recursive:
        result.append("--recursive")
    return result


def _validate_i3_closure(path: Path) -> dict[str, Any]:
    """Validate the persisted I3 closure using the producer's nested schema.

    I3.2 writes phase authorization under ``decision`` and authority boundaries
    under ``authority``.  This consumer intentionally validates that exact
    cross-phase contract and fails closed on flattened, partial, or premature
    authorization evidence.
    """

    closure = _read_json(path.expanduser().resolve())
    if str(closure.get("phase")) != "I3" or str(closure.get("status")) != "pass" or not bool(closure.get("pass")):
        raise ReconstructionValidationError("I4.1 requires a passing I3 closure record.")

    checks = closure.get("checks")
    if not isinstance(checks, Mapping):
        raise ReconstructionValidationError("I4.1 requires I3 closure check evidence.")
    for name in ("automated_requalification", "fixed_view_review", "reconstruction_ui_sanity"):
        if str(checks.get(name)) != "pass":
            raise ReconstructionValidationError(f"I4.1 requires I3 closure check {name}=pass.")

    authority = closure.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("I4.1 requires I3 closure authority evidence.")
    if not bool(authority.get("canonical_geometry_authoritative")) or not bool(authority.get("canonical_persistence_promoted")):
        raise ReconstructionValidationError("I4.1 requires promoted canonical I3 geometry and persistence authority.")
    if bool(authority.get("metric_scale_claimed")) or bool(authority.get("measurement_authoritative")) or bool(authority.get("surface_mesh_generated")):
        raise ReconstructionValidationError("I4.1 refuses premature metric/measurement/mesh authority in I3 closure evidence.")

    decision = closure.get("decision")
    if not isinstance(decision, Mapping):
        raise ReconstructionValidationError("I4.1 requires nested I3 closure decision evidence.")
    if not bool(decision.get("i3_closure_authorized")):
        raise ReconstructionValidationError("I4.1 requires decision.i3_closure_authorized=True.")
    if not bool(decision.get("i4_connector_orbit_authorized")):
        raise ReconstructionValidationError("I4.1 requires decision.i4_connector_orbit_authorized=True.")
    if bool(decision.get("mesh_authorization")):
        raise ReconstructionValidationError("I4.1 refuses an unexpected pre-authorized mesh state.")
    return closure


def _write_degraded(
    output: Path,
    *,
    failed_stage: str,
    reason: str,
    i3_closure: Path,
    source_i1_dir: Path | None = None,
    details: Mapping[str, Any] | None = None,
) -> int:
    summary = {
        "phase": "I4.1",
        "schema_version": I4_1_SCHEMA_VERSION,
        "status": "degraded",
        "failed_stage": failed_stage,
        "reason": reason,
        "i3_closure": str(i3_closure),
        "source_i1_dir": None if source_i1_dir is None else str(source_i1_dir),
        "details": dict(details or {}),
        "authority": {
            "filter_policy_changed": False,
            "persistence_changed": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_manual_orbit_review": False,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
        },
    }
    path = _write_json(output / "i4_1_summary.json", summary)
    print("\nI4.1 object/connector orbit campaign: DEGRADED")
    print(f"failed_stage={failed_stage} reason={reason}")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"summary={path}")
    return 2


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    closure_path = args.i3_closure.expanduser().resolve()
    closure = _validate_i3_closure(closure_path)
    print("I4.1 I3 closure preflight: PASS")
    print("i3_closure_authorized=True i4_connector_orbit_authorized=True mesh_authorization=False")

    if args.reuse_i1_dir is not None:
        i1_dir = args.reuse_i1_dir.expanduser().resolve()
        if not (i1_dir / "i1_campaign_summary.json").is_file():
            raise ReconstructionValidationError("--reuse-i1-dir does not contain i1_campaign_summary.json")
        print(f"I4.1 reusing I1 object baseline: {i1_dir}")
    else:
        i1_dir = output / "01_i1_baseline"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_i1_offline_dense_campaign.py"),
            "--output-dir", str(i1_dir),
            "--sfm-python", str(args.sfm_python.expanduser().resolve()),
            "--da3-python", str(args.da3_python.expanduser().resolve()),
            "--da3-repo", str(args.da3_repo.expanduser().resolve()),
            "--profile", str(args.profile),
            "--matching", str(args.matching),
            "--max-keyframes", str(args.max_keyframes),
            "--max-dense-views", str(args.max_dense_views),
            "--process-res", str(args.process_res),
            "--sfm-timeout-minutes", str(args.sfm_timeout_minutes),
            "--da3-timeout-seconds", str(args.da3_timeout_seconds),
            *_source_args(args),
        ]
        code = _run(command, log=output / "logs/01_i1_baseline.log")
        if code != 0:
            evidence: dict[str, Any] = {"returncode": code}
            summary_path = i1_dir / "i1_campaign_summary.json"
            if summary_path.is_file():
                i1_failed = _read_json(summary_path)
                evidence.update({
                    "i1_status": i1_failed.get("status"),
                    "i1_failed_stage": i1_failed.get("failed_stage"),
                    "i1_reason": i1_failed.get("reason"),
                    "minimum_qualified_views": i1_failed.get("minimum_qualified_views"),
                    "selected_path": i1_failed.get("selected_path"),
                })
            return _write_degraded(
                output,
                failed_stage="object_orbit_i1_baseline",
                reason="I1 baseline did not qualify; capture/SfM/dense evidence must be fixed rather than relaxing I4 gates.",
                i3_closure=closure_path,
                source_i1_dir=i1_dir,
                details=evidence,
            )

    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    if str(i1.get("status")) != "pass" or not bool(i1.get("pipeline_execution_complete")):
        return _write_degraded(
            output,
            failed_stage="object_orbit_i1_baseline",
            reason="Reused I1 baseline is not a completed passing campaign.",
            i3_closure=closure_path,
            source_i1_dir=i1_dir,
        )

    baseline_quality_dir = output / "02_baseline_quality"
    code = _run(
        [
            sys.executable,
            str(ROOT / "scripts/run_i2_full_video_quality_qualification.py"),
            "--i1-dir", str(i1_dir),
            "--output-dir", str(baseline_quality_dir),
        ],
        log=output / "logs/02_baseline_quality.log",
    )
    if code != 0:
        i2_path = baseline_quality_dir / "i2_summary.json"
        details = {"returncode": code}
        if i2_path.is_file():
            i2_failed = _read_json(i2_path)
            details.update({"automated_pass": i2_failed.get("automated_pass"), "automated_gates": i2_failed.get("automated_gates")})
        return _write_degraded(
            output,
            failed_stage="object_orbit_baseline_quality",
            reason="Ordinary I2 baseline quality gates failed; orbit-specific gates are not evaluated on an unqualified baseline.",
            i3_closure=closure_path,
            source_i1_dir=i1_dir,
            details=details,
        )

    i2 = _read_json(baseline_quality_dir / "i2_summary.json")
    sfm_summary = _read_json(i1_dir / "01_sfm" / "real_sfm_summary.json")
    sfm_report = _read_json(i1_dir / "01_sfm" / "real_sfm_report.json")
    camera_ids, camera_matrices = camera_poses_from_sfm_report(sfm_report)

    f_summary = _read_json(i1_dir / "05_filtering" / "f_summary.json")
    extended_path = Path(str(f_summary["artifacts"]["extended_npz"])).expanduser().resolve()
    extended = _load_npz(extended_path)
    points = np.asarray(extended["points_reconstruction"], dtype=np.float64)
    colors = np.asarray(extended["colors_rgb"], dtype=np.uint8)
    classes = np.asarray(extended["class_code"], dtype=np.uint8).reshape(-1)
    if points.shape != colors.shape or classes.shape != (len(points),):
        raise ReconstructionValidationError("I4.1 Extended/Core geometry arrays are misaligned.")
    core_points = points[classes == 3]
    selected_keyframe_ids = tuple(str(value) for value in f_summary.get("selected_keyframe_ids", []))

    assessment = build_i4_1_orbit_assessment(
        sfm_summary=sfm_summary,
        i1_summary=i1,
        i2_summary=i2,
        registered_camera_matrices=camera_matrices,
        registered_camera_ids=camera_ids,
        selected_keyframe_ids=selected_keyframe_ids,
        core_points=core_points,
    )
    passed = bool(assessment["pass"])
    selected_ids, selected_matrices = select_camera_poses(camera_ids, camera_matrices, selected_keyframe_ids)
    diagnostics = output / "i4_1_orbit_diagnostics.npz"
    np.savez_compressed(
        diagnostics,
        target_reconstruction=np.asarray(assessment["target_reconstruction"], dtype=np.float64),
        registered_camera_matrices=np.asarray(camera_matrices, dtype=np.float64),
        registered_camera_ids=np.asarray(camera_ids, dtype="U160"),
        selected_camera_matrices=np.asarray(selected_matrices, dtype=np.float64),
        selected_camera_ids=np.asarray(selected_ids, dtype="U160"),
        schema_version=np.asarray(I4_1_SCHEMA_VERSION, dtype=np.int32),
    )

    scene = i2.get("scene", {})
    retained = i2.get("retained_geometry", {})
    spatial = i2.get("spatial", {})
    summary = {
        "phase": "I4.1",
        "schema_version": I4_1_SCHEMA_VERSION,
        "status": "ready_for_manual_orbit_review" if passed else "degraded",
        "automated_pass": passed,
        "i3_closure": str(closure_path),
        "i3_scene_id": str(closure.get("scene_id", "")),
        "source_i1_dir": str(i1_dir),
        "baseline_quality_dir": str(baseline_quality_dir),
        "source": dict(i1.get("source", {})),
        "scene": {
            "scene_id": str(scene.get("scene_id", i1.get("persistent_scene", {}).get("scene_id", ""))),
            "fingerprint": str(scene.get("fingerprint", i1.get("persistent_scene", {}).get("fingerprint", ""))),
            "registered_views": int(scene.get("registered_views", assessment["counts"]["registered_views"])),
            "dense_alignment_accepted_views": int(scene.get("dense_alignment_accepted_views", assessment["counts"]["dense_alignment_accepted_views"])),
            "selected_qualified_views": int(scene.get("selected_qualified_views", assessment["counts"]["selected_qualified_views"])),
            "selected_workspace_orders": list(scene.get("selected_workspace_orders", [])),
            "sfm_scale_status": "arbitrary",
        },
        "baseline_quality": {
            "automated_pass": bool(i2.get("automated_pass")),
            "extended_points": int(retained.get("extended_points", len(points))),
            "core_points": int(retained.get("core_points", len(core_points))),
            "largest_supported_component_fraction": retained.get("largest_supported_component_fraction"),
            "extended_full_to_robust": spatial.get("extended_envelope", {}).get("full_to_robust_diagonal_ratio"),
        },
        "orbit_assessment": assessment,
        "capture_model": {
            "object_static_required": True,
            "camera_moves_around_object": True,
            "turntable_object_rotation_supported": False,
            "matching_strategy": str(sfm_summary.get("matching_strategy", args.matching)),
        },
        "artifacts": {
            "orbit_diagnostics_npz": str(diagnostics.resolve()),
            "extended_npz": str(extended_path),
            "i2_spatial_diagnostics_npz": str(i2.get("artifacts", {}).get("spatial_diagnostics_npz", "")),
        },
        "authority": {
            "filter_policy_changed": False,
            "persistence_changed": False,
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_manual_orbit_review": passed,
            "authorizes_i4_2_detail_preservation": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Review Dense RGB + selected camera orbit fixed views. If capture coverage is visually object-centric, proceed to I4.2 thin-detail/manifold-filter preservation diagnosis."
                if passed
                else "Improve the object-orbit capture or investigate failed orbit gates; do not tune detail filtering yet."
            ),
        },
    }
    summary_path = _write_json(output / "i4_1_summary.json", summary)

    selected = assessment["selected_orbit"]
    counts = assessment["counts"]
    print("\nI4.1 object/connector orbit campaign: " + ("READY_FOR_MANUAL_ORBIT_REVIEW" if passed else "DEGRADED"))
    print(
        f"views registered={counts['registered_views']}/{counts['input_views']} "
        f"aligned={counts['dense_alignment_accepted_views']}/{counts['registered_views']} "
        f"selected={counts['selected_qualified_views']}"
    )
    print(
        f"orbit selected_coverage={selected['azimuth_coverage_degrees']:.1f}deg "
        f"bins={selected['occupied_azimuth_bins']}/{selected['azimuth_bin_count']} "
        f"opposed={selected['maximum_opposed_view_angle_degrees']:.1f}deg"
    )
    print(
        f"target_facing={selected['target_facing_fraction']:.3f} "
        f"standoff_p90/p10={selected['standoff_p90_to_p10_ratio']:.3f} "
        f"plane_residual_p90={selected['orbit_plane_residual_fraction']['p90']:.3f}"
    )
    print(f"orbit_gates_pass={passed} ordinary_i2_quality_pass={bool(i2.get('automated_pass'))}")
    print("filter_policy_changed=False persistence_changed=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"manual_orbit_review_required={passed} i4_2_detail_preservation_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
