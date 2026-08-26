#!/usr/bin/env python3
"""I3.1 second indoor/static scene end-to-end transfer qualification.

I3.1 begins the second real-scene campaign without changing the locked I2
production geometry.  It first proves that a generic scene-relative evidence
builder exactly reproduces the closed I2 d12_sparse manifold mask.  It then
runs/reuses I1 on a *different* indoor/static sequence, performs ordinary I2
baseline quality qualification, and evaluates the same d12 predicate as a
counterfactual on the second scene.

The I3.1 candidate is deliberately NOT written back into formal F/G.  Manual
review and second-scene transfer qualification must pass before any production
promotion.  SfM remains arbitrary scale; metric, measurement and mesh
authority are never promoted here.
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
from core.reconstruction.offline_dense_i3_transfer import (  # noqa: E402
    I3_TRANSFER_SCHEMA_VERSION,
    I3TransferSafetyPolicy,
    build_scene_relative_manifold_evidence,
    exact_reference_transfer_preflight,
    transfer_candidate_assessment,
)

_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DEFAULT_REFERENCE_I1 = Path("/tmp/eia_i1_offline_dense_campaign_recovered")
_DEFAULT_REFERENCE_I252 = Path("/tmp/eia_i2_5_2_extremum_local_geometry_manifold_diagnosis")
_DEFAULT_REFERENCE_I26 = Path("/tmp/eia_i2_6_manifold_filter_promotion")
_DEFAULT_OUTPUT = Path("/tmp/eia_i3_second_scene_campaign")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=False)
    source.add_argument("--video", type=Path)
    source.add_argument("--images", type=Path)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--reuse-i1-dir", type=Path, default=None, help="Reuse a completed I1 directory for the new scene.")
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--reference-i2-i1-dir", type=Path, default=_DEFAULT_REFERENCE_I1)
    p.add_argument("--reference-i2-5-2-dir", type=Path, default=_DEFAULT_REFERENCE_I252)
    p.add_argument("--reference-i2-6-dir", type=Path, default=_DEFAULT_REFERENCE_I26)
    p.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    p.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    p.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    p.add_argument("--profile", choices=("efficient", "balanced", "high_quality"), default="efficient")
    p.add_argument("--matching", choices=("sequential", "exhaustive"), default="sequential")
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
        p.error("I3.1 requires at least 10 keyframes/dense views")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I3.1 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I3.1 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I3.1 required NPZ is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _filtered_extended_from_f_dir(f_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    summary = _read_json(f_dir / "f_summary.json")
    artifact = Path(str(summary["artifacts"]["extended_npz"])).expanduser().resolve()
    return _load_npz(artifact), summary


def _voxel_keys(points: np.ndarray, voxel_size: float) -> list[tuple[int, int, int]]:
    keys = np.floor(np.asarray(points, dtype=np.float64) / float(voxel_size)).astype(np.int64)
    result = [tuple(map(int, row)) for row in keys]
    if len(set(result)) != len(result):
        raise ReconstructionValidationError("I3.1 expects one retained point per voxel key.")
    return result


def _i2_reference_preflight(reference_i1: Path, reference_i252: Path, reference_i26: Path) -> dict[str, Any]:
    source, source_f = _filtered_extended_from_f_dir(reference_i1 / "05_filtering")
    i252 = _read_json(reference_i252 / "i2_5_2_summary.json")
    i26 = _read_json(reference_i26 / "i2_6_summary.json")
    promoted_f_dir = reference_i26 / "01_promoted_filtering"
    promoted, promoted_f = _filtered_extended_from_f_dir(promoted_f_dir)
    if str(i252.get("status")) != "diagnosis_ready" or str(i26.get("status")) != "pass":
        raise ReconstructionValidationError("I3.1 reference preflight requires completed I2.5.2 and passing I2.6 evidence.")
    promotion = i26.get("promotion")
    if not isinstance(promotion, Mapping):
        raise ReconstructionValidationError("I3.1 reference I2.6 promotion evidence is missing.")
    eq = promotion.get("exact_counterfactual_equivalence")
    if not isinstance(eq, Mapping) or not bool(eq.get("exact_mask_match")):
        raise ReconstructionValidationError("I3.1 reference I2.6 mask equivalence is not exact.")

    evidence = build_scene_relative_manifold_evidence(source)
    source_points = np.asarray(source["points_reconstruction"], dtype=np.float64)
    promoted_points = np.asarray(promoted["points_reconstruction"], dtype=np.float64)
    voxel = float(np.asarray(source["voxel_size_reconstruction_units"]).item())
    promoted_voxel = float(np.asarray(promoted["voxel_size_reconstruction_units"]).item())
    if not np.isclose(voxel, promoted_voxel, rtol=0.0, atol=1e-12):
        raise ReconstructionValidationError("I3.1 reference I2 source/promoted voxel sizes disagree.")
    promoted_set = set(_voxel_keys(promoted_points, voxel))
    source_keys = _voxel_keys(source_points, voxel)
    expected_reject = np.asarray([key not in promoted_set for key in source_keys], dtype=bool)

    thresholds = i252.get("reference", {}).get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ReconstructionValidationError("I3.1 I2.5.2 reference thresholds are missing.")
    preflight = exact_reference_transfer_preflight(
        evidence,
        expected_reject_mask=expected_reject,
        expected_anchor_reference_q90=float(thresholds["anchor_distance_high"]),
        expected_knn_reference_q90=float(thresholds["knn_spacing_high"]),
        atol=1e-8,
    )
    preflight.update({
        "reference_scene_id": str(i26.get("scene_id", i26.get("persistent_scene", {}).get("scene_id", ""))),
        "source_extended_points": int(len(source_points)),
        "promoted_extended_points": int(len(promoted_points)),
        "formal_removed_points": int(np.count_nonzero(expected_reject)),
        "reference_i2_5_2": str(reference_i252),
        "reference_i2_6": str(reference_i26),
        "source_f_pass": bool(source_f.get("pass")),
        "promoted_f_pass": bool(promoted_f.get("pass")),
    })
    if not bool(preflight["pass"]):
        raise ReconstructionValidationError(
            "I3.1 generic scene-relative manifold evidence does not exactly reproduce the closed I2 production rule."
        )
    return preflight


def _run(command: list[str], *, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("\nI3.1 command: " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line); handle.flush()
        return int(process.wait())


def _run_or_reuse_i1(args: argparse.Namespace, output: Path) -> Path:
    if args.reuse_i1_dir is not None:
        i1 = args.reuse_i1_dir.expanduser().resolve()
        summary = _read_json(i1 / "i1_campaign_summary.json")
        if str(summary.get("status")) != "pass" or not bool(summary.get("pipeline_execution_complete")):
            raise ReconstructionValidationError("I3.1 --reuse-i1-dir must point to a completed passing I1 campaign.")
        return i1
    i1 = output / "01_i1_baseline"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_i1_offline_dense_campaign.py"),
        "--output-dir", str(i1),
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
    ]
    if args.video is not None:
        command += ["--video", str(args.video.expanduser().resolve()), "--video-target-fps", str(args.video_target_fps)]
    else:
        command += ["--images", str(args.images.expanduser().resolve()), "--image-sequence-fps", str(args.image_sequence_fps)]
        if args.recursive:
            command.append("--recursive")
    code = _run(command, log=output / "logs/01_i1_baseline.log")
    if code != 0:
        raise ReconstructionValidationError(f"I3.1 second-scene I1 baseline failed with exit code {code}.")
    return i1


def _run_baseline_quality(i1: Path, output: Path) -> tuple[Path, dict[str, Any], int]:
    quality = output / "02_baseline_quality"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_i2_full_video_quality_qualification.py"),
        "--i1-dir", str(i1),
        "--output-dir", str(quality),
    ]
    code = _run(command, log=output / "logs/02_baseline_quality.log")
    summary = _read_json(quality / "i2_summary.json")
    return quality, summary, code


def _slice_aligned_payload(source: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    count = len(np.asarray(source["points_reconstruction"]))
    selected = np.asarray(mask, dtype=bool).reshape(-1)
    if selected.shape != (count,):
        raise ReconstructionValidationError("I3.1 candidate slice mask is misaligned.")
    out: dict[str, np.ndarray] = {}
    for name, values in source.items():
        array = np.asarray(values)
        if array.ndim >= 1 and array.shape[0] == count and name not in (
            "selected_workspace_orders", "selected_keyframe_ids", "sparse_landmarks_reconstruction", "sparse_landmark_colors_rgb"
        ):
            out[name] = array[selected]
        else:
            out[name] = array.copy()
    return out


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I3.1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    preflight = _i2_reference_preflight(
        args.reference_i2_i1_dir.expanduser().resolve(),
        args.reference_i2_5_2_dir.expanduser().resolve(),
        args.reference_i2_6_dir.expanduser().resolve(),
    )
    _write_json(output / "00_i2_transfer_preflight.json", preflight)
    print("I3.1 I2 transfer preflight: PASS")
    print(
        f"exact_mask={preflight['exact_mask_match']} thresholds_exact="
        f"{preflight['anchor_reference_match'] and preflight['knn_reference_match']} "
        f"formal_removed={preflight['formal_removed_points']}"
    )

    i1_dir = _run_or_reuse_i1(args, output)
    quality_dir, quality, quality_code = _run_baseline_quality(i1_dir, output)
    if quality_code != 0 or not bool(quality.get("automated_pass")):
        final = {
            "phase": "I3.1", "schema_version": I3_TRANSFER_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "second_scene_baseline_quality", "i2_transfer_preflight": preflight,
            "source_i1_dir": str(i1_dir), "baseline_quality_dir": str(quality_dir),
            "baseline_quality": {"status": quality.get("status"), "automated_pass": bool(quality.get("automated_pass"))},
            "authority": {"geometry_changed": False, "production_f_policy_changed": False, "persistence_changed": False,
                          "metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
            "decision": {"authorizes_manual_transfer_review": False, "authorizes_i3_promotion": False,
                         "authorizes_mesh_reconstruction": False, "next_action": "Investigate second-scene baseline quality before manifold transfer."},
        }
        path = _write_json(output / "i3_1_summary.json", final)
        print("\nI3.1 second-scene transfer campaign: DEGRADED")
        print("failed_stage=second_scene_baseline_quality")
        print(f"summary={path}")
        return 2

    source, f_summary = _filtered_extended_from_f_dir(i1_dir / "05_filtering")
    evidence = build_scene_relative_manifold_evidence(source)
    assessment = transfer_candidate_assessment(evidence, policy=I3TransferSafetyPolicy())
    reject = np.asarray(evidence["reject"], dtype=bool)
    core = np.asarray(evidence["core"], dtype=bool)
    keep = ~reject

    transfer_dir = output / "03_manifold_transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = transfer_dir / "i3_1_transfer_diagnostics.npz"
    np.savez_compressed(
        diagnostics,
        points_reconstruction=np.asarray(evidence["points"], dtype=np.float64),
        colors_rgb=np.asarray(evidence["colors"], dtype=np.uint8),
        is_core=core.astype(np.uint8),
        is_rejected_by_d12_transfer=reject.astype(np.uint8),
        is_peripheral=np.asarray(evidence["peripheral"], dtype=np.uint8),
        is_far_peripheral=np.asarray(evidence["far"], dtype=np.uint8),
        is_extremum_driver=np.asarray(evidence["extremum_drivers"], dtype=np.uint8),
        is_strong_manifold_anchor=np.asarray(evidence["strong_anchor"], dtype=np.uint8),
        is_reference_central_core=np.asarray(evidence["reference"], dtype=np.uint8),
        nearest_strong_anchor_distance_voxels=np.asarray(evidence["nearest_strong_anchor_distance_voxels"], dtype=np.float32),
        knn_median_distance_voxels=np.asarray(evidence["knn_median_distance_voxels"], dtype=np.float32),
        voxel_size_reconstruction_units=np.asarray(float(evidence["voxel_size"]), dtype=np.float64),
        schema_version=np.asarray(I3_TRANSFER_SCHEMA_VERSION, dtype=np.int32),
    )
    candidate_ext = _slice_aligned_payload(source, keep)
    candidate_core = _slice_aligned_payload(source, core & keep)
    np.savez_compressed(transfer_dir / "i3_1_candidate_extended.npz", **candidate_ext)
    np.savez_compressed(transfer_dir / "i3_1_candidate_core.npz", **candidate_core)

    scene = quality["scene"]
    safe = bool(assessment["safe_for_manual_transfer_review"])
    summary = {
        "phase": "I3.1",
        "schema_version": I3_TRANSFER_SCHEMA_VERSION,
        "status": "ready_for_manual_transfer_review" if safe else "degraded",
        "source_i1_dir": str(i1_dir),
        "baseline_quality_dir": str(quality_dir),
        "scene": {
            "scene_id": str(scene["scene_id"]),
            "fingerprint": str(scene["fingerprint"]),
            "source": dict(quality.get("source", {})),
            "registered_views": int(scene["registered_views"]),
            "dense_alignment_accepted_views": int(scene["dense_alignment_accepted_views"]),
            "selected_qualified_views": int(scene["selected_qualified_views"]),
            "voxel_size_reconstruction_units": float(scene["voxel_size_reconstruction_units"]),
            "sfm_scale_status": "arbitrary",
        },
        "i2_transfer_preflight": preflight,
        "baseline_quality": {
            "status": str(quality["status"]),
            "automated_pass": bool(quality["automated_pass"]),
            "retained_geometry": dict(quality["retained_geometry"]),
            "spatial": dict(quality["spatial"]),
        },
        "transfer_evidence": {
            "reference_thresholds": dict(evidence["reference_thresholds"]),
            "resolved_rule": dict(evidence["resolved_rule"]),
            "strong_anchor_count": int(np.count_nonzero(np.asarray(evidence["strong_anchor"], dtype=bool))),
            "reference_count": int(np.count_nonzero(np.asarray(evidence["reference"], dtype=bool))),
            "extremum_driver_count": int(np.count_nonzero(np.asarray(evidence["extremum_drivers"], dtype=bool))),
            "peripheral_count": int(np.count_nonzero(np.asarray(evidence["peripheral"], dtype=bool))),
            "far_peripheral_count": int(np.count_nonzero(np.asarray(evidence["far"], dtype=bool))),
            "filter_predicate_uses_peripheral_or_far_labels": False,
            "filter_predicate_uses_extremum_labels": False,
            "reference_qualification_uses_diagnostic_central_core_non_driver": True,
        },
        "candidate": assessment,
        "artifacts": {
            "diagnostics_npz": str(diagnostics.resolve()),
            "candidate_extended_npz": str((transfer_dir / "i3_1_candidate_extended.npz").resolve()),
            "candidate_core_npz": str((transfer_dir / "i3_1_candidate_core.npz").resolve()),
        },
        "authority": {
            "geometry_changed": False,
            "production_f_policy_changed": False,
            "persistence_changed": False,
            "sfm_rerun": args.reuse_i1_dir is None,
            "da3_rerun": args.reuse_i1_dir is None,
            "fusion_rerun": args.reuse_i1_dir is None,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_manual_transfer_review": safe,
            "authorizes_i3_production_promotion": False,
            "authorizes_i4_connector_orbit": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": (
                "Review baseline/candidate fixed views and rejection decision map. Promote only if the second-scene "
                "transfer remains visually safe and useful."
                if safe else
                "Investigate failed second-scene transfer safety gates; do not promote the manifold refinement."
            ),
        },
    }
    path = _write_json(output / "i3_1_summary.json", summary)

    comp = assessment["component_lineage"]
    env = assessment["envelope"]
    print(f"\nI3.1 second-scene transfer campaign: {summary['status'].upper()}")
    print(
        f"views registered={scene['registered_views']} aligned={scene['dense_alignment_accepted_views']} "
        f"selected={scene['selected_qualified_views']}"
    )
    print(
        f"candidate rejected={assessment['rejected_points']} "
        f"extended_loss={assessment['extended_points']['loss_fraction']:.4f} "
        f"core_loss={assessment['core_points']['loss_fraction']:.4f}"
    )
    print(
        f"components baseline={comp['baseline']['largest_component_fraction']:.6f} "
        f"candidate={comp['promoted']['largest_component_fraction']:.6f} "
        f"removed_from_dominant={comp['removed_from_baseline_dominant']} "
        f"newly_detached={comp['newly_detached_voxels']}"
    )
    print(
        f"full/robust={env['baseline']['full_to_robust_diagonal_ratio']:.3f}->"
        f"{env['candidate']['full_to_robust_diagonal_ratio']:.3f} "
        f"improvement={env['full_to_robust_improvement_fraction']:.3f} "
        f"robust_shrink={env['robust_diagonal_shrink_fraction']:.3f}"
    )
    print(f"transfer_safety_gates_pass={safe}")
    print("production_F_policy_changed=False persistence_changed=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("manual_transfer_review_required=True i3_production_promotion=False mesh_authorization=False")
    print(f"summary={path}")
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
