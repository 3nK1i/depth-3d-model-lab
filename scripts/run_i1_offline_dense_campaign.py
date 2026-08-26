#!/usr/bin/env python3
"""I1 end-to-end offline Dense RGB quality-campaign foundation.

The runner generalizes the already-qualified B-E primitives to a complete
recorded video/image sequence, then deliberately reuses the unchanged formal F
filter and G persistence stages through an E3-compatible campaign adapter.

I1 validates execution/orchestration.  Visual scene-quality acceptance belongs
to I2 and later campaign steps.  Arbitrary SfM gauge is preserved throughout;
metric scale, measurement authority, certified metrology, and mesh authority
are never promoted here.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseFusionPolicy,
    DenseSceneProfile,
    SfmScaleStatus,
    arbitrary_scale_reference,
    camera_geometry_from_estimated_camera,
    fuse_dense_frames,
    reconstruction_camera_from_sfm_model,
)
from core.reconstruction.da3 import DA3_EFFICIENT_MODEL_ID, probe_da3_environment  # noqa: E402
from core.reconstruction.da3_provider import DA3ExternalWorkerRunner, DA3ProviderProfile  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_campaign import (  # noqa: E402
    I1_CAMPAIGN_SCHEMA_VERSION,
    StageTelemetry,
    pair_key,
    select_longest_monotonic_pass_path,
    uniform_order_subset,
)
from scripts.run_d1_three_view_canonical_accumulation import (  # noqa: E402
    _all_sparse_landmarks,
    _sfm_result_from_payload,
)
from scripts.run_e1_qualified_ten_view_voxel_fusion import (  # noqa: E402
    _save_cloud_npz,
    _support_histogram,
    _write_binary_ply,
)
from scripts.run_e3_qualified_scene_fusion_expansion import (  # noqa: E402
    _build_dense_fusion_frames,
    _diagnose_pair,
    _generate_candidate_view,
    _load_cloud_record,
    _robust_diagonal,
)

_DEFAULT_SFM_PYTHON = Path.home() / "anaconda3/envs/eia-pycolmap-probe/bin/python"
_DEFAULT_DA3_PYTHON = Path.home() / "anaconda3/envs/eia-da3/bin/python"
_DEFAULT_DA3_REPO = Path.home() / "Desktop/eia-da3-backends/depth-anything-3"
_DA3_WORKER = ROOT / "scripts/da3_workers/da3_worker.py"

_DENSE_THRESHOLDS = {
    "consistency_relative_tolerance": 0.12,
    "occlusion_relative_margin": 0.20,
    "minimum_comparable_samples": 1_000,
    "minimum_consistent_ratio": 0.50,
    "maximum_median_relative_residual": 0.10,
    "maximum_p90_relative_residual": 0.25,
    "maximum_front_conflict_ratio": 0.25,
    "nn_support_scene_fraction": 0.05,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--images", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sfm-python", type=Path, default=_DEFAULT_SFM_PYTHON)
    parser.add_argument("--da3-python", type=Path, default=_DEFAULT_DA3_PYTHON)
    parser.add_argument("--da3-repo", type=Path, default=_DEFAULT_DA3_REPO)
    parser.add_argument("--da3-model", default=DA3_EFFICIENT_MODEL_ID)
    parser.add_argument("--da3-model-revision", default="")
    parser.add_argument("--profile", default="efficient", choices=("efficient", "balanced", "high_quality"))
    parser.add_argument("--matching", default="sequential", choices=("sequential", "exhaustive"))
    parser.add_argument("--max-keyframes", type=int, default=40)
    parser.add_argument("--max-dense-views", type=int, default=40)
    parser.add_argument("--video-target-fps", type=float, default=2.0)
    parser.add_argument("--image-sequence-fps", type=float, default=10.0)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    parser.add_argument("--process-res", type=int, default=392)
    parser.add_argument("--da3-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--minimum-correspondences", type=int, default=8)
    parser.add_argument("--maximum-relative-residual", type=float, default=0.35)
    parser.add_argument("--max-bridge-gap", type=int, default=3)
    parser.add_argument("--minimum-qualified-views", type=int, default=10)
    parser.add_argument("--max-points-per-direction", type=int, default=120_000)
    parser.add_argument("--nn-max-points-per-view", type=int, default=60_000)
    parser.add_argument("--voxel-scene-fraction", type=float, default=0.004)
    parser.add_argument("--max-voxels", type=int, default=1_500_000)
    parser.add_argument("--minimum-supported-views", type=int, default=2)
    parser.add_argument("--high-support-views", type=int, default=3)
    parser.add_argument("--minimum-supported-voxels", type=int, default=1_000)
    parser.add_argument("--sfm-focal-relative-uncertainty", type=float, default=0.20)
    parser.add_argument("--reuse-sfm-dir", type=Path, default=None)
    parser.add_argument(
        "--reuse-dense-dir",
        type=Path,
        default=None,
        help=(
            "Reuse previously generated I1 02_dense_views candidate artifacts. "
            "Pair qualification/fusion/filtering/persistence are recomputed."
        ),
    )
    parser.add_argument(
        "--stop-after",
        choices=("sfm", "qualified-fusion", "filtering", "persistence"),
        default="persistence",
    )
    args = parser.parse_args()
    if args.max_keyframes < 3 or args.max_dense_views < 3:
        parser.error("--max-keyframes and --max-dense-views must be at least 3")
    if args.max_bridge_gap < 1 or args.max_bridge_gap > 6:
        parser.error("--max-bridge-gap must be in [1,6]")
    if args.minimum_qualified_views < 10:
        parser.error("--minimum-qualified-views must be at least 10 because the unchanged formal F gate requires 10+ views")
    if not 0.0 < args.voxel_scene_fraction < 0.1:
        parser.error("--voxel-scene-fraction must be in (0,0.1)")
    if args.max_voxels <= 0 or args.minimum_supported_voxels <= 0:
        parser.error("voxel limits must be positive")
    return args


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read required I1 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I1 JSON root must be an object: {path}")
    return payload


def _stream_command(command: list[str], *, log_path: Path) -> int:
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
        return process.wait()


def _source_args(args: argparse.Namespace) -> list[str]:
    if args.video is not None:
        return ["--video", str(args.video.expanduser().resolve()), "--video-target-fps", str(args.video_target_fps)]
    result = ["--images", str(args.images.expanduser().resolve()), "--image-sequence-fps", str(args.image_sequence_fps)]
    if args.recursive:
        result.append("--recursive")
    return result


def _sfm_command(args: argparse.Namespace, sfm_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/run_reconstruction_pycolmap_smoke.py"),
        *_source_args(args),
        "--sfm-python", str(args.sfm_python.expanduser().resolve()),
        "--output-dir", str(sfm_dir),
        "--profile", str(args.profile),
        "--matching", str(args.matching),
        "--max-keyframes", str(args.max_keyframes),
        "--timeout-minutes", str(args.sfm_timeout_minutes),
    ]


def _run_stage_command(stage: str, command: list[str], *, log_path: Path) -> tuple[dict[str, Any], int]:
    print(f"\nI1 stage {stage}: {' '.join(command)}", flush=True)
    with StageTelemetry(stage) as telemetry:
        code = _stream_command(command, log_path=log_path)
    record = telemetry.as_dict()
    record.update({"returncode": int(code), "log": str(log_path.resolve())})
    return record, code


def _fusion_support_masks(counts: np.ndarray, *, minimum: int, high: int, maximum_views: int) -> tuple[np.ndarray, np.ndarray]:
    if not 2 <= minimum <= high <= maximum_views:
        raise ReconstructionValidationError("I1 support gates are incompatible with selected view count")
    return counts >= minimum, counts >= high


def _raw_point_count(records: Iterable[Mapping[str, Any]]) -> int:
    return int(sum(int(np.asarray(record["points"]).shape[0]) for record in records))


def _load_reused_dense_candidate(
    *,
    order: int,
    dense_dir: Path,
    sfm_result: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Restore one previously generated I1/E3 candidate without rerunning DA3.

    The canonical cloud already contains the aligned reconstruction-unit depth
    and exact SfM pose used by the original candidate. ``_load_cloud_record``
    revalidates that pose/depth shape against the reused SfM result before the
    candidate is admitted to pair qualification.
    """

    root = dense_dir / f"candidate_order_{int(order):02d}"
    alignment_path = root / "alignment.json"
    if not alignment_path.is_file():
        raise ReconstructionValidationError(
            f"--reuse-dense-dir is missing candidate alignment evidence for order {order}: {alignment_path}"
        )
    alignment = _read_json(alignment_path)

    workspace_matches = [
        item for item in sfm_result.workspace.images
        if int(item.order) == int(order)
    ]
    if len(workspace_matches) != 1:
        raise ReconstructionValidationError(
            f"Reused dense candidate order {order} does not map to exactly one SfM workspace image"
        )
    workspace = workspace_matches[0]
    estimates = {item.keyframe_id: item for item in sfm_result.cameras}
    estimate = estimates.get(workspace.keyframe_id)
    if estimate is None:
        raise ReconstructionValidationError(
            f"Reused dense candidate order {order} is not registered in the reused SfM result"
        )

    alignment_payload = alignment.get("alignment")
    scale = None
    relative_uncertainty = None
    if isinstance(alignment_payload, Mapping):
        value = alignment_payload.get("scale_to_reconstruction_units")
        if value is not None:
            scale = float(value)
        uncertainty = alignment_payload.get("relative_scale_uncertainty")
        if uncertainty is not None:
            relative_uncertainty = float(uncertainty)

    cloud_path = root / "canonical_cloud.npz"
    accepted = str(alignment.get("status", "")) == "accepted" and cloud_path.is_file()
    summary: dict[str, Any] = {
        "workspace_order": int(order),
        "source_sequence_index": int(workspace.source_sequence_index),
        "keyframe_id": str(estimate.keyframe_id),
        "frame_id": str(estimate.frame_id),
        "camera_id": str(estimate.camera_id),
        "camera_model_name": str(alignment.get("camera_model_name", "")),
        "candidate_correspondences": int(alignment.get("candidate_observations", 0)),
        "valid_correspondences": int(alignment.get("valid_correspondences", 0)),
        "inlier_correspondences": int(alignment.get("inlier_correspondences", 0)),
        "median_relative_residual": alignment.get("median_relative_residual"),
        "p90_relative_residual": alignment.get("p90_relative_residual"),
        "alignment_evidence_json": str(alignment_path.resolve()),
        "alignment_accepted": bool(accepted),
        "rejection_code": alignment.get("rejection_code"),
        "reused_saved_candidate": True,
        "reuse_source_dir": str(root.resolve()),
    }
    if not accepted:
        return None, summary
    if scale is None or not isfinite(scale) or scale <= 0.0:
        raise ReconstructionValidationError(
            f"Reused accepted candidate order {order} is missing a finite positive alignment scale"
        )

    record = _load_cloud_record(
        order=int(order),
        keyframe_id=str(estimate.keyframe_id),
        cloud_npz=cloud_path,
        sfm_result=sfm_result,
        scale_to_reconstruction_units=scale,
        source="i1_reused_dense_candidate",
    )
    summary.update({
        "scale_to_reconstruction_units": float(scale),
        "relative_scale_uncertainty": relative_uncertainty,
        "point_count": int(np.asarray(record["points"]).shape[0]),
        "cloud_npz": str(cloud_path.resolve()),
    })
    return record, summary


def _build_compatibility_summary(
    *,
    output_dir: Path,
    sfm_result: Any,
    selected_records: list[dict[str, Any]],
    candidate_summaries: list[dict[str, Any]],
    pair_summaries: dict[tuple[int, int], dict[str, Any]],
    selected_path: Any,
    fusion_result: Any,
    voxel_size: float,
    sparse_points: np.ndarray,
    sparse_colors: np.ndarray,
    minimum_supported_views: int,
    high_support_views: int,
    minimum_supported_voxels: int,
    max_bridge_gap: int,
    candidate_orders: tuple[int, ...],
) -> dict[str, Any]:
    cloud = fusion_result.point_cloud
    counts = np.asarray(cloud.view_observation_counts, dtype=np.uint32)
    selected_orders = tuple(int(item["workspace_order"]) for item in selected_records)
    selected_keyframes = tuple(str(item["keyframe_id"]) for item in selected_records)
    supported_mask, high_mask = _fusion_support_masks(
        counts,
        minimum=minimum_supported_views,
        high=high_support_views,
        maximum_views=len(selected_orders),
    )
    all_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_all.npz",
        points=cloud.points_reconstruction,
        colors=cloud.colors_rgb,
        counts=counts,
        mean_confidence=cloud.mean_confidence,
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    supported_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_supported.npz",
        points=cloud.points_reconstruction[supported_mask],
        colors=cloud.colors_rgb[supported_mask],
        counts=counts[supported_mask],
        mean_confidence=cloud.mean_confidence[supported_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    high_npz = _save_cloud_npz(
        output_dir / "e3_expanded_voxel_fused_high_support.npz",
        points=cloud.points_reconstruction[high_mask],
        colors=cloud.colors_rgb[high_mask],
        counts=counts[high_mask],
        mean_confidence=cloud.mean_confidence[high_mask],
        voxel_size=voxel_size,
        selected_keyframe_ids=selected_keyframes,
        selected_workspace_orders=selected_orders,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
    )
    all_ply = _write_binary_ply(output_dir / "e3_expanded_voxel_fused_all.ply", cloud.points_reconstruction, cloud.colors_rgb)
    supported_ply = _write_binary_ply(output_dir / "e3_expanded_voxel_fused_supported.ply", cloud.points_reconstruction[supported_mask], cloud.colors_rgb[supported_mask])

    frames = []
    for index, record in enumerate(selected_records):
        workspace = record["workspace"]
        frames.append({
            "fusion_index": index,
            "workspace_order": int(record["workspace_order"]),
            "keyframe_id": str(record["keyframe_id"]),
            "frame_id": str(record["frame_id"]),
            "timestamp_ns": int(workspace.timestamp_ns),
            "source": "i1_full_scene_qualified_view",
            "cloud_npz": str(Path(record["cloud_npz"]).resolve()),
        })

    supported_count = int(np.count_nonzero(supported_mask))
    high_count = int(np.count_nonzero(high_mask))
    capacity_clean = int(fusion_result.statistics.dropped_new_voxels_capacity) == 0
    passed = (
        len(selected_orders) >= 10
        and capacity_clean
        and int(fusion_result.statistics.frames_integrated) == len(selected_orders)
        and supported_count >= int(minimum_supported_voxels)
        and fusion_result.scale_status is SfmScaleStatus.ARBITRARY
        and fusion_result.unit_to_metre is None
    )
    summary = {
        "phase": "E3",
        "pass": bool(passed),
        "status": "pass" if passed else "degraded",
        "campaign_adapter": {
            "source_phase": "I1",
            "purpose": "reuse unchanged formal F/G kernels for the generic full-scene quality campaign",
            "not_an_e3_revalidation": True,
            "selection_policy": "longest monotonic pass-only dense-consistency path; no threshold relaxation",
        },
        "source_sfm_session_id": str(sfm_result.session_id),
        "selected_workspace_orders": list(selected_orders),
        "selected_keyframe_ids": list(selected_keyframes),
        "selected_view_count": len(selected_orders),
        "new_qualified_view_count": len(selected_orders),
        "frames": frames,
        "candidate_policy": {
            "candidate_orders": list(candidate_orders),
            "max_bridge_gap": int(max_bridge_gap),
            "dense_thresholds_reused_unchanged_from_d2_d4": dict(_DENSE_THRESHOLDS),
        },
        "candidate_views": candidate_summaries,
        "edge_diagnostics": [pair_summaries[key] for key in sorted(pair_summaries)],
        "campaign_path": selected_path.as_dict(),
        "fusion": {
            "voxel_size_reconstruction_units": float(voxel_size),
            "all_voxel_count": int(len(counts)),
            "supported_voxel_count": supported_count,
            "high_support_voxel_count": high_count,
            "support_histogram": _support_histogram(counts),
            "capacity_clean": bool(capacity_clean),
            "capacity_drops": int(fusion_result.statistics.dropped_new_voxels_capacity),
            "frames_integrated": int(fusion_result.statistics.frames_integrated),
            "sampled_pixels": int(fusion_result.statistics.sampled_pixels),
        },
        "decision": {
            "authorizes_next_confidence_filtering_milestone": bool(passed),
            "next_milestone": "F - Confidence / Consistency / Artifact Filtering" if passed else None,
            "primary_output": "e3_expanded_voxel_fused_supported.npz",
        },
        "authority": {
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": False,
            "canonical_geometry_authoritative": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "confidence_artifact_filtering_formalized": False,
        },
        "artifacts": {
            "all_npz": str(all_npz.resolve()),
            "all_ply": str(all_ply.resolve()),
            "supported_npz": str(supported_npz.resolve()),
            "supported_ply": str(supported_ply.resolve()),
            "high_support_npz": str(high_npz.resolve()),
        },
    }
    _write_json(output_dir / "e3_summary.json", summary)
    return summary


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stages: dict[str, Any] = {}

    # ------------------------------------------------------------------ SfM
    sfm_dir = output / "01_sfm"
    if args.reuse_sfm_dir is not None:
        sfm_source = args.reuse_sfm_dir.expanduser().resolve()
        report_path = sfm_source / "real_sfm_report.json"
        if not report_path.is_file():
            raise ReconstructionValidationError("--reuse-sfm-dir does not contain real_sfm_report.json")
        sfm_dir = sfm_source
        stages["sfm"] = {"reused": True, "source_dir": str(sfm_source)}
    else:
        sfm_dir.mkdir(parents=True, exist_ok=True)
        record, code = _run_stage_command("sfm", _sfm_command(args, sfm_dir), log_path=output / "logs/01_sfm.log")
        stages["sfm"] = record
        if code != 0:
            _write_json(output / "i1_campaign_summary.json", {"phase": "I1", "status": "failed", "failed_stage": "sfm", "stages": stages})
            return 2
    sfm_report = _read_json(sfm_dir / "real_sfm_report.json")
    sfm_result = _sfm_result_from_payload(sfm_report["sfm_result"])
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY or sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I1 requires arbitrary-scale SfM with no unit-to-metre promotion")
    registered_ids = {camera.keyframe_id for camera in sfm_result.cameras}
    registered_workspace = sorted(
        (item for item in sfm_result.workspace.images if item.keyframe_id in registered_ids),
        key=lambda item: (int(item.order), item.keyframe_id),
    )
    registered_orders = tuple(int(item.order) for item in registered_workspace)
    candidate_orders = uniform_order_subset(registered_orders, int(args.max_dense_views))
    if args.stop_after == "sfm":
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "pass",
            "pipeline_execution_complete": False, "stopped_after": "sfm",
            "sfm": {"registered_view_count": len(registered_orders), "candidate_orders": list(candidate_orders)},
            "stages": stages,
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 0

    # --------------------------------------------------- Dense view alignment
    dense_dir = output / "02_dense_views"
    dense_dir.mkdir(parents=True, exist_ok=True)
    reuse_dense_dir = None if args.reuse_dense_dir is None else args.reuse_dense_dir.expanduser().resolve()
    runner = None
    probe = None
    if reuse_dense_dir is not None:
        if not reuse_dense_dir.is_dir():
            raise ReconstructionValidationError(f"--reuse-dense-dir is not a directory: {reuse_dense_dir}")
        print(f"\nI1 dense-view qualification: reusing saved candidates from {reuse_dense_dir}", flush=True)
    else:
        da3_python = args.da3_python.expanduser().resolve()
        da3_repo = args.da3_repo.expanduser().resolve()
        probe = probe_da3_environment(da3_python, da3_repo)
        if not probe.available:
            raise ReconstructionValidationError("I1 isolated DA3 environment is unavailable: " + json.dumps(probe.as_dict(), sort_keys=True))
        runner = DA3ExternalWorkerRunner(
            python_executable=da3_python,
            worker_script=_DA3_WORKER.resolve(),
            profile=DA3ProviderProfile(
                model_id=str(args.da3_model),
                model_revision=str(args.da3_model_revision),
                process_res=int(args.process_res),
                process_res_method="upper_bound_resize",
            ),
            timeout_s=float(args.da3_timeout_seconds),
        )
        print("\nI1 dense-view qualification:", flush=True)

    records: dict[int, dict[str, Any]] = {}
    candidate_summaries: list[dict[str, Any]] = []
    with StageTelemetry("dense_alignment_projection") as telemetry:
        for order in candidate_orders:
            if reuse_dense_dir is not None:
                record, summary = _load_reused_dense_candidate(
                    order=order, dense_dir=reuse_dense_dir, sfm_result=sfm_result
                )
            else:
                assert runner is not None and probe is not None
                record, summary = _generate_candidate_view(
                    order=order,
                    sfm_result=sfm_result,
                    b2_summary={},
                    output_dir=dense_dir,
                    runner=runner,
                    da3_probe=probe,
                    minimum_correspondences=int(args.minimum_correspondences),
                    maximum_relative_residual=float(args.maximum_relative_residual),
                )
            candidate_summaries.append(summary)
            if record is not None:
                records[order] = record
                reuse_label = " REUSED" if reuse_dense_dir is not None else ""
                print(
                    f"  order={order} ALIGNMENT PASS{reuse_label} "
                    f"scale={float(summary['scale_to_reconstruction_units']):.6f}",
                    flush=True,
                )
            else:
                print(f"  order={order} ALIGNMENT REJECTED reason={summary.get('rejection_code')}", flush=True)
    stages["dense_alignment_projection"] = telemetry.as_dict()
    if reuse_dense_dir is not None:
        stages["dense_alignment_projection"].update({
            "reused": True,
            "source_dir": str(reuse_dense_dir),
            "da3_rerun": False,
        })
    stages["dense_alignment_projection"].update({
        "candidate_view_count": len(candidate_orders),
        "alignment_accepted_count": len(records),
        "alignment_rejected_count": len(candidate_orders) - len(records),
    })
    if len(records) < int(args.minimum_qualified_views):
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "dense_alignment_projection",
            "reason": "insufficient_alignment_accepted_views",
            "minimum_qualified_views": int(args.minimum_qualified_views),
            "stages": stages,
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 2

    # ------------------------------------------ Dense consistency graph/path
    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    scene_scale = _robust_diagonal(sparse_points)
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("I1 robust sparse scene diagonal is invalid")
    pair_summaries: dict[tuple[int, int], dict[str, Any]] = {}
    aligned_orders = tuple(sorted(records))
    pair_dir = output / "03_pair_diagnostics"
    pair_dir.mkdir(parents=True, exist_ok=True)
    print("\nI1 local dense-consistency graph:", flush=True)
    with StageTelemetry("dense_pair_qualification") as telemetry:
        for index, order_a in enumerate(aligned_orders):
            for order_b in aligned_orders[index + 1:]:
                if order_b - order_a > int(args.max_bridge_gap):
                    break
                pair = _diagnose_pair(
                    a=records[order_a], b=records[order_b], thresholds=_DENSE_THRESHOLDS,
                    scene_scale=scene_scale, sparse_landmarks=sparse_points, output_dir=pair_dir,
                    max_points_per_direction=int(args.max_points_per_direction),
                    nn_max_points_per_view=int(args.nn_max_points_per_view),
                )
                pair["campaign_stage"] = "I1"
                pair_summaries[pair_key(order_a, order_b)] = pair
                print(f"  edge {order_a}<->{order_b} status={pair['status']}", flush=True)
    stages["dense_pair_qualification"] = telemetry.as_dict()
    selected_path = select_longest_monotonic_pass_path(
        aligned_orders, pair_summaries, max_bridge_gap=int(args.max_bridge_gap)
    )
    print(f"\nI1 selected pass-only path={'->'.join(map(str, selected_path.orders))}", flush=True)
    print(
        f"qualified_views={selected_path.view_count} span={selected_path.span} "
        f"worst_consistency={selected_path.worst_consistency_ratio:.3f} "
        f"worst_median_residual={selected_path.worst_median_relative_residual:.4f}",
        flush=True,
    )
    if selected_path.view_count < int(args.minimum_qualified_views):
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "dense_pair_qualification", "reason": "insufficient_connected_qualified_views",
            "selected_path": selected_path.as_dict(), "minimum_qualified_views": int(args.minimum_qualified_views),
            "stages": stages,
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 2

    # ------------------------------------------------------- Qualified fusion
    fusion_dir = output / "04_qualified_fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    selected_records = [records[order] for order in selected_path.orders]
    frames, _, camera_id = _build_dense_fusion_frames(selected_records, sfm_result=sfm_result)
    model = {item.camera_id: item for item in sfm_result.camera_models}[camera_id]
    reconstruction_camera = reconstruction_camera_from_sfm_model(
        model, keyframe_id=str(selected_records[0]["keyframe_id"]), reconstruction_frame_id="offline-sfm-reconstruction"
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"i1-{camera_id}-sfm-estimated",
        camera=reconstruction_camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes="I1 campaign camera profile from globally optimized SfM; arbitrary gauge only.",
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"i1-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="I1 preserves arbitrary offline-SfM reconstruction gauge; no physical unit claim.",
    )
    voxel_size = float(scene_scale * float(args.voxel_scene_fraction))
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=1,
        minimum_view_observations=1,
        max_voxels=int(args.max_voxels),
    )
    print(f"\nI1 qualified fusion: views={len(frames)} voxel={voxel_size:.6f}", flush=True)
    with StageTelemetry("qualified_voxel_fusion") as telemetry:
        fusion_result = fuse_dense_frames(
            frames,
            session_id=f"{sfm_result.session_id}:i1-qualified-fusion",
            camera_profile=camera_profile,
            scale_profile=scale_profile,
            policy=policy,
            reconstruction_frame_id="offline-sfm-reconstruction",
        )
        compatibility = _build_compatibility_summary(
            output_dir=fusion_dir, sfm_result=sfm_result, selected_records=selected_records,
            candidate_summaries=candidate_summaries, pair_summaries=pair_summaries, selected_path=selected_path,
            fusion_result=fusion_result, voxel_size=voxel_size, sparse_points=sparse_points, sparse_colors=sparse_colors,
            minimum_supported_views=int(args.minimum_supported_views), high_support_views=int(args.high_support_views),
            minimum_supported_voxels=int(args.minimum_supported_voxels), max_bridge_gap=int(args.max_bridge_gap),
            candidate_orders=candidate_orders,
        )
    stages["qualified_voxel_fusion"] = telemetry.as_dict()
    stages["qualified_voxel_fusion"].update({
        "status": compatibility["status"],
        "selected_view_count": compatibility["selected_view_count"],
        "all_voxel_count": compatibility["fusion"]["all_voxel_count"],
        "supported_voxel_count": compatibility["fusion"]["supported_voxel_count"],
        "high_support_voxel_count": compatibility["fusion"]["high_support_voxel_count"],
        "capacity_drops": compatibility["fusion"]["capacity_drops"],
    })
    if not compatibility["pass"]:
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "qualified_voxel_fusion", "stages": stages,
            "selected_path": selected_path.as_dict(),
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 2
    if args.stop_after == "qualified-fusion":
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "pass",
            "pipeline_execution_complete": False, "stopped_after": "qualified-fusion", "stages": stages,
            "selected_path": selected_path.as_dict(),
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 0

    # ------------------------------------------------------- Formal F filter
    filter_dir = output / "05_filtering"
    filter_command = [
        sys.executable, str(ROOT / "scripts/run_f_confidence_artifact_filtering.py"),
        "--e3-dir", str(fusion_dir), "--output-dir", str(filter_dir),
    ]
    record, code = _run_stage_command("formal_filtering", filter_command, log_path=output / "logs/05_filtering.log")
    stages["formal_filtering"] = record
    if code != 0:
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "formal_filtering", "stages": stages, "selected_path": selected_path.as_dict(),
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 2
    f_summary = _read_json(filter_dir / "f_summary.json")
    stages["formal_filtering"].update({
        "core_voxels": int(f_summary["classification"]["core_voxels"]),
        "extended_voxels": int(f_summary["classification"]["extended_voxels"]),
        "rejected_supported_voxels": int(f_summary["classification"]["rejected_supported_voxels"]),
    })
    if args.stop_after == "filtering":
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "pass",
            "pipeline_execution_complete": False, "stopped_after": "filtering", "stages": stages,
            "selected_path": selected_path.as_dict(),
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 0

    # --------------------------------------------------------- G persistence
    persistent_dir = output / "06_persistent_scene"
    g_command = [
        sys.executable, str(ROOT / "scripts/run_g_persistent_dense_rgb_geometry.py"),
        "--f-dir", str(filter_dir), "--output-dir", str(persistent_dir),
    ]
    record, code = _run_stage_command("canonical_persistence", g_command, log_path=output / "logs/06_persistence.log")
    stages["canonical_persistence"] = record
    if code != 0:
        final = {
            "phase": "I1", "schema_version": I1_CAMPAIGN_SCHEMA_VERSION, "status": "degraded",
            "failed_stage": "canonical_persistence", "stages": stages, "selected_path": selected_path.as_dict(),
            "authority": {"metric_scale_claimed": False, "measurement_authoritative": False, "surface_mesh_generated": False},
        }
        _write_json(output / "i1_campaign_summary.json", final)
        return 2
    g_summary = _read_json(persistent_dir / "g_summary.json")
    stages["canonical_persistence"].update({
        "scene_id": str(g_summary["scene_id"]),
        "scene_fingerprint_sha256": str(g_summary["scene_fingerprint_sha256"]),
        "extended_points": int(g_summary["geometry"]["primary_point_count"]),
        "core_points": int(g_summary["geometry"]["core_point_count"]),
        "self_contained": bool(g_summary["persistence"]["bundle_self_contained"]),
    })

    final = {
        "phase": "I1",
        "schema_version": I1_CAMPAIGN_SCHEMA_VERSION,
        "status": "pass",
        "pipeline_execution_complete": True,
        "quality_campaign_status": "pending_I2_real_scene_visual_and_coverage_review",
        "source": {
            "kind": "video" if args.video is not None else "image_sequence",
            "path": str((args.video if args.video is not None else args.images).expanduser().resolve()),
        },
        "sfm": {
            "session_id": str(sfm_result.session_id),
            "registered_view_count": len(registered_orders),
            "registered_orders": list(registered_orders),
            "candidate_dense_orders": list(candidate_orders),
            "scale_status": "arbitrary",
        },
        "dense_qualification": {
            "alignment_accepted_views": len(records),
            "selected_pass_path": selected_path.as_dict(),
            "thresholds": dict(_DENSE_THRESHOLDS),
            "threshold_relaxation": False,
        },
        "fusion": compatibility["fusion"],
        "filtering": {
            "core_voxels": int(f_summary["classification"]["core_voxels"]),
            "extended_voxels": int(f_summary["classification"]["extended_voxels"]),
            "rejected_supported_voxels": int(f_summary["classification"]["rejected_supported_voxels"]),
        },
        "persistent_scene": {
            "scene_id": str(g_summary["scene_id"]),
            "fingerprint": str(g_summary["scene_fingerprint_sha256"]),
            "scene_dir": str(persistent_dir.resolve()),
            "extended_points": int(g_summary["geometry"]["primary_point_count"]),
            "core_points": int(g_summary["geometry"]["core_point_count"]),
            "self_contained": True,
        },
        "stages": stages,
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_current_full_video_validation": True,
            "next_milestone": "I2 - Current Full Office Video Validation",
            "note": "I1 proves generic offline execution/orchestration only; scene-quality acceptance remains I2.",
        },
    }
    summary_path = _write_json(output / "i1_campaign_summary.json", final)
    print("\nI1 offline Dense RGB campaign foundation: PASS")
    print(f"registered_views={len(registered_orders)} dense_alignment_pass={len(records)} selected_qualified_views={selected_path.view_count}")
    print(f"supported_voxels={compatibility['fusion']['supported_voxel_count']} core={final['filtering']['core_voxels']} extended={final['filtering']['extended_voxels']}")
    print(f"persistent_scene={persistent_dir}")
    print(f"scene_id={final['persistent_scene']['scene_id']}")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("quality_campaign_status=pending_I2_real_scene_visual_and_coverage_review")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
