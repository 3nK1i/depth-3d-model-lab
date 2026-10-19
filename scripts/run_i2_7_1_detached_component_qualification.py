#!/usr/bin/env python3
"""I2.7.1 detached-component qualification after the I2.7 component gate failure.

Diagnostic-only.  Recomputes baseline and promoted 26-neighbour components on
identical voxel semantics, traces component lineage through the exact I2.6
subset deletion, and characterizes pre-existing versus newly detached geometry.
No filtering/persistence/authority state is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_detached_component_qualification import (  # noqa: E402
    I2_7_1_SCHEMA_VERSION,
    classify_component_gate_failure,
    compare_component_lineage,
    detached_distance_statistics,
    mask_statistics,
)

_DEFAULT_I2 = Path("/tmp/depth3d_i2_full_video_quality")
_DEFAULT_I27 = Path("/tmp/depth3d_i2_7_final_requalification")
_DEFAULT_OUT = Path("/tmp/depth3d_i2_7_1_detached_component_qualification")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i2-baseline-dir", type=Path, default=_DEFAULT_I2)
    p.add_argument("--i2-7-dir", type=Path, default=_DEFAULT_I27)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I2.7.1 JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I2.7.1 JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_diagnostics(path: Path, *, baseline: bool) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise ReconstructionValidationError(f"I2.7.1 diagnostics file is missing: {path}")
    with np.load(path, allow_pickle=False) as p:
        result: dict[str, np.ndarray | float] = {
            "points": np.asarray(p["points_reconstruction"], dtype=np.float64),
            "colors": np.asarray(p["colors_rgb"], dtype=np.uint8),
            "support": np.asarray(p["view_observation_counts"], dtype=np.uint32),
            "confidence": np.asarray(p["filter_confidence"], dtype=np.float32),
            "dispersion": np.asarray(p["p90_dispersion_voxels"], dtype=np.float32),
            "is_core": np.asarray(p["is_core"], dtype=bool),
            "peripheral": np.asarray(p["is_peripheral"], dtype=bool),
            "far": np.asarray(p["is_far_peripheral"], dtype=bool),
            "weak": np.asarray(p["is_weak_peripheral"], dtype=bool),
            "voxel_size": float(np.asarray(p["voxel_size_reconstruction_units"]).item()),
        }
        if baseline:
            result["baseline_dominant_hint"] = np.asarray(p["is_dominant_component"], dtype=bool)
    n = len(np.asarray(result["points"]))
    if n == 0 or np.asarray(result["colors"]).shape != (n, 3):
        raise ReconstructionValidationError("I2.7.1 diagnostics geometry is invalid.")
    for name in ("support", "confidence", "dispersion", "is_core", "peripheral", "far", "weak"):
        if np.asarray(result[name]).shape != (n,):
            raise ReconstructionValidationError(f"I2.7.1 field {name} is not aligned.")
    return result


def _keys(points: np.ndarray, voxel: float) -> list[tuple[int, int, int]]:
    return [tuple(map(int, row)) for row in np.floor(points / voxel).astype(np.int64)]


def main() -> int:
    a = _args()
    i2_dir = a.i2_baseline_dir.expanduser().resolve()
    i27_dir = a.i2_7_dir.expanduser().resolve()
    out = a.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I2.7.1 output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    baseline_summary = _read_json(i2_dir / "i2_summary.json")
    i27 = _read_json(i27_dir / "i2_7_summary.json")
    if str(baseline_summary.get("phase")) != "I2" or not bool(baseline_summary.get("automated_pass")):
        raise ReconstructionValidationError("I2.7.1 requires the original automated-passing I2 baseline.")
    if str(i27.get("phase")) != "I2.7" or bool(i27.get("automated_pass")):
        raise ReconstructionValidationError("I2.7.1 is only for a failed/degraded I2.7 automated requalification.")
    failed_gates = sorted(name for name, value in i27.get("automated_closure_gates", {}).items() if not bool(value))
    if failed_gates != ["largest_component_preserved"]:
        raise ReconstructionValidationError(
            "I2.7.1 requires largest_component_preserved to be the only failed I2.7 automated gate; "
            f"found {failed_gates}."
        )

    baseline_diag = _load_diagnostics(Path(str(baseline_summary["artifacts"]["spatial_diagnostics_npz"])).expanduser().resolve(), baseline=True)
    promoted_diag = _load_diagnostics(Path(str(i27["artifacts"]["spatial_diagnostics_npz"])).expanduser().resolve(), baseline=False)
    baseline_points = np.asarray(baseline_diag["points"], dtype=np.float64)
    promoted_points = np.asarray(promoted_diag["points"], dtype=np.float64)
    voxel = float(baseline_diag["voxel_size"])
    if not np.isclose(voxel, float(promoted_diag["voxel_size"]), atol=1e-12, rtol=0.0):
        raise ReconstructionValidationError("I2.7.1 baseline/promoted voxel sizes disagree.")

    lineage = compare_component_lineage(baseline_points=baseline_points, promoted_points=promoted_points, voxel_size=voxel)
    masks = lineage.pop("masks")
    promoted_baseline_indices = np.asarray(masks["promoted_baseline_indices"], dtype=np.int64)
    removed_baseline_indices = np.asarray(masks["removed_baseline_indices"], dtype=np.int64)
    promoted_dominant = np.asarray(masks["promoted_dominant"], dtype=bool)
    newly_detached = np.asarray(masks["newly_detached"], dtype=bool)
    preexisting_detached = np.asarray(masks["surviving_preexisting_detached"], dtype=bool)

    # Verify I2.6/I2.7 is an exact subset promotion, not a coordinate/attribute rewrite.
    surviving_equivalence = {
        "points": bool(np.array_equal(baseline_points[promoted_baseline_indices], promoted_points)),
        "colors": bool(np.array_equal(np.asarray(baseline_diag["colors"])[promoted_baseline_indices], np.asarray(promoted_diag["colors"]))),
        "support": bool(np.array_equal(np.asarray(baseline_diag["support"])[promoted_baseline_indices], np.asarray(promoted_diag["support"]))),
        "confidence": bool(np.array_equal(np.asarray(baseline_diag["confidence"])[promoted_baseline_indices], np.asarray(promoted_diag["confidence"]))),
        "dispersion": bool(np.array_equal(np.asarray(baseline_diag["dispersion"])[promoted_baseline_indices], np.asarray(promoted_diag["dispersion"]))),
        "core_membership": bool(np.array_equal(np.asarray(baseline_diag["is_core"])[promoted_baseline_indices], np.asarray(promoted_diag["is_core"]))),
    }
    if not all(surviving_equivalence.values()):
        raise ReconstructionValidationError(f"I2.7.1 surviving baseline/promoted evidence is not exact: {surviving_equivalence}")

    promoted_fields = {
        "is_core": np.asarray(promoted_diag["is_core"], dtype=bool),
        "peripheral": np.asarray(promoted_diag["peripheral"], dtype=bool),
        "far": np.asarray(promoted_diag["far"], dtype=bool),
        "weak": np.asarray(promoted_diag["weak"], dtype=bool),
        "support": np.asarray(promoted_diag["support"], dtype=np.uint32),
        "confidence": np.asarray(promoted_diag["confidence"], dtype=np.float32),
        "dispersion": np.asarray(promoted_diag["dispersion"], dtype=np.float32),
    }
    promoted_detached = ~promoted_dominant
    stats = {
        "promoted_all_detached": mask_statistics(promoted_detached, **promoted_fields),
        "surviving_preexisting_detached": mask_statistics(preexisting_detached, **promoted_fields),
        "newly_detached": mask_statistics(newly_detached, **promoted_fields),
    }
    distances = {
        "promoted_all_detached": detached_distance_statistics(promoted_points, promoted_dominant, promoted_detached, voxel),
        "surviving_preexisting_detached": detached_distance_statistics(promoted_points, promoted_dominant, preexisting_detached, voxel),
        "newly_detached": detached_distance_statistics(promoted_points, promoted_dominant, newly_detached, voxel),
    }

    threshold = float(i27["closure_policy"]["minimum_largest_component_fraction_of_extended"])
    baseline_fraction = float(lineage["baseline"]["largest_component_fraction"])
    promoted_fraction = float(lineage["promoted"]["largest_component_fraction"])
    failure_classification = classify_component_gate_failure(
        baseline_fraction=baseline_fraction,
        promoted_fraction=promoted_fraction,
        threshold=threshold,
        newly_detached_voxels=int(lineage["newly_detached_voxels"]),
    )

    promoted_component_records = list(lineage["promoted_component_records"])
    new_components = [item for item in promoted_component_records if item["origin"] == "newly_detached_from_baseline_dominant"]
    touching_new_components = [item for item in new_components if int(item["touching_removed_voxels"]) > 0]

    # Build a baseline-universe lineage map for deterministic visual inspection.
    n_base = len(baseline_points)
    category = np.full(n_base, -1, dtype=np.int8)  # -1 should never survive below.
    category[promoted_baseline_indices[promoted_dominant]] = 0  # promoted dominant
    category[promoted_baseline_indices[preexisting_detached]] = 1  # pre-existing detached
    category[promoted_baseline_indices[newly_detached]] = 2  # newly detached
    newly_core = newly_detached & np.asarray(promoted_diag["is_core"], dtype=bool)
    category[promoted_baseline_indices[newly_core]] = 3  # newly detached Core
    category[removed_baseline_indices] = 4  # d12 removed by production promotion
    if np.any(category < 0):
        raise ReconstructionValidationError("I2.7.1 lineage visualization accounting is incomplete.")

    diagnostic_npz = out / "i2_7_1_component_lineage_diagnostics.npz"
    np.savez_compressed(
        diagnostic_npz,
        baseline_points_reconstruction=baseline_points,
        baseline_colors_rgb=np.asarray(baseline_diag["colors"], dtype=np.uint8),
        lineage_category_code=category,
        promoted_points_reconstruction=promoted_points,
        promoted_colors_rgb=np.asarray(promoted_diag["colors"], dtype=np.uint8),
        promoted_is_core=np.asarray(promoted_diag["is_core"], dtype=np.uint8),
        promoted_is_dominant_component=promoted_dominant.astype(np.uint8),
        promoted_is_newly_detached=newly_detached.astype(np.uint8),
        promoted_is_preexisting_detached=preexisting_detached.astype(np.uint8),
        voxel_size_reconstruction_units=np.asarray(voxel, dtype=np.float64),
        schema_version=np.asarray(I2_7_1_SCHEMA_VERSION, dtype=np.int32),
    )

    summary = {
        "phase": "I2.7.1",
        "schema_version": I2_7_1_SCHEMA_VERSION,
        "status": "diagnosis_ready",
        "source_scene": {
            "scene_id": str(i27["scene"]["scene_id"]),
            "fingerprint": str(i27["scene"]["fingerprint"]),
            "voxel_size_reconstruction_units": voxel,
        },
        "trigger": {
            "i2_7_status": str(i27["status"]),
            "failed_automated_gates": failed_gates,
            "absolute_largest_component_threshold": threshold,
        },
        "component_comparison": lineage,
        "failure_classification": failure_classification,
        "baseline_absolute_gate_pass": baseline_fraction >= threshold,
        "promoted_absolute_gate_pass": promoted_fraction >= threshold,
        "surviving_evidence_equivalence": surviving_equivalence,
        "detached_region_statistics": stats,
        "distance_to_promoted_dominant_component_vox": distances,
        "promotion_cut_relationship": {
            "newly_detached_component_count": len(new_components),
            "newly_detached_components_touching_removed_voxels": len(touching_new_components),
            "top_newly_detached_components": new_components[:10],
        },
        "authority": {
            "geometry_changed": False,
            "filter_policy_changed": False,
            "persistence_changed": False,
            "sfm_rerun": False,
            "da3_rerun": False,
            "fusion_rerun": False,
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "component_gate_reassessment_authorized": False,
            "final_manual_review_authorized": False,
            "i2_closure_authorized": False,
            "i3_authorized": False,
            "mesh_authorization": False,
            "next_action": "Review baseline-vs-promoted component lineage and detached-fragment evidence before changing any I2.7 gate.",
        },
        "artifacts": {"component_lineage_diagnostics_npz": str(diagnostic_npz.resolve())},
    }
    summary_path = _write_json(out / "i2_7_1_summary.json", summary)

    print("I2.7.1 detached component qualification: DIAGNOSIS_READY")
    print(
        f"baseline extended={lineage['baseline']['retained_voxels']} components={lineage['baseline']['component_count']} "
        f"largest={lineage['baseline']['largest_component_voxels']} fraction={baseline_fraction:.6f} detached={lineage['baseline']['detached_voxels']}"
    )
    print(
        f"promoted extended={lineage['promoted']['retained_voxels']} components={lineage['promoted']['component_count']} "
        f"largest={lineage['promoted']['largest_component_voxels']} fraction={promoted_fraction:.6f} detached={lineage['promoted']['detached_voxels']}"
    )
    print(
        f"delta largest_fraction={lineage['largest_component_fraction_delta']:+.6f} "
        f"detached_voxels={lineage['detached_voxel_delta']:+d} threshold={threshold:.3f}"
    )
    print(
        f"removed={lineage['removed_voxels']} removed_from_baseline_dominant={lineage['removed_from_baseline_dominant']} "
        f"removed_from_preexisting_detached={lineage['removed_from_preexisting_detached']}"
    )
    print(
        f"promoted_detached={lineage['promoted_detached_voxels']} preexisting_survivors={lineage['surviving_preexisting_detached_voxels']} "
        f"newly_detached={lineage['newly_detached_voxels']}"
    )
    print(
        f"newly_detached_components={len(new_components)} touching_removed={len(touching_new_components)} "
        f"dominant_lineage_fraction={lineage['promoted_dominant_from_single_baseline_component_fraction']:.6f}"
    )
    new_stats = stats["newly_detached"]
    print(
        f"newly_detached core={new_stats['core_fraction']:.3f} peripheral={new_stats['peripheral_fraction']:.3f} "
        f"far={new_stats['far_fraction']:.3f} weak={new_stats['weak_fraction']:.3f}"
    )
    print(
        "newly_detached_distance_to_dominant_vox="
        f"p50={distances['newly_detached']['p50_vox']} p90={distances['newly_detached']['p90_vox']} max={distances['newly_detached']['max_vox']}"
    )
    print(f"failure_classification={failure_classification}")
    print(f"surviving_evidence_equivalence={all(surviving_equivalence.values())}")
    print("geometry_changed=False filter_policy_changed=False sfm_rerun=False da3_rerun=False fusion_rerun=False")
    print("component_gate_reassessment_authorized=False i2_closure_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
