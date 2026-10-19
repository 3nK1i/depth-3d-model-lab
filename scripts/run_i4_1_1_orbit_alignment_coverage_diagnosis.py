#!/usr/bin/env python3
"""I4.1.1 diagnosis for a DEGRADED object-orbit baseline.

Reuses the completed I4.1 Horse evidence.  It does not rerun SfM, DA3,
alignment, fusion, filtering, or persistence.  It evaluates the exact I2 gate
vector, alignment rejection/scale evidence, and registered/aligned/selected
camera-orbit coverage so the next decision is evidence-based.
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
from core.reconstruction.offline_dense_i4_orbit_alignment_coverage_diagnosis import (  # noqa: E402
    I4_1_1_SCHEMA_VERSION,
    build_i4_1_1_diagnosis,
)
from core.reconstruction.offline_dense_i4_orbit_qualification import (  # noqa: E402
    camera_poses_from_sfm_report,
    select_camera_poses,
)

_DEFAULT_I4 = Path("/tmp/depth3d_i4_object_orbit_campaign")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_1_orbit_alignment_coverage_diagnosis")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-dir", type=Path, default=_DEFAULT_I4)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.1 cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.1 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _resolve_artifact(raw: object, *, fallback: Path) -> Path:
    value = str(raw or "").strip()
    if value:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            return candidate
    candidate = fallback.expanduser().resolve()
    if candidate.is_file():
        return candidate
    raise ReconstructionValidationError(f"I4.1.1 required artifact is missing: {candidate}")


def main() -> int:
    args = _args()
    i4_dir = args.i4_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.1 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_i4 = _read_json(i4_dir / "i4_1_summary.json")
    i1_dir = i4_dir / "01_i1_baseline"
    i2_dir = i4_dir / "02_baseline_quality"
    i1 = _read_json(i1_dir / "i1_campaign_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    sfm_summary = _read_json(i1_dir / "01_sfm" / "real_sfm_summary.json")
    sfm_report = _read_json(i1_dir / "01_sfm" / "real_sfm_report.json")
    e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    f_summary = _read_json(i1_dir / "05_filtering" / "f_summary.json")

    extended_path = _resolve_artifact(
        f_summary.get("artifacts", {}).get("extended_npz") if isinstance(f_summary.get("artifacts"), Mapping) else None,
        fallback=i1_dir / "05_filtering" / "f_filtered_extended.npz",
    )
    with np.load(extended_path, allow_pickle=False) as payload:
        points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
        classes = np.asarray(payload["class_code"], dtype=np.uint8).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or classes.shape != (len(points),):
        raise ReconstructionValidationError("I4.1.1 Extended/Core geometry arrays are misaligned.")
    core_points = points[classes == 3]

    camera_ids, camera_matrices = camera_poses_from_sfm_report(sfm_report)
    diagnosis = build_i4_1_1_diagnosis(
        source_i4_summary=source_i4,
        sfm_summary=sfm_summary,
        i1_summary=i1,
        i2_summary=i2,
        e3_summary=e3,
        registered_camera_ids=camera_ids,
        registered_camera_matrices=camera_matrices,
        core_points=core_points,
    )

    aligned_ids = tuple(str(v) for v in diagnosis["alignment_diagnosis"]["accepted_keyframe_ids"])
    rejected_ids = tuple(str(v) for v in diagnosis["alignment_diagnosis"]["rejected_keyframe_ids"])
    selected_ids = tuple(str(v) for v in e3.get("selected_keyframe_ids", []))
    _, aligned_matrices = select_camera_poses(camera_ids, camera_matrices, aligned_ids)
    _, selected_matrices = select_camera_poses(camera_ids, camera_matrices, selected_ids)
    if rejected_ids:
        _, rejected_matrices = select_camera_poses(camera_ids, camera_matrices, rejected_ids)
    else:
        rejected_matrices = np.empty((0, 4, 4), dtype=np.float64)

    diagnostics_npz = output / "i4_1_1_orbit_alignment_coverage_diagnostics.npz"
    np.savez_compressed(
        diagnostics_npz,
        target_reconstruction=np.asarray(diagnosis["orbit_diagnosis"]["target_reconstruction"], dtype=np.float64),
        registered_camera_ids=np.asarray(camera_ids, dtype="U160"),
        registered_camera_matrices=np.asarray(camera_matrices, dtype=np.float64),
        aligned_camera_ids=np.asarray(aligned_ids, dtype="U160"),
        aligned_camera_matrices=np.asarray(aligned_matrices, dtype=np.float64),
        rejected_camera_ids=np.asarray(rejected_ids, dtype="U160"),
        rejected_camera_matrices=np.asarray(rejected_matrices, dtype=np.float64),
        selected_camera_ids=np.asarray(selected_ids, dtype="U160"),
        selected_camera_matrices=np.asarray(selected_matrices, dtype=np.float64),
        schema_version=np.asarray(I4_1_1_SCHEMA_VERSION, dtype=np.int32),
    )

    diagnosis["source"] = {
        "i4_dir": str(i4_dir),
        "i1_dir": str(i1_dir),
        "i2_dir": str(i2_dir),
        "extended_npz": str(extended_path),
    }
    diagnosis["artifacts"] = {
        "diagnostics_npz": str(diagnostics_npz.resolve()),
        "extended_npz": str(extended_path),
    }
    summary_path = _write_json(output / "i4_1_1_summary.json", diagnosis)

    counts = diagnosis["counts"]
    alignment = diagnosis["alignment_diagnosis"]
    orbit = diagnosis["orbit_diagnosis"]
    policy = diagnosis["policy_compatibility"]
    print("I4.1.1 Orbit Alignment & Coverage Diagnosis: DIAGNOSIS_READY")
    print("i2_failed_gates=" + ",".join(diagnosis["i2_policy_diagnosis"]["failed_gate_names"]))
    print(f"i2_geometry_contract_pass={diagnosis['i2_policy_diagnosis']['geometry_and_contract_gates_pass']}")
    print(
        f"views registered={counts['registered_views']}/{counts['input_views']} "
        f"candidates={counts['candidate_dense_views']} aligned={counts['alignment_accepted_views']} "
        f"selected={counts['selected_qualified_views']}"
    )
    print(
        f"alignment_fraction i2_candidates={counts['alignment_fraction_of_candidates_i2_semantics']:.3f} "
        f"i4_registered={counts['alignment_fraction_of_registered_i4_semantics']:.3f} "
        f"denominators_match={counts['alignment_denominator_semantics_match_for_this_scene']}"
    )
    print("alignment_rejected_orders=" + ",".join(str(v) for v in alignment["rejected_orders"]))
    print(f"alignment_rejection_reasons={alignment['rejection_reason_histogram']}")
    print(f"alignment_rejected_clusters={alignment['rejected_clusters']}")
    print(
        f"accepted_scale min={alignment['accepted_scale_min']:.6f} max={alignment['accepted_scale_max']:.6f} "
        f"max/min={alignment['accepted_scale_max_to_min_ratio']:.3f} "
        f"largest_adjacent_ratio={alignment['largest_adjacent_accepted_scale_ratio']:.3f}"
    )
    for name in ("registered", "aligned", "selected"):
        metrics = orbit[name]
        print(
            f"{name}_orbit coverage={metrics['azimuth_coverage_degrees']:.1f}deg "
            f"bins={metrics['occupied_azimuth_bins']}/{metrics['azimuth_bin_count']} "
            f"opposed={metrics['maximum_opposed_view_angle_degrees']:.1f}deg "
            f"facing={metrics['target_facing_fraction']:.3f} "
            f"standoff={metrics['standoff_p90_to_p10_ratio']:.3f}"
        )
    print(f"selected_orbit_coverage_pass={orbit['selected_orbit_coverage_pass']}")
    print(f"i4_capture_without_i2_prerequisite_pass={orbit['i4_capture_gates_without_i2_prerequisite_pass']}")
    print(f"failure_classification={policy['failure_classification']}")
    print(f"policy_reassessment_candidate={policy['policy_reassessment_candidate']} authorized=False threshold_relaxation=False")
    print("sfm_rerun=False da3_rerun=False fusion_rerun=False filter_policy_changed=False persistence_changed=False")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=False mesh_authorization=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
