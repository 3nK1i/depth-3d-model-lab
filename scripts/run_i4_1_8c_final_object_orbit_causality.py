#!/usr/bin/env python3
"""I4.1.8C final retained-Core target binding + 118-vs-40 orbit causality.

Lightweight evidence replay only.  No reconstruction stage or production
binding is rerun.  The isolated I4.1.8B object target is independently
reproduced, then the unchanged I4.1 orbit policy is evaluated on the complete
qualified decoupled path and the actual bound production path.
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
from core.reconstruction.offline_dense_i4_final_object_orbit_causality import (  # noqa: E402
    I4_1_8C_SCHEMA_VERSION,
    build_i4_1_8c_final_object_orbit_causality,
    apply_i4_1_8c_manual_closure,
)

_DEFAULT_I4_1_7B = Path("/tmp/depth3d_i4_1_7b_box_bounded_target_01")
_DEFAULT_I4_1_8 = Path("/tmp/depth3d_i4_1_8_box_production_binding_01")
_DEFAULT_I4_1_8B = Path("/tmp/depth3d_i4_1_8b_box_object_support_01")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_8c_final_object_orbit_causality")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7b-dir", type=Path, default=_DEFAULT_I4_1_7B)
    p.add_argument("--i4-1-8-dir", type=Path, default=_DEFAULT_I4_1_8)
    p.add_argument("--i4-1-8b-dir", type=Path, default=_DEFAULT_I4_1_8B)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--confirm-manual-orbit-review", action="store_true", help="Close I4.1 only if automated qualification passes and the user has manually reviewed the final orbit.")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.8C cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.8C JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_i417b(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            poses = np.asarray(payload["selected_camera_matrices"], dtype=np.float64)
            sources = np.asarray(payload["selected_source_sequence_indices"], dtype=np.int64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8C cannot read I4.1.7B pose evidence: {path}") from exc
    return poses, sources


def _load_i418b(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            poses = np.asarray(payload["production_selected_camera_matrices"], dtype=np.float64)
            sources = np.asarray(payload["production_selected_source_sequence_indices"], dtype=np.int64)
            isolated = np.asarray(payload["isolated_core_points_reconstruction"], dtype=np.float64)
            voxel = float(np.asarray(payload["voxel_size_reconstruction_units"]).item())
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.8C cannot read I4.1.8B object-support evidence: {path}") from exc
    return poses, sources, isolated, voxel


def main() -> int:
    args = _args()
    b7_dir = args.i4_1_7b_dir.expanduser().resolve()
    i8_dir = args.i4_1_8_dir.expanduser().resolve()
    b8_dir = args.i4_1_8b_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.8C output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    s7_path = b7_dir / "i4_1_7b_summary.json"
    s8_path = i8_dir / "i4_1_8_summary.json"
    sb_path = b8_dir / "i4_1_8b_summary.json"
    s7 = _read_json(s7_path)
    s8 = _read_json(s8_path)
    sb = _read_json(sb_path)

    a7 = s7.get("artifacts")
    ab = sb.get("artifacts")
    if not isinstance(a7, Mapping) or not isinstance(ab, Mapping):
        raise ReconstructionValidationError("I4.1.8C source artifact maps are missing.")
    p7 = Path(str(a7.get("target_correction_npz", ""))).expanduser().resolve()
    pb = Path(str(ab.get("object_support_diagnostics_npz", ""))).expanduser().resolve()
    q_poses, q_sources = _load_i417b(p7)
    b_poses, b_sources, isolated, voxel = _load_i418b(pb)

    result = build_i4_1_8c_final_object_orbit_causality(
        source_i4_1_7b_summary=s7,
        source_i4_1_8_summary=s8,
        source_i4_1_8b_summary=sb,
        qualified_camera_matrices=q_poses,
        qualified_source_sequence_indices=q_sources,
        production_bound_camera_matrices=b_poses,
        production_bound_source_sequence_indices=b_sources,
        isolated_core_points=isolated,
        retained_core_voxel_size=voxel,
    )
    result = apply_i4_1_8c_manual_closure(result, manual_review_confirmed=bool(args.confirm_manual_orbit_review))
    result["source_artifacts"] = {
        "i4_1_7b_summary": str(s7_path),
        "i4_1_8_summary": str(s8_path),
        "i4_1_8b_summary": str(sb_path),
        "qualified_path_npz": str(p7),
        "object_support_npz": str(pb),
    }
    diag = output / "i4_1_8c_orbit_causality.npz"
    np.savez_compressed(
        diag,
        final_object_target_reconstruction=np.asarray(result["target"]["target_reconstruction"], dtype=np.float64),
        qualified_camera_matrices=q_poses,
        qualified_source_sequence_indices=q_sources,
        production_bound_camera_matrices=b_poses,
        production_bound_source_sequence_indices=b_sources,
        isolated_core_points_reconstruction=isolated,
        voxel_size_reconstruction_units=np.asarray(voxel, dtype=np.float64),
        schema_version=np.asarray(I4_1_8C_SCHEMA_VERSION, dtype=np.int32),
    )
    result["artifacts"] = {"orbit_causality_npz": str(diag.resolve())}
    summary = _write_json(output / "i4_1_8c_summary.json", result)

    q = result["pose_sets"]["qualified_path"]["metrics"]
    b = result["pose_sets"]["production_bound"]["metrics"]
    cause = result["causality"]
    decision = result["decision"]
    print("I4.1.8C Final Object-Target Binding + Orbit Causality: " + result["status"].upper())
    print(
        f"classification={cause['classification']} target_mode={result['target']['mode']} "
        f"target_reproduced_exactly={result['target']['saved_target_reproduced_exactly']}"
    )
    print(
        f"qualified_path views={result['pose_sets']['qualified_path']['view_count']} "
        f"azimuth={q['azimuth_coverage_degrees']:.1f}deg bins={q['occupied_azimuth_bins']}/12 "
        f"opposed={q['maximum_opposed_view_angle_degrees']:.1f}deg facing={q['target_facing_fraction']:.3f} "
        f"standoff={q['standoff_p90_to_p10_ratio']:.3f} pass={cause['qualified_path_pass']}"
    )
    print(
        f"production_bound views={result['pose_sets']['production_bound']['view_count']} "
        f"azimuth={b['azimuth_coverage_degrees']:.1f}deg bins={b['occupied_azimuth_bins']}/12 "
        f"opposed={b['maximum_opposed_view_angle_degrees']:.1f}deg facing={b['target_facing_fraction']:.3f} "
        f"standoff={b['standoff_p90_to_p10_ratio']:.3f} pass={cause['production_bound_pass']}"
    )
    print(
        f"final_i4_1_automated_qualification_pass={decision['final_i4_1_automated_qualification_pass']} "
        f"recapture_required={decision['recapture_required']} "
        f"production_binding_correction_required={decision['production_binding_correction_required']} "
        f"manual_orbit_review_required={decision['manual_orbit_review_required']}"
    )
    print("camera_subset_search_performed=False orbit_metrics_used_to_select_target=False orbit_thresholds_changed=False")
    print("sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False filtering_rerun=False persistence_rerun=False production_binding_rerun=False")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"next={decision['next_action']}")
    print(f"summary={summary}")
    if result["status"] == "i4_1_closed":
        print("I4.1 FINAL CLOSURE: CLOSED; i4_2_detail_preservation_authorized=True")
    return 0 if decision["final_i4_1_automated_qualification_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
