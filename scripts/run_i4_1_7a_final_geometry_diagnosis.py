#!/usr/bin/env python3
"""I4.1.7A lightweight final orbit-geometry diagnosis.

Reuses a completed DEGRADED I4.1.7 campaign. No SfM, DA3, dense A/B,
fusion, filtering, or persistence stage is rerun.
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
from core.reconstruction.offline_dense_i4_clean_static_orbit_diagnosis import (  # noqa: E402
    build_i4_1_7a_orbit_geometry_diagnosis,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import selected_plan_payload  # noqa: E402
from core.reconstruction.offline_dense_i4_decoupled_dense_ab import compare_dense_plan_to_baseline  # noqa: E402

_DEFAULT_I4_1_7 = Path("/tmp/depth3d_i4_1_7_box_clean_static_orbit_03_final")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_7a_final_geometry_diagnosis")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7-dir", type=Path, default=_DEFAULT_I4_1_7)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.7A cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.7A JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_orbit_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            target = np.asarray(payload["target_reconstruction"], dtype=np.float64)
            registered = np.asarray(payload["registered_camera_matrices"], dtype=np.float64)
            selected = np.asarray(payload["selected_camera_matrices"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.7A cannot read orbit diagnostics: {path}") from exc
    return target, registered, selected


def _retained_core_points(i1_dir: Path) -> np.ndarray | None:
    path = i1_dir / "05_filtering" / "f_filtered_extended.npz"
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            points = np.asarray(payload["points_reconstruction"], dtype=np.float64)
            classes = np.asarray(payload["class_code"], dtype=np.uint8).reshape(-1)
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.7A cannot read retained Core geometry: {path}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or classes.shape != (len(points),):
        raise ReconstructionValidationError("I4.1.7A retained Extended/Core arrays are misaligned.")
    core = points[classes == 3]
    return core if len(core) >= 64 else None


def _coverage_range(record: Mapping[str, Any]) -> str:
    return f"{float(record['azimuth_coverage_min_degrees']):.1f}-{float(record['azimuth_coverage_max_degrees']):.1f}"


def main() -> int:
    args = _args()
    i4_dir = args.i4_1_7_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.7A output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_summary_path = i4_dir / "i4_1_7_summary.json"
    diagnostics_path = i4_dir / "i4_1_7_orbit_diagnostics.npz"
    source = _read_json(source_summary_path)
    target, registered, selected = _load_orbit_diagnostics(diagnostics_path)
    core = _retained_core_points(i4_dir / "01_i1_baseline")

    diagnosis = build_i4_1_7a_orbit_geometry_diagnosis(
        source_i4_1_7_summary=source,
        registered_camera_matrices=registered,
        selected_camera_matrices=selected,
        provisional_target_reconstruction=target,
        retained_core_points=core,
    )

    b_summary = _read_json(i4_dir / "03_real_dense_ab" / "i4_1_6b_summary.json")
    decision = b_summary.get("decision")
    baseline = b_summary.get("baseline_global_only")
    if not isinstance(decision, Mapping) or not isinstance(baseline, Mapping):
        raise ReconstructionValidationError("I4.1.7A real dense A/B comparison evidence is incomplete.")
    plan_name = str(decision.get("recommended_plan", ""))
    plan = selected_plan_payload(b_summary, plan_name)
    comparison = compare_dense_plan_to_baseline(
        selected_views=int(plan.get("selected_qualified_views", 0)),
        pair_pass_fraction=float(plan.get("pair_pass_fraction")),
        baseline_selected_views=int(baseline.get("selected_qualified_views", 0)),
        baseline_pair_pass_fraction=float(baseline.get("pair_pass_fraction")),
    )
    diagnosis["dense_ab_comparison_replay"] = {
        "baseline_selected_qualified_views": int(baseline.get("selected_qualified_views", 0)),
        "baseline_pair_pass_fraction": float(baseline.get("pair_pass_fraction")),
        "plan_name": plan_name,
        "plan_selected_qualified_views": int(plan.get("selected_qualified_views", 0)),
        "plan_pair_pass_fraction": float(plan.get("pair_pass_fraction")),
        "plan_dense_gate_pass": bool(plan.get("dense_gate_pass")),
        **comparison,
        "real_dense_ab_validation_pass_after_saturation_fix": bool(
            plan.get("dense_gate_pass") and comparison["comparison_pass"]
        ),
        "production_binding_authorized": False,
    }
    diagnosis["source"] = {
        "i4_1_7_dir": str(i4_dir),
        "i4_1_7_summary": str(source_summary_path.resolve()),
        "orbit_diagnostics_npz": str(diagnostics_path.resolve()),
        "retained_core_available": core is not None,
    }
    summary = _write_json(output / "i4_1_7a_summary.json", diagnosis)

    selected_eval = diagnosis["camera_set_evaluations"]["selected"]
    current = selected_eval["targets"]["provisional_i4_1_7"]["current_metric"]
    trajectory = selected_eval["trajectory_centered"]
    diag = diagnosis["diagnosis"]

    print("I4.1.7A Final Orbit Geometry Diagnosis: DIAGNOSIS_READY")
    print(
        f"selected current azimuth={current['azimuth_coverage_degrees']:.1f}deg "
        f"bins={current['occupied_azimuth_bins']}/12 "
        f"opposed={current['maximum_opposed_view_angle_degrees']:.1f}deg"
    )
    print(
        f"selected trajectory_centered azimuth={trajectory['azimuth_coverage_degrees']:.1f}deg "
        f"bins={trajectory['occupied_azimuth_bins']}/12"
    )
    for target_name, evaluation in selected_eval["targets"].items():
        print(
            f"selected target={target_name} invariant_azimuth={_coverage_range(evaluation)}deg "
            f"bins={evaluation['occupied_azimuth_bins_min']}-{evaluation['occupied_azimuth_bins_max']}/12 "
            f"opposed={evaluation['opposed_view_angle_min_degrees']:.1f}-{evaluation['opposed_view_angle_max_degrees']:.1f}deg "
            f"robust_locked_geometry_pass={evaluation['robust_locked_orbit_geometry_pass']}"
        )
    replay = diagnosis["dense_ab_comparison_replay"]
    print(
        f"dense_ab_replay baseline_selected={replay['baseline_selected_qualified_views']} "
        f"baseline_pair={replay['baseline_pair_pass_fraction']:.3f} "
        f"plan_selected={replay['plan_selected_qualified_views']} "
        f"plan_pair={replay['plan_pair_pass_fraction']:.3f} "
        f"comparison_pass={replay['comparison_pass']} rule={replay['comparison_rule']}"
    )
    print(
        f"current_metric_reproduction_exact={diagnosis['current_i4_1_7_metric_reproduction_exact']} "
        f"classification={diag['classification']}"
    )
    print(
        f"robust_alternative_targets={','.join(diag['robust_alternative_targets']) or 'none'} "
        f"orbit_target_correction_candidate={diag['orbit_target_correction_candidate']} authorized=False"
    )
    print("sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False filtering_rerun=False persistence_rerun=False")
    print("threshold_relaxation=False metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"next={diagnosis['decision']['next_action']}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
