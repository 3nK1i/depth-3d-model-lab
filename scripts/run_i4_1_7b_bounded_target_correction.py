#!/usr/bin/env python3
"""I4.1.7B bounded provisional orbit-target correction and lightweight replay.

Consumes a completed I4.1.7 campaign plus its I4.1.7A diagnosis.  No heavy
reconstruction stage is rerun.
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
from core.reconstruction.offline_dense_i4_clean_static_orbit_qualification import (  # noqa: E402
    selected_plan_payload,
)
from core.reconstruction.offline_dense_i4_clean_static_orbit_target_correction import (  # noqa: E402
    I4_1_7B_SCHEMA_VERSION,
    build_i4_1_7b_bounded_target_correction,
)

_DEFAULT_I4_1_7 = Path("/tmp/depth3d_i4_1_7_box_clean_static_orbit_03_final")
_DEFAULT_I4_1_7A = Path("/tmp/depth3d_i4_1_7a_box_final_geometry_diag_01")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_7b_bounded_target_correction")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7-dir", type=Path, default=_DEFAULT_I4_1_7)
    p.add_argument("--i4-1-7a-dir", type=Path, default=_DEFAULT_I4_1_7A)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.7B cannot read JSON evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.7B JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _load_orbit_diagnostics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            target = np.asarray(payload["target_reconstruction"], dtype=np.float64)
            registered = np.asarray(payload["registered_camera_matrices"], dtype=np.float64)
            selected = np.asarray(payload["selected_camera_matrices"], dtype=np.float64)
            sources = [int(v) for v in np.asarray(payload["selected_source_sequence_indices"], dtype=np.int64).tolist()]
            provenance = [str(v) for v in np.asarray(payload["selected_pose_provenance"]).tolist()]
    except (OSError, KeyError, ValueError) as exc:
        raise ReconstructionValidationError(f"I4.1.7B cannot read orbit diagnostics: {path}") from exc
    return target, registered, selected, sources, provenance


def main() -> int:
    args = _args()
    i4_dir = args.i4_1_7_dir.expanduser().resolve()
    a_dir = args.i4_1_7a_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.7B output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_summary_path = i4_dir / "i4_1_7_summary.json"
    a_summary_path = a_dir / "i4_1_7a_summary.json"
    diagnostics_path = i4_dir / "i4_1_7_orbit_diagnostics.npz"
    b_summary_path = i4_dir / "03_real_dense_ab" / "i4_1_6b_summary.json"
    source = _read_json(source_summary_path)
    a_summary = _read_json(a_summary_path)
    b_summary = _read_json(b_summary_path)
    old_target, registered, selected, selected_sources, selected_provenance = _load_orbit_diagnostics(diagnostics_path)

    decision = b_summary.get("decision")
    baseline = b_summary.get("baseline_global_only")
    if not isinstance(decision, Mapping) or not isinstance(baseline, Mapping):
        raise ReconstructionValidationError("I4.1.7B real dense A/B evidence is incomplete.")
    plan_name = str(source.get("qualification_plan") or decision.get("recommended_plan", ""))
    plan = selected_plan_payload(b_summary, plan_name)

    result = build_i4_1_7b_bounded_target_correction(
        source_i4_1_7_summary=source,
        source_i4_1_7a_summary=a_summary,
        registered_camera_matrices=registered,
        selected_camera_matrices=selected,
        plan_payload=plan,
        baseline_payload=baseline,
        selected_source_indices=selected_sources,
        selected_pose_provenance=selected_provenance,
    )
    result["source_artifacts"] = {
        "i4_1_7_dir": str(i4_dir),
        "i4_1_7_summary": str(source_summary_path),
        "i4_1_7a_summary": str(a_summary_path),
        "i4_1_6b_summary": str(b_summary_path),
        "orbit_diagnostics_npz": str(diagnostics_path),
    }
    corrected_target = np.asarray(result["target_correction"]["target_reconstruction"], dtype=np.float64)
    correction_npz = output / "i4_1_7b_target_correction.npz"
    np.savez_compressed(
        correction_npz,
        source_provisional_target_reconstruction=np.asarray(old_target, dtype=np.float64),
        corrected_provisional_target_reconstruction=corrected_target,
        registered_camera_matrices=registered,
        selected_camera_matrices=selected,
        selected_source_sequence_indices=np.asarray(selected_sources, dtype=np.int64),
        schema_version=np.asarray(I4_1_7B_SCHEMA_VERSION, dtype=np.int32),
    )
    result["artifacts"] = {"target_correction_npz": str(correction_npz.resolve())}
    summary_path = _write_json(output / "i4_1_7b_summary.json", result)

    assessment = result["assessment"]
    selected_orbit = assessment.get("selected_orbit") or {}
    decision_out = result["decision"]
    print("I4.1.7B Bounded Provisional Orbit-Target Correction: " + ("QUALIFIED" if result["status"] == "qualified_for_i4_1_8" else "DEGRADED"))
    print(
        f"preconditions_pass={result['correction_preconditions_pass']} "
        f"source_failed_gates={','.join(result['source_failed_gates']) or 'none'} "
        f"unrelated_source_failures={','.join(result['unrelated_source_gate_failures']) or 'none'}"
    )
    print(
        f"target_mode={assessment['target']['mode']} "
        f"selected_azimuth={selected_orbit.get('azimuth_coverage_degrees', float('nan')):.1f}deg "
        f"bins={selected_orbit.get('occupied_azimuth_bins', 0)}/12 "
        f"opposed={selected_orbit.get('maximum_opposed_view_angle_degrees', float('nan')):.1f}deg "
        f"facing={selected_orbit.get('target_facing_fraction', float('nan')):.3f} "
        f"standoff_p90_p10={selected_orbit.get('standoff_p90_to_p10_ratio', float('nan')):.3f}"
    )
    counts = assessment["counts"]
    print(
        f"dense_ab_requalified={assessment['gates']['real_dense_plan_improves_over_global_only']} "
        f"baseline_selected={counts['baseline_selected_qualified_views']} selected={counts['selected_qualified_views']} "
        f"baseline_pair={counts['baseline_pair_pass_fraction']:.3f} pair={counts['pair_pass_fraction']:.3f} "
        f"rule={counts['dense_baseline_comparison_rule']}"
    )
    print(
        f"i4_1_8_production_binding_authorized={decision_out['i4_1_8_production_binding_authorized']} "
        "production_sampling_integration_authorized=False i4_1_8_core_target_requalification_required=True"
    )
    print("sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False filtering_rerun=False persistence_rerun=False")
    print("orbit_thresholds_changed=False dense_consistency_threshold_relaxation_authorized=False")
    print("metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
