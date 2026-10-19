#!/usr/bin/env python3
"""I2.3 balanced cross-segment exact counterfactual campaign.

Consumes I2.2 exact single-view evidence, derives a tiny cross product of early
far-tail suspects and late spatial-envelope suspects, delegates exact refusion
and unchanged formal F filtering to the proven I2.2 engine, then evaluates
combined artifact reduction, envelope improvement, and Core-loss overlap.

Diagnostic only: source geometry/persistence is unchanged and no metric,
measurement, I2-closure, I3, or mesh authority is promoted automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_balanced_counterfactuals import (  # noqa: E402
    I23_BALANCED_SCHEMA_VERSION,
    balanced_pareto_front,
    derive_balanced_cross_segment_scenarios,
    evaluate_balanced_result,
    single_by_order,
)


_DEFAULT_I1 = Path("/tmp/depth3d_i1_offline_dense_campaign_recovered")
_DEFAULT_I2 = Path("/tmp/depth3d_i2_full_video_quality")
_DEFAULT_I21 = Path("/tmp/depth3d_i2_1_targeted_peripheral_diagnosis")
_DEFAULT_I22 = Path("/tmp/depth3d_i2_2_narrow_exact_counterfactual_refusion")
_DEFAULT_OUT = Path("/tmp/depth3d_i2_3_balanced_cross_segment_counterfactuals")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i1-dir", type=Path, default=_DEFAULT_I1)
    parser.add_argument("--i2-dir", type=Path, default=_DEFAULT_I2)
    parser.add_argument("--i2-1-dir", type=Path, default=_DEFAULT_I21)
    parser.add_argument("--i2-2-dir", type=Path, default=_DEFAULT_I22)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--max-early", type=int, default=3)
    parser.add_argument("--max-late", type=int, default=2)
    parser.add_argument("--max-scenarios", type=int, default=6)
    parser.add_argument("--max-voxels", type=int, default=1_500_000)
    args = parser.parse_args()
    if min(args.max_early, args.max_late, args.max_scenarios, args.max_voxels) <= 0:
        parser.error("candidate limits and --max-voxels must be positive")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I2.3 JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I2.3 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _run_exact_engine(
    *,
    i1_dir: Path,
    i2_dir: Path,
    i21_dir: Path,
    output_dir: Path,
    scenarios: tuple[Any, ...],
    max_voxels: int,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_i2_2_narrow_exact_counterfactual_refusion.py"),
        "--i1-dir", str(i1_dir),
        "--i2-dir", str(i2_dir),
        "--i2-1-dir", str(i21_dir),
        "--output-dir", str(output_dir),
        "--max-voxels", str(int(max_voxels)),
    ]
    for scenario in scenarios:
        a, b = scenario.excluded_orders
        command.extend(["--scenario", f"{scenario.name}={a},{b}"])
    process = subprocess.run(command, cwd=ROOT, check=False)
    if process.returncode != 0:
        raise ReconstructionValidationError(
            f"I2.3 exact cross-segment refusion engine failed with exit code {process.returncode}."
        )


def main() -> int:
    args = _args()
    i1_dir = args.i1_dir.expanduser().resolve()
    i2_dir = args.i2_dir.expanduser().resolve()
    i21_dir = args.i2_1_dir.expanduser().resolve()
    i22_dir = args.i2_2_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I2.3 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_i22 = _read_json(i22_dir / "i2_2_summary.json")
    i2 = _read_json(i2_dir / "i2_summary.json")
    baseline_e3 = _read_json(i1_dir / "04_qualified_fusion" / "e3_summary.json")
    selected_orders = tuple(int(v) for v in baseline_e3["selected_workspace_orders"])
    if str(source_i22.get("scene_id")) != str(i2.get("scene", {}).get("scene_id")):
        raise ReconstructionValidationError("I2.3 detected I2/I2.2 scene identity drift.")

    scenarios = derive_balanced_cross_segment_scenarios(
        source_i22,
        selected_orders,
        max_early=int(args.max_early),
        max_late=int(args.max_late),
        max_scenarios=int(args.max_scenarios),
    )
    singles = single_by_order(source_i22)
    exact_out = output / "exact_runs"

    print("I2.3 balanced cross-segment counterfactuals:")
    print(
        f"baseline_views={len(selected_orders)} baseline_full/robust="
        f"{float(source_i22['baseline']['extended_envelope']['full_to_robust_diagonal_ratio']):.3f}"
    )
    print("scenarios=" + ",".join(s.name for s in scenarios))
    print("strategy=early far-tail suspect + late envelope suspect | exact refusion + unchanged F")
    print("sfm_rerun=False da3_rerun=False source_geometry_changed=False", flush=True)

    _run_exact_engine(
        i1_dir=i1_dir,
        i2_dir=i2_dir,
        i21_dir=i21_dir,
        output_dir=exact_out,
        scenarios=scenarios,
        max_voxels=int(args.max_voxels),
    )
    exact_summary = _read_json(exact_out / "i2_2_summary.json")
    exact_by_name = {str(item["name"]): item for item in exact_summary["scenarios"]}

    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = exact_by_name.get(scenario.name)
        if result is None:
            raise ReconstructionValidationError(f"I2.3 exact engine omitted scenario {scenario.name}.")
        early = singles.get(scenario.early_order)
        late = singles.get(scenario.late_order)
        if early is None or late is None:
            raise ReconstructionValidationError(f"I2.3 cannot resolve component singles for {scenario.name}.")
        evaluation = evaluate_balanced_result(result, early, late, source_i22["baseline"])
        record = {
            **scenario.as_dict(),
            "exact_i2_2_result": result,
            "balanced_evaluation": evaluation,
        }
        records.append(record)
        print(
            f"{scenario.name}: core_loss={evaluation['combined_core_loss_fraction']:.3f} "
            f"peripheral_reduction={evaluation['combined_peripheral_reduction_fraction']:.3f} "
            f"far_reduction={evaluation['combined_far_peripheral_reduction_fraction']:.3f} "
            f"full/robust={evaluation['combined_full_to_robust_ratio']:.3f} "
            f"ratio_improvement={evaluation['combined_full_to_robust_ratio_improvement_fraction']:.3f} "
            f"core_overlap_savings={evaluation['core_overlap_savings_fraction']:.3f} "
            f"balanced={evaluation['balanced_candidate_for_manual_review']}"
        )

    ranked = sorted(records, key=lambda r: float(r["balanced_evaluation"]["balanced_score"]), reverse=True)
    pareto = balanced_pareto_front(records)
    balanced = [r["name"] for r in ranked if bool(r["balanced_evaluation"]["balanced_candidate_for_manual_review"])]
    priority = balanced[:3] if balanced else [name for name in pareto[:3]]

    summary = {
        "phase": "I2.3",
        "schema_version": I23_BALANCED_SCHEMA_VERSION,
        "status": "balanced_counterfactuals_ready",
        "scene_id": str(i2["scene"]["scene_id"]),
        "source_i1_dir": str(i1_dir),
        "source_i2_dir": str(i2_dir),
        "source_i2_1_dir": str(i21_dir),
        "source_i2_2_dir": str(i22_dir),
        "exact_runs_dir": str(exact_out.resolve()),
        "baseline": source_i22["baseline"],
        "candidate_generation": {
            "max_early": int(args.max_early),
            "max_late": int(args.max_late),
            "max_scenarios": int(args.max_scenarios),
            "scenarios": [scenario.as_dict() for scenario in scenarios],
            "principle": "cross one early far-tail-reduction single with one late envelope-improvement single",
        },
        "scenarios": records,
        "ranking": {
            "by_balanced_score": [r["name"] for r in ranked],
            "balanced_pareto_front": list(pareto),
            "balanced_candidates_for_manual_review": balanced,
            "manual_review_priority": priority,
            "diagnostic_only": True,
        },
        "authority": {
            "source_geometry_changed": False,
            "source_persistent_scene_changed": False,
            "counterfactual_geometry_generated": True,
            "sfm_rerun": False,
            "da3_rerun": False,
            "exact_fusion_rerun_per_scenario": True,
            "formal_f_filtering_rerun_per_scenario": True,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_closure": False,
            "authorizes_i3": False,
            "authorizes_mesh_reconstruction": False,
            "next_action": "Visually review only the top balanced/Pareto cross-segment scenarios before choosing any production correction.",
        },
    }
    path = _write_json(output / "i2_3_summary.json", summary)
    print("\nI2.3 balanced cross-segment counterfactuals: BALANCED_COUNTERFACTUALS_READY")
    print("ranking=" + ",".join(r["name"] for r in ranked))
    print("pareto_front=" + (",".join(pareto) if pareto else "none"))
    print("balanced_candidates=" + (",".join(balanced) if balanced else "none"))
    print("manual_review_priority=" + (",".join(priority) if priority else "none"))
    print("source_geometry_changed=False sfm_rerun=False da3_rerun=False mesh_authorization=False")
    print(f"summary={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
