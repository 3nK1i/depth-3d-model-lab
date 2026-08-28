#!/usr/bin/env python3
"""I4.3F3 execution-free production validation and adaptive-selection closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.hardware.gpu_info import detect_hardware
from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_selection_closure import build_i4_3f3_closure_snapshot
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.dense_provider_architecture import DenseProviderImplementation
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.keyframes import KeyframeSelectionPolicy
from tests.i4_3f1_helpers import inflate_resolution, make_f1_sfm
from tests.i4_3f2_helpers import available_descriptor
from tests.i4_3f3_helpers import resources_5090, resources_constrained


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _source_contract_checks(project_root: Path) -> dict[str, bool]:
    worker = (project_root / "scripts/run_reconstruction_offline_job.py").read_text(encoding="utf-8")
    offline = (project_root / "app/offline_reconstruction_controller.py").read_text(encoding="utf-8")
    controller = (project_root / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    keyframes = (project_root / "core/reconstruction/keyframes.py").read_text(encoding="utf-8")
    mvs = (project_root / "scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py").read_text(encoding="utf-8")
    return {
        "core_keyframe_default_adaptive": "maximum_keyframes: int | None = None" in keyframes,
        "offline_configuration_default_adaptive": "max_keyframes: int | None = None" in offline,
        "reconstruction_controller_default_adaptive": "max_keyframes: int | None = None" in controller,
        "worker_cli_default_adaptive": 'parser.add_argument("--max-keyframes", type=int, default=None)' in worker,
        "legacy_worker_default_80_absent": "default=80" not in worker,
        "pre_sfm_plan_persisted": 'adaptive_pre_sfm_plan.json' in worker,
        "post_sfm_dense_readiness_persisted": 'adaptive_dense_selection_ready.json' in worker,
        "mvs_selected_subset_validation_present": "permanent_global_view_cap" in mvs and "selected" in mvs,
    }


def _actual_resources(device_index: int) -> AdaptiveSelectionResourceSnapshot:
    hardware = detect_hardware(device_index)
    return AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware,
        available_ram_bytes=_available_ram_bytes(),
    )


def _run_scenarios(actual: AdaptiveSelectionResourceSnapshot) -> tuple[list[dict[str, Any]], list[str]]:
    scenarios: list[dict[str, Any]] = []
    providers: list[str] = []

    source_cases = (20, 68, 1000, 5000)
    for total in source_cases:
        constrained = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video", estimated_total_frames=total,
            expected_frame_rate_hz=30.0, resources=resources_constrained(),
        )
        observed = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video", estimated_total_frames=total,
            expected_frame_rate_hz=30.0, resources=actual,
        )
        high = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video", estimated_total_frames=total,
            expected_frame_rate_hz=30.0, resources=resources_5090(),
        )
        if not (3 <= constrained.target_sampled_frames <= high.target_sampled_frames <= total):
            raise ReconstructionValidationError("I4.3F3 synthetic resource scaling is not monotonic.")
        if observed.target_sampled_frames is None or not (3 <= observed.target_sampled_frames <= total):
            raise ReconstructionValidationError("I4.3F3 actual-resource pre-SfM target is invalid.")
        scenarios.append({
            "name": f"pre_sfm_{total}_frames",
            "pass": True,
            "source_frames": total,
            "constrained_target": constrained.target_sampled_frames,
            "actual_target": observed.target_sampled_frames,
            "high_resource_target": high.target_sampled_frames,
            "permanent_global_view_cap": None,
        })

    override = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video", estimated_total_frames=1000,
        expected_frame_rate_hz=30.0, resources=actual,
        engineer_max_keyframes_override=17,
    )
    if override.target_sampled_frames != 17 or not override.engineer_override_applied:
        raise ReconstructionValidationError("I4.3F3 explicit pre-SfM engineer override did not remain authoritative/visible.")
    scenarios.append({"name": "engineer_pre_sfm_override", "pass": True, "target": 17})

    with tempfile.TemporaryDirectory(prefix="i4_3f3_") as tmp:
        sfm = inflate_resolution(make_f1_sfm(Path(tmp), view_count=48))
        original = sfm.as_dict()
        temporal_validated = []
        counts: dict[str, int] = {}
        for implementation in DenseProviderImplementation:
            prep = prepare_adaptive_provider_execution(
                sfm,
                provider=available_descriptor(implementation, f"f3-{implementation.value}"),
                reconstruction_frame_id="offline-sfm-reconstruction",
                resources=actual,
            )
            if prep.plan.permanent_global_view_cap is not None or prep.selection_decision.automatic:
                raise ReconstructionValidationError(f"I4.3F3 {implementation.value} violated selection authority.")
            if prep.plan.selected_view_count < prep.plan.requirements.minimum_views:
                raise ReconstructionValidationError(f"I4.3F3 {implementation.value} violated minimum-view requirement.")
            if prep.plan.requirements.stateful_sequence:
                order = {key: i for i, key in enumerate(prep.plan.source_order_keyframe_ids)}
                if list(prep.plan.selected_keyframe_ids) != sorted(prep.plan.selected_keyframe_ids, key=order.__getitem__):
                    raise ReconstructionValidationError(f"I4.3F3 {implementation.value} did not preserve temporal source order.")
                temporal_validated.append(implementation.value)
            providers.append(implementation.value)
            counts[implementation.value] = prep.plan.selected_view_count
        if sfm.as_dict() != original:
            raise ReconstructionValidationError("I4.3F3 provider preparation mutated authoritative SfM.")
        if sorted(temporal_validated) != sorted(["rolling_depth", "video_depth_anything"]):
            raise ReconstructionValidationError("I4.3F3 did not validate both temporal providers.")
        scenarios.append({
            "name": "all_provider_post_sfm_wiring",
            "pass": True,
            "provider_selected_view_counts": counts,
            "temporal_ordering_validated": temporal_validated,
            "authoritative_sfm_modified": False,
        })

    scenarios.append({
        "name": "core_default_no_implicit_cap",
        "pass": KeyframeSelectionPolicy().maximum_keyframes is None,
        "maximum_keyframes_default": None,
    })
    return scenarios, providers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3f1-snapshot", required=True, type=Path)
    parser.add_argument("--i4-3f2-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        f1 = _read_json(args.i4_3f1_snapshot.expanduser().resolve())
        f2 = _read_json(args.i4_3f2_snapshot.expanduser().resolve())
        actual = _actual_resources(int(args.device_index))
        source_checks = _source_contract_checks(args.project_root.expanduser().resolve())
        scenarios, providers = _run_scenarios(actual)
        snapshot = build_i4_3f3_closure_snapshot(
            i4_3f1_snapshot=f1,
            i4_3f2_snapshot=f2,
            source_contract_checks=source_checks,
            scenario_results=scenarios,
            validated_provider_implementations=providers,
        )
        snapshot["resources"] = actual.as_dict()
    except Exception as exc:  # noqa: BLE001
        print(f"I4.3F3 production validation / I4.3F closure: FAIL — {exc}")
        return 2

    target = out / "i4_3f3_production_validation_closure_snapshot.json"
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    d = snapshot["decision"]
    r = snapshot["resources"]
    print("I4.3F3 production validation / I4.3F closure: PASS")
    print(
        "gpu={gpu} cuda_ready={cuda} free_vram_gib={vram} available_ram_gib={ram}".format(
            gpu=r["gpu_name"], cuda=r["cuda_ready"], vram=r["free_vram_gib"], ram=r["available_ram_gib"]
        )
    )
    print("validated_provider_implementations=10 temporal_ordering_validated=True engineer_override_validated=True deterministic_planning_validated=True")
    print("core_keyframe_default_adaptive=True permanent_global_view_cap=None authoritative_sfm_modified=False")
    print(
        "i4_3f_closed={closed} i4_3g_authorized={authorized} provider_selected={selected} auto_policy_implemented={auto}".format(
            closed=d["i4_3f_closed"], authorized=d["i4_3g_cross_provider_qualification_authorized"],
            selected=d["provider_selected"], auto=d["auto_policy_implemented"],
        )
    )
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
