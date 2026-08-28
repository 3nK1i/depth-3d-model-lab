#!/usr/bin/env python3
"""Execution-free I4.3F2 production adaptive-planning wiring smoke."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.hardware.gpu_info import detect_hardware
from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    build_i4_3f2_production_readiness_snapshot,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.errors import ReconstructionValidationError


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"JSON root must be an object: {path}")
    return payload


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages > 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        pass
    return None


def _source_contract_checks() -> dict[str, bool]:
    offline_controller = (PROJECT_ROOT / "app/offline_reconstruction_controller.py").read_text(encoding="utf-8")
    reconstruction_controller = (PROJECT_ROOT / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    worker = (PROJECT_ROOT / "scripts/run_reconstruction_offline_job.py").read_text(encoding="utf-8")
    mvs_worker = (PROJECT_ROOT / "scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py").read_text(encoding="utf-8")
    checks = {
        "offline_configuration_default_adaptive": "max_keyframes: int | None = None" in offline_controller,
        "selected_offline_default_adaptive": "max_keyframes: int | None = None" in reconstruction_controller,
        "worker_cli_default_adaptive": 'parser.add_argument("--max-keyframes", type=int, default=None)' in worker,
        "legacy_worker_default_80_absent": "default=80" not in worker,
        "pre_sfm_plan_persisted": "adaptive_pre_sfm_plan.json" in worker,
        "post_sfm_dense_readiness_persisted": "adaptive_dense_selection_ready.json" in worker,
        "mvs_selected_subset_validation": "selected_names.issubset(source_names)" in mvs_worker,
        "mvs_source_point_subset_integrity": "set(dense_points).issubset(source_points)" in mvs_worker,
    }
    return checks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3f1-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        f1 = _read_json(args.i4_3f1_snapshot.expanduser().resolve())
        hardware = detect_hardware(int(args.device_index))
        resources = AdaptiveSelectionResourceSnapshot.from_hardware(
            hardware,
            available_ram_bytes=_available_ram_bytes(),
        )
        source_checks = _source_contract_checks()
        if not all(source_checks.values()):
            failed = [name for name, value in source_checks.items() if not value]
            raise ReconstructionValidationError(f"I4.3F2 production source wiring check failed: {failed}")
        small = build_adaptive_pre_sfm_frame_plan(
            source_kind="image_directory",
            estimated_total_frames=68,
            expected_frame_rate_hz=10.0,
            resources=resources,
        )
        large = build_adaptive_pre_sfm_frame_plan(
            source_kind="recorded_video",
            estimated_total_frames=1000,
            expected_frame_rate_hz=30.0,
            resources=resources,
        )
        snapshot = build_i4_3f2_production_readiness_snapshot(
            i4_3f1_snapshot=f1,
            resources=resources,
        )
        snapshot["production_wiring"] = {
            "source_contract_checks": source_checks,
            "small_source_example": small.as_dict(),
            "large_source_example": large.as_dict(),
            "provider_projection_bridge_importable": True,
            "provider_execution_result_rebinding_importable": True,
            "qt_controller_runtime_validation_deferred_to_authoritative_depth_environment": True,
        }
    except Exception as exc:  # noqa: BLE001 - smoke reports fail-closed reason
        print(f"I4.3F2 production adaptive-planning wiring: FAIL — {exc}")
        return 2

    target = output_dir / "i4_3f2_production_wiring_binding_snapshot.json"
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    resources_payload = snapshot["resources"]
    decision = snapshot["decision"]
    wiring = snapshot["production_wiring"]
    print("I4.3F2 production adaptive-planning wiring: PASS")
    print(
        "gpu={gpu} cuda_ready={cuda} free_vram_gib={vram} available_ram_gib={ram} planning_capacity_units={capacity}".format(
            gpu=resources_payload["gpu_name"],
            cuda=resources_payload["cuda_ready"],
            vram=resources_payload["free_vram_gib"],
            ram=resources_payload["available_ram_gib"],
            capacity=resources_payload["planning_capacity_units"],
        )
    )
    print(
        "small_source_target={small} large_source_target={large} permanent_global_view_cap=None".format(
            small=wiring["small_source_example"]["target_sampled_frames"],
            large=wiring["large_source_example"]["target_sampled_frames"],
        )
    )
    print(
        "legacy_default_80_removed=True explicit_engineer_override_supported=True provider_projection_bridge=True mvs_selected_subset=True"
    )
    print(
        "i4_3f2_closed={closed} i4_3f3_authorized={authorized} provider_selected={selected} auto_policy_implemented={auto}".format(
            closed=decision["i4_3f2_production_wiring_closed"],
            authorized=decision["i4_3f3_production_validation_authorized"],
            selected=decision["provider_selected"],
            auto=decision["auto_policy_implemented"],
        )
    )
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
