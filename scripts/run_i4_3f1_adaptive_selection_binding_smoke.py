#!/usr/bin/env python3
"""Execution-free I4.3F1 adaptive-selection contract binding smoke."""
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
from core.reconstruction.adaptive_view_selection import (
    AdaptiveSelectionResourceSnapshot,
    build_i4_3f1_binding_snapshot,
)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="I4.3F1 execution-free adaptive frame/view-selection contract binding smoke."
    )
    parser.add_argument("--i4-3a-snapshot", required=True, type=Path)
    parser.add_argument("--i4-3e6-snapshot", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        i43a = _read_json(args.i4_3a_snapshot.expanduser().resolve())
        i43e6 = _read_json(args.i4_3e6_snapshot.expanduser().resolve())
        hardware = detect_hardware(int(args.device_index))
        resources = AdaptiveSelectionResourceSnapshot.from_hardware(
            hardware,
            available_ram_bytes=_available_ram_bytes(),
        )
        snapshot = build_i4_3f1_binding_snapshot(
            i4_3a_snapshot=i43a,
            i4_3e6_snapshot=i43e6,
            resources=resources,
        )
    except Exception as exc:  # noqa: BLE001 - smoke must report fail-closed reason
        print(f"I4.3F1 adaptive-selection contract binding: FAIL — {exc}")
        return 2

    target = output_dir / "i4_3f1_adaptive_selection_binding_snapshot.json"
    target.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    resources_payload = snapshot["resources"]
    decision = snapshot["decision"]
    selection = snapshot["selection"]
    print("I4.3F1 adaptive-selection contract binding: PASS")
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
        "adaptive_selection_required=True engineer_override_supported={override} permanent_global_view_cap={cap}".format(
            override=selection["engineer_override_supported"],
            cap=selection["permanent_global_view_cap"],
        )
    )
    print(
        "i4_3f1_closed={closed} i4_3f2_authorized={authorized} provider_selected={selected} auto_policy_implemented={auto}".format(
            closed=decision["i4_3f1_adaptive_selection_contract_closed"],
            authorized=decision["i4_3f2_production_wiring_authorized"],
            selected=decision["provider_selected"],
            auto=decision["auto_policy_implemented"],
        )
    )
    print("provider_requirement_count={}".format(len(snapshot["provider_requirements"])))
    print(
        "metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False"
    )
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
