#!/usr/bin/env python3
"""Close I4.3D temporal-provider architecture from passing D1/D2 binding snapshots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.temporal_provider_closure import build_i4_3d3_temporal_provider_closure


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3d1-snapshot", required=True)
    parser.add_argument("--i4-3d2-snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"I4.3D3 output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i4_3d3_temporal_provider_closure(
        i4_3d1_snapshot=_read(Path(args.i4_3d1_snapshot).expanduser().resolve()),
        i4_3d2_snapshot=_read(Path(args.i4_3d2_snapshot).expanduser().resolve()),
    )
    path = out / "i4_3d3_temporal_provider_closure.json"
    path.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    contract = closure["temporal_provider_contract"]
    decision = closure["decision"]
    authority = closure["authority"]
    qualification = closure["qualification"]
    print("I4.3D3 temporal-provider architecture closure: PASS")
    print(
        f"family={contract['family']} implementations={','.join(contract['implementations'])} "
        f"frame_binding={contract['frame_binding']} camera_pose_authority_owner={contract['camera_pose_authority_owner']}"
    )
    print(
        f"i4_3d_closed={decision['i4_3d_temporal_provider_architecture_closed']} "
        f"i4_3e_authorized={decision['i4_3e_authorized']} provider_selected={decision['provider_selected']} "
        f"provider_promoted={decision['provider_promoted']} auto_policy_implemented={decision['auto_policy_implemented']}"
    )
    print(
        f"permanent_global_view_cap={contract['permanent_global_view_cap']} "
        f"adaptive_view_selection_deferred_to={contract['adaptive_view_selection_deferred_to']}"
    )
    print(
        f"temporal_ab_pending={qualification['temporal_cross_provider_comparison_pending_i4_3g_i5']} "
        f"universal_provider_winner_assumed={qualification['universal_provider_winner_assumed']}"
    )
    print(
        f"metric_scale_claimed={authority['metric_scale_claimed']} "
        f"measurement_authoritative={authority['measurement_authoritative']} "
        f"certified_metrology={authority['certified_metrology']} "
        f"surface_mesh_generated={authority['surface_mesh_generated']}"
    )
    print(f"snapshot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
