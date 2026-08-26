#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_provider_architecture import (
    build_i4_3a_architecture_snapshot,
    read_i4_2f_closure,
    write_i4_3a_architecture_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="I4.3A execution-free dense-provider architecture gate.")
    parser.add_argument("--i4-2f-closure", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    closure_path = Path(args.i4_2f_closure).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    closure = read_i4_2f_closure(closure_path)
    snapshot = build_i4_3a_architecture_snapshot(i4_2f_closure=closure)
    snapshot_path = write_i4_3a_architecture_snapshot(output_dir / "i4_3a_architecture_snapshot.json", snapshot)

    architecture = snapshot["architecture"]
    progress = snapshot["progressive_preview_contract"]
    qualification = snapshot["qualification_plan"]
    authority = snapshot["authority"]

    print("I4.3A Dense Provider Architecture Foundation v2: PASS")
    print("selection_modes=" + ",".join(architecture["selection_modes"]))
    print("provider_families=" + ",".join(architecture["provider_families"]))
    print("provider_implementations=" + ",".join(architecture["provider_implementations"]))
    print(
        "auto_policy_only="
        + str(architecture["auto_is_selection_policy_only"])
        + " permanent_global_view_cap="
        + str(architecture["permanent_global_view_cap"])
    )
    print(
        "authoritative_camera_frame_owner="
        + architecture["authoritative_camera_frame_owner"]
        + " single_authoritative_sfm_frame_required="
        + str(architecture["single_authoritative_sfm_frame_required"])
    )
    print(
        "progressive_preview_required_for_v1_0="
        + str(progress["required_for_v1_0"])
        + " provisional_geometry_authoritative="
        + str(progress["provisional_geometry_authoritative"])
    )
    print(
        "complete_room_mvs_vs_da3_required="
        + str(qualification["complete_room_mvs_vs_da3_required"])
        + " hybrid_implementation_authorized="
        + str(qualification["hybrid_implementation_authorized"])
    )
    print(
        "backend_started="
        + str(authority["backend_started"])
        + " provider_selected="
        + str(authority["provider_selected"])
        + " provider_promoted="
        + str(authority["provider_promoted"])
    )
    print(
        "metric_scale_claimed="
        + str(authority["metric_scale_claimed"])
        + " measurement_authoritative="
        + str(authority["measurement_authoritative"])
        + " certified_metrology="
        + str(authority["certified_metrology"])
        + " surface_mesh_generated="
        + str(authority["surface_mesh_generated"])
    )
    print(f"snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
