#!/usr/bin/env python3
"""I4.3C execution-free binding smoke for the isolated DA3 runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.da3 import probe_da3_environment
from core.reconstruction.da3_dense_provider import validate_i4_3c_binding
from core.reconstruction.errors import ReconstructionValidationError


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3a-snapshot", type=Path, required=True)
    parser.add_argument(
        "--da3-python",
        type=Path,
        default=Path.home() / "anaconda3/envs/eia-da3/bin/python3.11",
    )
    parser.add_argument(
        "--da3-repo",
        type=Path,
        default=Path.home() / "Desktop/eia-da3-backends/depth-anything-3",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{path} must contain a JSON object.")
    return payload


def main() -> int:
    args = _args()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.3C binding output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = _read_json(args.i4_3a_snapshot.expanduser().resolve())
    probe = probe_da3_environment(
        args.da3_python.expanduser().resolve(),
        args.da3_repo.expanduser().resolve(),
    )
    try:
        result = validate_i4_3c_binding(i4_3a_snapshot=snapshot, environment=probe)
    except ReconstructionValidationError as exc:
        print(f"I4.3C DA3 production-provider binding: FAIL — {exc}", file=sys.stderr)
        return 2
    path = output / "i4_3c_binding_snapshot.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provider = result["provider"]
    architecture = result["architecture"]
    authority = result["authority"]
    qualification = result["qualification"]
    print("I4.3C DA3 production-provider binding: PASS")
    print(
        f"provider_id={provider['provider_id']} implementation={provider['implementation']} "
        f"family={provider['family']} qualification_state={provider['qualification_state']} available={provider['available']}"
    )
    print(
        f"source_revision={probe.source_revision} gpu={','.join(probe.device_names)} "
        f"runtime_lock_valid={probe.runtime_lock_valid}"
    )
    print(
        f"camera_pose_authority_owner={architecture['camera_pose_authority_owner']} "
        f"frame_binding={architecture['frame_binding']} permanent_global_view_cap={architecture['permanent_global_view_cap']}"
    )
    print(
        f"backend_started={authority['backend_started']} provider_selected={authority['provider_selected']} "
        f"provider_promoted={authority['provider_promotion_performed']} da3_pose_authority_promoted={authority['da3_pose_authority_promoted']}"
    )
    print(
        f"mvs_vs_da3_pending_i4_3g={qualification['mvs_vs_da3_production_comparison_pending_i4_3g']} "
        f"universal_provider_winner_assumed={qualification['universal_provider_winner_assumed']}"
    )
    print(
        f"metric_scale_claimed={authority['metric_scale_claimed']} "
        f"measurement_authoritative={authority['measurement_authoritative']} "
        f"certified_metrology={authority['certified_metrology']} surface_mesh_generated={authority['surface_mesh_generated']}"
    )
    print(f"snapshot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
