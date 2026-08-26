#!/usr/bin/env python3
"""Read-only real-evidence smoke for the I4.3B COLMAP MVS provider binding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_environment import probe_pycolmap_mvs_environment
from core.reconstruction.mvs_provider import validate_existing_i4_2_mvs_binding


def _read_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-2f-closure", type=Path, required=True)
    parser.add_argument("--i4-3a-snapshot", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"I4.3B binding output must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    closure = _read_json(args.i4_2f_closure.expanduser().resolve(), "I4.2F closure")
    architecture = _read_json(args.i4_3a_snapshot.expanduser().resolve(), "I4.3A snapshot")
    environment = probe_pycolmap_mvs_environment(str(args.python.expanduser().resolve()))
    try:
        snapshot = validate_existing_i4_2_mvs_binding(
            i4_2f_closure=closure,
            i4_3a_snapshot=architecture,
            environment=environment,
        )
    except ReconstructionValidationError as exc:
        print(f"I4.3B COLMAP MVS production-provider binding: FAIL — {exc}")
        return 2

    snapshot_path = output / "i4_3b_binding_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provider = snapshot["provider"]
    authority = snapshot["authority"]
    frozen = snapshot["frozen_mvs_evidence"]
    print("I4.3B COLMAP MVS production-provider binding: PASS")
    print(
        f"provider_id={provider['provider_id']} implementation={provider['implementation']} "
        f"qualification_state={provider['qualification_state']} available={provider['available']}"
    )
    print(
        f"frozen_mvs_points={frozen['fused_points']} backend_started={authority['backend_started']} "
        f"provider_selected={authority['provider_selected']} provider_promoted={authority['provider_promotion_performed']}"
    )
    print("camera_pose_authority_owner=pycolmap_sfm permanent_global_view_cap=None adaptive_view_selection_deferred_to=I4.3F")
    print("production_operating_envelope_pending_i4_3g=True universal_provider_winner_assumed=False")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print(f"snapshot={snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
