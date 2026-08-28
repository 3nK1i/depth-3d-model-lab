#!/usr/bin/env python3
"""Real-runtime binding smoke for I4.3D2 RollingDepth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.reconstruction.rolling_depth_dense_provider import (
    probe_rolling_depth_environment,
    validate_i4_3d2_binding,
)


def _read(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected JSON object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3a-snapshot", required=True)
    parser.add_argument("--i4-3d1-snapshot", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    environment = probe_rolling_depth_environment(
        python_executable=Path(args.python),
        repository_root=Path(args.repository),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )
    snapshot = validate_i4_3d2_binding(
        i4_3a_snapshot=_read(Path(args.i4_3a_snapshot)),
        i4_3d1_snapshot=_read(Path(args.i4_3d1_snapshot)),
        environment=environment,
    )
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "i4_3d2_rolling_depth_binding_snapshot.json"
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    provider = snapshot["provider"]
    architecture = snapshot["architecture"]
    authority = snapshot["authority"]
    qualification = snapshot["qualification"]
    print("I4.3D2 RollingDepth production-provider binding: PASS")
    print(
        f"provider_id={provider['provider_id']} implementation={provider['implementation']} "
        f"family={provider['family']} qualification_state={provider['qualification_state']} "
        f"available={provider['available']}"
    )
    runtime = provider.get("metadata", {})
    print(
        f"source_revision={environment.source_revision} cuda_available={environment.cuda_available} "
        f"gpu={environment.device_names[0] if environment.device_names else '<none>'} "
        f"runtime_lock_valid={environment.runtime_lock_valid} modified_diffusers_ready={environment.modified_diffusers_ready}"
    )
    print(
        f"camera_pose_authority_owner={architecture['camera_pose_authority_owner']} "
        f"frame_binding={architecture['frame_binding']} temporal_mode={architecture['temporal_mode']} "
        f"permanent_global_view_cap={architecture['permanent_global_view_cap']}"
    )
    print(
        f"sequence_level_alignment={architecture['sequence_level_alignment']} "
        f"per_frame_alignment={architecture['per_frame_alignment']}"
    )
    print(
        f"backend_started={authority['backend_started']} provider_selected={authority['provider_selected']} "
        f"provider_promoted={authority['provider_promotion_performed']} "
        f"rolling_depth_pose_authority_promoted={authority['rolling_depth_pose_authority_promoted']}"
    )
    print(
        f"vda_binding_retained={qualification['video_depth_anything_binding_retained']} "
        f"temporal_closure_pending={qualification['temporal_provider_closure_pending_i4_3d3']}"
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
