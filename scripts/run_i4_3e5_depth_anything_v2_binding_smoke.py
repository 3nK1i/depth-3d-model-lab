#!/usr/bin/env python3
"""I4.3E5 execution-free Depth Anything V2 production-provider binding smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.depth_anything_v2_dense_provider import (
    probe_depth_anything_v2_environment,
    validate_i4_3e5_binding,
)
from core.reconstruction.errors import ReconstructionValidationError


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3a-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3d3-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3e1-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3e2-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3e3-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3e4-snapshot", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
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
        raise ReconstructionValidationError(
            f"I4.3E5 binding output directory must be empty or absent: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    probe = probe_depth_anything_v2_environment(
        project_root=args.project_root.expanduser().resolve(),
        python_executable=args.python.expanduser().resolve(),
        variant=args.variant,
        source_directory=args.source_dir.expanduser().resolve() if args.source_dir else None,
        checkpoint_path=args.checkpoint.expanduser().resolve() if args.checkpoint else None,
    )
    try:
        result = validate_i4_3e5_binding(
            i4_3a_snapshot=_read_json(args.i4_3a_snapshot.expanduser().resolve()),
            i4_3d3_snapshot=_read_json(args.i4_3d3_snapshot.expanduser().resolve()),
            i4_3e1_snapshot=_read_json(args.i4_3e1_snapshot.expanduser().resolve()),
            i4_3e2_snapshot=_read_json(args.i4_3e2_snapshot.expanduser().resolve()),
            i4_3e3_snapshot=_read_json(args.i4_3e3_snapshot.expanduser().resolve()),
            i4_3e4_snapshot=_read_json(args.i4_3e4_snapshot.expanduser().resolve()),
            environment=probe,
        )
    except ReconstructionValidationError as exc:
        print(
            f"I4.3E5 Depth Anything V2 production-provider binding: FAIL — {exc}",
            file=sys.stderr,
        )
        print(json.dumps(probe.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 2

    path = output / "i4_3e5_depth_anything_v2_binding_snapshot.json"
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    provider = result["provider"]
    architecture = result["architecture"]
    authority = result["authority"]
    qualification = result["qualification"]
    print("I4.3E5 Depth Anything V2 production-provider binding: PASS")
    print(
        f"provider_id={provider['provider_id']} implementation={provider['implementation']} "
        f"family={provider['family']} qualification_state={provider['qualification_state']} "
        f"available={provider['available']}"
    )
    print(
        f"source_revision={probe.source_revision} "
        f"source_checkout_matches_marker={probe.source_checkout_matches_marker} "
        f"installer_ref_commit_pinned={probe.installer_ref_commit_pinned} "
        f"variant={probe.variant} encoder={probe.encoder} "
        f"checkpoint_repository={probe.checkpoint_repository} "
        f"checkpoint_declared_revision={probe.checkpoint_declared_revision} "
        f"checkpoint_revision_commit_pinned={probe.checkpoint_revision_commit_pinned} "
        f"checkpoint_ready={probe.checkpoint_ready} checkpoint_size_bytes={probe.checkpoint_size_bytes} "
        f"checkpoint_sha256={probe.checkpoint_sha256} cuda_available={probe.cuda_available} "
        f"gpu={','.join(probe.device_names) or '<none>'} "
        f"official_runtime_importable={probe.official_runtime_importable} "
        f"official_runtime_source_matches_checkout={probe.official_runtime_source_matches_checkout}"
    )
    print(
        f"frame_binding={architecture['frame_binding']} "
        f"native_depth_scale={architecture['native_depth_scale']} "
        f"depth_representation={architecture['native_depth_representation']} "
        f"intrinsics={architecture['native_intrinsics_source']} "
        f"confidence={architecture['native_confidence_kind']} "
        f"per_view_alignment={architecture['per_view_alignment']} "
        f"permanent_global_view_cap={architecture['permanent_global_view_cap']}"
    )
    print(
        f"camera_pose_authority_owner={authority['camera_pose_authority_owner']} "
        f"relative_depth_metric_promoted={authority['relative_depth_promoted_to_metric_scale']} "
        f"model_intrinsics_canonical={authority['estimated_model_intrinsics_consumed_for_canonical_projection']} "
        f"adapter_points_canonical={authority['adapter_projected_points_consumed_for_canonical_geometry']}"
    )
    print(
        f"qualification_state={qualification['state']} "
        f"predecessor_e1_verified={qualification['predecessor_i4_3e1_verified']} "
        f"predecessor_e2_verified={qualification['predecessor_i4_3e2_verified']} "
        f"predecessor_e3_verified={qualification['predecessor_i4_3e3_verified']} "
        f"predecessor_e4_verified={qualification['predecessor_i4_3e4_verified']} "
        f"cross_provider_pending={qualification['cross_provider_qualification_pending_i4_3g']} "
        f"remaining_i4_3e={','.join(qualification['remaining_i4_3e_providers_pending'])}"
    )
    print(
        f"metric_scale_claimed={authority['metric_scale_claimed']} "
        f"measurement_authoritative={authority['measurement_authoritative']} "
        f"certified_metrology={authority['certified_metrology']} "
        f"surface_mesh_generated={authority['surface_mesh_generated']}"
    )
    print(f"license_status={provider['metadata']['license_status']}")
    print(f"snapshot={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
