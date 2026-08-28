#!/usr/bin/env python3
"""I4.3G3A explicit temporal-source/runtime binding preflight; starts no heavy backend."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.rolling_depth_dense_provider import probe_rolling_depth_environment  # noqa: E402
from core.reconstruction.temporal_provider_qualification import (  # noqa: E402
    I43G3A_SCENARIOS,
    TemporalScenarioBinding,
    build_i4_3g3a_preflight_snapshot,
    build_temporal_video_identity,
)
from core.reconstruction.video_depth_anything_dense_provider import probe_video_depth_anything_environment  # noqa: E402

SNAPSHOT_NAME = "i4_3g3a_temporal_preflight_snapshot.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3g1-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g2-closure", type=Path, required=True)
    parser.add_argument("--static-sweep-video", type=Path, required=True)
    parser.add_argument("--motion-video", type=Path, required=True)
    parser.add_argument("--low-texture-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vda-runtime-json", type=Path)
    parser.add_argument("--vda-project-root", type=Path, default=ROOT)
    parser.add_argument("--vda-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--vda-checkpoint", type=Path, default=Path("checkpoints/video-depth-anything/metric_video_depth_anything_vits.pth"))
    parser.add_argument("--rolling-runtime-json", type=Path)
    parser.add_argument("--rolling-python", type=Path)
    parser.add_argument("--rolling-repository", type=Path)
    parser.add_argument("--rolling-checkpoint", type=Path)
    return parser


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _vda_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.vda_runtime_json is not None:
        return _read_json(args.vda_runtime_json, label="VDA runtime probe")
    checkpoint = args.vda_checkpoint.expanduser()
    project = args.vda_project_root.expanduser().resolve()
    if not checkpoint.is_absolute():
        checkpoint = project / checkpoint
    return probe_video_depth_anything_environment(
        project,
        python_executable=args.vda_python.expanduser().resolve(),
        checkpoint=checkpoint.resolve(),
    ).as_dict()


def _rolling_runtime(args: argparse.Namespace) -> dict[str, Any]:
    if args.rolling_runtime_json is not None:
        return _read_json(args.rolling_runtime_json, label="RollingDepth runtime probe")
    if args.rolling_python is None or args.rolling_repository is None:
        raise ReconstructionValidationError(
            "I4.3G3A requires --rolling-runtime-json or both --rolling-python and --rolling-repository."
        )
    return probe_rolling_depth_environment(
        python_executable=args.rolling_python.expanduser().resolve(),
        repository_root=args.rolling_repository.expanduser().resolve(),
        checkpoint_path=args.rolling_checkpoint.expanduser().resolve() if args.rolling_checkpoint else None,
    ).as_dict()


def main() -> int:
    args = _parser().parse_args()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.3G3A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    g1_path = args.i4_3g1_snapshot.expanduser().resolve()
    g2_path = args.i4_3g2_closure.expanduser().resolve()
    g1 = _read_json(g1_path, label="I4.3G1 snapshot")
    g2 = _read_json(g2_path, label="I4.3G2 final closure")

    source_paths = {
        "g3_temporal_static_sweep": args.static_sweep_video,
        "g3_temporal_motion": args.motion_video,
        "g3_temporal_low_texture": args.low_texture_video,
    }
    bindings: list[TemporalScenarioBinding] = []
    source_dir = out / "source_bindings"
    for sid in I43G3A_SCENARIOS:
        identity = build_temporal_video_identity(source_paths[sid])
        manifest_payload = {
            "schema_version": 1,
            "phase": "I4.3G3A",
            "scenario_id": sid,
            "semantic_binding_inferred": False,
            "semantic_binding_method": "explicit_scenario_specific_cli_binding",
            "source_identity": identity.as_dict(),
            "temporal_order_semantics": "recorded_video_decode_sequence_index",
        }
        manifest_path = _write_json(source_dir / f"{sid}_source_manifest.json", manifest_payload)
        bindings.append(TemporalScenarioBinding(
            scenario_id=sid,
            source_identity=identity,
            manifest_path=str(manifest_path),
            manifest_sha256=_sha256(manifest_path),
            semantic_binding_method="explicit_scenario_specific_cli_binding",
        ))

    snapshot = build_i4_3g3a_preflight_snapshot(
        i4_3g1_snapshot=g1,
        i4_3g2_closure=g2,
        scenario_bindings=bindings,
        video_depth_anything_runtime=_vda_runtime(args),
        rolling_depth_runtime=_rolling_runtime(args),
    )
    snapshot["predecessor_evidence"] = {
        "i4_3g1_snapshot": str(g1_path), "i4_3g1_sha256": _sha256(g1_path),
        "i4_3g2_closure": str(g2_path), "i4_3g2_sha256": _sha256(g2_path),
    }
    target = _write_json(out / SNAPSHOT_NAME, snapshot)

    summary = snapshot["binding_summary"]
    runtime = snapshot["runtime"]
    decision = snapshot["decision"]
    authority = snapshot["authority"]
    print(f"I4.3G3A temporal source/runtime binding preflight: {'PASS' if snapshot['pass'] else 'NOT READY'}")
    print(f"bound_scenarios={summary['bound_scenario_count']}/3 all_g3_sources_bound={summary['all_g3_sources_bound']}")
    print(f"vda_ready={runtime['video_depth_anything_ready']} rolling_depth_ready={runtime['rolling_depth_ready']}")
    print(
        "g3b_authorized={authorized} sfm_started=False vda_started=False rolling_depth_started=False matched_view_started=False".format(
            authorized=decision["i4_3g3b_production_native_execution_authorized"]
        )
    )
    print(
        "camera_pose_authority_owner={owner} provider_selected={selected} provider_promoted={promoted} "
        "auto_policy_implemented={auto} hybrid_authorized={hybrid} metric_scale_claimed={metric} "
        "measurement_authoritative={measurement} certified_metrology={metrology} surface_mesh_generated={mesh}".format(
            owner=authority["camera_pose_authority_owner"], selected=authority["provider_selected"],
            promoted=authority["provider_promoted"], auto=authority["auto_policy_implemented"],
            hybrid=authority["hybrid_authorized"], metric=authority["metric_scale_claimed"],
            measurement=authority["measurement_authoritative"], metrology=authority["certified_metrology"],
            mesh=authority["surface_mesh_generated"],
        )
    )
    print(f"snapshot={target}")
    return 0 if snapshot["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
