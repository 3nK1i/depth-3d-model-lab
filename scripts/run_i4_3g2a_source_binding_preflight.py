#!/usr/bin/env python3
"""I4.3G2A read-only source/evidence binding discovery and preflight."""
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
from core.reconstruction.mvs_da3_qualification import build_g2a_preflight_snapshot  # noqa: E402

SNAPSHOT_NAME = "i4_3g2a_source_binding_preflight_snapshot.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3g1-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temple-ring-dir", type=Path, default=Path.home() / "Desktop/datasets/templeRing_official/templeRing")
    parser.add_argument("--i4-2f-closure", type=Path, default=Path.home() / "Desktop/i4_2_evidence_20260824/i4_2f_closure_attempt01/i4_2f_closure.json")
    parser.add_argument("--house-dir", type=Path, default=Path.home() / "Downloads/House")
    parser.add_argument("--house-historical-closure", type=Path, default=Path("/tmp/depth3d_i4_1_9a_house_mvs_positive_control/a3_visual_ab/i4_1_9a_a3_1_closure.json"))
    parser.add_argument("--complete-room-dir", type=Path)
    parser.add_argument("--low-texture-dir", type=Path)
    parser.add_argument("--weak-overlap-dir", type=Path)
    parser.add_argument(
        "--fresh-binding-snapshot",
        type=Path,
        help="Optional ready I4.3G2A.3 fresh-scenario binding snapshot; its bound source bytes are reverified.",
    )
    parser.add_argument("--discovery-root", type=Path, action="append", default=[])
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read I4.3G1 snapshot {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError("I4.3G1 snapshot must be a JSON object.")
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


def main() -> int:
    args = _parser().parse_args()
    g1_path = args.i4_3g1_snapshot.expanduser().resolve()
    g1 = _read_json(g1_path)
    fresh_binding_path = args.fresh_binding_snapshot.expanduser().resolve() if args.fresh_binding_snapshot else None
    fresh_binding = _read_json(fresh_binding_path) if fresh_binding_path is not None else None
    roots = list(args.discovery_root) or [
        Path.home() / "Desktop/datasets",
        Path.home() / "Downloads",
        Path.home() / "Desktop/phase8_3c_south_building",
    ]
    snapshot = build_g2a_preflight_snapshot(
        i4_3g1_snapshot=g1,
        temple_ring_dir=args.temple_ring_dir,
        i4_2f_closure=args.i4_2f_closure,
        house_dir=args.house_dir,
        house_historical_closure=args.house_historical_closure,
        complete_room_dir=args.complete_room_dir,
        low_texture_dir=args.low_texture_dir,
        weak_overlap_dir=args.weak_overlap_dir,
        fresh_binding_snapshot=fresh_binding,
        discovery_roots=roots,
    )
    snapshot["predecessor_evidence"] = {"i4_3g1_snapshot": str(g1_path), "i4_3g1_sha256": _sha256(g1_path)}
    if fresh_binding_path is not None:
        snapshot["predecessor_evidence"].update({
            "i4_3g2a3_fresh_binding_snapshot": str(fresh_binding_path),
            "i4_3g2a3_fresh_binding_sha256": _sha256(fresh_binding_path),
        })
    target = _write_json(args.output_dir.expanduser().resolve() / SNAPSHOT_NAME, snapshot)

    summary = snapshot["binding_summary"]
    decision = snapshot["decision"]
    authority = snapshot["authority"]
    print("I4.3G2A MVS/DA3 source-binding preflight: PASS")
    print(
        "fresh_bindings_ready={fresh}/4 historical_verified={hist} all_g2_sources_bound={all_bound}".format(
            fresh=summary["fresh_scenario_bindings_ready"],
            hist=summary["historical_binding_verified"],
            all_bound=summary["all_g2_sources_bound"],
        )
    )
    print(f"missing_explicit_fresh_bindings={summary['missing_explicit_fresh_bindings']}")
    print(f"discovered_candidate_directories={len(snapshot['discovered_candidate_directories'])}")
    for candidate in snapshot["discovered_candidate_directories"][:20]:
        print(f"candidate images={candidate['image_count']} path={candidate['directory']}")
    print(
        "g2b_authorized={authorized} real_provider_execution_started=False mvs_started=False da3_started=False sfm_started=False".format(
            authorized=decision["i4_3g2b_real_execution_authorized"]
        )
    )
    print(
        "camera_pose_authority_owner={owner} permanent_global_view_cap={cap} provider_selected={selected} "
        "auto_policy_implemented={auto} hybrid_authorized={hybrid} metric_scale_claimed={metric} "
        "measurement_authoritative={measurement} certified_metrology={metrology} surface_mesh_generated={mesh}".format(
            owner=authority["camera_pose_authority_owner"], cap=authority["permanent_global_view_cap"],
            selected=authority["provider_selected"], auto=authority["auto_policy_implemented"],
            hybrid=authority["hybrid_authorized"], metric=authority["metric_scale_claimed"],
            measurement=authority["measurement_authoritative"], metrology=authority["certified_metrology"],
            mesh=authority["surface_mesh_generated"],
        )
    )
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
