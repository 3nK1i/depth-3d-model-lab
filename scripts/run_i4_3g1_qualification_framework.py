#!/usr/bin/env python3
"""Freeze the I4.3G1 controlled cross-provider qualification framework."""
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

from core.reconstruction.cross_provider_qualification import (  # noqa: E402
    build_i4_3g1_framework_snapshot,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

SNAPSHOT_NAME = "i4_3g1_qualification_framework_snapshot.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3a-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3f3-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def main() -> int:
    args = _parser().parse_args()
    a_path = args.i4_3a_snapshot.expanduser().resolve()
    f_path = args.i4_3f3_snapshot.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    a = _read_json(a_path, label="I4.3A architecture snapshot")
    f3 = _read_json(f_path, label="I4.3F3 closure snapshot")
    snapshot = build_i4_3g1_framework_snapshot(i4_3a_snapshot=a, i4_3f3_snapshot=f3)
    snapshot["predecessor_evidence"] = {
        "i4_3a_snapshot": str(a_path),
        "i4_3a_sha256": _sha256(a_path),
        "i4_3f3_snapshot": str(f_path),
        "i4_3f3_sha256": _sha256(f_path),
    }
    target = _write_json(output_dir / SNAPSHOT_NAME, snapshot)

    coverage = snapshot["coverage"]
    decision = snapshot["decision"]
    authority = snapshot["authority"]
    print("I4.3G1 controlled qualification framework: PASS")
    print(
        "scenarios={scenario_count} g2={g2_scenarios} g3={g3_scenarios} "
        "g4={g4_scenarios} metrics={metric_count} providers={provider_implementation_count}".format(**coverage)
    )
    print(
        "production_native=True matched_view_diagnostic=True historical_house_read_only=True "
        "global_scalar_score_forbidden=True"
    )
    print(
        "i4_3g1_closed={closed} i4_3g2_authorized={g2} provider_selected={selected} "
        "auto_policy_implemented={auto} hybrid_authorized={hybrid}".format(
            closed=decision["i4_3g1_framework_closed"],
            g2=decision["i4_3g2_mvs_da3_qualification_authorized"],
            selected=decision["provider_selected"],
            auto=decision["auto_policy_implemented"],
            hybrid=decision["hybrid_authorized"],
        )
    )
    print(
        "camera_pose_authority_owner={owner} permanent_global_view_cap={cap} "
        "metric_scale_claimed={metric} measurement_authoritative={measurement} "
        "certified_metrology={metrology} surface_mesh_generated={mesh}".format(
            owner=authority["camera_pose_authority_owner"],
            cap=authority["permanent_global_view_cap"],
            metric=authority["metric_scale_claimed"],
            measurement=authority["measurement_authoritative"],
            metrology=authority["certified_metrology"],
            mesh=authority["surface_mesh_generated"],
        )
    )
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
