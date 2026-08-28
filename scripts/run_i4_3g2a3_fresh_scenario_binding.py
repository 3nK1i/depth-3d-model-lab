#!/usr/bin/env python3
"""I4.3G2A.3 fresh scenario binding + visual/diagnostic freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.mvs_da3_fresh_scenario_binding import (  # noqa: E402
    build_fresh_scenario_binding_snapshot,
)

SNAPSHOT_NAME = "i4_3g2a3_fresh_scenario_binding_snapshot.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-room-dir", type=Path, required=True)
    parser.add_argument("--low-texture-dir", type=Path)
    parser.add_argument("--weak-overlap-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-complete-room", action="store_true")
    parser.add_argument("--confirm-low-texture", action="store_true")
    parser.add_argument("--confirm-weak-overlap", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    out = args.output_dir.expanduser().resolve()
    snapshot = build_fresh_scenario_binding_snapshot(
        complete_room_dir=args.complete_room_dir,
        low_texture_dir=args.low_texture_dir,
        weak_overlap_dir=args.weak_overlap_dir,
        output_dir=out,
        confirm_complete_room=args.confirm_complete_room,
        confirm_low_texture=args.confirm_low_texture,
        confirm_weak_overlap=args.confirm_weak_overlap,
    )
    target = out / SNAPSHOT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    summary = snapshot["binding_summary"]
    print("I4.3G2A.3 fresh-scenario binding: PASS")
    print(
        f"ready_fresh_bindings={summary['ready_fresh_bindings']}/3 "
        f"all_fresh_bindings_ready={summary['all_fresh_bindings_ready']}"
    )
    print(f"missing_or_unqualified={summary['missing_or_unqualified']}")
    for binding in snapshot["fresh_bindings"]:
        diag = binding["diagnostics"]
        review = binding["visual_review"]
        print(
            f"scenario={binding['scenario_id']} status={binding['status']} "
            f"confirmed={binding['semantic_confirmation']['confirmed']} "
            f"objective_proxy_qualified={diag.get('objective_proxy_qualified')} "
            f"images={binding['source_identity']['image_count']} "
            f"contact_sheet={review['path']}"
        )
        if "component_ratios_to_reference" in diag:
            ratios = diag["component_ratios_to_reference"]
            print(
                f"  low_texture_methodology={diag['methodology']} "
                f"amendment={diag['methodology_amendment_id']}"
            )
            print(
                "  component_ratios "
                f"laplacian={ratios['laplacian_variance']:.6f} "
                f"contrast={ratios['contrast_stddev']:.6f} "
                f"edge={ratios['edge_density']:.6f} "
                f"margin={diag['qualification_margin_ratio']:.6f}"
            )
            print(
                f"  legacy_log_proxy_ratio={diag['median_ratio_to_reference']:.6f} "
                "authoritative=False"
            )
        elif "median_ratio_to_reference" in diag:
            print(f"  median_ratio_to_reference={diag['median_ratio_to_reference']:.6f}")
        if "minimum_good_matches" in diag:
            print(f"  minimum_good_matches={diag['minimum_good_matches']}")
        if "capture_spacing_factor" in diag:
            print(f"  capture_spacing_factor={diag['capture_spacing_factor']}")
    print("real_provider_execution_started=False sfm_started=False mvs_started=False da3_started=False")
    print(f"snapshot={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
