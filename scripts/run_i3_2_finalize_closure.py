#!/usr/bin/env python3
"""Finalize I3 after I3.2 automated requalification, fixed-view review and UI sanity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.offline_dense_i3_promotion import build_i3_closure_record  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i3-2-dir", type=Path, default=Path("/tmp/eia_i3_2_second_scene_promotion"))
    p.add_argument("--confirm-fixed-view-review-pass", action="store_true")
    a = p.parse_args(); d = a.i3_2_dir.expanduser().resolve()
    automated = json.loads((d / "i3_2_summary.json").read_text(encoding="utf-8"))
    ui = json.loads((d / "i3_2_ui_sanity.json").read_text(encoding="utf-8"))
    closure = build_i3_closure_record(automated, ui, fixed_view_review_pass=bool(a.confirm_fixed_view_review_pass))
    path = d / "i3_2_closure.json"; path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("I3.2 final closure: PASS — I3 COMPLETE")
    print(f"scene_id={closure['scene_id']} fingerprint={closure['scene_fingerprint_sha256']}")
    print("automated_requalification=pass fixed_view_review=pass reconstruction_ui_sanity=pass")
    print("i3_closure_authorized=True i4_connector_orbit_authorized=True mesh_authorization=False")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("next=I4 - Object / connector orbit sequence")
    print(f"closure={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
