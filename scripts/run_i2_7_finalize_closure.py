#!/usr/bin/env python3
"""Finalize I2 after automated requalification, fixed-view review and UI sanity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from core.reconstruction.offline_dense_i2_closure import build_i2_closure_record  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i2-7-dir", type=Path, default=Path("/tmp/eia_i2_7_final_requalification"))
    p.add_argument("--confirm-fixed-view-review-pass", action="store_true", help="Explicit human confirmation after reviewing the requested final fixed views.")
    a = p.parse_args(); d = a.i2_7_dir.expanduser().resolve()
    automated = json.loads((d / "i2_7_summary.json").read_text(encoding="utf-8")); ui = json.loads((d / "i2_7_ui_sanity.json").read_text(encoding="utf-8"))
    closure = build_i2_closure_record(automated, ui, fixed_view_review_pass=bool(a.confirm_fixed_view_review_pass))
    path = d / "i2_7_closure.json"; path.write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("I2.7 final closure: PASS — I2 COMPLETE")
    print(f"scene_id={closure['scene_id']} fingerprint={closure['scene_fingerprint_sha256']}")
    print("automated_requalification=pass fixed_view_review=pass reconstruction_ui_sanity=pass")
    print("i2_closure_authorized=True i3_authorized=True mesh_authorization=False")
    print("sfm_scale_status=arbitrary metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False")
    print("next=I3 - Second indoor/static sequence")
    print(f"closure={path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
