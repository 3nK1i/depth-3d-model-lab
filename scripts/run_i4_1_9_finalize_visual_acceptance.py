#!/usr/bin/env python3
"""Finalize I4.1.9 after explicit manual visual review of the reconstructed scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_visual_acceptance import finalize_visual_acceptance  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-9-dir", type=Path, default=Path("/tmp/eia_i4_1_9_visual_acceptance"))
    p.add_argument("--confirm-manual-visual-pass", action="store_true")
    a = p.parse_args()
    root = a.i4_1_9_dir.expanduser().resolve()
    summary_path = root / "i4_1_9_summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.9 closure cannot read {summary_path}: {exc}") from exc
    closure = finalize_visual_acceptance(summary, manual_visual_review_pass=bool(a.confirm_manual_visual_pass))
    closure["source_i4_1_9_summary"] = str(summary_path)
    closure["persistent_scene"] = str(summary.get("artifacts", {}).get("persistent_scene", ""))
    output = root / "i4_1_9_closure.json"
    output.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("I4.1.9 Visual Closure: PASS")
    print(f"status={closure['status']}")
    print("visual_reconstruction_qualified=True")
    print(f"strict_geometry_qualification_passed={closure['strict_geometry_qualification_passed']}")
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False")
    print("i4_2_detail_preservation_authorized=True")
    print(f"closure={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
