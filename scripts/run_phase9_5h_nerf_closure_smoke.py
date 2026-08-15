#!/usr/bin/env python3
"""Dependency-light Phase 9.5H closure invariant smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase9_5h_real_scene_qualification import validate_phase9_5_chain
from tests.phase9_5h_helpers import make_valid_phase9_5h_evidence


def main() -> int:
    evidence = make_valid_phase9_5h_evidence()
    passed, reasons = validate_phase9_5_chain(evidence)
    payload = {
        "phase": "9.5H",
        "status": "PASS" if passed else "FAIL",
        "reasons": list(reasons),
        "real_backend_started": False,
        "training_started": False,
        "rendering_started": False,
        "canonical_geometry_authoritative": True,
        "nerf_scene_role": "appearance_only",
        "nerf_measurement_authoritative": False,
        "certified_metrology": False,
        "quality_characterization_required_for_closure": True,
        "full_regression_required_for_closure": True,
        "real_nerf_campaign_required_for_closure": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
