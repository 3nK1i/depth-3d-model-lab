#!/usr/bin/env python3
"""Dependency-light Phase 9.4H closure invariant smoke."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase9_4h_real_scene_qualification import validate_phase9_4_chain
from tests.phase9_4h_helpers import make_valid_phase9_4h_evidence


def main() -> int:
    evidence = make_valid_phase9_4h_evidence()
    passed, reasons = validate_phase9_4_chain(evidence)
    payload = {
        "phase": "9.4H",
        "status": "PASS" if passed else "FAIL",
        "reasons": list(reasons),
        "real_backend_started": False,
        "training_started": False,
        "render_backend_started": False,
        "canonical_geometry_authoritative": True,
        "gaussian_scene_role": "appearance_only",
        "gaussian_measurement_authoritative": False,
        "certified_metrology": False,
        "full_regression_required_for_closure": True,
        "real_gaussian_campaign_required_for_closure": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
