#!/usr/bin/env python3
"""Print the Phase 5 hardware snapshot and resolved execution profiles."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware import ComputeProfile, detect_hardware, resolve_profile


def main() -> None:
    hardware = detect_hardware()
    automatic = resolve_profile(ComputeProfile.AUTO, hardware)

    print("=== Hardware ===")
    print(hardware.details)
    print()
    print("=== Automatic profile ===")
    print(f"Requested : {automatic.requested.display_name}")
    print(f"Effective : {automatic.effective.display_name}")
    print(f"Device    : {automatic.device}")
    print(f"Precision : {automatic.precision.display_name}")
    print(f"Model     : {automatic.spec.model_repo}")
    print(f"Tokens    : {automatic.spec.num_tokens}")
    print(f"Points    : {automatic.spec.point_budget:,}")
    if automatic.warning:
        print(f"Warning   : {automatic.warning}")


if __name__ == "__main__":
    main()