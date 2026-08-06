#!/usr/bin/env python3
"""Create an editable Phase 7.5 known-size benchmark manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.validation import create_benchmark_template  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "phase7_5" / "known_size_benchmark.json",
    )
    args = parser.parse_args()
    path = create_benchmark_template(args.output)
    print(f"status=success output={path}")
    print("Review the provisional thresholds, add at least three real cases, then set thresholds_confirmed=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
