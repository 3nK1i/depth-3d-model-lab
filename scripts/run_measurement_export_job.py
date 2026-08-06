#!/usr/bin/env python3
"""Isolated Step 9 inspection export job."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle


def _event(stage: str, progress: int, message: str, **extra) -> None:
    print("EVENT " + json.dumps({"stage": stage, "progress": progress, "message": message, **extra}, separators=(",", ":")), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--observation-id", action="append", default=[])
    parser.add_argument("--locked-id", action="append", default=[])
    args = parser.parse_args()
    try:
        _event("export_loading", 10, "Loading the approved inspection bundle")
        summary, arrays = load_inspection_bundle(args.bundle)
        _event("export_writing", 35, "Writing images, masks, geometry and measurements")
        result = export_inspection_bundle(
            summary,
            arrays,
            InspectionExportOptions(
                output_dir=Path(args.output),
                observation_ids=tuple(args.observation_id),
                locked_ids=tuple(args.locked_id),
            ),
        )
        _event("export_complete", 100, "Professional inspection export completed", object_count=result.object_count, file_count=result.file_count)
        print(f"RESULT {result.export_dir}", flush=True)
        return 0
    except Exception as exc:
        _event("export_failed", 100, str(exc))
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
