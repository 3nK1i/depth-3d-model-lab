#!/usr/bin/env python3
"""Create and validate a deterministic Step 9 export using the development inspection bundle."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="/tmp/phase7_5_step9_export")
    args = parser.parse_args()
    root = ROOT
    output = Path(args.output).expanduser().resolve()
    shutil.rmtree(output, ignore_errors=True)
    with tempfile.TemporaryDirectory(prefix="step9_smoke_") as temp:
        bundle = Path(temp) / "bundle"
        image = np.zeros((72, 96, 3), dtype=np.uint8)
        image[..., 1] = 110
        input_path = Path(temp) / "frame.npy"
        np.save(input_path, image)
        config = Path(temp) / "depth.json"
        config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}))
        subprocess.run([
            sys.executable, str(root / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(input_path), "--output", str(bundle), "--depth-model", "development-mock", "--depth-config", str(config),
            "--perception-model", "development-perception-mock", "--max-objects", "2",
        ], cwd=root, check=True)
        summary, arrays = load_inspection_bundle(bundle)
        result = export_inspection_bundle(summary, arrays, InspectionExportOptions(output_dir=output))
    print(f"status=success objects={result.object_count} files={result.file_count} output={result.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
