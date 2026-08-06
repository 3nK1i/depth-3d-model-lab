from __future__ import annotations

import subprocess
import sys
import json
import numpy as np
from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle


def test_spawned_mock_job_exports_without_rerunning_models(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    bundle = tmp_path / "bundle"
    image = np.zeros((72, 96, 3), dtype=np.uint8)
    image[..., 1] = 110
    input_path = tmp_path / "frame.npy"
    np.save(input_path, image)
    config = tmp_path / "depth.json"
    config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}))
    subprocess.run([
        sys.executable, str(root / "scripts" / "run_measurement_inspection_job.py"),
        "--input", str(input_path), "--output", str(bundle), "--depth-model", "development-mock", "--depth-config", str(config),
        "--perception-model", "development-perception-mock", "--max-objects", "2",
    ], cwd=root, check=True, capture_output=True, text=True)
    summary, arrays = load_inspection_bundle(bundle)
    result = export_inspection_bundle(summary, arrays, InspectionExportOptions(tmp_path / "export"))
    assert result.object_count >= 1
    assert result.file_count >= 10
    assert (result.export_dir / "report.pdf").stat().st_size > 1000
