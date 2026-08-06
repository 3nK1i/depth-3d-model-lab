from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from core.inspection import load_inspection_bundle

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_measurement_inspection_job.py"


def _run(*args: str):
    completed = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT, text=True, capture_output=True, check=False, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


def test_mock_job_manual_revision_and_safe_automatic_reset(tmp_path: Path):
    height, width = 72, 96
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.linspace(20, 180, width, dtype=np.uint8)
    image[..., 1] = 90
    input_path = tmp_path / "frame.npy"
    np.save(input_path, image)
    config = tmp_path / "depth.json"
    config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}))

    initial = tmp_path / "initial"
    _run(
        "--input", str(input_path), "--output", str(initial),
        "--depth-model", "development-mock", "--depth-config", str(config),
        "--perception-model", "development-perception-mock", "--max-objects", "1",
    )
    summary, arrays = load_inspection_bundle(initial)
    assert summary.objects
    obj = summary.objects[0]
    assert arrays[obj.filtered_points_key].shape[1] == 3
    automatic_mask = arrays[obj.mask_key].copy()

    edited_mask = automatic_mask.copy()
    edited_mask[:5, :5] = True
    mask_path = tmp_path / "manual.npy"
    np.save(mask_path, edited_mask)
    manual = tmp_path / "manual"
    _run(
        "--output", str(manual), "--base-state", str(summary.state_path),
        "--manual-mask", str(mask_path), "--observation-id", obj.observation_id,
        "--manual-revision", "3", "--box-method", "camera_aligned",
    )
    manual_summary, manual_arrays = load_inspection_bundle(manual)
    manual_obj = manual_summary.objects[0]
    assert manual_obj.mask_source == "manual_correction"
    assert manual_obj.mask_revision == 3
    assert manual_obj.approval_status == "approved"
    assert np.array_equal(manual_arrays[manual_obj.mask_key], edited_mask)

    reset = tmp_path / "reset"
    _run(
        "--output", str(reset), "--base-state", str(manual_summary.state_path),
        "--reset-manual", "--box-method", "auto",
    )
    reset_summary, reset_arrays = load_inspection_bundle(reset)
    reset_obj = reset_summary.objects[0]
    assert reset_obj.mask_source != "manual_correction"
    assert np.array_equal(reset_arrays[reset_obj.mask_key], automatic_mask)
