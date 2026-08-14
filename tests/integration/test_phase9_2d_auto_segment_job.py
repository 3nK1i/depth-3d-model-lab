from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_9_2d_isolated_job_exports_reviewable_mask_payload(tmp_path: Path) -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[10:80, 10:70] = 180
    image[35:110, 90:150] = 220
    input_path = tmp_path / "frame.npy"
    output = tmp_path / "result"
    np.save(input_path, image)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_auto_segment_job.py"),
            "--input", str(input_path),
            "--output", str(output),
            "--model", "development-perception-mock",
            "--profile", "efficient",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(output / "masks.npz", allow_pickle=False)
    assert manifest["model_id"] == "development-perception-mock"
    assert manifest["kept_proposals"] == 2
    assert len(manifest["proposals"]) == 2
    assert len(arrays.files) == 2
    assert all(arrays[key].shape == (120, 160) for key in arrays.files)
