import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_snapshot_cli_runs_mock(tmp_path):
    root = Path(__file__).resolve().parents[2]
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image_path = tmp_path / "frame.npy"
    config_path = tmp_path / "config.json"
    output = tmp_path / "out"
    np.save(image_path, image)
    config_path.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0}), encoding="utf-8")
    completed = subprocess.run([
        sys.executable, str(root / "scripts" / "run_snapshot_job.py"),
        "--model", "development-mock", "--config", str(config_path),
        "--input", str(image_path), "--output", str(output),
    ], cwd=root, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert (output / "result.npz").is_file()
    with np.load(output / "result.npz", allow_pickle=False) as data:
        assert int(data["point_stride"].reshape(-1)[0]) >= 1
    assert (output / "snapshot_manifest.json").is_file()
