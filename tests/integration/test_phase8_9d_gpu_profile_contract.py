from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_phase8_9d_gpu_production_profile.py"


def test_d_policy_readiness_cli_is_explicitly_not_real_measurement(tmp_path: Path) -> None:
    out = tmp_path / "profile"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(out), "--policy-readiness-only"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["measurement_scope"] == "none_policy_readiness_only"
    assert payload["claims"] == {
        "rtx2060_measured": False,
        "rtx4090_measured": False,
        "rtx5090_measured": False,
    }
    assert payload["readiness_matrix"]["rtx4090"]["measured"] is False
    assert payload["readiness_matrix"]["rtx5090"]["measured"] is False
