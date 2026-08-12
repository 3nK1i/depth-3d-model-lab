from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_g1_inventory_cli_emits_complete_non_strict_report(tmp_path):
    out = tmp_path / "g1"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_phase8_9g1_regression_inventory.py",
            "--non-strict-git",
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["planned_g2"]["scope"] == "complete_current_tree_regression"
    assert payload["planned_g3"]["scope"] == "explicit_backward_compatibility_and_final_regression_gate"
    assert payload["truthful_scope"]["heavy_backend_started_by_8_9g1"] is False
