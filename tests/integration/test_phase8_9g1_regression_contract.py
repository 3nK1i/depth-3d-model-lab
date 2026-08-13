from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_g1_inventory_cli_preserves_frozen_pre_release_gate_after_v080_transition(tmp_path):
    """The frozen G1 CLI must truthfully refuse the post-transition 0.8.0 tree."""
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
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert proc.returncode == 1, proc.stdout
    assert payload["pass"] is False

    blockers = set(payload["blockers"])
    assert "release_version_still_0_7_5" in blockers
    # Before I3 only the frozen-version blocker is present.  After the
    # annotated v0.8.0 tag exists, the equally historical tag-absence gate may
    # also fail.  No unrelated G1 invariant may regress.
    assert blockers <= {"release_version_still_0_7_5", "v0_8_0_tag_absent"}

    checks = payload["checks"]
    assert checks["release_version_still_0_7_5"] is False
    for name, passed in checks.items():
        if name not in {"release_version_still_0_7_5", "v0_8_0_tag_absent"}:
            assert passed is True, f"{name} unexpectedly failed"

    release = payload["inventory"]["release"]
    assert release["core_version"] == "0.8.0"
    assert release["config_version"] == "0.8.0"

    assert payload["planned_g2"]["scope"] == "complete_current_tree_regression"
    assert payload["planned_g3"]["scope"] == "explicit_backward_compatibility_and_final_regression_gate"
    assert payload["truthful_scope"]["heavy_backend_started_by_8_9g1"] is False
