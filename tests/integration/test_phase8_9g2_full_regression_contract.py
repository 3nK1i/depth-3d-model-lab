from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_g2_plan_only_cli_preserves_frozen_pre_release_gate_after_v080_transition(tmp_path):
    """The frozen G2 plan remains inspectable but must reject post-transition metadata."""
    out = tmp_path / "g2-plan"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_phase8_9g2_full_regression.py",
            "--non-strict-git",
            "--plan-only",
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
    assert "repository_preflight:version_0_7_5" in blockers
    assert blockers == {"repository_preflight:version_0_7_5"}

    pre = payload["repository_pre"]
    assert pre["checks"]["version_0_7_5"] is False
    for name, passed in pre["checks"].items():
        if name != "version_0_7_5":
            assert passed is True, f"{name} unexpectedly failed"

    release = pre["release"]
    assert release["core_version"] == "0.8.0"
    assert release["config_version"] == "0.8.0"

    assert payload["plan_only"] is True
    assert payload["execution_started"] is False
    assert [item["name"] for item in payload["matrix"]] == [
        "compileall",
        "unit",
        "ui",
        "integration",
        "stress",
        "phase8_8_protocol",
        "phase8_9_runner_chain",
    ]
    chain = payload["matrix"][-1]
    assert len(chain["runners"]) == 12
    assert chain["runners"][0].endswith("run_phase8_9a1_tests.sh")
    assert chain["runners"][-1].endswith("run_phase8_9f6_tests.sh")
    assert payload["heavy_real_training_or_inference_started"] is False
    assert payload["production_code_changed_by_8_9g2"] is False
