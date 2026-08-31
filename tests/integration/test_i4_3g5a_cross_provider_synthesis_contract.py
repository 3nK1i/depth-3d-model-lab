from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_3g5a_cross_provider_synthesis.py"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_g5a_runner_hash_binds_all_four_frozen_predecessors(tmp_path: Path) -> None:
    inputs = {
        "g1": passing_g1(),
        "g2": passing_g2(),
        "g3": passing_g3(),
        "g4": passing_g4(),
    }
    paths = {name: tmp_path / f"{name}.json" for name in inputs}
    hashes = {name: _write(paths[name], payload) for name, payload in inputs.items()}
    out = tmp_path / "out"
    cmd = [
        sys.executable, str(RUNNER),
        "--i4-3g1-snapshot", str(paths["g1"]), "--i4-3g1-sha256", hashes["g1"],
        "--i4-3g2-closure", str(paths["g2"]), "--i4-3g2-sha256", hashes["g2"],
        "--i4-3g3c-closure", str(paths["g3"]), "--i4-3g3c-sha256", hashes["g3"],
        "--i4-3g4c-closure", str(paths["g4"]), "--i4-3g4c-sha256", hashes["g4"],
        "--output-dir", str(out),
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "I4.3G5A cross-provider evidence synthesis: PASS" in completed.stdout
    payload = json.loads((out / "i4_3g5a_cross_provider_synthesis.json").read_text())
    assert payload["decision"]["i4_3g5b_authorized"] is True
    assert payload["decision"]["i4_3g_closed"] is False
    assert payload["scope"]["provider_execution_started"] is False
    assert payload["scope"]["sfm_rerun"] is False
    for name, expected in hashes.items():
        key = {"g1": "i4_3g1_snapshot", "g2": "i4_3g2_closure", "g3": "i4_3g3c_closure", "g4": "i4_3g4c_closure"}[name]
        assert payload["evidence_bindings"][key]["sha256"] == expected


def test_g5a_runner_fails_closed_on_hash_drift_before_output(tmp_path: Path) -> None:
    paths = {}
    hashes = {}
    for name, payload in (("g1", passing_g1()), ("g2", passing_g2()), ("g3", passing_g3()), ("g4", passing_g4())):
        paths[name] = tmp_path / f"{name}.json"
        hashes[name] = _write(paths[name], payload)
    out = tmp_path / "out"
    completed = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3g1-snapshot", str(paths["g1"]), "--i4-3g1-sha256", "0" * 64,
        "--i4-3g2-closure", str(paths["g2"]), "--i4-3g2-sha256", hashes["g2"],
        "--i4-3g3c-closure", str(paths["g3"]), "--i4-3g3c-sha256", hashes["g3"],
        "--i4-3g4c-closure", str(paths["g4"]), "--i4-3g4c-sha256", hashes["g4"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "SHA256 mismatch" in completed.stderr
    assert not out.exists()
