from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.cross_provider_evidence_synthesis import G5AEvidenceBinding, build_i4_3g5a_cross_provider_synthesis
from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_3g5b_final_g_closure.py"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain(tmp_path: Path):
    raw = {"g1": passing_g1(), "g2": passing_g2(), "g3": passing_g3(), "g4": passing_g4()}
    paths = {name: (tmp_path / f"{name}.json").resolve() for name in raw}
    hashes = {name: _write(paths[name], payload) for name, payload in raw.items()}
    g5a = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=raw["g1"], i4_3g2_closure=raw["g2"],
        i4_3g3c_closure=raw["g3"], i4_3g4c_closure=raw["g4"],
        evidence_bindings={
            {"g1": "i4_3g1_snapshot", "g2": "i4_3g2_closure", "g3": "i4_3g3c_closure", "g4": "i4_3g4c_closure"}[name]:
                G5AEvidenceBinding(path=str(paths[name]), sha256=hashes[name])
            for name in raw
        },
    )
    paths["g5a"] = (tmp_path / "g5a.json").resolve()
    hashes["g5a"] = _write(paths["g5a"], g5a)
    return paths, hashes, g5a


def _cmd(paths, hashes, out):
    return [
        sys.executable, str(RUNNER),
        "--i4-3g1-snapshot", str(paths["g1"]), "--i4-3g1-sha256", hashes["g1"],
        "--i4-3g2-closure", str(paths["g2"]), "--i4-3g2-sha256", hashes["g2"],
        "--i4-3g3c-closure", str(paths["g3"]), "--i4-3g3c-sha256", hashes["g3"],
        "--i4-3g4c-closure", str(paths["g4"]), "--i4-3g4c-sha256", hashes["g4"],
        "--i4-3g5a-synthesis", str(paths["g5a"]), "--i4-3g5a-sha256", hashes["g5a"],
        "--output-dir", str(out),
    ]


def test_g5b_runner_rebuilds_verified_chain_and_formally_closes_g(tmp_path: Path) -> None:
    paths, hashes, _ = _chain(tmp_path)
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "I4.3G cross-provider qualification: FORMALLY CLOSED" in completed.stdout
    payload = json.loads((out / "i4_3g5b_final_g_closure.json").read_text())
    assert payload["decision"]["i4_3g_closed"] is True
    assert payload["decision"]["i4_3h_authorized"] is True
    assert payload["closure_chain"]["g5a_deterministically_rebuilt_from_predecessors"] is True
    assert payload["scope"]["provider_execution_started"] is False
    assert len(payload["evidence_bindings"]) == 5


def test_g5b_runner_fails_if_frozen_g5a_no_longer_matches_predecessor_chain(tmp_path: Path) -> None:
    paths, hashes, g5a = _chain(tmp_path)
    g5a["next"] = "tampered but otherwise structurally valid"
    hashes["g5a"] = _write(paths["g5a"], g5a)
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "does not match a deterministic semantic rebuild" in completed.stderr
    assert not out.exists()


def test_g5b_runner_fails_closed_on_predecessor_hash_drift(tmp_path: Path) -> None:
    paths, hashes, _ = _chain(tmp_path)
    hashes["g3"] = "0" * 64
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "SHA256 mismatch" in completed.stderr
    assert not out.exists()


def test_g5b_runner_accepts_semantically_identical_event_order_after_json_reserialization(tmp_path: Path) -> None:
    paths, hashes, g5a = _chain(tmp_path)
    events = g5a["cross_family_synthesis"]["terminal_quality_events"]
    g5a["cross_family_synthesis"]["terminal_quality_events"] = list(reversed(events))
    hashes["g5a"] = _write(paths["g5a"], g5a)
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((out / "i4_3g5b_final_g_closure.json").read_text())
    assert payload["decision"]["i4_3g_closed"] is True
