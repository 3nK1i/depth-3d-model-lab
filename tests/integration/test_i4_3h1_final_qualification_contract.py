from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.cross_provider_evidence_synthesis import G5AEvidenceBinding, build_i4_3g5a_cross_provider_synthesis
from core.reconstruction.cross_provider_qualification_closure import G5BClosureBinding, build_i4_3g5b_final_closure
from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4
from tests.i4_3h1_helpers import passing_regression_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_3h1_final_qualification.py"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chain(tmp_path: Path):
    raw = {"g1": passing_g1(), "g2": passing_g2(), "g3": passing_g3(), "g4": passing_g4()}
    paths = {name: (tmp_path / f"{name}.json").resolve() for name in raw}
    hashes = {name: _write(paths[name], payload) for name, payload in raw.items()}
    names = {"g1": "i4_3g1_snapshot", "g2": "i4_3g2_closure", "g3": "i4_3g3c_closure", "g4": "i4_3g4c_closure"}
    g5a = build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=raw["g1"], i4_3g2_closure=raw["g2"],
        i4_3g3c_closure=raw["g3"], i4_3g4c_closure=raw["g4"],
        evidence_bindings={
            names[name]: G5AEvidenceBinding(path=str(paths[name]), sha256=hashes[name])
            for name in raw
        },
    )
    paths["g5a"] = (tmp_path / "g5a.json").resolve(); hashes["g5a"] = _write(paths["g5a"], g5a)
    g5b_bindings = {
        names[name]: G5BClosureBinding(path=str(paths[name]), sha256=hashes[name])
        for name in raw
    }
    g5b_bindings["i4_3g5a_synthesis"] = G5BClosureBinding(path=str(paths["g5a"]), sha256=hashes["g5a"])
    g5b = build_i4_3g5b_final_closure(
        i4_3g5a_synthesis=g5a, evidence_bindings=g5b_bindings, chain_rebuild_verified=True
    )
    paths["g5b"] = (tmp_path / "g5b.json").resolve(); hashes["g5b"] = _write(paths["g5b"], g5b)
    paths["receipt"] = (tmp_path / "receipt.json").resolve(); hashes["receipt"] = _write(paths["receipt"], passing_regression_receipt())
    return paths, hashes


def _cmd(paths, hashes, out):
    return [
        sys.executable, str(RUNNER),
        "--i4-3g5b-closure", str(paths["g5b"]), "--i4-3g5b-sha256", hashes["g5b"],
        "--regression-receipt", str(paths["receipt"]), "--regression-receipt-sha256", hashes["receipt"],
        "--output-dir", str(out),
    ]


def test_h1_runner_reverifies_g_chain_and_writes_qualification_audit(tmp_path: Path) -> None:
    paths, hashes = _chain(tmp_path); out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "I4.3H1 final qualification / authority / provenance / regression audit: PASS" in completed.stdout
    payload = json.loads((out / "i4_3h1_final_qualification_audit.json").read_text())
    assert payload["decision"]["i4_3h1_qualification_complete"] is True
    assert payload["decision"]["i4_3h2_authorized"] is True
    assert payload["decision"]["i4_3_closed"] is False
    assert payload["closure_chain_audit"]["g5b_predecessor_sha256_bindings_reverified"] is True
    assert payload["regression_audit"]["predecessor_total_tests"] == 668
    assert payload["architecture_audit"]["provider_implementation_count"] == 10


def test_h1_runner_fails_closed_on_g5b_hash_drift(tmp_path: Path) -> None:
    paths, hashes = _chain(tmp_path); hashes["g5b"] = "0" * 64
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "SHA256 mismatch" in completed.stderr
    assert not out.exists()


def test_h1_runner_rehashes_g5b_predecessors_not_only_g5b_itself(tmp_path: Path) -> None:
    paths, hashes = _chain(tmp_path)
    paths["g3"].write_text(paths["g3"].read_text() + "\n", encoding="utf-8")
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "G5B predecessor i4_3g3c_closure SHA256 mismatch" in completed.stderr
    assert not out.exists()


def test_h1_runner_fails_closed_on_regression_receipt_hash_drift(tmp_path: Path) -> None:
    paths, hashes = _chain(tmp_path); hashes["receipt"] = "f" * 64
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "regression receipt SHA256 mismatch" in completed.stderr
    assert not out.exists()


def test_h1_runner_rejects_hash_valid_but_incomplete_regression_receipt(tmp_path: Path) -> None:
    paths, hashes = _chain(tmp_path)
    receipt = passing_regression_receipt(); receipt["counts"]["predecessor_integration_tests"] = 85
    hashes["receipt"] = _write(paths["receipt"], receipt)
    out = tmp_path / "out"
    completed = subprocess.run(_cmd(paths, hashes, out), cwd=ROOT, text=True, capture_output=True, check=False)
    assert completed.returncode == 2
    assert "predecessor_integration_tests contract drift" in completed.stderr
    assert not out.exists()
