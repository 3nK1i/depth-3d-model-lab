from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.i4_3_final_qualification import audit_i4_3_provider_architecture
from tests.i4_3h2_helpers import passing_h1
from tests.i4_3h1_helpers import passing_g5b, passing_regression_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_3h2_final_closure.py"


def write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize_h1_chain(tmp_path: Path) -> tuple[Path, str]:
    h1 = passing_h1()

    predecessor_dir = tmp_path / "pred"
    predecessor_dir.mkdir()
    predecessor_bindings = {}
    for index, key in enumerate(h1["g5b_predecessor_bindings"]):
        pred = predecessor_dir / f"{index}_{key}.json"
        pred.write_text("{}\n", encoding="utf-8")
        digest = hashlib.sha256(pred.read_bytes()).hexdigest()
        predecessor_bindings[key] = {"path": str(pred), "sha256": digest}
        h1["g5b_predecessor_bindings"][key] = dict(predecessor_bindings[key])

    g5b = passing_g5b()
    g5b["evidence_bindings"] = {key: dict(value) for key, value in predecessor_bindings.items()}
    g5b_path = tmp_path / "g5b.json"
    g5b_sha = write_json(g5b_path, g5b)

    receipt = passing_regression_receipt(focused_tests=40)
    receipt_path = tmp_path / "regression.json"
    receipt_sha = write_json(receipt_path, receipt)

    h1["evidence_bindings"]["i4_3g5b_closure"] = {"path": str(g5b_path), "sha256": g5b_sha}
    h1["evidence_bindings"]["i4_3h1_regression_receipt"] = {
        "path": str(receipt_path),
        "sha256": receipt_sha,
    }
    h1_path = tmp_path / "h1.json"
    h1_sha = write_json(h1_path, h1)
    return h1_path, h1_sha


def run_runner(tmp_path: Path, h1_path: Path, h1_sha: str) -> subprocess.CompletedProcess[str]:
    out = tmp_path / "out"
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--i4-3h1-audit", str(h1_path),
            "--i4-3h1-sha256", h1_sha,
            "--output-dir", str(out),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_h2_runner_rehashes_h1_chain_and_writes_formal_closure(tmp_path: Path) -> None:
    h1_path, h1_sha = materialize_h1_chain(tmp_path)
    result = run_runner(tmp_path, h1_path, h1_sha)
    assert result.returncode == 0, result.stderr
    payload = json.loads((tmp_path / "out" / "i4_3h2_final_i4_3_closure.json").read_text())
    assert payload["decision"]["i4_3_closed"] is True
    assert payload["decision"]["i4_4_authorized"] is True
    assert payload["closure_chain"]["current_repository_architecture_matches_h1"] is True


def test_h2_runner_fails_closed_on_h1_hash_drift(tmp_path: Path) -> None:
    h1_path, _ = materialize_h1_chain(tmp_path)
    result = run_runner(tmp_path, h1_path, "0" * 64)
    assert result.returncode == 2
    assert "SHA256 mismatch" in result.stderr


def test_h2_runner_rehashes_h1_bound_g5b_and_regression_receipt(tmp_path: Path) -> None:
    h1_path, h1_sha = materialize_h1_chain(tmp_path)
    h1 = json.loads(h1_path.read_text())
    Path(h1["evidence_bindings"]["i4_3h1_regression_receipt"]["path"]).write_text("changed\n")
    result = run_runner(tmp_path, h1_path, h1_sha)
    assert result.returncode == 2
    assert "H1 bound artifact" in result.stderr


def test_h2_runner_rehashes_g5b_predecessor_bindings_recorded_by_h1(tmp_path: Path) -> None:
    h1_path, h1_sha = materialize_h1_chain(tmp_path)
    h1 = json.loads(h1_path.read_text())
    first = next(iter(h1["g5b_predecessor_bindings"].values()))
    Path(first["path"]).write_text("changed\n")
    result = run_runner(tmp_path, h1_path, h1_sha)
    assert result.returncode == 2
    assert "G5B predecessor" in result.stderr


def test_h2_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    h1_path, h1_sha = materialize_h1_chain(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    result = run_runner(tmp_path, h1_path, h1_sha)
    assert result.returncode == 2
    assert "output directory" in result.stderr


def test_h2_architecture_probe_is_execution_free() -> None:
    audit = audit_i4_3_provider_architecture()
    assert audit["provider_implementation_count"] == 10
    assert all(
        item["backend_runtime_probe_started"] is False
        for item in audit["provider_inventory"].values()
    )


def test_h2_runner_help_probe() -> None:
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--i4-3h1-audit" in result.stdout
