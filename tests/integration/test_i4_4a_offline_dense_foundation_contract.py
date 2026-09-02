from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tests.i4_4a_helpers import passing_h2, passing_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_4a_focused_regression.py"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path, *, h2_sha: str | None = None, receipt_sha: str | None = None):
    h2 = tmp_path / "h2.json"; receipt = tmp_path / "receipt.json"; out = tmp_path / "out"
    real_h2_sha = _write(h2, passing_h2()); real_receipt_sha = _write(receipt, passing_receipt())
    return subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", h2_sha or real_h2_sha,
        "--regression-receipt", str(receipt), "--regression-receipt-sha256", receipt_sha or real_receipt_sha,
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False), out


def test_i44a_runner_hash_binds_h2_and_regression_receipt_and_writes_result(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "i4_4a_focused_regression.json").read_text())
    assert payload["pass"] is True
    assert payload["decision"]["i4_4b_authorized"] is True


def test_i44a_runner_fails_closed_on_h2_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, h2_sha="0" * 64)
    assert result.returncode == 2
    assert "I4.3H2 closure SHA256 mismatch" in result.stderr


def test_i44a_runner_fails_closed_on_receipt_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, receipt_sha="0" * 64)
    assert result.returncode == 2
    assert "regression receipt SHA256 mismatch" in result.stderr


def test_i44a_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    h2 = tmp_path / "h2.json"; receipt = tmp_path / "receipt.json"; out = tmp_path / "out"; out.mkdir()
    (out / "keep.txt").write_text("keep")
    h2_sha = _write(h2, passing_h2()); receipt_sha = _write(receipt, passing_receipt())
    result = subprocess.run([
        sys.executable, str(RUNNER), "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", h2_sha,
        "--regression-receipt", str(receipt), "--regression-receipt-sha256", receipt_sha,
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output directory" in result.stderr


def test_i44a_runner_help_is_execution_free() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--i4-3h2-closure" in result.stdout
