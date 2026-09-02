from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tests.i4_4b_helpers import passing_g2, passing_g3, passing_g4, passing_h2, passing_i4_4a, passing_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_4b_cross_provider_visual_verification.py"


def _write(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path, *, h2_sha: str | None = None, i44a_sha: str | None = None, receipt_sha: str | None = None):
    h2 = tmp_path / "h2.json"
    i44a = tmp_path / "i44a.json"
    g2 = tmp_path / "g2.json"
    g3 = tmp_path / "g3.json"
    g4 = tmp_path / "g4.json"
    receipt = tmp_path / "receipt.json"
    out = tmp_path / "out"
    real_h2_sha = _write(h2, passing_h2())
    real_i44a_sha = _write(i44a, passing_i4_4a())
    real_g2_sha = _write(g2, passing_g2())
    real_g3_sha = _write(g3, passing_g3())
    real_g4_sha = _write(g4, passing_g4())
    real_receipt_sha = _write(receipt, passing_receipt())
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", h2_sha or real_h2_sha,
        "--i4-4a-result", str(i44a), "--i4-4a-sha256", i44a_sha or real_i44a_sha,
        "--i4-3g2-closure", str(g2), "--i4-3g2-sha256", real_g2_sha,
        "--i4-3g3c-closure", str(g3), "--i4-3g3c-sha256", real_g3_sha,
        "--i4-3g4c-closure", str(g4), "--i4-3g4c-sha256", real_g4_sha,
        "--visual-receipt", str(receipt), "--visual-receipt-sha256", receipt_sha or real_receipt_sha,
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    return result, out


def test_i44b_runner_hash_binds_all_inputs_and_writes_result(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "i4_4b_cross_provider_visual_verification.json").read_text())
    assert payload["pass"] is True
    assert payload["decision"]["i4_4c_authorized"] is True


def test_i44b_runner_fails_closed_on_i44a_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, i44a_sha="0" * 64)
    assert result.returncode == 2
    assert "I4.4A result SHA256 mismatch" in result.stderr


def test_i44b_runner_fails_closed_on_receipt_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, receipt_sha="0" * 64)
    assert result.returncode == 2
    assert "visual receipt SHA256 mismatch" in result.stderr


def test_i44b_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    h2 = tmp_path / "h2.json"; i44a = tmp_path / "i44a.json"; g2 = tmp_path / "g2.json"; g3 = tmp_path / "g3.json"; g4 = tmp_path / "g4.json"; receipt = tmp_path / "receipt.json"
    out = tmp_path / "out"; out.mkdir(); (out / "keep.txt").write_text("keep", encoding="utf-8")
    h2_sha = _write(h2, passing_h2()); i44a_sha = _write(i44a, passing_i4_4a()); g2_sha = _write(g2, passing_g2()); g3_sha = _write(g3, passing_g3()); g4_sha = _write(g4, passing_g4()); receipt_sha = _write(receipt, passing_receipt())
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", h2_sha,
        "--i4-4a-result", str(i44a), "--i4-4a-sha256", i44a_sha,
        "--i4-3g2-closure", str(g2), "--i4-3g2-sha256", g2_sha,
        "--i4-3g3c-closure", str(g3), "--i4-3g3c-sha256", g3_sha,
        "--i4-3g4c-closure", str(g4), "--i4-3g4c-sha256", g4_sha,
        "--visual-receipt", str(receipt), "--visual-receipt-sha256", receipt_sha,
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output directory" in result.stderr


def test_i44b_runner_help_is_execution_free() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--i4-4a-result" in result.stdout
