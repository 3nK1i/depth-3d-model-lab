from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.offline_dense_cross_provider_visual_verification import (
    VisualVerificationEvidenceBinding,
    build_i4_4b_cross_provider_visual_verification,
)
from core.reconstruction.offline_dense_foundation_regression import (
    RegressionEvidenceBinding,
    build_i4_4a_focused_regression,
)
from core.reconstruction.offline_dense_provenance_authority_audit import (
    ProvenanceAuthorityEvidenceBinding,
    build_i4_4c_provenance_authority_audit,
)
from tests.i4_4a_helpers import passing_h2, passing_receipt as passing_i44a_receipt
from tests.i4_4b_helpers import passing_g2, passing_g3, passing_g4, passing_receipt as passing_i44b_receipt
from tests.i4_4c_helpers import passing_receipt as passing_i44c_receipt
from tests.i4_4d_helpers import passing_receipt as passing_i44d_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_4d_final_i4_closure.py"


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _frozen_i44_chain(tmp_path: Path) -> tuple[Path, Path]:
    pred = tmp_path / "pred"
    h2 = pred / "h2.json"; _write(h2, passing_h2())
    h2_payload = json.loads(h2.read_text())

    a_receipt = pred / "a_receipt.json"; _write(a_receipt, passing_i44a_receipt())
    a = pred / "a.json"
    a_payload = build_i4_4a_focused_regression(
        i4_3h2_closure=h2_payload,
        i4_3h2_binding=RegressionEvidenceBinding(str(h2.resolve()), _binding(h2)["sha256"]),
        regression_receipt=json.loads(a_receipt.read_text()),
        regression_binding=RegressionEvidenceBinding(str(a_receipt.resolve()), _binding(a_receipt)["sha256"]),
    )
    _write(a, a_payload)

    g2 = pred / "g2.json"; _write(g2, passing_g2())
    g3 = pred / "g3.json"; _write(g3, passing_g3())
    g4 = pred / "g4.json"; _write(g4, passing_g4())
    b_receipt = pred / "b_receipt.json"; _write(b_receipt, passing_i44b_receipt())
    b = pred / "b.json"
    b_payload = build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=h2_payload,
        i4_3h2_binding=VisualVerificationEvidenceBinding(str(h2.resolve()), _binding(h2)["sha256"]),
        i4_4a_result=a_payload,
        i4_4a_binding=VisualVerificationEvidenceBinding(str(a.resolve()), _binding(a)["sha256"]),
        i4_3g2_closure=json.loads(g2.read_text()),
        i4_3g2_binding=VisualVerificationEvidenceBinding(str(g2.resolve()), _binding(g2)["sha256"]),
        i4_3g3c_closure=json.loads(g3.read_text()),
        i4_3g3c_binding=VisualVerificationEvidenceBinding(str(g3.resolve()), _binding(g3)["sha256"]),
        i4_3g4c_closure=json.loads(g4.read_text()),
        i4_3g4c_binding=VisualVerificationEvidenceBinding(str(g4.resolve()), _binding(g4)["sha256"]),
        visual_receipt=json.loads(b_receipt.read_text()),
        visual_receipt_binding=VisualVerificationEvidenceBinding(str(b_receipt.resolve()), _binding(b_receipt)["sha256"]),
    )
    _write(b, b_payload)

    c_receipt = pred / "c_receipt.json"; _write(c_receipt, passing_i44c_receipt())
    c = pred / "c.json"
    c_payload = build_i4_4c_provenance_authority_audit(
        i4_3h2_closure=h2_payload,
        i4_3h2_binding=ProvenanceAuthorityEvidenceBinding(str(h2.resolve()), _binding(h2)["sha256"]),
        i4_4a_result=a_payload,
        i4_4a_binding=ProvenanceAuthorityEvidenceBinding(str(a.resolve()), _binding(a)["sha256"]),
        i4_4b_result=b_payload,
        i4_4b_binding=ProvenanceAuthorityEvidenceBinding(str(b.resolve()), _binding(b)["sha256"]),
        audit_receipt=json.loads(c_receipt.read_text()),
        audit_receipt_binding=ProvenanceAuthorityEvidenceBinding(str(c_receipt.resolve()), _binding(c_receipt)["sha256"]),
        h2_h1_chain_reverified=True,
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        i4_4a_chain_reverified=True,
        i4_4b_chain_reverified=True,
        current_repository_authority_matches_frozen_h1=True,
    )
    _write(c, c_payload)
    d_receipt = pred / "d_receipt.json"; _write(d_receipt, passing_i44d_receipt())
    return c, d_receipt


def _run(tmp_path: Path, *, c_sha: str | None = None, receipt_sha: str | None = None):
    c, receipt = _frozen_i44_chain(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-4c-result", str(c), "--i4-4c-sha256", c_sha or _binding(c)["sha256"],
        "--closure-receipt", str(receipt), "--closure-receipt-sha256", receipt_sha or _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    return result, out


def test_i44d_runner_reverifies_i44c_immediate_graph_and_writes_final_closure(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "i4_4d_final_i4_closure.json").read_text())
    assert payload["pass"] is True
    assert payload["decision"]["i4_closed"] is True
    assert payload["decision"]["i5_authorized"] is True


def test_i44d_runner_fails_closed_on_i44c_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, c_sha="0" * 64)
    assert result.returncode == 2
    assert "I4.4C result SHA256 mismatch" in result.stderr


def test_i44d_runner_fails_closed_if_i44c_bound_artifact_is_tampered(tmp_path: Path) -> None:
    c, receipt = _frozen_i44_chain(tmp_path)
    c_payload = json.loads(c.read_text())
    a = Path(c_payload["evidence_bindings"]["i4_4a_result"]["path"])
    a.write_text(a.read_text() + "\n", encoding="utf-8")
    out = tmp_path / "tampered"
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-4c-result", str(c), "--i4-4c-sha256", _binding(c)["sha256"],
        "--closure-receipt", str(receipt), "--closure-receipt-sha256", _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "I4.4C bound i4_4a_result SHA256 mismatch" in result.stderr


def test_i44d_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    c, receipt = _frozen_i44_chain(tmp_path)
    out = tmp_path / "out"; out.mkdir(); (out / "keep.txt").write_text("keep", encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-4c-result", str(c), "--i4-4c-sha256", _binding(c)["sha256"],
        "--closure-receipt", str(receipt), "--closure-receipt-sha256", _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output directory" in result.stderr


def test_i44d_runner_help_is_execution_free() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--i4-4c-result" in result.stdout
