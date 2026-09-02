from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from core.reconstruction.i4_3_final_closure import H2EvidenceBinding, build_i4_3h2_final_closure
from core.reconstruction.i4_3_final_qualification import H1EvidenceBinding, build_i4_3h1_final_qualification_audit
from core.reconstruction.offline_dense_cross_provider_visual_verification import VisualVerificationEvidenceBinding, build_i4_4b_cross_provider_visual_verification
from core.reconstruction.offline_dense_foundation_regression import RegressionEvidenceBinding, build_i4_4a_focused_regression
from tests.i4_3g5a_helpers import passing_g1
from tests.i4_3g5b_helpers import passing_g5a
from tests.i4_3h1_helpers import passing_regression_receipt
from tests.i4_4a_helpers import passing_receipt as passing_i44a_receipt
from tests.i4_4b_helpers import (
    passing_g2 as passing_visual_g2,
    passing_g3 as passing_visual_g3,
    passing_g4 as passing_visual_g4,
    passing_receipt as passing_i44b_receipt,
)
from tests.i4_4c_helpers import passing_receipt as passing_i44c_receipt

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_i4_4c_provenance_authority_audit.py"


def _write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _frozen_chain(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    pred = tmp_path / "pred"
    g1 = pred / "g1.json"; g2 = pred / "g2.json"; g3 = pred / "g3.json"; g4 = pred / "g4.json"; g5a = pred / "g5a.json"
    _write(g1, passing_g1()); _write(g2, passing_visual_g2()); _write(g3, passing_visual_g3()); _write(g4, passing_visual_g4()); _write(g5a, passing_g5a())

    from core.reconstruction.cross_provider_qualification_closure import G5BClosureBinding, build_i4_3g5b_final_closure
    g5a_payload = json.loads(g5a.read_text())
    g5b = pred / "g5b.json"
    g5b_payload = build_i4_3g5b_final_closure(
        i4_3g5a_synthesis=g5a_payload,
        evidence_bindings={
            "i4_3g1_snapshot": G5BClosureBinding(str(g1.resolve()), _binding(g1)["sha256"]),
            "i4_3g2_closure": G5BClosureBinding(str(g2.resolve()), _binding(g2)["sha256"]),
            "i4_3g3c_closure": G5BClosureBinding(str(g3.resolve()), _binding(g3)["sha256"]),
            "i4_3g4c_closure": G5BClosureBinding(str(g4.resolve()), _binding(g4)["sha256"]),
            "i4_3g5a_synthesis": G5BClosureBinding(str(g5a.resolve()), _binding(g5a)["sha256"]),
        },
        chain_rebuild_verified=True,
    )
    _write(g5b, g5b_payload)

    h1_receipt = pred / "h1_receipt.json"; _write(h1_receipt, passing_regression_receipt())
    h1 = pred / "h1.json"
    h1_payload = build_i4_3h1_final_qualification_audit(
        i4_3g5b_closure=g5b_payload,
        regression_receipt=json.loads(h1_receipt.read_text()),
        evidence_bindings={
            "i4_3g5b_closure": H1EvidenceBinding(str(g5b.resolve()), _binding(g5b)["sha256"]),
            "i4_3h1_regression_receipt": H1EvidenceBinding(str(h1_receipt.resolve()), _binding(h1_receipt)["sha256"]),
        },
        g5b_predecessor_hashes_reverified=True,
    )
    _write(h1, h1_payload)

    h2 = pred / "h2.json"
    h2_payload = build_i4_3h2_final_closure(
        i4_3h1_audit=h1_payload,
        evidence_binding=H2EvidenceBinding(str(h1.resolve()), _binding(h1)["sha256"]),
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        current_architecture_matches_h1=True,
    )
    _write(h2, h2_payload)

    i44a_receipt = pred / "i44a_receipt.json"; _write(i44a_receipt, passing_i44a_receipt())
    i44a = pred / "i44a.json"
    i44a_payload = build_i4_4a_focused_regression(
        i4_3h2_closure=h2_payload,
        i4_3h2_binding=RegressionEvidenceBinding(str(h2.resolve()), _binding(h2)["sha256"]),
        regression_receipt=json.loads(i44a_receipt.read_text()),
        regression_binding=RegressionEvidenceBinding(str(i44a_receipt.resolve()), _binding(i44a_receipt)["sha256"]),
    )
    _write(i44a, i44a_payload)

    i44b_receipt = pred / "i44b_receipt.json"; _write(i44b_receipt, passing_i44b_receipt())
    i44b = pred / "i44b.json"
    i44b_payload = build_i4_4b_cross_provider_visual_verification(
        i4_3h2_closure=h2_payload,
        i4_3h2_binding=VisualVerificationEvidenceBinding(str(h2.resolve()), _binding(h2)["sha256"]),
        i4_4a_result=i44a_payload,
        i4_4a_binding=VisualVerificationEvidenceBinding(str(i44a.resolve()), _binding(i44a)["sha256"]),
        i4_3g2_closure=json.loads(g2.read_text()),
        i4_3g2_binding=VisualVerificationEvidenceBinding(str(g2.resolve()), _binding(g2)["sha256"]),
        i4_3g3c_closure=json.loads(g3.read_text()),
        i4_3g3c_binding=VisualVerificationEvidenceBinding(str(g3.resolve()), _binding(g3)["sha256"]),
        i4_3g4c_closure=json.loads(g4.read_text()),
        i4_3g4c_binding=VisualVerificationEvidenceBinding(str(g4.resolve()), _binding(g4)["sha256"]),
        visual_receipt=json.loads(i44b_receipt.read_text()),
        visual_receipt_binding=VisualVerificationEvidenceBinding(str(i44b_receipt.resolve()), _binding(i44b_receipt)["sha256"]),
    )
    _write(i44b, i44b_payload)

    i44c_receipt = pred / "i44c_receipt.json"; _write(i44c_receipt, passing_i44c_receipt())
    return h2, i44a, i44b, i44c_receipt


def _run(tmp_path: Path, *, i44b_sha: str | None = None):
    h2, i44a, i44b, receipt = _frozen_chain(tmp_path)
    out = tmp_path / "out"
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", _binding(h2)["sha256"],
        "--i4-4a-result", str(i44a), "--i4-4a-sha256", _binding(i44a)["sha256"],
        "--i4-4b-result", str(i44b), "--i4-4b-sha256", i44b_sha or _binding(i44b)["sha256"],
        "--audit-receipt", str(receipt), "--audit-receipt-sha256", _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    return result, out


def test_i44c_runner_rehashes_formal_graph_and_writes_result(tmp_path: Path) -> None:
    result, out = _run(tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads((out / "i4_4c_provenance_authority_audit.json").read_text())
    assert payload["pass"] is True
    assert payload["provenance_chain_audit"]["g5b_predecessor_artifacts_reverified"] is True
    assert payload["decision"]["i4_4d_authorized"] is True


def test_i44c_runner_fails_closed_on_i44b_hash_drift(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, i44b_sha="0" * 64)
    assert result.returncode == 2
    assert "I4.4B result SHA256 mismatch" in result.stderr


def test_i44c_runner_fails_closed_if_bound_predecessor_is_tampered(tmp_path: Path) -> None:
    h2, i44a, i44b, receipt = _frozen_chain(tmp_path)
    h2_payload = json.loads(h2.read_text())
    g2 = Path(h2_payload["g5b_predecessor_bindings"]["i4_3g2_closure"]["path"])
    g2.write_text(g2.read_text() + "\n", encoding="utf-8")
    out = tmp_path / "tampered-out"
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", _binding(h2)["sha256"],
        "--i4-4a-result", str(i44a), "--i4-4a-sha256", _binding(i44a)["sha256"],
        "--i4-4b-result", str(i44b), "--i4-4b-sha256", _binding(i44b)["sha256"],
        "--audit-receipt", str(receipt), "--audit-receipt-sha256", _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "G5B predecessor i4_3g2_closure SHA256 mismatch" in result.stderr


def test_i44c_runner_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    h2, i44a, i44b, receipt = _frozen_chain(tmp_path)
    out = tmp_path / "out"; out.mkdir(); (out / "keep.txt").write_text("keep", encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(RUNNER),
        "--i4-3h2-closure", str(h2), "--i4-3h2-sha256", _binding(h2)["sha256"],
        "--i4-4a-result", str(i44a), "--i4-4a-sha256", _binding(i44a)["sha256"],
        "--i4-4b-result", str(i44b), "--i4-4b-sha256", _binding(i44b)["sha256"],
        "--audit-receipt", str(receipt), "--audit-receipt-sha256", _binding(receipt)["sha256"],
        "--output-dir", str(out),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "output directory" in result.stderr


def test_i44c_runner_help_is_execution_free() -> None:
    result = subprocess.run([sys.executable, str(RUNNER), "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    assert "--i4-4b-result" in result.stdout
