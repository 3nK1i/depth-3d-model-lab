#!/usr/bin/env python3
"""Independently reverify I4.4C immediate evidence and formally close/freeze I4."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_cross_provider_visual_verification import validate_i4_4a_result_for_i4_4b  # noqa: E402
from core.reconstruction.offline_dense_final_i4_closure import (  # noqa: E402
    FinalI4EvidenceBinding,
    build_i4_4d_final_i4_closure,
    validate_i4_4c_result_for_i4_4d,
    validate_i4_4d_closure_receipt,
)
from core.reconstruction.offline_dense_foundation_regression import validate_i4_3h2_for_i4_4a  # noqa: E402
from core.reconstruction.offline_dense_provenance_authority_audit import (  # noqa: E402
    audit_current_offline_dense_authority_contracts,
    validate_i4_4b_result_for_i4_4c,
    validate_i4_4c_audit_receipt,
)

OUTPUT_NAME = "i4_4d_final_i4_closure.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-4c-result", type=Path, required=True)
    parser.add_argument("--i4-4c-sha256", required=True)
    parser.add_argument("--closure-receipt", type=Path, required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I4.4D {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I4.4D {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(f"I4.4D {label} SHA256 mismatch: expected={expected_text} actual={actual}")
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.4D {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.4D {label} root must be an object.")
    return payload


def _binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.4D {label} binding is missing or malformed.")
    path = str(value.get("path", "")).strip()
    sha = str(value.get("sha256", "")).strip().lower()
    if not path or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha):
        raise ReconstructionValidationError(f"I4.4D {label} binding is incomplete.")
    return {"path": str(Path(path).expanduser().resolve()), "sha256": sha}


def _verify_binding(value: Any, *, label: str) -> tuple[Path, str, dict[str, Any]]:
    binding = _binding(value, label=label)
    path, sha = _verify(Path(binding["path"]), binding["sha256"], label=label)
    return path, sha, _read(path, label=label)


def _same_binding(value: Any, *, path: Path, sha256: str, label: str) -> None:
    binding = _binding(value, label=label)
    expected = {"path": str(path.expanduser().resolve()), "sha256": sha256}
    if binding != expected:
        raise ReconstructionValidationError(f"I4.4D {label} does not match the independently verified artifact.")


def run(args: argparse.Namespace) -> Path:
    c_path, c_sha = _verify(args.i4_4c_result, args.i4_4c_sha256, label="I4.4C result")
    receipt_path, receipt_sha = _verify(args.closure_receipt, args.closure_receipt_sha256, label="I4.4D closure receipt")
    c = _read(c_path, label="I4.4C result")
    receipt = _read(receipt_path, label="I4.4D closure receipt")
    validated_c = validate_i4_4c_result_for_i4_4d(c)
    validate_i4_4d_closure_receipt(receipt)

    # I4.4D independently rehashes I4.4C's immediate bindings. C already
    # rehashed the deeper H2->H1->G5B/predecessor graph; D preserves that
    # conclusion without falsely claiming a recursive raw provider-artifact rehash.
    bindings = validated_c["evidence_bindings"]
    h2_path, h2_sha, h2 = _verify_binding(bindings["i4_3h2_closure"], label="I4.4C bound i4_3h2_closure")
    a_path, a_sha, a = _verify_binding(bindings["i4_4a_result"], label="I4.4C bound i4_4a_result")
    b_path, b_sha, b = _verify_binding(bindings["i4_4b_result"], label="I4.4C bound i4_4b_result")
    _, _, c_receipt = _verify_binding(bindings["i4_4c_audit_receipt"], label="I4.4C bound i4_4c_audit_receipt")

    validate_i4_3h2_for_i4_4a(h2)
    validate_i4_4a_result_for_i4_4b(a)
    validate_i4_4b_result_for_i4_4c(b)
    validate_i4_4c_audit_receipt(c_receipt)

    a_bindings = a.get("evidence_bindings")
    b_bindings = b.get("evidence_bindings")
    if not isinstance(a_bindings, Mapping) or not isinstance(b_bindings, Mapping):
        raise ReconstructionValidationError("I4.4D A/B evidence bindings are missing.")
    _same_binding(a_bindings.get("i4_3h2_closure"), path=h2_path, sha256=h2_sha, label="I4.4A H2 binding")
    _same_binding(b_bindings.get("i4_3h2_closure"), path=h2_path, sha256=h2_sha, label="I4.4B H2 binding")
    _same_binding(b_bindings.get("i4_4a_result"), path=a_path, sha256=a_sha, label="I4.4B I4.4A binding")
    _same_binding(bindings["i4_4b_result"], path=b_path, sha256=b_sha, label="I4.4C I4.4B binding")

    current_authority = audit_current_offline_dense_authority_contracts()
    if current_authority != validated_c["authority_audit"]:
        raise ReconstructionValidationError("I4.4D current repository authority drifted from I4.4C.")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.4D output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    closure = build_i4_4d_final_i4_closure(
        i4_4c_result=c,
        i4_4c_binding=FinalI4EvidenceBinding(str(c_path), c_sha),
        closure_receipt=receipt,
        closure_receipt_binding=FinalI4EvidenceBinding(str(receipt_path), receipt_sha),
        i4_4c_bound_artifacts_reverified=True,
        current_repository_authority_matches_i4_4c=True,
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(closure, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I4.4D Final I4 Closure + Offline Dense Foundation Freeze: PASS")
    print("i4_4a=True i4_4b=True i4_4c=True i4_4d=True i4_closed=True i5_authorized=True i5_started=False")
    print("i4_4c_bound_artifacts_reverified=True current_authority_matches_i4_4c=True")
    print("camera_authority=pycolmap_sfm reconstruction_scale=arbitrary permanent_global_view_cap=None")
    print("provider_execution=False sfm_rerun=False matched_view_rerun=False manual_review_rerun=False auto=False hybrid=False metric_scale=False metrology=False")
    print("raw_nested_provider_artifact_tree_recursively_rehashed=False")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I4.4D ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
