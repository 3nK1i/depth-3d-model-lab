#!/usr/bin/env python3
"""Rehash the formal I4 provenance graph and build I4.4C authority-audit evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i4_3_final_closure import validate_i4_3h1_qualification_audit  # noqa: E402
from core.reconstruction.i4_3_final_qualification import (  # noqa: E402
    audit_i4_3_provider_architecture,
    validate_i4_3g5b_closure,
    validate_i4_3h1_regression_receipt,
)
from core.reconstruction.offline_dense_cross_provider_visual_verification import (  # noqa: E402
    validate_i4_4a_result_for_i4_4b,
)
from core.reconstruction.offline_dense_foundation_regression import validate_i4_3h2_for_i4_4a  # noqa: E402
from core.reconstruction.offline_dense_provenance_authority_audit import (  # noqa: E402
    ProvenanceAuthorityEvidenceBinding,
    build_i4_4c_provenance_authority_audit,
    validate_i4_4b_result_for_i4_4c,
    validate_i4_4c_audit_receipt,
)

OUTPUT_NAME = "i4_4c_provenance_authority_audit.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i4-3h2-closure", type=Path, required=True)
    parser.add_argument("--i4-3h2-sha256", required=True)
    parser.add_argument("--i4-4a-result", type=Path, required=True)
    parser.add_argument("--i4-4a-sha256", required=True)
    parser.add_argument("--i4-4b-result", type=Path, required=True)
    parser.add_argument("--i4-4b-sha256", required=True)
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--audit-receipt-sha256", required=True)
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
        raise ReconstructionValidationError(f"I4.4C {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I4.4C {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(f"I4.4C {label} SHA256 mismatch: expected={expected_text} actual={actual}")
    return resolved, actual


def _read(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I4.4C {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.4C {label} root must be an object.")
    return payload


def _binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.4C {label} binding is missing or malformed.")
    path = str(value.get("path", "")).strip()
    sha = str(value.get("sha256", "")).strip().lower()
    if not path or len(sha) != 64:
        raise ReconstructionValidationError(f"I4.4C {label} binding is incomplete.")
    return {"path": str(Path(path).expanduser().resolve()), "sha256": sha}


def _verify_binding(value: Any, *, label: str) -> tuple[Path, str]:
    binding = _binding(value, label=label)
    return _verify(Path(binding["path"]), binding["sha256"], label=label)


def _same_binding(value: Any, *, path: Path, sha256: str, label: str) -> None:
    binding = _binding(value, label=label)
    expected_path = str(path.expanduser().resolve())
    if binding != {"path": expected_path, "sha256": sha256}:
        raise ReconstructionValidationError(
            f"I4.4C {label} does not match the explicitly verified artifact: {binding!r}"
        )


def _base_architecture_from_h1(architecture: Mapping[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(dict(architecture))
    inventory = normalized.get("provider_inventory")
    if isinstance(inventory, dict):
        for item in inventory.values():
            if isinstance(item, dict):
                item.pop("terminal_quality_events", None)
                item.pop("no_terminal_quality_event_implies_universal_robustness", None)
    return normalized


def run(args: argparse.Namespace) -> Path:
    h2_path, h2_sha = _verify(args.i4_3h2_closure, args.i4_3h2_sha256, label="I4.3H2 closure")
    i44a_path, i44a_sha = _verify(args.i4_4a_result, args.i4_4a_sha256, label="I4.4A result")
    i44b_path, i44b_sha = _verify(args.i4_4b_result, args.i4_4b_sha256, label="I4.4B result")
    receipt_path, receipt_sha = _verify(args.audit_receipt, args.audit_receipt_sha256, label="I4.4C audit receipt")

    h2 = _read(h2_path, label="I4.3H2 closure")
    i44a = _read(i44a_path, label="I4.4A result")
    i44b = _read(i44b_path, label="I4.4B result")
    receipt = _read(receipt_path, label="I4.4C audit receipt")
    validate_i4_3h2_for_i4_4a(h2)
    validate_i4_4a_result_for_i4_4b(i44a)
    validate_i4_4b_result_for_i4_4c(i44b)
    validate_i4_4c_audit_receipt(receipt)

    # H2 -> H1, then H1's two immediate bindings.
    h1_binding = _binding(h2.get("evidence_binding", {}).get("i4_3h1_audit"), label="H2 bound I4.3H1 audit")
    h1_path, _ = _verify(Path(h1_binding["path"]), h1_binding["sha256"], label="H2 bound I4.3H1 audit")
    h1 = _read(h1_path, label="I4.3H1 audit")
    validated_h1 = validate_i4_3h1_qualification_audit(h1)

    h2_h1_bound = h2.get("h1_bound_evidence")
    if validated_h1["evidence_bindings"] != h2_h1_bound:
        raise ReconstructionValidationError("I4.4C H1 evidence bindings disagree with frozen H2.")
    for key, value in validated_h1["evidence_bindings"].items():
        _verify_binding(value, label=f"H1 bound artifact {key}")

    g5b_binding = validated_h1["evidence_bindings"]["i4_3g5b_closure"]
    g5b_path, _ = _verify_binding(g5b_binding, label="H1 bound I4.3G5B closure")
    g5b = _read(g5b_path, label="I4.3G5B closure")
    validated_g5b = validate_i4_3g5b_closure(g5b)
    h1_receipt_binding = validated_h1["evidence_bindings"]["i4_3h1_regression_receipt"]
    h1_receipt_path, _ = _verify_binding(h1_receipt_binding, label="H1 bound regression receipt")
    validate_i4_3h1_regression_receipt(_read(h1_receipt_path, label="H1 regression receipt"))

    h2_predecessors = h2.get("g5b_predecessor_bindings")
    if validated_h1["g5b_predecessor_bindings"] != h2_predecessors:
        raise ReconstructionValidationError("I4.4C H1 G5B predecessor bindings disagree with H2.")
    if validated_g5b["predecessor_bindings"] != h2_predecessors:
        raise ReconstructionValidationError("I4.4C G5B predecessor bindings disagree with H2.")
    for key, value in validated_h1["g5b_predecessor_bindings"].items():
        _verify_binding(value, label=f"G5B predecessor {key}")

    # I4.4A must bind this exact H2 and its own receipt must still hash-verify.
    i44a_bindings = i44a.get("evidence_bindings")
    if not isinstance(i44a_bindings, Mapping):
        raise ReconstructionValidationError("I4.4C I4.4A evidence bindings are missing.")
    _same_binding(i44a_bindings.get("i4_3h2_closure"), path=h2_path, sha256=h2_sha, label="I4.4A H2 binding")
    _verify_binding(i44a_bindings.get("i4_4a_regression_receipt"), label="I4.4A regression receipt")

    # I4.4B must bind the exact H2/A artifacts and the same G2/G3/G4 artifacts
    # already frozen in the H2 -> G5B predecessor graph.
    i44b_bindings = i44b.get("evidence_bindings")
    if not isinstance(i44b_bindings, Mapping):
        raise ReconstructionValidationError("I4.4C I4.4B evidence bindings are missing.")
    _same_binding(i44b_bindings.get("i4_3h2_closure"), path=h2_path, sha256=h2_sha, label="I4.4B H2 binding")
    _same_binding(i44b_bindings.get("i4_4a_result"), path=i44a_path, sha256=i44a_sha, label="I4.4B I4.4A binding")
    for b_key, h2_key in (
        ("i4_3g2_closure", "i4_3g2_closure"),
        ("i4_3g3c_closure", "i4_3g3c_closure"),
        ("i4_3g4c_closure", "i4_3g4c_closure"),
    ):
        expected = _binding(validated_h1["g5b_predecessor_bindings"][h2_key], label=f"H2 predecessor {h2_key}")
        actual = _binding(i44b_bindings.get(b_key), label=f"I4.4B {b_key}")
        if actual != expected:
            raise ReconstructionValidationError(f"I4.4C I4.4B {b_key} binding drifted from H2/G5B provenance.")
        _verify(Path(actual["path"]), actual["sha256"], label=f"I4.4B bound {b_key}")
    _verify_binding(i44b_bindings.get("i4_4b_visual_receipt"), label="I4.4B visual receipt")

    # Re-audit current repository provider authority and require exact H1 match.
    current_architecture = audit_i4_3_provider_architecture()
    frozen_architecture = _base_architecture_from_h1(validated_h1["architecture_audit"])
    if current_architecture != frozen_architecture:
        raise ReconstructionValidationError("I4.4C current provider architecture drifted from frozen H1.")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I4.4C output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    audit = build_i4_4c_provenance_authority_audit(
        i4_3h2_closure=h2,
        i4_3h2_binding=ProvenanceAuthorityEvidenceBinding(str(h2_path), h2_sha),
        i4_4a_result=i44a,
        i4_4a_binding=ProvenanceAuthorityEvidenceBinding(str(i44a_path), i44a_sha),
        i4_4b_result=i44b,
        i4_4b_binding=ProvenanceAuthorityEvidenceBinding(str(i44b_path), i44b_sha),
        audit_receipt=receipt,
        audit_receipt_binding=ProvenanceAuthorityEvidenceBinding(str(receipt_path), receipt_sha),
        h2_h1_chain_reverified=True,
        h1_bound_artifacts_reverified=True,
        g5b_predecessor_artifacts_reverified=True,
        i4_4a_chain_reverified=True,
        i4_4b_chain_reverified=True,
        current_repository_authority_matches_frozen_h1=True,
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I4.4C Provenance / Authority Audit: PASS")
    print("h2_h1_chain=True h1_bound_artifacts=True g5b_predecessors=True i4_4a_chain=True i4_4b_chain=True architecture_match=True")
    print("camera_authority=pycolmap_sfm reconstruction_scale=arbitrary permanent_global_view_cap=None")
    print("provider_execution=False sfm_rerun=False matched_view_rerun=False manual_review_rerun=False auto=False hybrid=False dense_metric_scale=False certified_metrology=False")
    print("raw_nested_provider_artifact_tree_recursively_rehashed=False")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I4.4C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
