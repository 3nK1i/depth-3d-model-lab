#!/usr/bin/env python3
"""Bind I5.2A historical weak-overlap evidence and freeze controlled stress slices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_low_overlap_weak_parallax import (  # noqa: E402
    build_i5_2a_low_overlap_weak_parallax_admission,
    build_ordered_source_identity,
)

OUTPUT_NAME = "i5_2a_low_overlap_weak_parallax_admission.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-1c-result", type=Path, required=True)
    p.add_argument("--i5-1c-sha256", required=True)
    p.add_argument("--i4-3g2-final", type=Path, required=True)
    p.add_argument("--i4-3g2-final-sha256", required=True)
    p.add_argument("--i4-3g2a3-doc", type=Path, required=True)
    p.add_argument("--i4-3g2a3-doc-sha256", required=True)
    p.add_argument("--i4-3g2b-doc", type=Path, required=True)
    p.add_argument("--i4-3g2b-doc-sha256", required=True)
    p.add_argument("--i4-3g2c-doc", type=Path, required=True)
    p.add_argument("--i4-3g2c-doc-sha256", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--confirm-ordered-static-multiview",
        action="store_true",
        help="Explicitly confirm the source is an ordered static multiview capture suitable for controlled stress slicing.",
    )
    p.add_argument("--admission-receipt", type=Path, required=True)
    p.add_argument("--admission-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.2A {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.2A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.2A {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.2A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.2A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    i51c_path, i51c_sha = _verify(args.i5_1c_result, args.i5_1c_sha256, label="I5.1C closure")
    g2_path, g2_sha = _verify(args.i4_3g2_final, args.i4_3g2_final_sha256, label="I4.3G2 final closure")
    g2a3_path, g2a3_sha = _verify(args.i4_3g2a3_doc, args.i4_3g2a3_doc_sha256, label="I4.3G2A.3 phase history")
    g2b_path, g2b_sha = _verify(args.i4_3g2b_doc, args.i4_3g2b_doc_sha256, label="I4.3G2B phase history")
    g2c_path, g2c_sha = _verify(args.i4_3g2c_doc, args.i4_3g2c_doc_sha256, label="I4.3G2C phase history")
    receipt_path, receipt_sha = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.2A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.2A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_identity = build_ordered_source_identity(args.dataset_dir)
    result = build_i5_2a_low_overlap_weak_parallax_admission(
        i5_1c_result=_json(i51c_path, label="I5.1C closure"),
        i5_1c_binding=I51EvidenceBinding(str(i51c_path), i51c_sha),
        i4_3g2_final_result=_json(g2_path, label="I4.3G2 final closure"),
        i4_3g2_final_binding=I51EvidenceBinding(str(g2_path), g2_sha),
        i4_3g2a3_doc=g2a3_path.read_text(encoding="utf-8"),
        i4_3g2a3_binding=I51EvidenceBinding(str(g2a3_path), g2a3_sha),
        i4_3g2b_doc=g2b_path.read_text(encoding="utf-8"),
        i4_3g2b_binding=I51EvidenceBinding(str(g2b_path), g2b_sha),
        i4_3g2c_doc=g2c_path.read_text(encoding="utf-8"),
        i4_3g2c_binding=I51EvidenceBinding(str(g2c_path), g2c_sha),
        source_identity=source_identity,
        semantic_confirmation=bool(args.confirm_ordered_static_multiview),
        admission_receipt=_json(receipt_path, label="I5.2A admission receipt"),
        admission_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    plan = result["stress_plan"]
    low = plan["low_overlap_proxy"]
    weak = plan["weak_parallax_proxy"]
    historical = result["historical_baseline"]
    print("I5.2A Low-Overlap / Weak-Parallax Scenario Admission: PASS")
    print(
        f"historical_weak_overlap_registered_scope={historical['historical_authoritative_registered_view_scope']} "
        f"historical_partial_registration_preserved={historical['historical_partial_registration_preserved']}"
    )
    print(
        f"fresh_source_images={source_identity['image_count']} asset_set_sha256={source_identity['asset_set_sha256']}"
    )
    print(
        f"low_overlap_proxy images={low['image_count']} spacing={low['capture_spacing_factor']} "
        f"span={low['source_span_fraction']:.3f} authoritative_overlap_measurement=False"
    )
    print(
        f"weak_parallax_proxy images={weak['image_count']} spacing={weak['capture_spacing_factor']} "
        f"span={weak['source_span_fraction']:.3f} authoritative_parallax_measurement=False"
    )
    print("sfm_rerun=False dense_provider_rerun=False thresholds_modified=False production_default_modified=False")
    print("pycolmap_authority=True auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.2B current frozen-foundation low-overlap / weak-parallax replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.2A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
