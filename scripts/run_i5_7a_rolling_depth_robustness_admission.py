#!/usr/bin/env python3
"""Build the execution-free I5.7A RollingDepth robustness admission artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_rolling_depth_robustness import (
    I57A_I56A_FROZEN_SHA256,
    I57A_I56C_FROZEN_SHA256,
    I57A_SCENARIO_IDS,
    build_i5_7a_rolling_depth_robustness_admission,
)
from core.reconstruction.single_image_provider_qualification import I43G3C_FROZEN_CLOSURE_SHA256

OUTPUT_NAME = "i5_7a_rolling_depth_robustness_admission.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.7A {label} is missing: {resolved}")
    text = str(expected).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ReconstructionValidationError(f"I5.7A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != text:
        raise ReconstructionValidationError(
            f"I5.7A {label} SHA256 mismatch: expected={text} actual={actual}"
        )
    return I51EvidenceBinding(str(resolved), actual)


def _json(path: str, *, label: str) -> dict:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.7A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.7A {label} root must be an object.")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-6a-admission", type=Path, required=True)
    parser.add_argument("--i5-6a-sha256", default=I57A_I56A_FROZEN_SHA256)
    parser.add_argument("--i5-6c-closure", type=Path, required=True)
    parser.add_argument("--i5-6c-sha256", default=I57A_I56C_FROZEN_SHA256)
    parser.add_argument("--i4-3g3c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g3c-sha256", default=I43G3C_FROZEN_CLOSURE_SHA256)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--admission-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> Path:
    i56a = _verify(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    i56c = _verify(args.i5_6c_closure, args.i5_6c_sha256, label="I5.6C closure")
    g3c = _verify(args.i4_3g3c_closure, args.i4_3g3c_sha256, label="I4.3G3C closure")
    receipt = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.7A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.7A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    result = build_i5_7a_rolling_depth_robustness_admission(
        i5_6a_admission=_json(i56a.path, label="I5.6A admission"),
        i5_6a_binding=i56a,
        i5_6c_closure=_json(i56c.path, label="I5.6C closure"),
        i5_6c_binding=i56c,
        i4_3g3c_closure=_json(g3c.path, label="I4.3G3C closure"),
        i4_3g3c_binding=g3c,
        admission_receipt=_json(receipt.path, label="I5.7A admission receipt"),
        admission_receipt_binding=receipt,
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.7A RollingDepth Robustness Admission + Envelope Design: PASS")
    print("provider=rolling_depth provider_id=rolling-depth-temporal stateful_temporal_sequence=True")
    for sid in I57A_SCENARIO_IDS:
        node = result["scenario_matrix"][sid]
        print(
            f"admitted {sid} registered={node['registered_images']}/{node['input_images']} "
            f"interpretation={node['scientific_interpretation']}"
        )
    anchors = result["historical_temporal_anchors"]
    print(
        "historical_g3="
        + ",".join(f"{sid}:{anchors[sid]['rolling_depth_production_status']}" for sid in anchors)
        + " rerun=False"
    )
    print("excluded blur=BLOCKED_UPSTREAM underexposure=BLOCKED_UPSTREAM clutter=NOT_FORMALLY_ADMITTED kitti=BLOCKED_UPSTREAM horse=DEFERRED_NOT_ADMITTED")
    print("rolling_depth_inference=False sfm_rerun=False mvs_rerun=False provider_comparison=False threshold_relaxation=False")
    print("pycolmap_authority=True metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.7B real frozen-foundation RollingDepth robustness replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.7A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
