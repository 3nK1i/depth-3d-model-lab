#!/usr/bin/env python3
"""Bind I5.4A forward-motion / KITTI-style source and freeze the real-replay window."""
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
from core.reconstruction.i5_forward_motion_kitti import (  # noqa: E402
    build_i5_4a_forward_motion_kitti_admission,
    build_i5_4a_source_identity,
)

OUTPUT_NAME = "i5_4a_forward_motion_kitti_admission.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-3c-closure", type=Path, required=True)
    p.add_argument("--i5-3c-sha256", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--source-label", required=True)
    p.add_argument(
        "--confirm-kitti-style-forward-motion-source",
        action="store_true",
        help="Confirm the ordered source is intentionally supplied as a KITTI-style forward-motion robustness case. This is provenance, not camera-geometry ground truth.",
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
        raise ReconstructionValidationError(f"I5.4A {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.4A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.4A {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.4A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.4A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    closure_path, closure_sha = _verify(args.i5_3c_closure, args.i5_3c_sha256, label="I5.3C closure")
    receipt_path, receipt_sha = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.4A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.4A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    identity = build_i5_4a_source_identity(args.dataset_dir)
    result = build_i5_4a_forward_motion_kitti_admission(
        i5_3c_result=_json(closure_path, label="I5.3C closure"),
        i5_3c_binding=I51EvidenceBinding(str(closure_path), closure_sha),
        source_identity=identity,
        source_label=args.source_label,
        kitti_style_forward_motion_confirmation=bool(args.confirm_kitti_style_forward_motion_source),
        admission_receipt=_json(receipt_path, label="I5.4A admission receipt"),
        admission_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    window = result["stress_plan"]["forward_motion_window"]
    print("I5.4A Forward-Motion / KITTI Scenario Admission: PASS")
    print(
        f"source_label={result['source_semantics']['label']} source_images={identity['image_count']} "
        f"source_asset_set_sha256={identity['asset_set_sha256']}"
    )
    print(
        f"forward_motion_window images={window['image_count']} indices={window['first_source_index']}..{window['last_source_index']} "
        f"stride={window['source_stride']} contiguous={window['contiguous']} window_sha256={window['window_asset_set_sha256']}"
    )
    print(
        "operator_kitti_style_forward_motion_confirmation=True "
        "authoritative_forward_motion_geometry_proven=False i5_4b_geometry_revalidation_required=True"
    )
    print(
        f"matching_policy={window['matching_policy']} sequential_overlap={window['sequential_overlap']} "
        f"nominal_image_sequence_fps={window['nominal_image_sequence_fps']}"
    )
    print("sfm_rerun=False images_transformed=False dense_provider_rerun=False thresholds_modified=False production_default_modified=False")
    print("pycolmap_authority=True auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.4B current frozen-foundation forward-motion / KITTI replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.4A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
