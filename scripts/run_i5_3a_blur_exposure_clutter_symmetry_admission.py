#!/usr/bin/env python3
"""Bind I5.3A and freeze controlled blur/exposure/clutter/symmetry stress recipes."""
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
from core.reconstruction.i5_blur_exposure_clutter_symmetry import (  # noqa: E402
    build_i5_3a_blur_exposure_clutter_symmetry_admission,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_low_overlap_weak_parallax import build_ordered_source_identity  # noqa: E402

OUTPUT_NAME = "i5_3a_blur_exposure_clutter_symmetry_admission.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-2c-closure", type=Path, required=True)
    p.add_argument("--i5-2c-sha256", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--confirm-controlled-source",
        action="store_true",
        help="Confirm the rebound source is the frozen ordered static I5.2 control capture used for source-derived stress design.",
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
        raise ReconstructionValidationError(f"I5.3A {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.3A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.3A {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.3A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.3A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    i52c_path, i52c_sha = _verify(args.i5_2c_closure, args.i5_2c_sha256, label="I5.2C closure")
    receipt_path, receipt_sha = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.3A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.3A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    source_identity = build_ordered_source_identity(args.dataset_dir)
    result = build_i5_3a_blur_exposure_clutter_symmetry_admission(
        i5_2c_result=_json(i52c_path, label="I5.2C closure"),
        i5_2c_binding=I51EvidenceBinding(str(i52c_path), i52c_sha),
        source_identity=source_identity,
        controlled_source_confirmation=bool(args.confirm_controlled_source),
        admission_receipt=_json(receipt_path, label="I5.3A admission receipt"),
        admission_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    plan = result["stress_plan"]
    print("I5.3A Blur / Exposure / Clutter / Symmetry Scenario Admission: PASS")
    print(
        f"source_images={source_identity['image_count']} asset_set_sha256={source_identity['asset_set_sha256']} "
        "same_control_source_as_i5_2=True"
    )
    print(
        f"blur_proxy kernel={plan['blur_proxy']['transform']['kernel_size']} "
        f"sigma={plan['blur_proxy']['transform']['sigma']} authoritative=False"
    )
    print(
        f"underexposure_proxy gain={plan['underexposure_proxy']['transform']['gain']} authoritative=False"
    )
    print(
        f"overexposure_proxy gain={plan['overexposure_proxy']['transform']['gain']} "
        f"bias={plan['overexposure_proxy']['transform']['bias']} authoritative=False"
    )
    print(
        f"clutter_proxy band_fraction={plan['clutter_proxy']['transform']['band_fraction']} "
        f"source_offset={plan['clutter_proxy']['transform']['source_offset']} authoritative=False natural_ground_truth=False"
    )
    print(
        f"symmetry_proxy band_fraction={plan['symmetry_proxy']['transform']['band_fraction']} "
        "authoritative=False natural_ground_truth=False"
    )
    print("stress_images_materialized=False sfm_rerun=False dense_provider_rerun=False thresholds_modified=False production_default_modified=False")
    print("pycolmap_authority=True auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.3B current frozen-foundation blur / exposure / clutter / symmetry replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.3A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
