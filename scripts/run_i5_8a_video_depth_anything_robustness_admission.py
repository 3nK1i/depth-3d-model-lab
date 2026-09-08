#!/usr/bin/env python3
"""Build the execution-free I5.8A Video Depth Anything robustness admission artifact."""
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
from core.reconstruction.i5_video_depth_anything_robustness import (
    I58A_G3C_FROZEN_SHA256,
    I58A_I56A_FROZEN_SHA256,
    I58A_I57C_FROZEN_SHA256,
    I58A_SCENARIO_IDS,
    build_i5_8a_video_depth_anything_robustness_admission,
)

OUTPUT_NAME = "i5_8a_video_depth_anything_robustness_admission.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.8A {label} is missing: {resolved}")
    text = str(expected).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ReconstructionValidationError(f"I5.8A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != text:
        raise ReconstructionValidationError(
            f"I5.8A {label} SHA256 mismatch: expected={text} actual={actual}"
        )
    return I51EvidenceBinding(str(resolved), actual)


def _json(path: str, *, label: str) -> dict:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionValidationError(f"I5.8A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.8A {label} root must be an object.")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-7c-closure", type=Path, required=True)
    parser.add_argument("--i5-7c-sha256", default=I58A_I57C_FROZEN_SHA256)
    parser.add_argument("--i5-6a-admission", type=Path, required=True)
    parser.add_argument("--i5-6a-sha256", default=I58A_I56A_FROZEN_SHA256)
    parser.add_argument("--i4-3g3c-closure", type=Path, required=True)
    parser.add_argument("--i4-3g3c-sha256", default=I58A_G3C_FROZEN_SHA256)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--admission-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> Path:
    i57c = _verify(args.i5_7c_closure, args.i5_7c_sha256, label="I5.7C closure")
    i56a = _verify(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    g3c = _verify(args.i4_3g3c_closure, args.i4_3g3c_sha256, label="I4.3G3C closure")
    receipt = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.8A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.8A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    result = build_i5_8a_video_depth_anything_robustness_admission(
        i5_7c_closure=_json(i57c.path, label="I5.7C closure"),
        i5_7c_binding=i57c,
        i5_6a_admission=_json(i56a.path, label="I5.6A admission"),
        i5_6a_binding=i56a,
        i4_3g3c_closure=_json(g3c.path, label="I4.3G3C closure"),
        i4_3g3c_binding=g3c,
        admission_receipt=_json(receipt.path, label="I5.8A admission receipt"),
        admission_receipt_binding=receipt,
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.8A Video Depth Anything Robustness Admission + Envelope Design: PASS")
    print("provider=video_depth_anything provider_id=video-depth-anything-temporal temporal_mode=official_streaming_cache")
    print("alignment=per_registered_frame_affine_to_authoritative_sfm sequence_global_alignment=False")
    print("runtime_profile=hardware_resolved_compute_profile_auto checkpoint_must_match_resolved_profile=True")
    for sid in I58A_SCENARIO_IDS:
        node = result["scenario_matrix"][sid]
        print(
            f"admitted {sid} registered={node['registered_images']}/{node['input_images']} "
            f"interpretation={node['scientific_interpretation']}"
        )
    anchors = result["historical_temporal_anchors"]
    print(
        "historical_g3="
        + ",".join(f"{sid}:{anchors[sid]['video_depth_anything_production_status']}" for sid in anchors)
        + " rerun=False"
    )
    print("excluded blur=BLOCKED_UPSTREAM underexposure=BLOCKED_UPSTREAM clutter=NOT_FORMALLY_ADMITTED kitti=BLOCKED_UPSTREAM horse=DEFERRED_NOT_ADMITTED")
    print("video_depth_anything_inference=False sfm_rerun=False mvs_rerun=False provider_comparison=False threshold_relaxation=False")
    print("pycolmap_authority=True native_metric_promoted=False model_intrinsics_authority=False permanent_global_view_cap=None")
    print("I5.8B real frozen-foundation Video Depth Anything robustness replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.8A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
