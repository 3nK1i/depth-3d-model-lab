#!/usr/bin/env python3
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
from core.reconstruction.i5_remaining_depth_provider_robustness import (
    I59A_B1_PROVIDERS,
    I59A_B2_PROVIDERS,
    build_i5_9a_remaining_depth_provider_robustness_admission,
)

OUTPUT_NAME = "i5_9a_remaining_depth_provider_robustness_admission.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return payload


def _binding(path_value: str, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    path = Path(path_value).expanduser().resolve()
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}"
        )
    return I51EvidenceBinding(str(path), actual)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="I5.9A execution-free remaining single-image depth-provider robustness inventory and admission."
    )
    parser.add_argument("--i5-8c-closure", required=True)
    parser.add_argument("--i5-8c-sha256", required=True)
    parser.add_argument("--i5-6a-admission", required=True)
    parser.add_argument("--i5-6a-sha256", required=True)
    parser.add_argument("--i4-3g4c-closure", required=True)
    parser.add_argument("--i4-3g4c-sha256", required=True)
    parser.add_argument("--admission-receipt", required=True)
    parser.add_argument("--admission-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    i58c_binding = _binding(args.i5_8c_closure, args.i5_8c_sha256, label="I5.8C closure")
    i56a_binding = _binding(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    g4c_binding = _binding(args.i4_3g4c_closure, args.i4_3g4c_sha256, label="I4.3G4C closure")
    receipt_binding = _binding(args.admission_receipt, args.admission_receipt_sha256, label="I5.9A admission receipt")

    result = build_i5_9a_remaining_depth_provider_robustness_admission(
        i5_8c_closure=_read_json(Path(i58c_binding.path), label="I5.8C closure"),
        i5_8c_binding=i58c_binding,
        i5_6a_admission=_read_json(Path(i56a_binding.path), label="I5.6A admission"),
        i5_6a_binding=i56a_binding,
        i4_3g4c_closure=_read_json(Path(g4c_binding.path), label="I4.3G4C closure"),
        i4_3g4c_binding=g4c_binding,
        admission_receipt=_read_json(Path(receipt_binding.path), label="I5.9A admission receipt"),
        admission_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    hist = result["historical_g4_anchors"]
    print("I5.9A Remaining Depth-Provider Robustness Inventory + Admission: PASS")
    print("providers=" + ",".join(result["provider_inventory"]))
    print("B1=" + ",".join(I59A_B1_PROVIDERS))
    print("B2=" + ",".join(I59A_B2_PROVIDERS))
    print("scenario_count=6 batches=I5.9B1_then_I5.9B2")
    print(
        "historical_g4="
        f"complete_room:{','.join(hist['complete_room'].values())};"
        "low_texture:"
        + ",".join(
            f"{provider}:{hist['low_texture'][provider]}"
            for provider in result["provider_inventory"]
        )
        + ";"
        f"weak_overlap:{','.join(hist['weak_overlap'].values())}"
    )
    print("historical_low_texture_rejections=depth_anything_v2,marigold preserved=True")
    print("production_max_voxels=1500000 minimum_accepted_views=2 minimum_correspondences=8 maximum_relative_residual=0.35")
    print("fusion_capacity_drops_change_single_image_status=False")
    print("execution_free=True single_image_provider_inference=False sfm_rerun=False mvs_rerun=False")
    print("provider_comparison=False provider_ranking=False threshold_relaxation=False production_default_modified=False")
    print("pycolmap_authority=True metric_scale=False metrology=False permanent_global_view_cap=None")
    print("I5.9B1 metric-like single-image provider robustness replay: AUTHORIZED")
    print("I5.9B2 relative-depth single-image provider robustness replay: AUTHORIZED AFTER B1 FREEZE")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.9A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
