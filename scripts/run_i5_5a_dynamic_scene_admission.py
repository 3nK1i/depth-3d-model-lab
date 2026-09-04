#!/usr/bin/env python3
"""Bind I5.5A static control and freeze a controlled independent-motion proxy for I5.5B."""
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
from core.reconstruction.i5_dynamic_scene_handling import (  # noqa: E402
    build_i5_5a_dynamic_scene_admission,
    build_i5_5a_source_identity,
)

OUTPUT_NAME = "i5_5a_dynamic_scene_admission.json"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-4c-closure", type=Path, required=True)
    p.add_argument("--i5-4c-sha256", required=True)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--source-label", required=True)
    p.add_argument(
        "--confirm-static-control-source",
        action="store_true",
        help=(
            "Confirm the bound source is intentionally supplied as the static multiview control for the "
            "paired source-derived dynamic-foreground proxy. This is source provenance, not natural dynamic-scene ground truth."
        ),
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
        raise ReconstructionValidationError(f"I5.5A {label} is missing: {resolved}")
    expected_text = str(expected).strip().lower()
    if len(expected_text) != 64 or any(ch not in "0123456789abcdef" for ch in expected_text):
        raise ReconstructionValidationError(f"I5.5A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected_text:
        raise ReconstructionValidationError(
            f"I5.5A {label} SHA256 mismatch: expected={expected_text} actual={actual}"
        )
    return resolved, actual


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.5A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.5A {label} root must be an object.")
    return payload


def run(args: argparse.Namespace) -> Path:
    closure_path, closure_sha = _verify(args.i5_4c_closure, args.i5_4c_sha256, label="I5.4C closure")
    receipt_path, receipt_sha = _verify(args.admission_receipt, args.admission_receipt_sha256, label="I5.5A admission receipt")

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.5A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    identity = build_i5_5a_source_identity(args.dataset_dir)
    result = build_i5_5a_dynamic_scene_admission(
        i5_4c_result=_json(closure_path, label="I5.4C closure"),
        i5_4c_binding=I51EvidenceBinding(str(closure_path), closure_sha),
        source_identity=identity,
        source_label=args.source_label,
        static_control_confirmation=bool(args.confirm_static_control_source),
        admission_receipt=_json(receipt_path, label="I5.5A admission receipt"),
        admission_receipt_binding=I51EvidenceBinding(str(receipt_path), receipt_sha),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    plan = result["stress_plan"]
    stress = plan["dynamic_foreground_proxy"]
    recipe = stress["materialization_recipe"]
    donor = recipe["donor_box_normalized"]
    print("I5.5A Dynamic-Scene Scenario Admission + Stress Design: PASS")
    print(
        f"source_label={result['source_semantics']['label']} source_images={identity['image_count']} "
        f"source_asset_set_sha256={identity['asset_set_sha256']}"
    )
    print(
        f"paired_scope control={plan['static_control']['image_count']} dynamic_proxy={stress['image_count']} "
        f"matching={plan['static_control']['matching_policy']} max_keyframes={plan['static_control']['max_keyframes']}"
    )
    print(
        f"dynamic_proxy donor_index={recipe['donor_source_index']} donor_name={recipe['donor_source_name']} "
        f"patch={donor['width']:.2f}x{donor['height']:.2f} patch_area_fraction={stress['patch_area_fraction']:.4f}"
    )
    print(
        f"trajectory_x={recipe['target_x_start_fraction']:.2f}->{recipe['target_x_end_fraction']:.2f} "
        f"target_y={recipe['target_y_fraction']:.2f} horizontal_displacement_fraction={stress['horizontal_displacement_fraction']:.2f} "
        f"recipe_sha256={stress['recipe_sha256']}"
    )
    print(
        "controlled_independent_foreground_motion_proxy=True natural_dynamic_scene_ground_truth=False "
        "real_world_dynamic_scene_robustness_claimed=False"
    )
    print(
        "stress_images_materialized=False sfm_rerun=False dense_provider_rerun=False dynamic_masking_added=False "
        "semantic_segmentation_added=False motion_outlier_filter_added=False"
    )
    print(
        "pycolmap_authority=True thresholds_modified=False production_default_modified=False mp_sfm=False "
        "auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print("I5.5B current frozen-foundation dynamic-scene replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.5A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
