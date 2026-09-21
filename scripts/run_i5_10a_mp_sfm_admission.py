#!/usr/bin/env python3
"""Generate the execution-free I5.10A MP-SfM inventory/admission artifact."""
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
from core.reconstruction.i5_mp_sfm_evaluation import (  # noqa: E402
    I510A_I59C_FROZEN_SHA256,
    build_i5_10a_mp_sfm_admission,
    verify_i5_10a_repository_anchors,
)

OUTPUT_NAME = "i5_10a_mp_sfm_inventory_and_admission.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.10A could not read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.10A {label} root must be an object: {path}")
    return payload


def _binding(path: Path, expected_sha: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.10A {label} missing: {resolved}")
    expected = str(expected_sha).strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise ReconstructionValidationError(f"I5.10A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.10A {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return I51EvidenceBinding(str(resolved), actual)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-9c-closure", type=Path, required=True)
    parser.add_argument("--i5-9c-sha256", default=I510A_I59C_FROZEN_SHA256)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--admission-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    closure_binding = _binding(args.i5_9c_closure, args.i5_9c_sha256, label="I5.9C closure")
    receipt_binding = _binding(args.admission_receipt, args.admission_receipt_sha256, label="I5.10A admission receipt")
    anchors = verify_i5_10a_repository_anchors(ROOT)

    result = build_i5_10a_mp_sfm_admission(
        i5_9c_closure=_read_json(Path(closure_binding.path), label="I5.9C closure"),
        i5_9c_binding=closure_binding,
        admission_receipt=_read_json(Path(receipt_binding.path), label="I5.10A admission receipt"),
        admission_receipt_binding=receipt_binding,
        repository_anchors=anchors,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    inventory = result["repository_inventory"]
    design = result["experiment_design"]
    candidate = design["candidate"]
    print("I5.10A MP-SfM Inventory + Experiment Admission: PASS")
    print("repository_meaning=matching_strategy_evaluation")
    print("sfm_backend=pycolmap-offline-sfm pycolmap_recommended_version=4.1.1")
    print("supported_matching_strategies=sequential,exhaustive feature_strategy=colmap_sift")
    print("new_matching_algorithm_introduced=False")
    print("historical_horse=exhaustive_global_sfm_completed_dense_bottleneck_not_primary_candidate")
    print("historical_kitti=sequential_overlap10_60_images_incremental_mapping_no_sparse_model")
    print("primary_candidate=kitti_forward_motion exhaustive exactly_once baseline_rerun=False")
    print(
        "candidate_bounds="
        f"profile={candidate['same_profile']} max_keyframes={candidate['same_max_keyframes']} "
        f"image_sequence_fps={candidate['same_image_sequence_fps']} same_source_bytes=True"
    )
    print("comparison=registration_sparse_health_and_scale_invariant_geometry_only absolute_cross_run_pose=False")
    print("execution_free=True sfm_rerun=False mvs_rerun=False provider_inference=False")
    print("threshold_relaxation=False retry_until_pass=False result_driven_retuning=False production_default_modified=False")
    print("pycolmap_authority=True metric_scale=False metrology=False auto=False hybrid=False")
    print("I5.10B MP-SfM runtime/evidence preflight: AUTHORIZED / NOT STARTED")
    print("I5.10C real evaluation: NOT AUTHORIZED UNTIL B FREEZE")
    print("I5.11-I5.14: DEFERRED")
    print(f"repository_anchor_count={len(result['repository_anchors'])}")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.10A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
