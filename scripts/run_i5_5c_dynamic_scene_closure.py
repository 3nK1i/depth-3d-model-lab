#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_dynamic_scene_handling import build_i5_5c_dynamic_scene_closure

OUTPUT_NAME = "i5_5c_dynamic_scene_closure.json"


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
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return I51EvidenceBinding(str(path), actual)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="I5.5C evidence-only dynamic-scene robustness qualification and closure.")
    parser.add_argument("--i5-5a-admission", required=True)
    parser.add_argument("--i5-5a-sha256", required=True)
    parser.add_argument("--i5-5b-replay", required=True)
    parser.add_argument("--i5-5b-sha256", required=True)
    parser.add_argument("--closure-receipt", required=True)
    parser.add_argument("--closure-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _binding(args.i5_5a_admission, args.i5_5a_sha256, label="I5.5A admission")
    b_binding = _binding(args.i5_5b_replay, args.i5_5b_sha256, label="I5.5B real replay")
    receipt_binding = _binding(args.closure_receipt, args.closure_receipt_sha256, label="I5.5C closure receipt")

    result = build_i5_5c_dynamic_scene_closure(
        i5_5a_result=_read_json(Path(a_binding.path), label="I5.5A admission"),
        i5_5a_binding=a_binding,
        i5_5b_result=_read_json(Path(b_binding.path), label="I5.5B replay"),
        i5_5b_binding=b_binding,
        qualification_receipt=_read_json(Path(receipt_binding.path), label="I5.5C closure receipt"),
        qualification_receipt_binding=receipt_binding,
    )
    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    q = result["qualification"]
    d = result["directional_quality_evidence"]
    c = result["paired_comparison"]
    print("I5.5C Dynamic-Scene Handling Robustness Qualification + Closure: PASS")
    print(
        f"quality_status={q['quality_status']} control={q['control_registration']} "
        f"dynamic_proxy={q['dynamic_proxy_registration']} "
        f"components={q['control_sparse_component_count']}->{q['dynamic_proxy_sparse_component_count']}"
    )
    print(
        f"landmarks={c.get('landmarks_control')}->{c.get('landmarks_dynamic_proxy')} "
        f"retention_ratio={d.get('landmark_retention_ratio')} "
        f"observations={c.get('observations_control')}->{c.get('observations_dynamic_proxy')} "
        f"retention_ratio={d.get('observation_retention_ratio')}"
    )
    print(
        f"reprojection_px={c.get('mean_reprojection_error_px_control')}->{c.get('mean_reprojection_error_px_dynamic_proxy')} "
        f"ratio={d.get('reprojection_error_stress_to_control_ratio')} "
        f"mean_track_length={c.get('mean_track_length_control')}->{c.get('mean_track_length_dynamic_proxy')} "
        f"ratio={d.get('mean_track_length_stress_to_control_ratio')}"
    )
    print(
        f"directional_sparse_quality_penalty_observed={d['directional_sparse_quality_penalty_observed']} "
        f"classification_threshold_used={d['classification_threshold_used']}"
    )
    print(f"operating_envelope={q['operating_envelope']}")
    print(
        "controlled_proxy=True natural_dynamic_scene_ground_truth=False "
        "real_world_dynamic_scene_robustness=False universal_dynamic_scene_robustness=False"
    )
    print(
        "evidence_only=True sfm_rerun=False source_rematerialized=False stress_images_rematerialized=False "
        "dynamic_masking=False semantic_segmentation=False motion_outlier_filter=False"
    )
    print(
        "pycolmap_authority=True threshold_relaxation=False production_default_modified=False "
        "mp_sfm=False auto=False hybrid=False metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print(f"I5.5 dynamic-scene handling: CLOSED / {q['quality_status'].upper()}")
    print("I5.6 MVS robustness envelope: AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.5C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
