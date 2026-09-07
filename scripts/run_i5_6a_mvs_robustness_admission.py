#!/usr/bin/env python3
"""Run I5.6A MVS robustness scenario admission without executing SfM/MVS."""
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
from core.reconstruction.i5_mvs_robustness import (
    I56A_SCENARIO_IDS,
    analyze_i5_6a_sfm_report,
    build_i5_6a_historical_mvs_anchor,
    build_i5_6a_mvs_robustness_admission,
)

OUTPUT_NAME = "i5_6a_mvs_robustness_admission.json"
_EXPECTED = {
    "castle_control": (19, 19),
    "castle_low_overlap": (7, 6),
    "castle_weak_parallax": (7, 7),
    "castle_overexposure": (19, 19),
    "castle_symmetry": (19, 12),
    "castle_dynamic_proxy": (19, 19),
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i5-5c-closure", type=Path, required=True)
    p.add_argument("--i5-5c-sha256", required=True)
    p.add_argument("--i5-2b-replay", type=Path, required=True)
    p.add_argument("--i5-2b-sha256", required=True)
    p.add_argument("--i5-3b-replay", type=Path, required=True)
    p.add_argument("--i5-3b-sha256", required=True)
    p.add_argument("--i5-4c-closure", type=Path, required=True)
    p.add_argument("--i5-4c-sha256", required=True)
    p.add_argument("--i5-5b-replay", type=Path, required=True)
    p.add_argument("--i5-5b-sha256", required=True)
    p.add_argument("--castle-control-report", type=Path, required=True)
    p.add_argument("--castle-low-overlap-report", type=Path, required=True)
    p.add_argument("--castle-weak-parallax-report", type=Path, required=True)
    p.add_argument("--castle-overexposure-report", type=Path, required=True)
    p.add_argument("--castle-symmetry-report", type=Path, required=True)
    p.add_argument("--castle-dynamic-proxy-report", type=Path, required=True)
    p.add_argument("--admission-receipt", type=Path, required=True)
    p.add_argument("--admission-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected: str, *, label: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6A {label} is missing: {resolved}")
    text = str(expected).strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ReconstructionValidationError(f"I5.6A {label} expected SHA256 is invalid.")
    actual = _sha256(resolved)
    if actual != text:
        raise ReconstructionValidationError(f"I5.6A {label} SHA256 mismatch: expected={text} actual={actual}")
    return resolved, actual


def _json(path: Path, *, label: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReconstructionValidationError(f"I5.6A {label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.6A {label} root must be an object.")
    return payload


def _report(path: Path, *, sid: str) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6A {sid} real_sfm_report is missing: {resolved}")
    n, r = _EXPECTED[sid]
    analysis = analyze_i5_6a_sfm_report(_json(resolved, label=f"{sid} report"), scenario_id=sid, expected_input=n, expected_registered=r)
    analysis["report_binding"] = {"path": str(resolved), "sha256": _sha256(resolved)}
    return analysis


def run(args: argparse.Namespace) -> Path:
    verified = {}
    for key, path, sha in (
        ("i55c", args.i5_5c_closure, args.i5_5c_sha256),
        ("i52b", args.i5_2b_replay, args.i5_2b_sha256),
        ("i53b", args.i5_3b_replay, args.i5_3b_sha256),
        ("i54c", args.i5_4c_closure, args.i5_4c_sha256),
        ("i55b", args.i5_5b_replay, args.i5_5b_sha256),
        ("receipt", args.admission_receipt, args.admission_receipt_sha256),
    ):
        verified[key] = _verify(path, sha, label=key)

    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ReconstructionValidationError(f"I5.6A output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    report_paths = {
        "castle_control": args.castle_control_report,
        "castle_low_overlap": args.castle_low_overlap_report,
        "castle_weak_parallax": args.castle_weak_parallax_report,
        "castle_overexposure": args.castle_overexposure_report,
        "castle_symmetry": args.castle_symmetry_report,
        "castle_dynamic_proxy": args.castle_dynamic_proxy_report,
    }
    reports = {sid: _report(report_paths[sid], sid=sid) for sid in I56A_SCENARIO_IDS}
    anchor = build_i5_6a_historical_mvs_anchor(
        ROOT / "docs/phase_history/I4_2D_TEMPLE_RING_RAW_MVS_DENSE_RECONSTRUCTION.md",
        ROOT / "docs/phase_history/I4_2F_FINAL_TEMPLE_RING_CLOSURE.md",
    )
    def payload(key: str) -> dict:
        return _json(verified[key][0], label=key)
    def binding(key: str) -> I51EvidenceBinding:
        p, s = verified[key]; return I51EvidenceBinding(str(p), s)

    result = build_i5_6a_mvs_robustness_admission(
        i5_5c_result=payload("i55c"), i5_5c_binding=binding("i55c"),
        i5_2b_result=payload("i52b"), i5_2b_binding=binding("i52b"),
        i5_3b_result=payload("i53b"), i5_3b_binding=binding("i53b"),
        i5_4c_result=payload("i54c"), i5_4c_binding=binding("i54c"),
        i5_5b_result=payload("i55b"), i5_5b_binding=binding("i55b"),
        scenario_reports=reports, historical_anchor=anchor,
        admission_receipt=payload("receipt"), admission_receipt_binding=binding("receipt"),
    )
    target = out / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.6A MVS Robustness Scenario Admission + Envelope Design: PASS")
    print(f"source_family=Castle-P19 source_sha256={result['source_family']['frozen_source_asset_set_sha256']}")
    for sid in I56A_SCENARIO_IDS:
        node = result["scenario_matrix"][sid]
        print(f"admitted {sid} registered={node['registered_images']}/{node['input_images']} quality={node['quality_status']} mvs_input_integrity=True")
    print("excluded castle_blur=BLOCKED_UPSTREAM castle_underexposure=BLOCKED_UPSTREAM castle_clutter=NOT_FORMALLY_ADMITTED kitti_forward_motion=BLOCKED_UPSTREAM")
    print("historical_anchor=TempleRing 47/47 evidence_only=True rerun=False")
    print("production_mvs=colmap_mvs adaptive_selection=True geometric_consistency=True sequential=True permanent_global_view_cap=None")
    print("sfm_rerun=False mvs_execution=False dense_provider_rerun=False da3_rerun=False threshold_relaxation=False production_default_modified=False")
    print("pycolmap_authority=True auto=False hybrid=False mp_sfm=False metric_scale=False metrology=False")
    print("I5.6B current frozen-foundation MVS robustness replay: AUTHORIZED")
    print(f"evidence={target}")
    return target


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"I5.6A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
