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
from core.reconstruction.i5_auto_policy_admission import build_i5_13a_auto_policy_admission

OUTPUT_NAME = "i5_13a_auto_policy_evidence_synthesis_and_admission.json"

ARTIFACTS = {
    "i4_3g5b": "i4_3g5b_final_g_closure.json",
    "i5_1c": "i5_1c_difficult_handheld_closure.json",
    "i5_2c": "i5_2c_low_overlap_weak_parallax_closure.json",
    "i5_3c": "i5_3c_blur_exposure_clutter_symmetry_closure.json",
    "i5_4c": "i5_4c_forward_motion_kitti_closure.json",
    "i5_5c": "i5_5c_dynamic_scene_closure.json",
    "i5_6c": "i5_6c_mvs_robustness_closure.json",
    "i5_7c": "i5_7c_rolling_depth_robustness_closure.json",
    "i5_8c": "i5_8c_video_depth_anything_robustness_closure.json",
    "i5_9c": "i5_9c_remaining_depth_provider_robustness_closure.json",
    "i5_10d": "i5_10d_mp_sfm_qualification_and_closure.json",
    "i5_11d": "i5_11d_failure_robustness_qualification_and_closure.json",
    "i5_12a": "i5_12a_hybrid_evidence_justification_and_admission.json",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Unable to read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object.")
    return value


def _discover_unique(root: Path, basename: str) -> Path:
    matches = sorted(path.resolve() for path in root.rglob(basename) if path.is_file())
    if len(matches) != 1:
        raise ReconstructionValidationError(
            f"I5.13A expected exactly one {basename!r} under {root}, found {len(matches)}: "
            + ", ".join(str(p) for p in matches)
        )
    return matches[0]


def _binding(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"Evidence file does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _verified_binding(path_value: str | Path, expected_sha: str, *, label: str) -> dict[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.13A evidence-only Auto-policy synthesis/admission review.")
    p.add_argument("--evidence-root", type=Path, required=True)
    p.add_argument("--post-i5-12a-delta-checkpoint", required=True)
    p.add_argument("--post-i5-12a-delta-sha256", required=True)
    p.add_argument("--admission-receipt", required=True)
    p.add_argument("--admission-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    evidence_root = args.evidence_root.expanduser().resolve()
    if not evidence_root.is_dir():
        raise ReconstructionValidationError(f"Evidence root does not exist: {evidence_root}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    discovered: dict[str, Path] = {key: _discover_unique(evidence_root, name) for key, name in ARTIFACTS.items()}
    bindings = {key: _binding(path) for key, path in discovered.items()}
    checkpoint = _verified_binding(
        args.post_i5_12a_delta_checkpoint,
        args.post_i5_12a_delta_sha256,
        label="post-I5.12A source checkpoint",
    )
    receipt = _verified_binding(args.admission_receipt, args.admission_receipt_sha256, label="I5.13A admission receipt")

    result = build_i5_13a_auto_policy_admission(
        i4_3g5b_closure=_read_json(discovered["i4_3g5b"], label="I4.3G5B closure"),
        i4_3g5b_binding=bindings["i4_3g5b"],
        i5_1c_closure=_read_json(discovered["i5_1c"], label="I5.1C closure"),
        i5_1c_binding=bindings["i5_1c"],
        i5_2c_closure=_read_json(discovered["i5_2c"], label="I5.2C closure"),
        i5_2c_binding=bindings["i5_2c"],
        i5_3c_closure=_read_json(discovered["i5_3c"], label="I5.3C closure"),
        i5_3c_binding=bindings["i5_3c"],
        i5_4c_closure=_read_json(discovered["i5_4c"], label="I5.4C closure"),
        i5_4c_binding=bindings["i5_4c"],
        i5_5c_closure=_read_json(discovered["i5_5c"], label="I5.5C closure"),
        i5_5c_binding=bindings["i5_5c"],
        i5_6c_closure=_read_json(discovered["i5_6c"], label="I5.6C closure"),
        i5_6c_binding=bindings["i5_6c"],
        i5_7c_closure=_read_json(discovered["i5_7c"], label="I5.7C closure"),
        i5_7c_binding=bindings["i5_7c"],
        i5_8c_closure=_read_json(discovered["i5_8c"], label="I5.8C closure"),
        i5_8c_binding=bindings["i5_8c"],
        i5_9c_closure=_read_json(discovered["i5_9c"], label="I5.9C closure"),
        i5_9c_binding=bindings["i5_9c"],
        i5_10d_closure=_read_json(discovered["i5_10d"], label="I5.10D closure"),
        i5_10d_binding=bindings["i5_10d"],
        i5_11d_closure=_read_json(discovered["i5_11d"], label="I5.11D closure"),
        i5_11d_binding=bindings["i5_11d"],
        i5_12a_closure=_read_json(discovered["i5_12a"], label="I5.12A closure"),
        i5_12a_binding=bindings["i5_12a"],
        post_i5_12a_checkpoint_binding=checkpoint,
        admission_receipt=_read_json(Path(receipt["path"]), label="I5.13A admission receipt"),
        admission_receipt_binding=receipt,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    sparse = result["evidence_synthesis"]["sparse_sfm_operating_envelope"]
    providers = result["evidence_synthesis"]["provider_operating_envelope"]
    design = result["auto_policy_v1_design"]
    decision = result["decision"]
    print("I5.13A Evidence Synthesis + Auto-Policy Admission/Design: PASS")
    print(f"policy_profile={result['policy_profile']}")
    print(
        "sparse_envelope="
        f"difficult_handheld:{sparse['difficult_handheld']['quality_status']} "
        f"low_overlap_weak_parallax:{sparse['low_overlap_weak_parallax']['quality_status']} "
        f"visual_stress:{sparse['visual_stress']['quality_status']} "
        f"forward_motion:{sparse['forward_motion']['quality_status']} "
        f"dynamic_proxy:{sparse['dynamic_scene_proxy']['quality_status']}"
    )
    print(
        "provider_envelope="
        f"mvs:{providers['mvs']['quality_status']} "
        f"rolling_depth:{providers['rolling_depth']['current_castle_quality_status']} "
        f"vda:{providers['video_depth_anything']['quality_status']} "
        "single_image:35_pass_1_degraded_with_historical_low_texture_rejections_preserved"
    )
    print("auto_principle=" + design["primary_principle"])
    print("abstention_first=True global_score=False cross_family_rank=False runtime_tiebreak=False")
    print("forward_motion_sparse_failure_hard_abstention=True")
    print("historical_rejections_preserved=True hybrid_family_forbidden=True matching_policy_change=False")
    print("provider_execution=False sfm_execution=False mvs_execution=False auto_engine_implemented=False provider_selected=False")
    print("production_auto_authorized=False production_default_modified=False metric_scale=False metrology=False")
    if decision["i5_13b_authorized"]:
        print("I5.13B deterministic evidence-bound Auto policy engine: AUTHORIZED / NOT STARTED (OFFLINE/SHADOW ONLY)")
    print("I5.13C offline replay: NOT AUTHORIZED UNTIL B FREEZE")
    print("I5.13D production integration decision: NOT AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.13A ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
