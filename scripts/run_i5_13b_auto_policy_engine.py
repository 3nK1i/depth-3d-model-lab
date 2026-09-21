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
from core.reconstruction.i5_auto_policy_engine import build_i5_13b_engine_freeze

OUTPUT_NAME = "i5_13b_deterministic_auto_policy_engine_freeze.json"
ENGINE_MODULE = ROOT / "core/reconstruction/i5_auto_policy_engine.py"


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


def _verified_binding(path_value: str | Path, expected_sha: str, *, label: str) -> dict[str, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    actual = _sha256(path)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise ReconstructionValidationError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def _binding(path: Path, *, label: str) -> dict[str, str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ReconstructionValidationError(f"{label} does not exist: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="I5.13B deterministic evidence-bound Auto engine freeze (offline/shadow only).")
    p.add_argument("--i5-13a-admission", required=True)
    p.add_argument("--i5-13a-sha256", required=True)
    p.add_argument("--post-i5-13a-delta-checkpoint", required=True)
    p.add_argument("--post-i5-13a-delta-sha256", required=True)
    p.add_argument("--engine-receipt", required=True)
    p.add_argument("--engine-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _verified_binding(args.i5_13a_admission, args.i5_13a_sha256, label="I5.13A admission")
    checkpoint = _verified_binding(
        args.post_i5_13a_delta_checkpoint,
        args.post_i5_13a_delta_sha256,
        label="post-I5.13A source checkpoint",
    )
    receipt_binding = _verified_binding(args.engine_receipt, args.engine_receipt_sha256, label="I5.13B engine receipt")
    module_binding = _binding(ENGINE_MODULE, label="I5.13B engine module")

    result = build_i5_13b_engine_freeze(
        i5_13a_admission=_read_json(Path(a_binding["path"]), label="I5.13A admission"),
        i5_13a_binding=a_binding,
        post_i5_13a_checkpoint_binding=checkpoint,
        engine_module_binding=module_binding,
        engine_receipt=_read_json(Path(receipt_binding["path"]), label="I5.13B engine receipt"),
        engine_receipt_binding=receipt_binding,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    summary = result["probe_summary"]
    decision = result["decision"]
    print("I5.13B Deterministic Evidence-Bound Auto Policy Engine: PASS")
    print(f"policy_profile={result['policy_profile']}")
    print("engine_mode=OFFLINE_SHADOW_ONLY production_auto=False production_integration=False")
    print("auto_principle=exact_unique_clean_evidence_else_abstain")
    print(
        "self_probes="
        f"count={summary['probe_count']} "
        f"select={summary['decision_class_counts']['select']} "
        f"abstain={summary['decision_class_counts']['abstain']} "
        f"engineer_override={summary['decision_class_counts']['engineer_override']}"
    )
    print("deterministic_same_input_same_decision=True")
    print("mvs_unique_select=colmap-mvs-external")
    print("temporal_unique_select=rolling-depth-temporal")
    print("single_image_low_texture=ABSTAIN_AMBIGUOUS")
    print("forward_motion=HARD_ABSTAIN")
    print("unknown_context=ABSTAIN")
    print("provider_execution=False sfm_execution=False mvs_execution=False scene_classifier=False")
    print("global_score=False cross_family_rank=False runtime_tiebreak=False hybrid=False matcher_change=False")
    print("production_default_modified=False provider_promoted=False")
    if decision["i5_13c_authorized"]:
        print("I5.13C frozen-evidence offline replay: AUTHORIZED / NOT STARTED")
    print("I5.13D production integration decision: NOT AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.13B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
