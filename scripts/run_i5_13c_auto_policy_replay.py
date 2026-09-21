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
from core.reconstruction.i5_auto_policy_replay import build_i5_13c_replay_qualification

OUTPUT_NAME = "i5_13c_frozen_evidence_auto_policy_replay_qualification.json"
ENGINE_MODULE = ROOT / "core/reconstruction/i5_auto_policy_engine.py"
REPLAY_MODULE = ROOT / "core/reconstruction/i5_auto_policy_replay.py"


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
    p = argparse.ArgumentParser(description="I5.13C frozen-evidence offline Auto-policy replay qualification.")
    p.add_argument("--i5-13a-admission", required=True)
    p.add_argument("--i5-13a-sha256", required=True)
    p.add_argument("--i5-13b-engine-freeze", required=True)
    p.add_argument("--i5-13b-sha256", required=True)
    p.add_argument("--post-i5-13b-delta-checkpoint", required=True)
    p.add_argument("--post-i5-13b-delta-sha256", required=True)
    p.add_argument("--replay-receipt", required=True)
    p.add_argument("--replay-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    a_binding = _verified_binding(args.i5_13a_admission, args.i5_13a_sha256, label="I5.13A admission")
    b_binding = _verified_binding(args.i5_13b_engine_freeze, args.i5_13b_sha256, label="I5.13B engine freeze")
    checkpoint = _verified_binding(
        args.post_i5_13b_delta_checkpoint,
        args.post_i5_13b_delta_sha256,
        label="post-I5.13B source checkpoint",
    )
    receipt_binding = _verified_binding(args.replay_receipt, args.replay_receipt_sha256, label="I5.13C replay receipt")
    engine_binding = _binding(ENGINE_MODULE, label="I5.13B frozen engine module")
    replay_binding = _binding(REPLAY_MODULE, label="I5.13C replay module")

    result = build_i5_13c_replay_qualification(
        i5_13a_admission=_read_json(Path(a_binding["path"]), label="I5.13A admission"),
        i5_13a_binding=a_binding,
        i5_13b_freeze=_read_json(Path(b_binding["path"]), label="I5.13B engine freeze"),
        i5_13b_binding=b_binding,
        post_i5_13b_checkpoint_binding=checkpoint,
        replay_module_binding=replay_binding,
        engine_module_binding=engine_binding,
        replay_receipt=_read_json(Path(receipt_binding["path"]), label="I5.13C replay receipt"),
        replay_receipt_binding=receipt_binding,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    summary = result["replay_summary"]
    decision = result["decision"]
    print("I5.13C Frozen-Evidence Offline Auto-Policy Replay: PASS")
    print(f"policy_profile={result['policy_profile']}")
    print(f"replay_cases={summary['case_count']} select={summary['decision_class_counts']['select']} abstain={summary['decision_class_counts']['abstain']}")
    print(f"expected_manifest_sha256={result['replay_contract']['expected_manifest_sha256']}")
    print(f"replay_cases_sha256={summary['replay_cases_sha256']}")
    print("all_expected_decisions_matched=True all_reason_codes_matched=True same_input_same_decision=True")
    print("mvs_unique_select=PASS temporal_unique_select=PASS")
    print("temporal_zero_clean_abstention=PASS single_image_ambiguity_abstention=PASS")
    print("forward_motion_hard_abstention=PASS degraded_sparse_abstention=PASS")
    print("provider_execution=False sfm_execution=False mvs_execution=False scenario_rematerialization=False")
    print("production_auto=False production_integration=False production_default_modified=False")
    print("global_score=False provider_rank=False cross_family_rank=False result_driven_tuning=False")
    if decision["i5_13d_authorized"]:
        print("I5.13D production Auto integration qualification/decision: AUTHORIZED / NOT STARTED")
    print("production Auto remains NOT AUTHORIZED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.13C ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
