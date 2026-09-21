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
from core.reconstruction.i5_auto_policy_production_qualification import (
    build_i5_13d_production_integration_decision,
    inspect_i5_13d_production_integration_surface,
)

OUTPUT_NAME = "i5_13d_auto_policy_production_integration_decision.json"
ANCHORS = {
    "dense_provider_architecture": ROOT / "core/reconstruction/dense_provider_architecture.py",
    "adaptive_production_planning": ROOT / "core/reconstruction/adaptive_production_planning.py",
    "reconstruction_page": ROOT / "app/widgets/reconstruction_page.py",
    "i5_13b_engine_module": ROOT / "core/reconstruction/i5_auto_policy_engine.py",
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
    p = argparse.ArgumentParser(description="I5.13D evidence-only production Auto integration qualification/decision.")
    p.add_argument("--i5-13c-replay-qualification", required=True)
    p.add_argument("--i5-13c-sha256", required=True)
    p.add_argument("--post-i5-13c-delta-checkpoint", required=True)
    p.add_argument("--post-i5-13c-delta-sha256", required=True)
    p.add_argument("--decision-receipt", required=True)
    p.add_argument("--decision-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p


def main() -> int:
    args = _parser().parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    c_binding = _verified_binding(args.i5_13c_replay_qualification, args.i5_13c_sha256, label="I5.13C replay qualification")
    checkpoint = _verified_binding(
        args.post_i5_13c_delta_checkpoint,
        args.post_i5_13c_delta_sha256,
        label="post-I5.13C source checkpoint",
    )
    receipt_binding = _verified_binding(args.decision_receipt, args.decision_receipt_sha256, label="I5.13D decision receipt")
    anchor_bindings = {name: _binding(path, label=name) for name, path in ANCHORS.items()}
    inventory = inspect_i5_13d_production_integration_surface(ROOT)

    result = build_i5_13d_production_integration_decision(
        i5_13c_qualification=_read_json(Path(c_binding["path"]), label="I5.13C replay qualification"),
        i5_13c_binding=c_binding,
        post_i5_13c_checkpoint_binding=checkpoint,
        integration_inventory=inventory,
        integration_anchor_bindings=anchor_bindings,
        decision_receipt=_read_json(Path(receipt_binding["path"]), label="I5.13D decision receipt"),
        decision_receipt_binding=receipt_binding,
    )

    target = output / OUTPUT_NAME
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    q = result["qualification"]
    d = result["decision"]
    inv = result["production_integration_inventory"]
    print("I5.13D Auto Policy Production Integration Qualification/Decision: PASS")
    print(f"policy_profile={result['policy_profile']}")
    print(f"offline_policy_qualified={q['offline_policy_qualified']} offline_replay_qualified={q['offline_replay_qualified']}")
    print(f"offline_replay=cases:{q['offline_replay_cases']} select:{q['offline_replay_selects']} abstain:{q['offline_replay_abstentions']}")
    print("production_auto_qualified=False production_auto_authorized=False")
    print("integration_decision=DEFER_PRODUCTION_AUTO")
    print(f"adaptive_planning={inv['adaptive_production_planning_mode']} ui_provider_control={inv['production_ui_provider_selection_mode']}")
    print(f"live_context_mapping_qualified={inv['live_context_to_evidence_token_mapping_qualified']} production_auto_entrypoint={inv['production_auto_orchestration_entrypoint_implemented']}")
    print("provider_execution=False sfm_execution=False mvs_execution=False scene_classifier_execution=False")
    print("production_default_modified=False provider_promoted=False global_score=False provider_rank=False")
    print("blocking_gaps=" + ",".join(result["blocking_gaps"]))
    print("I5.13: CLOSED / PASS (OFFLINE/SHADOW AUTO QUALIFIED; PRODUCTION AUTO DEFERRED)")
    if d["i5_14_authorized"]:
        print("I5.14 final I5 qualification and closure: AUTHORIZED / NOT STARTED")
    print(f"evidence={target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.13D ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
