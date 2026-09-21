#!/usr/bin/env python3
"""Formal I5.11B controlled fault-injection runtime/evidence preflight."""
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
from core.reconstruction.i5_failure_robustness_preflight import (
    build_i5_11b_preflight,
    discover_live_i5_11_fault_workers,
    observe_disk_space,
    probe_failure_classifier_without_fault_execution,
    verify_i5_11b_repository_anchors,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_path(path: Path, expected: str, label: str) -> tuple[Path, str]:
    p = path.expanduser().resolve()
    if not p.is_file():
        raise ReconstructionValidationError(f"I5.11B {label} missing: {p}")
    actual = sha256(p)
    if actual != expected.lower():
        raise ReconstructionValidationError(
            f"I5.11B {label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return p, actual


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I5.11B JSON must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i5-11a-admission", type=Path, required=True)
    parser.add_argument("--i5-11a-sha256", required=True)
    parser.add_argument("--post-i5-10d-delta-checkpoint", type=Path, required=True)
    parser.add_argument("--post-i5-10d-delta-sha256", required=True)
    parser.add_argument("--preflight-receipt", type=Path, required=True)
    parser.add_argument("--preflight-receipt-sha256", required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a_path, a_sha = verified_path(args.i5_11a_admission, args.i5_11a_sha256, "I5.11A admission")
    delta_path, delta_sha = verified_path(
        args.post_i5_10d_delta_checkpoint,
        args.post_i5_10d_delta_sha256,
        "post-I5.10D delta checkpoint",
    )
    receipt_path, receipt_sha = verified_path(args.preflight_receipt, args.preflight_receipt_sha256, "preflight receipt")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ReconstructionValidationError(f"I5.11B output directory already exists: {output_dir}")

    anchors = verify_i5_11b_repository_anchors(ROOT)
    classifier = probe_failure_classifier_without_fault_execution()
    disk = observe_disk_space(args.disk_path)
    live = discover_live_i5_11_fault_workers()

    payload = build_i5_11b_preflight(
        i5_11a_admission=read_json(a_path),
        i5_11a_binding={"path": str(a_path), "sha256": a_sha},
        post_i5_10d_delta_binding={"path": str(delta_path), "sha256": delta_sha},
        preflight_receipt=read_json(receipt_path),
        preflight_receipt_binding={"path": str(receipt_path), "sha256": receipt_sha},
        repository_anchors=anchors,
        classifier_probe=classifier,
        disk_observation=disk,
        live_fault_workers=live,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    artifact = output_dir / "i5_11b_failure_injection_runtime_evidence_preflight.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("I5.11B Controlled Fault-Injection Runtime/Evidence Preflight: PASS")
    print(
        "classifier="
        f"cuda_oom:{classifier['cuda_oom']['observed_kind']} first_class={classifier['cuda_oom']['first_class']} "
        f"enospc:{classifier['disk_enospc']['observed_kind']} first_class={classifier['disk_enospc']['first_class']} "
        f"gap_preserved={classifier['disk_enospc']['gap_preserved']}"
    )
    print(
        f"disk_free_bytes={disk['free_bytes']} minimum_free_bytes_for_c={disk['minimum_free_bytes_for_i5_11c']} "
        f"safety_floor_met={disk['safety_floor_met']} pressure_performed={disk['pressure_performed']}"
    )
    print("live_fault_workers=0")
    print("faults=CUDA_OOM_SYNTHETIC,ENOSPC_SYNTHETIC,CRASH_ORPHAN_DESCENDANT")
    print("fault_attempts=3 recovery_attempts=3 each_exactly_once=True")
    print("fault_worker=scripts/i5_11_bounded_fault_worker.py fault_worker_executed=False")
    print("real_gpu_exhaustion=False real_disk_fill=False")
    print("fault_injection=False supervised_fault_worker_execution=False")
    print("production_classifier_modified=False recovery_engine_modified=False production_default_modified=False")
    print("I5.11C real bounded campaign: AUTHORIZED / NOT STARTED")
    print("I5.11D and I5.12-I5.14: DEFERRED")
    print(f"repository_anchor_count={len(anchors)}")
    print(f"evidence={artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
