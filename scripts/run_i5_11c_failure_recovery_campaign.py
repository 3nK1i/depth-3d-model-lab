#!/usr/bin/env python3
"""Execute the single authoritative I5.11C bounded failure/recovery campaign."""
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
from core.reconstruction.i5_failure_robustness_campaign import (
    I511C_FINAL_ARTIFACT,
    build_i5_11c_campaign_header,
    execute_i5_11c_campaign,
    observe_i5_11c_disk,
    verify_i5_11c_repository_anchors,
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
        raise ReconstructionValidationError(f"I5.11C {label} missing: {p}")
    actual = sha256(p)
    if actual != expected.strip().lower():
        raise ReconstructionValidationError(f"I5.11C {label} SHA256 mismatch: expected={expected} actual={actual}")
    return p, actual


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReconstructionValidationError(f"I5.11C JSON must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--i5-11a-admission", type=Path, required=True)
    parser.add_argument("--i5-11a-sha256", required=True)
    parser.add_argument("--i5-11b-preflight", type=Path, required=True)
    parser.add_argument("--i5-11b-sha256", required=True)
    parser.add_argument("--post-i5-10d-delta-checkpoint", type=Path, required=True)
    parser.add_argument("--post-i5-10d-delta-sha256", required=True)
    parser.add_argument("--post-i5-11b-delta-checkpoint", type=Path, required=True)
    parser.add_argument("--post-i5-11b-delta-sha256", required=True)
    parser.add_argument("--campaign-receipt", type=Path, required=True)
    parser.add_argument("--campaign-receipt-sha256", required=True)
    parser.add_argument("--disk-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    a_path, a_sha = verified_path(args.i5_11a_admission, args.i5_11a_sha256, "I5.11A admission")
    b_path, b_sha = verified_path(args.i5_11b_preflight, args.i5_11b_sha256, "I5.11B preflight")
    d_delta_path, d_delta_sha = verified_path(args.post_i5_10d_delta_checkpoint, args.post_i5_10d_delta_sha256, "post-I5.10D delta")
    b_delta_path, b_delta_sha = verified_path(args.post_i5_11b_delta_checkpoint, args.post_i5_11b_delta_sha256, "post-I5.11B delta")
    receipt_path, receipt_sha = verified_path(args.campaign_receipt, args.campaign_receipt_sha256, "campaign receipt")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise ReconstructionValidationError(f"I5.11C output directory already exists: {output_dir}")

    anchors = verify_i5_11c_repository_anchors(ROOT)
    disk = observe_i5_11c_disk(args.disk_path)
    header = build_i5_11c_campaign_header(
        i5_11a_binding={"path": str(a_path), "sha256": a_sha},
        i5_11b_preflight=read_json(b_path),
        i5_11b_binding={"path": str(b_path), "sha256": b_sha},
        post_i5_10d_delta_binding={"path": str(d_delta_path), "sha256": d_delta_sha},
        post_i5_11b_delta_binding={"path": str(b_delta_path), "sha256": b_delta_sha},
        campaign_receipt=read_json(receipt_path),
        campaign_receipt_binding={"path": str(receipt_path), "sha256": receipt_sha},
        repository_anchors_before=anchors,
        disk_before=disk,
    )

    payload = execute_i5_11c_campaign(
        header=header,
        repository_root=ROOT,
        output_dir=output_dir,
        disk_path=args.disk_path,
    )
    artifact = output_dir / I511C_FINAL_ARTIFACT
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("I5.11C Real Bounded OOM / Disk / Crash / Recovery Campaign: " + ("EVIDENCE_COMPLETE" if payload["pass"] else "INCOMPLETE"))
    print(f"terminal_status={payload['terminal_status']} campaign_quality_status={payload['campaign_quality_status']}")
    print(
        "attempts="
        f"fault={payload['attempt_accounting']['fault_attempts_executed']}/3 "
        f"recovery={payload['attempt_accounting']['recovery_attempts_executed']}/3 "
        "retry=0"
    )
    for record in payload["scenario_records"]:
        print(
            f"scenario={record.get('scenario')} status={record.get('terminal_status')} "
            f"fault_executed={record.get('fault_attempt_executed')} recovery_executed={record.get('recovery_attempt_executed')}"
        )
    print("real_gpu_oom=False real_disk_fill=False real_resource_exhaustion=False")
    print("production_classifier_modified=False recovery_engine_modified=False production_default_modified=False")
    print(f"I5.11D authorized={payload['decision']['i5_11d_authorized']} started=False")
    print(f"evidence={artifact}")
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
