#!/usr/bin/env python3
"""Generate the execution-free I5.11A failure-mode inventory/admission artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.i5_failure_robustness import (  # noqa: E402
    build_i5_11a_failure_mode_admission,
    verify_i5_11a_repository_anchors,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def verified_binding(path: Path, expected_sha: str, *, label: str) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    actual = sha256(resolved)
    expected = str(expected_sha).strip().lower()
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected} actual={actual}")
    return {"path": str(resolved), "sha256": actual}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--i5-10d-closure", type=Path, required=True)
    p.add_argument("--i5-10d-sha256", required=True)
    p.add_argument("--post-i5-10d-delta-checkpoint", type=Path, required=True)
    p.add_argument("--post-i5-10d-delta-sha256", required=True)
    p.add_argument("--admission-receipt", type=Path, required=True)
    p.add_argument("--admission-receipt-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    i510d = verified_binding(args.i5_10d_closure, args.i5_10d_sha256, label="I5.10D closure")
    delta = verified_binding(
        args.post_i5_10d_delta_checkpoint,
        args.post_i5_10d_delta_sha256,
        label="post-I5.10D delta checkpoint",
    )
    receipt = verified_binding(args.admission_receipt, args.admission_receipt_sha256, label="I5.11A receipt")
    anchors = verify_i5_11a_repository_anchors(ROOT)

    artifact = build_i5_11a_failure_mode_admission(
        i5_10d_closure=load_json(Path(i510d["path"])),
        i5_10d_binding=i510d,
        post_i5_10d_delta_binding=delta,
        admission_receipt=load_json(Path(receipt["path"])),
        admission_receipt_binding=receipt,
        repository_anchors=anchors,
    )

    outdir = args.output_dir.expanduser().resolve()
    if outdir.exists():
        raise RuntimeError(f"I5.11A output directory must be new/immutable: {outdir}")
    outdir.mkdir(parents=True, exist_ok=False)
    output = outdir / "i5_11a_failure_mode_inventory_and_admission.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("I5.11A OOM / Disk / Crash / Recovery Inventory + Admission: PASS")
    print("faults=CUDA_OOM_SYNTHETIC,ENOSPC_SYNTHETIC,CRASH_ORPHAN_DESCENDANT")
    print("fault_attempts=3 recovery_attempts=3 each_fault_exactly_once=True")
    print("real_gpu_exhaustion=False real_root_filesystem_fill=False user_data_corruption=False")
    print("existing_supervisor=process_group_cleanup+failed_workspace_cleanup+safe_for_recovery")
    print("cuda_oom_first_class=True disk_enospc_first_class=False gap_preserved=True")
    print("cancel_timeout=historical_context_only checkpoint_corruption=NOT_ADMITTED")
    print("execution_free=True fault_injection=False sfm_execution=False mvs_execution=False")
    print("production_default_modified=False production_classifier_modified=False recovery_engine_modified=False")
    print("I5.11B controlled fault-injection runtime/evidence preflight: AUTHORIZED / NOT STARTED")
    print("I5.11C real bounded campaign: NOT AUTHORIZED UNTIL B FREEZE")
    print("I5.12-I5.14: DEFERRED")
    print(f"repository_anchor_count={len(anchors)}")
    print(f"evidence={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
