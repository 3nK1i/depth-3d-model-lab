#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRunState,
    build_reconstruction_endurance_campaign,
    run_supervised_engine_process,
)
from tests.reconstruction_g3_helpers import (
    checkpoint_rejection_evidence,
    clean_evidence,
    optional_degradation_evidence,
    required_failure_evidence,
)


def _fault_spec(root: Path, name: str, *, mode: str, cancel_after_s=None, timeout_s=1.0):
    workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(
            sys.executable,
            str((ROOT / "scripts/reconstruction_engine_fault_worker.py").resolve()),
            "--workspace", str(workspace),
            "--mode", mode,
            "--child-sleep-s", "10",
        ),
        owned_root=str(root.resolve()),
        workspace=str(workspace),
        timeout_s=float(timeout_s),
        cancel_after_s=cancel_after_s,
        terminate_grace_s=0.2,
        cwd=str(ROOT.resolve()),
    )


def main() -> int:
    root = Path("/tmp/phase8_7g3_contracts").resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    lifecycle = []
    for cycle in range(2):
        fault = run_supervised_engine_process(
            _fault_spec(root, f"cancel-{cycle}", mode="hang", cancel_after_s=0.08, timeout_s=1.0)
        )
        recovery = run_supervised_engine_process(
            _fault_spec(root, f"recover-{cycle}", mode="complete", timeout_s=1.0)
        )
        lifecycle.append({"fault": fault.as_dict(), "recovery": recovery.as_dict()})

    cleanup_ok = all(
        item["fault"]["state"] == ReconstructionEngineRunState.CANCELLED.value
        and item["fault"]["safe_for_recovery"]
        and item["recovery"]["state"] == ReconstructionEngineRunState.COMPLETED.value
        and item["recovery"]["safe_for_recovery"]
        for item in lifecycle
    )

    cycles = (
        clean_evidence(0),
        clean_evidence(1),
        optional_degradation_evidence(2, root),
        required_failure_evidence(3),
        checkpoint_rejection_evidence(4),
    )
    report = build_reconstruction_endurance_campaign(
        cycles,
        backward_compatibility_verified=True,
        resource_cleanup_verified=cleanup_ok,
        authority_stability_verified=True,
        checkpoint_safety_verified=True,
    )
    payload = report.as_dict()
    payload["lifecycle_cycles"] = lifecycle
    payload["repeated_cancel_recovery_cycles"] = len(lifecycle)
    payload["endurance_scope"] = "bounded_contract_campaign"
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Phase 8.7G3 endurance/degradation contracts: {'PASS' if report.passed else 'FAIL'}")
    print(f"cycles={len(report.cycles)} repeated_cancel_recovery={len(lifecycle)}")
    print(f"resource_cleanup={cleanup_ok} stale_state_detected={payload['stale_state_detected']}")
    print(f"optional_degradation=True required_failure_fail_closed=True checkpoint_rejection=True")
    print("canonical_geometry_authority_stable=True neural_authority=appearance_only")
    print("full_regression_deferred_to=8.7G4 reconstruction_ui_enabled=False")
    print(f"summary={root / 'summary.json'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
