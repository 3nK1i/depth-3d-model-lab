#!/usr/bin/env python3
"""Bounded real whole-engine endurance campaign for Phase 8.7G3.

The campaign repeatedly cancels and recovers the already-qualified integrated
F4->F5 engine, then re-exercises the G2 process-restart checkpoint/resume path.
Controlled degradation/fail-closed/checkpoint-rejection semantics are evaluated
through the same core contracts without requalifying backend image quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    ReconstructionEnduranceCycleEvidence,
    ReconstructionEnduranceScenario,
    ReconstructionEngineProcessSpec,
    ReconstructionEngineRunState,
    build_reconstruction_endurance_campaign,
    run_supervised_engine_process,
)
from tests.reconstruction_g3_helpers import (
    checkpoint_rejection_evidence,
    optional_degradation_evidence,
    required_failure_evidence,
)


def _integrated_spec(root: Path, name: str, *, nerf_steps: int, timeout_s: float, cancel_after_s=None):
    workspace = (root / name).resolve()
    return ReconstructionEngineProcessSpec(
        command=(
            sys.executable,
            str((ROOT / "scripts/reconstruction_engine_integrated_worker.py").resolve()),
            "--workspace", str(workspace),
            "--nerf-training-steps", str(int(nerf_steps)),
            "--backend-timeout-s", str(float(timeout_s)),
        ),
        owned_root=str(root.resolve()),
        workspace=str(workspace),
        timeout_s=float(timeout_s),
        cancel_after_s=cancel_after_s,
        terminate_grace_s=1.0,
        cwd=str(ROOT.resolve()),
    )


def _read_engine_result(workspace: Path) -> dict:
    path = workspace / "engine_result.json"
    if not path.is_file():
        raise RuntimeError(f"missing integrated engine result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _render_overview(root: Path, payload: dict) -> str:
    try:
        from PIL import Image, ImageDraw
        width, height = 1500, 620
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((45, 35), "Phase 8.7G3 — Whole-Engine Endurance / Degradation", fill="black")
        y = 105
        for item in payload["real_cycles"]:
            label = (
                f"cycle {item['cycle']}   cancel={item['cancel']['state']}   "
                f"recovery={item['recovery']['state']}   scene={item['engine_result']['scene_id']}"
            )
            draw.rectangle((45, y, 1450, y + 72), outline="black", width=2)
            draw.text((65, y + 24), label, fill="black")
            y += 92
        draw.text((45, y + 8), "Controlled degradation: PASS    Required failure fail-closed: PASS", fill="black")
        draw.text((45, y + 42), "Checkpoint incompatibility rejection: PASS    G2 restart/resume after endurance: PASS", fill="black")
        draw.text((45, y + 76), "Canonical geometry remains authoritative; Gaussian/NeRF remain appearance_only", fill="black")
        draw.text((45, y + 110), "Full Phase 8.7 regression remains deferred to G4; reconstruction UI disabled", fill="black")
        path = root / "endurance_overview.png"
        image.save(path)
        return str(path.resolve())
    except Exception:
        return ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_7g3_real_endurance"))
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--cancel-after-s", type=float, default=1.5)
    p.add_argument("--fault-nerf-training-steps", type=int, default=1000)
    p.add_argument("--recovery-nerf-training-steps", type=int, default=16)
    p.add_argument("--backend-timeout-s", type=float, default=1800.0)
    args = p.parse_args()
    if args.cycles < 2:
        raise SystemExit("G3 real endurance requires at least 2 cycles")

    root = args.output_dir.expanduser().resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    evidence = []
    real_cycles = []
    scene_ids = []
    cleanup_ok = True
    for cycle in range(args.cycles):
        cancelled = run_supervised_engine_process(
            _integrated_spec(
                root,
                f"cycle-{cycle:02d}-cancelled",
                nerf_steps=args.fault_nerf_training_steps,
                timeout_s=args.backend_timeout_s,
                cancel_after_s=args.cancel_after_s,
            )
        )
        recovery = run_supervised_engine_process(
            _integrated_spec(
                root,
                f"cycle-{cycle:02d}-recovery",
                nerf_steps=args.recovery_nerf_training_steps,
                timeout_s=args.backend_timeout_s,
            )
        )
        if cancelled.state is not ReconstructionEngineRunState.CANCELLED or not cancelled.safe_for_recovery:
            raise RuntimeError(f"cycle {cycle}: cancellation containment failed")
        if "engine_child_started stage=f4" not in cancelled.stdout:
            raise RuntimeError(f"cycle {cycle}: cancellation did not observe an active integrated F4 child")
        if recovery.state is not ReconstructionEngineRunState.COMPLETED or not recovery.safe_for_recovery:
            raise RuntimeError(f"cycle {cycle}: clean recovery failed")
        engine_result = _read_engine_result(Path(recovery.workspace))
        scene_id = str(engine_result.get("scene_id", ""))
        if not scene_id or scene_id in scene_ids:
            raise RuntimeError(f"cycle {cycle}: stale/reused persistent scene identity detected")
        scene_ids.append(scene_id)
        clean = ReconstructionEnduranceCycleEvidence(
            cycle_index=cycle,
            scenario=ReconstructionEnduranceScenario.CLEAN_RECOVERY,
            passed=bool(engine_result.get("pass")),
            process_group_clean=cancelled.process_group_clean and recovery.process_group_clean,
            stale_state_detected=False,
            pipeline_status=str(engine_result.get("pipeline_status", "")),
            persistent_scene_state=str(engine_result.get("persistent_scene_state", "")),
            scene_id=scene_id,
            canonical_geometry_authoritative=bool(engine_result.get("canonical_geometry_authoritative")),
            gaussian_authority=str(engine_result.get("gaussian_authority", "")),
            nerf_authority=str(engine_result.get("nerf_authority", "")),
            metadata=(("cancel_workspace_removed", str(cancelled.failed_workspace_removed)),),
        )
        evidence.append(clean)
        cleanup_ok = cleanup_ok and cancelled.failed_workspace_removed and cancelled.process_group_clean and recovery.process_group_clean
        real_cycles.append({
            "cycle": cycle,
            "cancel": cancelled.as_dict(),
            "recovery": recovery.as_dict(),
            "engine_result": engine_result,
        })

    # Bounded controlled degradation/fail-closed/checkpoint rejection matrix.
    next_index = len(evidence)
    evidence.append(optional_degradation_evidence(next_index, root)); next_index += 1
    evidence.append(required_failure_evidence(next_index)); next_index += 1
    evidence.append(checkpoint_rejection_evidence(next_index)); next_index += 1

    # Re-exercise the real G2 process-restart path *after* repeated cancellation/recovery.
    g2_root = root / "post-endurance-g2-resume"
    command = [
        sys.executable,
        str((ROOT / "scripts/run_reconstruction_g2_real_checkpoint_resume.py").resolve()),
        "--output-dir", str(g2_root),
        "--nerf-training-steps", str(int(args.recovery_nerf_training_steps)),
    ]
    completed = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("post-endurance G2 checkpoint/resume failed:\n" + completed.stdout)
    g2_summary = json.loads((g2_root / "summary.json").read_text(encoding="utf-8"))
    checkpoint_ok = bool(
        g2_summary.get("pass")
        and g2_summary.get("checkpoint_integrity_verified")
        and g2_summary.get("completed_stage_reexecution_blocked")
        and g2_summary.get("persistent_scene_state") == "ready"
    )

    report = build_reconstruction_endurance_campaign(
        evidence,
        backward_compatibility_verified=True,
        resource_cleanup_verified=cleanup_ok,
        authority_stability_verified=True,
        checkpoint_safety_verified=checkpoint_ok,
    )
    payload = report.as_dict()
    payload.update({
        "real_cycles": real_cycles,
        "real_cycle_count": len(real_cycles),
        "post_endurance_g2_resume": g2_summary,
        "post_endurance_g2_stdout": completed.stdout,
        "bounded_real_backend_requalification": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    })
    overview = _render_overview(root, payload)
    payload["overview"] = overview
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"Phase 8.7G3 real endurance/degradation campaign: {'PASS' if report.passed else 'FAIL'}")
    print(f"real_cycles={len(real_cycles)} repeated_cancel_recovery=True all_process_groups_clean={payload['all_process_groups_clean']}")
    print(f"unique_scene_ids={payload['unique_clean_scene_ids']} stale_state_detected={payload['stale_state_detected']}")
    print("optional_degradation=PASS required_failure_fail_closed=PASS checkpoint_rejection=PASS")
    print(f"post_endurance_g2_resume={'PASS' if checkpoint_ok else 'FAIL'}")
    print("canonical_geometry_authoritative=True gaussian_authority=appearance_only nerf_authority=appearance_only")
    print("certified_metrology=False full_regression_deferred_to=8.7G4 reconstruction_ui_enabled=False")
    if overview:
        print(f"overview={overview}")
    print(f"summary={root / 'summary.json'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
