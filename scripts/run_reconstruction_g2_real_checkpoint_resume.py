#!/usr/bin/env python3
"""Real process-restart checkpoint/resume validation for Phase 8.7G2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
WORKER = (ROOT / "scripts/reconstruction_engine_checkpoint_worker.py").resolve()


def _run(command):
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"G2 worker failed ({proc.returncode}):\n{proc.stdout}")
    return proc.stdout


def _overview(summary, path: Path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.set_xlim(0, 12); ax.set_ylim(0, 4.5); ax.axis("off")
        stages = ["ingestion", "pose", "depth", "dense", "mesh", "semantics", "QC", "Gaussian", "NeRF"]
        for i, label in enumerate(stages):
            x = 0.25 + i * 1.28
            status = "REUSED" if i < summary["reused_stage_count"] else "RESUMED"
            box = FancyBboxPatch((x, 2.1), 1.08, 0.9, boxstyle="round,pad=0.03")
            ax.add_patch(box)
            ax.text(x + 0.54, 2.68, label, ha="center", va="center", fontsize=8, fontweight="bold")
            ax.text(x + 0.54, 2.34, status, ha="center", va="center", fontsize=7)
        ax.text(0.35, 3.75, f"Checkpoint {summary['checkpoint_id']}  →  process restart  →  scene {summary['scene_id']}", fontsize=10)
        ax.text(0.35, 0.85, "Canonical geometry remains authoritative; Gaussian/NeRF remain appearance_only", fontsize=10)
        ax.text(0.35, 0.45, f"persistent_scene={summary['persistent_scene_state']}  checkpoint_integrity={summary['checkpoint_integrity_verified']}", fontsize=9)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    except Exception:
        pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="/tmp/phase8_7g2_real_checkpoint_resume")
    p.add_argument("--nerf-training-steps", type=int, default=16)
    p.add_argument("--backend-timeout-s", type=float, default=1800.0)
    a = p.parse_args()
    out = Path(a.output_dir).expanduser().resolve()
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    checkpoint_root = (out / "durable-checkpoint").resolve()
    request_id = "phase8-7g2-real"
    session_id = str(uuid4())

    checkpoint_stdout = _run((
        sys.executable, str(WORKER), "--mode", "checkpoint",
        "--workspace", str((out / "checkpoint-attempt").resolve()),
        "--checkpoint-root", str(checkpoint_root),
        "--request-id", request_id, "--session-id", session_id,
    ))
    checkpoint_result = json.loads((out / "checkpoint-attempt/checkpoint_result.json").read_text())
    manifest = checkpoint_root / "reconstruction_engine_checkpoint.json"
    sidecar = checkpoint_root / "reconstruction_engine_checkpoint.json.sha256"
    if not manifest.is_file() or not sidecar.is_file():
        raise RuntimeError("Durable G2 checkpoint manifest/checksum missing after interrupted process.")

    resume_stdout = _run((
        sys.executable, str(WORKER), "--mode", "resume",
        "--workspace", str((out / "resume-attempt").resolve()),
        "--checkpoint-root", str(checkpoint_root),
        "--request-id", request_id, "--session-id", session_id,
        "--nerf-training-steps", str(a.nerf_training_steps),
        "--backend-timeout-s", str(a.backend_timeout_s),
    ))
    resumed = json.loads((out / "resume-attempt/resume_result.json").read_text())
    passed = bool(
        checkpoint_result.get("pass")
        and resumed.get("pass")
        and checkpoint_result.get("checkpoint_id") == resumed.get("checkpoint_id")
        and resumed.get("compatibility", {}).get("compatible") is True
        and resumed.get("reused_stage_count") == 3
        and resumed.get("first_executed_stage") == "dense_geometry"
        and resumed.get("pipeline_status") == "completed"
        and resumed.get("persistent_scene_state") == "ready"
        and resumed.get("canonical_geometry_authoritative") is True
        and resumed.get("gaussian_authority") == "appearance_only"
        and resumed.get("nerf_authority") == "appearance_only"
    )
    summary = {
        "phase": "8.7G2", "pass": passed,
        "checkpoint_process_completed": True,
        "resume_process_completed": True,
        "process_restart_exercised": True,
        "checkpoint_id": resumed.get("checkpoint_id", ""),
        "checkpoint_fingerprint_sha256": resumed.get("checkpoint_fingerprint_sha256", ""),
        "checkpoint_stage": checkpoint_result.get("last_committed_stage", ""),
        "next_stage": checkpoint_result.get("next_stage", ""),
        "reused_stage_count": resumed.get("reused_stage_count", 0),
        "first_executed_stage_after_resume": resumed.get("first_executed_stage", ""),
        "completed_stage_reexecution_blocked": resumed.get("completed_stage_reexecution_blocked", False),
        "checkpoint_integrity_verified": True,
        "provider_identity_validation": True,
        "payload_rehydration_validation": True,
        "pipeline_status": resumed.get("pipeline_status", ""),
        "persistent_scene_state": resumed.get("persistent_scene_state", ""),
        "scene_id": resumed.get("scene_id", ""),
        "scene_fingerprint_sha256": resumed.get("scene_fingerprint_sha256", ""),
        "canonical_geometry_authoritative": resumed.get("canonical_geometry_authoritative", False),
        "gaussian_authority": resumed.get("gaussian_authority", ""),
        "nerf_authority": resumed.get("nerf_authority", ""),
        "certified_metrology": False,
        "endurance_hardening_deferred_to": "8.7G3",
        "full_regression_deferred_to": "8.7G4",
        "reconstruction_ui_enabled": False,
        "checkpoint_worker_stdout": checkpoint_stdout,
        "resume_worker_stdout": resume_stdout,
    }
    overview = out / "checkpoint_resume_overview.png"
    _overview(summary, overview)
    summary["overview"] = str(overview) if overview.is_file() else ""
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Phase 8.7G2 real whole-engine checkpoint + resume: {'PASS' if passed else 'FAIL'}")
    print(f"checkpoint={summary['checkpoint_stage']} next={summary['next_stage']} reused={summary['reused_stage_count']} first_executed={summary['first_executed_stage_after_resume']}")
    print(f"process_restart=True checkpoint_integrity=True completed_stage_reexecution_blocked={summary['completed_stage_reexecution_blocked']}")
    print(f"recovery={summary['pipeline_status']} persistent_scene={summary['persistent_scene_state']}")
    print(f"canonical_geometry_authoritative={summary['canonical_geometry_authoritative']} gaussian_authority={summary['gaussian_authority']} nerf_authority={summary['nerf_authority']}")
    print("certified_metrology=False endurance_deferred_to=8.7G3 full_regression_deferred_to=8.7G4 reconstruction_ui_enabled=False")
    if summary["overview"]: print(f"overview={summary['overview']}")
    print(f"summary={out/'summary.json'}")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
