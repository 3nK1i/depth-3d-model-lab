#!/usr/bin/env python3
"""Final bounded real whole-engine sanity after the G4 full regression passes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PHASE = "8.7G4"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/phase8_7g4_real_e2e"))
    parser.add_argument(
        "--regression-summary",
        type=Path,
        default=Path("/tmp/phase8_7g4_full_regression/summary.json"),
    )
    parser.add_argument("--nerf-training-steps", type=int, default=16)
    parser.add_argument("--backend-timeout-s", type=float, default=1800.0)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    regression_summary = args.regression_summary.expanduser().resolve()
    if not regression_summary.is_file():
        raise RuntimeError("G4 real E2E requires the full-regression summary first.")
    regression = _load(regression_summary)
    if not (
        regression.get("pass") is True
        and regression.get("full_app_regression_performed") is True
        and regression.get("full_app_regression_passed") is True
        and regression.get("ui_readiness_gate_passed") is True
    ):
        raise RuntimeError("G4 full regression did not satisfy the UI-readiness prerequisite.")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_workspace = output_dir / "integrated-engine"
    command = [
        sys.executable,
        str((ROOT / "scripts/reconstruction_engine_integrated_worker.py").resolve()),
        "--workspace",
        str(engine_workspace),
        "--nerf-training-steps",
        str(int(args.nerf_training_steps)),
        "--backend-timeout-s",
        str(float(args.backend_timeout_s)),
    ]
    proc = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (output_dir / "engine.log").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"G4 integrated real engine failed with exit code {proc.returncode}.")

    engine_result_path = engine_workspace / "engine_result.json"
    engine = _load(engine_result_path)
    f4 = _load(engine_workspace / "f4/summary.json")
    f5 = _load(engine_workspace / "f5/summary.json")

    authority_ok = bool(
        engine.get("canonical_geometry_authoritative") is True
        and engine.get("gaussian_authority") == "appearance_only"
        and engine.get("nerf_authority") == "appearance_only"
        and engine.get("certified_metrology") is False
    )
    state_ok = bool(
        engine.get("pass") is True
        and engine.get("pipeline_status") == "completed"
        and engine.get("persistent_scene_state") == "ready"
    )
    ui_ok = bool(engine.get("reconstruction_ui_enabled") is False)
    passed = bool(authority_ok and state_ok and ui_ok and f4.get("pass") and f5.get("pass"))

    payload = {
        "phase": PHASE,
        "pass": passed,
        "phase8_7_complete": passed,
        "phase8_8_ui_handoff_ready": passed,
        "full_regression_passed": True,
        "full_regression_summary": str(regression_summary),
        "pipeline_status": engine.get("pipeline_status"),
        "persistent_scene_state": engine.get("persistent_scene_state"),
        "scene_id": engine.get("scene_id"),
        "scene_fingerprint_sha256": engine.get("scene_fingerprint_sha256"),
        "canonical_geometry_authoritative": engine.get("canonical_geometry_authoritative"),
        "gaussian_authority": engine.get("gaussian_authority"),
        "nerf_authority": engine.get("nerf_authority"),
        "certified_metrology": engine.get("certified_metrology"),
        "reconstruction_ui_enabled": engine.get("reconstruction_ui_enabled"),
        "appearance_contact_sheet": str(engine_workspace / "f4/appearance_contact_sheet.png"),
        "persistent_scene_overview": str(engine_workspace / "f5/persistent_scene_overview.png"),
        "new_reconstruction_algorithms_added": False,
    }
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"G4 final real E2E invariants failed; summary={summary}")

    print("Phase 8.7G4 final real whole-engine sanity: PASS")
    print("full_regression=PASS pipeline=completed persistent_scene=ready")
    print("canonical_geometry_authoritative=True gaussian_authority=appearance_only nerf_authority=appearance_only")
    print("certified_metrology=False reconstruction_ui_enabled=False")
    print("phase8_7_complete=True phase8_8_ui_handoff_ready=True")
    print(f"appearance_contact_sheet={payload['appearance_contact_sheet']}")
    print(f"persistent_scene_overview={payload['persistent_scene_overview']}")
    print(f"summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
