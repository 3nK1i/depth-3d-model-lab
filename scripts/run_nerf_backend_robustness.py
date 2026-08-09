#!/usr/bin/env python3
"""Run Phase 8.7E4 NeRF backend crash/cancel/timeout/recovery validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    NERFSTUDIO_METHOD,
    NerfBackendKind,
    NerfBackendRobustnessEvidence,
    NerfBackendRobustnessReport,
    NerfBackendRobustnessScenario,
    NerfBackendRunState,
    NerfHardwareProfile,
    NerfLifecycleEvidence,
    NerfOptimizationProfile,
    neural_scene_dataset_fingerprint,
    read_neural_scene_dataset_manifest,
)
from scripts.nerf_backend_lifecycle import run_supervised_process  # noqa: E402
from scripts.nerf_e4_support import (  # noqa: E402
    default_env_python,
    isolated_environment,
    load_e3_split,
    preflight,
    worker_environment,
    write_json,
)

DEFAULT_OUTPUT = Path("/tmp/phase8_7e4_nerf_robustness")
SCENARIOS = tuple(NerfBackendRobustnessScenario)


def _pid_exists(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{int(pid)}").exists()


def _load_marker(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fault_mode(scenario: NerfBackendRobustnessScenario) -> str:
    if scenario in {NerfBackendRobustnessScenario.CANCELLATION, NerfBackendRobustnessScenario.TIMEOUT}:
        return "hang"
    if scenario is NerfBackendRobustnessScenario.CRASH:
        return "crash"
    return "partial_artifact"


def _expected_state(scenario: NerfBackendRobustnessScenario) -> NerfBackendRunState:
    if scenario is NerfBackendRobustnessScenario.CANCELLATION:
        return NerfBackendRunState.CANCELLED
    if scenario is NerfBackendRobustnessScenario.TIMEOUT:
        return NerfBackendRunState.TIMED_OUT
    return NerfBackendRunState.CRASHED


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError("refusing to reset E4 path outside lifecycle root") from exc
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def _valid_recovery(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not bool(payload.get("complete")):
        return False
    artifact = payload.get("artifact", {})
    try:
        config = Path(str(artifact["config_path"]))
        checkpoint = Path(str(artifact["checkpoint_path"]))
    except Exception:
        return False
    return bool(
        config.is_file()
        and config.stat().st_size > 0
        and checkpoint.is_file()
        and checkpoint.stat().st_size > 0
        and payload.get("performance")
        and payload.get("artifact_integrity", {}).get("render_probe_passed") is True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--profile", choices=[value.value for value in NerfOptimizationProfile], default="efficient")
    parser.add_argument("--recovery-training-steps", type=int, default=16)
    parser.add_argument("--cancel-after-s", type=float, default=0.5)
    parser.add_argument("--fault-timeout-s", type=float, default=0.8)
    parser.add_argument("--recovery-timeout-s", type=float, default=600.0)
    parser.add_argument("--nerf-python", type=Path, default=default_env_python())
    parser.add_argument("--nerfstudio-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/nerfstudio")
    parser.add_argument("--tiny-cuda-nn-root", type=Path, default=Path.home() / "Desktop/eia-nerf-backends/tiny-cuda-nn")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not 1 <= args.recovery_training_steps <= 1000:
        raise RuntimeError("recovery-training-steps must lie in [1, 1000]")
    if args.cancel_after_s <= 0.0 or args.fault_timeout_s <= 0.0 or args.recovery_timeout_s <= 0.0:
        raise RuntimeError("lifecycle durations must be positive")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    lifecycle_root = output / "lifecycle"
    lifecycle_root.mkdir(parents=True, exist_ok=True)

    python = args.nerf_python.expanduser().resolve()
    nerfstudio_root = args.nerfstudio_root.expanduser().resolve()
    tcnn_root = args.tiny_cuda_nn_root.expanduser().resolve()
    dataset_path = args.dataset_manifest.expanduser().resolve()
    dataset = read_neural_scene_dataset_manifest(dataset_path)
    dataset_fingerprint = neural_scene_dataset_fingerprint(dataset)
    training_ids, validation_ids, _split = load_e3_split(args.split_manifest.expanduser().resolve(), dataset_fingerprint)
    profile = NerfOptimizationProfile(args.profile)

    summary: dict = {
        "phase": "8.7E4",
        "pass": False,
        "scope": "nerf_backend_process_robustness_only",
        "scenario_count": len(SCENARIOS),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
        "whole_engine_hardening_deferred_to": "8.7G",
    }

    try:
        with isolated_environment(python):
            pf = preflight(
                python=python,
                nerfstudio_root=nerfstudio_root,
                tcnn_root=tcnn_root,
                hardware_profile=NerfHardwareProfile.RTX_2060_6GB,
            )
            summary["environment_probe"] = pf["probe"]
            summary["preflight_reasons"] = pf["reasons"]
            if pf["reasons"]:
                raise RuntimeError("preflight failed: " + ",".join(pf["reasons"]))

            evidence_values: list[NerfBackendRobustnessEvidence] = []
            worker = ROOT / "scripts/nerf_workers/nerfstudio_e4_worker.py"
            for scenario in SCENARIOS:
                attempt = lifecycle_root / scenario.value
                _safe_reset(attempt, lifecycle_root)
                marker_path = attempt / "fault_marker.json"
                fault_result = attempt / "worker_result.json"
                request = {
                    "phase": "8.7E4",
                    "request_id": f"phase8-7e4-{scenario.value}",
                    "backend_kind": NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL.value,
                    "method_name": NERFSTUDIO_METHOD,
                    "dataset_manifest_path": str(dataset_path),
                    "dataset_fingerprint_sha256": dataset_fingerprint,
                    "training_frame_ids": list(training_ids),
                    "validation_frame_ids": list(validation_ids),
                    "profile": profile.value,
                    "hardware_profile": NerfHardwareProfile.RTX_2060_6GB.value,
                    "training_steps": int(args.recovery_training_steps),
                    "render_repeats": 3,
                    "output_dir": str(attempt / "worker-output"),
                    "lifecycle_fault_mode": _fault_mode(scenario),
                    "lifecycle_fault_marker_path": str(marker_path),
                }
                request_path = write_json(attempt / "fault_request.json", request)
                outcome = run_supervised_process(
                    [str(python), str(worker), "--request", str(request_path), "--result", str(fault_result)],
                    timeout_s=(float(args.fault_timeout_s) if scenario is NerfBackendRobustnessScenario.TIMEOUT else 15.0),
                    cancel_after_s=(float(args.cancel_after_s) if scenario is NerfBackendRobustnessScenario.CANCELLATION else None),
                    env=worker_environment(
                        nerfstudio_root=nerfstudio_root,
                        tcnn_root=tcnn_root,
                        allow_faults=True,
                    ),
                )

                marker = _load_marker(marker_path)
                child_pid = int(marker.get("child_pid", 0) or 0)
                child_clean = child_pid <= 0 or not _pid_exists(child_pid)
                process_group_clean = bool(outcome.process_group_clean and child_clean)

                partial_detected = False
                if scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT and fault_result.is_file():
                    try:
                        partial_payload = json.loads(fault_result.read_text(encoding="utf-8"))
                        partial_detected = partial_payload.get("partial") is True and partial_payload.get("complete") is False
                    except Exception:
                        partial_detected = False

                expected = _expected_state(scenario)
                if scenario is NerfBackendRobustnessScenario.CANCELLATION:
                    observed_state = NerfBackendRunState.CANCELLED if outcome.cancelled else NerfBackendRunState.FAILED
                elif scenario is NerfBackendRobustnessScenario.TIMEOUT:
                    observed_state = NerfBackendRunState.TIMED_OUT if outcome.timed_out else NerfBackendRunState.FAILED
                else:
                    crashed = (
                        not outcome.cancelled
                        and not outcome.timed_out
                        and outcome.exit_code is not None
                        and outcome.exit_code != 0
                    )
                    observed_state = NerfBackendRunState.CRASHED if crashed else NerfBackendRunState.FAILED
                fault_lifecycle = NerfLifecycleEvidence(
                    state=observed_state,
                    worker_exit_code=outcome.exit_code,
                    cancellation_requested=(scenario is NerfBackendRobustnessScenario.CANCELLATION),
                    error=("" if observed_state is expected else f"unexpected lifecycle outcome: {outcome}"),
                )

                # Remove only fault-attempt artifacts, then run a real backend recovery
                # in the same attempt directory.  This proves the failure did not poison
                # the isolated runtime or leave unusable artifacts behind.
                for candidate in (fault_result, attempt / "worker-output"):
                    if candidate.is_dir():
                        shutil.rmtree(candidate)
                    elif candidate.exists():
                        candidate.unlink()
                partial_cleaned = not fault_result.exists() and not (attempt / "worker-output").exists()

                recovery_request = dict(request)
                recovery_request["lifecycle_fault_mode"] = "none"
                recovery_request.pop("lifecycle_fault_marker_path", None)
                recovery_request["request_id"] = f"phase8-7e4-{scenario.value}-recovery"
                recovery_request["output_dir"] = str(attempt / "recovery-worker-output")
                recovery_request_path = write_json(attempt / "recovery_request.json", recovery_request)
                recovery_result = attempt / "recovery_result.json"
                recovery = run_supervised_process(
                    [str(python), str(worker), "--request", str(recovery_request_path), "--result", str(recovery_result)],
                    timeout_s=float(args.recovery_timeout_s),
                    env=worker_environment(nerfstudio_root=nerfstudio_root, tcnn_root=tcnn_root),
                )
                recovery_valid = bool(
                    recovery.exit_code == 0
                    and not recovery.cancelled
                    and not recovery.timed_out
                    and recovery.process_group_clean
                    and _valid_recovery(recovery_result)
                )
                recovery_lifecycle = NerfLifecycleEvidence(
                    state=NerfBackendRunState.RECOVERED if recovery_valid else NerfBackendRunState.FAILED,
                    worker_exit_code=recovery.exit_code,
                    recovery_attempted=True,
                    recovery_succeeded=recovery_valid,
                    error="" if recovery_valid else "real Nerfacto recovery failed",
                )
                evidence = NerfBackendRobustnessEvidence(
                    scenario=scenario,
                    fault_lifecycle=fault_lifecycle,
                    process_group_clean=process_group_clean,
                    partial_artifacts_detected=(partial_detected if scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT else False),
                    partial_artifacts_cleaned=(partial_cleaned if scenario is NerfBackendRobustnessScenario.PARTIAL_ARTIFACT else False),
                    recovery_lifecycle=recovery_lifecycle,
                    recovery_result_valid=recovery_valid,
                )
                evidence_values.append(evidence)
                print(
                    f"nerfacto {scenario.value}: fault={expected.value} "
                    f"cleanup={process_group_clean} recovery={'PASS' if recovery_valid else 'FAIL'}"
                )

        report = NerfBackendRobustnessReport(evidence=tuple(evidence_values))
        report_path = write_json(output / "robustness_report.json", report.as_dict())
        if not report.passed:
            raise RuntimeError("E4 backend robustness report did not pass")
        summary.update(
            {
                "pass": True,
                "backend_robustness_complete": True,
                "report": report.as_dict(),
                "report_path": str(report_path),
            }
        )
    except Exception as exc:
        summary["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(summary_path, summary)
        print("Phase 8.7E4 NeRF backend robustness: FAIL")
        print(summary["failure"])
        print(f"summary={summary_path}")
        return 1

    write_json(summary_path, summary)
    print("Phase 8.7E4 NeRF backend robustness: PASS")
    print("scenarios=cancellation,timeout,crash,partial_artifact")
    print("process_group_cleanup=True recovery_after_each_fault=True")
    print("backend_process_scope_only=True whole_engine_hardening_deferred_to=8.7G")
    print("canonical_geometry_authoritative=False certified_metrology=False reconstruction_ui_enabled=False")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
