#!/usr/bin/env python3
"""Real isolated-backend lifecycle robustness smoke for Phase 8.7D3 Step 4."""
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

from core.reconstruction import (
    GaussianBackendKind,
    GaussianBackendRobustnessEvidence,
    GaussianBackendRobustnessReport,
    GaussianBackendRobustnessScenario,
    GaussianBenchmarkHardwareProfile,
    GaussianBenchmarkRunState,
    GaussianLifecycleEvidence,
    GaussianOptimizationProfile,
    RealSceneGaussianBenchmarkSpec,
    read_neural_scene_dataset_manifest,
)
from scripts.gaussian_backend_lifecycle import run_supervised_process
from scripts.gaussian_benchmark_workers.common import LIFECYCLE_FAULT_ENV
from scripts.run_gaussian_backend_qualification import _isolated_backend_environment
from scripts.run_gaussian_heldout_quality_smoke import (
    _default_env_python,
    prepare_heldout_smoke_inputs,
)
from scripts.run_gaussian_real_scene_benchmark import _backend_request, _preflight_backend, _write_json

DEFAULT_OUTPUT = Path("/tmp/phase8_7d3_gaussian_backend_robustness")
SCENARIOS = (
    GaussianBackendRobustnessScenario.CANCELLATION,
    GaussianBackendRobustnessScenario.TIMEOUT,
    GaussianBackendRobustnessScenario.CRASH,
    GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT,
)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_marker(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _safe_reset_attempt(path: Path, *, lifecycle_root: Path) -> None:
    target = path.resolve()
    root = lifecycle_root.resolve()
    if root not in target.parents:
        raise RuntimeError(f"refusing to clean outside lifecycle root: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def _validate_recovery_result(path: Path, expected_backend: GaussianBackendKind) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend_kind") != expected_backend.value:
            return False
        if payload.get("partial") is True:
            return False
        scene_path = Path(str(payload["scene_path"])).resolve()
        manifest_path = Path(str(payload["render_manifest_path"])).resolve()
        if not scene_path.is_file() or not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("backend_kind") != expected_backend.value:
            return False
        if not manifest.get("renders"):
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _fault_mode(scenario: GaussianBackendRobustnessScenario) -> str:
    if scenario in {GaussianBackendRobustnessScenario.CANCELLATION, GaussianBackendRobustnessScenario.TIMEOUT}:
        return "hang_with_child"
    if scenario is GaussianBackendRobustnessScenario.CRASH:
        return "crash"
    return "partial_artifact_crash"


def _expected_state(scenario: GaussianBackendRobustnessScenario) -> GaussianBenchmarkRunState:
    return {
        GaussianBackendRobustnessScenario.CANCELLATION: GaussianBenchmarkRunState.CANCELLED,
        GaussianBackendRobustnessScenario.TIMEOUT: GaussianBenchmarkRunState.TIMED_OUT,
        GaussianBackendRobustnessScenario.CRASH: GaussianBenchmarkRunState.CRASHED,
        GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT: GaussianBenchmarkRunState.CRASHED,
    }[scenario]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("gsplat", "graphdeco", "both"), default="both")
    parser.add_argument("--gsplat-python", type=Path, default=_default_env_python("depth3d-gaussian-gsplat"))
    parser.add_argument("--graphdeco-python", type=Path, default=_default_env_python("depth3d-gaussian-graphdeco"))
    parser.add_argument(
        "--graphdeco-root", type=Path,
        default=Path.home() / "Desktop" / "depth3d-gaussian-backends" / "gaussian-splatting",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--recovery-training-steps", type=int, default=16)
    parser.add_argument("--cancel-after-s", type=float, default=0.75)
    parser.add_argument("--timeout-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.recovery_training_steps < 1 or args.recovery_training_steps > 1000:
        raise RuntimeError("recovery-training-steps must lie in [1, 1000]")
    if args.cancel_after_s <= 0.0 or args.timeout_s <= 0.0:
        raise RuntimeError("cancel/timeout durations must be positive")

    output = args.output_dir.expanduser().resolve()
    manifest_path, seed_path = prepare_heldout_smoke_inputs(output)
    lifecycle_root = output / "lifecycle"
    lifecycle_root.mkdir(parents=True, exist_ok=True)
    dataset = read_neural_scene_dataset_manifest(manifest_path)
    dataset_ids = tuple(frame.frame_id for frame in dataset.frames)
    validation_ids = ("heldout-smoke-4",)
    training_ids = tuple(value for value in dataset_ids if value not in set(validation_ids))

    selected: list[tuple[str, GaussianBackendKind, Path, Path | None]] = []
    if args.backend in {"gsplat", "both"}:
        selected.append(("gsplat", GaussianBackendKind.GSPLAT_EXTERNAL, args.gsplat_python, None))
    if args.backend in {"graphdeco", "both"}:
        selected.append((
            "graphdeco", GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
            args.graphdeco_python, args.graphdeco_root,
        ))

    spec = RealSceneGaussianBenchmarkSpec(
        benchmark_id="phase8-7d3-backend-robustness",
        source_label="deterministic-backend-lifecycle-smoke",
        dataset=dataset,
        training_frame_ids=training_ids,
        validation_frame_ids=validation_ids,
        backends=tuple(item[1] for item in selected),
        optimization_profile=GaussianOptimizationProfile.EFFICIENT,
        hardware_profile=GaussianBenchmarkHardwareProfile.CUSTOM,
        hardware_label="backend-lifecycle-only",
    )

    all_evidence: list[GaussianBackendRobustnessEvidence] = []
    for backend_name, backend_kind, python_path, root_path in selected:
        backend_python = python_path.expanduser().resolve()
        graphdeco_root = root_path.expanduser().resolve() if root_path is not None else None
        _preflight_backend(backend_kind, backend_python, graphdeco_root)
        worker_script = ROOT / "scripts" / "gaussian_benchmark_workers" / (
            "gsplat_worker.py" if backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL else "graphdeco_worker.py"
        )
        for scenario in SCENARIOS:
            attempt_dir = lifecycle_root / backend_name / scenario.value
            _safe_reset_attempt(attempt_dir, lifecycle_root=lifecycle_root)
            marker_path = attempt_dir / "fault_marker.json"
            fault_result = attempt_dir / "backend_result.json"
            request_payload = _backend_request(
                spec=spec,
                backend_kind=backend_kind,
                dataset_manifest=manifest_path,
                seed_path=seed_path,
                output_dir=attempt_dir / "worker-output",
                training_steps=int(args.recovery_training_steps),
                measure_performance=False,
                render_repeats=1,
            )
            request_payload.update({
                "lifecycle_fault_mode": _fault_mode(scenario),
                "lifecycle_fault_marker_path": str(marker_path),
            })
            request_path = _write_json(attempt_dir / "fault_request.json", request_payload)

            with _isolated_backend_environment(backend_python):
                env = os.environ.copy()
                env[LIFECYCLE_FAULT_ENV] = "1"
                if graphdeco_root is not None:
                    env["DEPTH3D_GRAPHDECO_ROOT"] = str(graphdeco_root)
                outcome = run_supervised_process(
                    [str(backend_python), str(worker_script), "--request", str(request_path), "--result", str(fault_result)],
                    timeout_s=(float(args.timeout_s) if scenario is GaussianBackendRobustnessScenario.TIMEOUT else 15.0),
                    cancel_after_s=(float(args.cancel_after_s) if scenario is GaussianBackendRobustnessScenario.CANCELLATION else None),
                    env=env,
                )

            marker = _load_marker(marker_path)
            child_pid = int(marker.get("child_pid", 0) or 0)
            child_clean = child_pid <= 0 or not _pid_exists(child_pid)
            process_group_clean = bool(outcome.process_group_clean and child_clean)
            partial_detected = bool(fault_result.is_file())
            if scenario is GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT:
                try:
                    partial_payload = json.loads(fault_result.read_text(encoding="utf-8"))
                    partial_detected = bool(partial_payload.get("partial") is True)
                except Exception:
                    partial_detected = False

            if scenario is GaussianBackendRobustnessScenario.CANCELLATION:
                fault_lifecycle = GaussianLifecycleEvidence(
                    state=GaussianBenchmarkRunState.CANCELLED,
                    worker_exit_code=outcome.exit_code,
                    cancellation_requested=True,
                    error="",
                )
            elif scenario is GaussianBackendRobustnessScenario.TIMEOUT:
                fault_lifecycle = GaussianLifecycleEvidence(
                    state=GaussianBenchmarkRunState.TIMED_OUT,
                    worker_exit_code=outcome.exit_code,
                    error="backend worker exceeded lifecycle timeout",
                )
            else:
                fault_lifecycle = GaussianLifecycleEvidence(
                    state=GaussianBenchmarkRunState.CRASHED,
                    worker_exit_code=outcome.exit_code,
                    error="intentional backend lifecycle fault",
                )

            # Cleanup only the failed attempt artifacts. Then reuse the same attempt
            # directory for a real worker run, proving the failure did not poison the backend.
            for candidate in (fault_result, attempt_dir / "worker-output"):
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                elif candidate.exists():
                    candidate.unlink()
            partial_cleaned = not fault_result.exists() and not (attempt_dir / "worker-output").exists()

            recovery_request = dict(request_payload)
            recovery_request["lifecycle_fault_mode"] = "none"
            recovery_request.pop("lifecycle_fault_marker_path", None)
            recovery_request_path = _write_json(attempt_dir / "recovery_request.json", recovery_request)
            recovery_result_path = attempt_dir / "backend_result.json"
            with _isolated_backend_environment(backend_python):
                recovery_env = os.environ.copy()
                recovery_env.pop(LIFECYCLE_FAULT_ENV, None)
                if graphdeco_root is not None:
                    recovery_env["DEPTH3D_GRAPHDECO_ROOT"] = str(graphdeco_root)
                recovery = run_supervised_process(
                    [str(backend_python), str(worker_script), "--request", str(recovery_request_path), "--result", str(recovery_result_path)],
                    timeout_s=1800.0,
                    env=recovery_env,
                )
            recovery_valid = bool(
                recovery.exit_code == 0
                and not recovery.cancelled
                and not recovery.timed_out
                and recovery.process_group_clean
                and _validate_recovery_result(recovery_result_path, backend_kind)
            )
            recovery_lifecycle = GaussianLifecycleEvidence(
                state=(GaussianBenchmarkRunState.RECOVERED if recovery_valid else GaussianBenchmarkRunState.FAILED),
                worker_exit_code=recovery.exit_code,
                recovery_attempted=True,
                recovery_succeeded=recovery_valid,
                error=("" if recovery_valid else "real backend recovery failed"),
            )
            evidence = GaussianBackendRobustnessEvidence(
                backend_kind=backend_kind,
                scenario=scenario,
                fault_lifecycle=fault_lifecycle,
                process_group_clean=process_group_clean,
                partial_artifacts_detected=(partial_detected if scenario is GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT else False),
                partial_artifacts_cleaned=(partial_cleaned if scenario is GaussianBackendRobustnessScenario.PARTIAL_ARTIFACT else False),
                recovery_lifecycle=recovery_lifecycle,
                recovery_result_valid=recovery_valid,
            )
            all_evidence.append(evidence)
            print(
                f"{backend_kind.value} {scenario.value}: "
                f"fault={_expected_state(scenario).value} cleanup={process_group_clean} "
                f"recovery={'PASS' if recovery_valid else 'FAIL'}"
            )

    report = GaussianBackendRobustnessReport(evidence=tuple(all_evidence))
    report_path = _write_json(output / "step4_robustness_report.json", report.as_dict())
    summary = {
        "phase": "8.7D3",
        "step": 4,
        "pass": report.passed,
        "backend_count": len(selected),
        "scenarios_per_backend": len(SCENARIOS),
        "backend_process_scope_only": True,
        "engine_level_recovery_deferred_to_phase8_7g": True,
        "report": report.as_dict(),
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }
    summary_path = _write_json(output / "step4_summary.json", summary)
    if not report.passed:
        print("Phase 8.7D3 Step 4 Gaussian backend robustness: FAIL")
        print(f"summary={summary_path}")
        return 1
    print("Phase 8.7D3 Step 4 Gaussian backend robustness: PASS")
    print("backend_process_scope_only=True engine_level_recovery_deferred_to_phase8_7g=True")
    print("canonical_geometry_authoritative=False certified_metrology=False")
    print(f"report={report_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
