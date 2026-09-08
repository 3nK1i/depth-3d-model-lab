#!/usr/bin/env python3
"""Run I5.7B real frozen-foundation RollingDepth robustness replay.

Preflight mode validates every frozen identity/runtime binding without model inference.
Production-native mode executes the six I5.7A-admitted scenarios sequentially through
RollingDepthProductionProviderAdapter and never reruns SfM/MVS or retries a result.
Recovery-production-native mode hash-verifies the preserved wrapper-failed castle_control
provider evidence, recovers that record without inference, then executes only scenarios 2-6.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import analyze_i5_6a_sfm_report, validate_i5_6a_for_i5_6b
from core.reconstruction.i5_rolling_depth_robustness_replay import (
    I57B_I57A_FROZEN_SHA256,
    I57B_RECOVERY_EXCEPTION_MESSAGE,
    I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS,
    I57B_SCENARIO_IDS,
    I57BReplayPolicy,
    analyze_i5_7b_provider_summary,
    build_i5_7b_recovered_control_record,
    build_i5_7b_rolling_depth_robustness_replay,
    validate_i5_7a_for_i5_7b,
    validate_i5_7b_receipt,
)
from core.reconstruction.rolling_depth_dense_provider import (
    RollingDepthProductionConfig,
    RollingDepthProductionProviderAdapter,
    RollingDepthProviderRunContext,
    probe_rolling_depth_environment,
    rolling_depth_dense_provider_descriptor,
)
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict

_EXPECTED = {
    "castle_control": (19, 19),
    "castle_low_overlap": (7, 6),
    "castle_weak_parallax": (7, 7),
    "castle_overexposure": (19, 19),
    "castle_symmetry": (19, 12),
    "castle_dynamic_proxy": (19, 19),
}
_INTERPRETATION = {
    "castle_control": "temporal_sequence_control",
    "castle_low_overlap": "temporal_support_spacing_stress_not_mvs_overlap_equivalence",
    "castle_weak_parallax": "narrow_baseline_temporal_geometry_stress",
    "castle_overexposure": "photometric_overexposure_temporal_stress",
    "castle_symmetry": "registered_subset_symmetry_temporal_stress",
    "castle_dynamic_proxy": "dynamic_motion_boundary_temporal_stress",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_binding(path: Path, expected_sha256: str, *, label: str) -> None:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    actual = _sha256(path)
    if actual.lower() != expected_sha256.strip().lower():
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected_sha256} actual={actual}")


def _binding(node: Mapping[str, Any], *, label: str) -> tuple[Path, str]:
    path = Path(str(node.get("path", ""))).expanduser().resolve()
    sha = str(node.get("sha256", "")).strip().lower()
    if len(sha) != 64:
        raise RuntimeError(f"{label} binding SHA256 is malformed.")
    _verify_binding(path, sha, label=label)
    return path, sha


def _system_ram_used_mib() -> float | None:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
        vals: dict[str, float] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            bits = rest.strip().split()
            if bits:
                vals[key] = float(bits[0]) / 1024.0
        total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
        return None if total is None or avail is None else max(0.0, total - avail)
    except (OSError, ValueError):
        return None


def _gpu_used_mib() -> float | None:
    try:
        cp = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            text=True, capture_output=True, check=False, timeout=5,
        )
        if cp.returncode != 0:
            return None
        return float(cp.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class _SystemResourceSampler:
    """Best-effort device/system observations; these are not qualification thresholds."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0
        self.baseline_vram_mib: float | None = None
        self.peak_vram_mib: float | None = None
        self.final_vram_mib: float | None = None
        self.baseline_system_ram_used_mib: float | None = None
        self.peak_system_ram_used_mib: float | None = None
        self.final_system_ram_used_mib: float | None = None

    def _sample(self) -> None:
        gpu = _gpu_used_mib()
        ram = _system_ram_used_mib()
        if self.samples == 0:
            self.baseline_vram_mib = gpu
            self.baseline_system_ram_used_mib = ram
        if gpu is not None:
            self.peak_vram_mib = gpu if self.peak_vram_mib is None else max(self.peak_vram_mib, gpu)
            self.final_vram_mib = gpu
        if ram is not None:
            self.peak_system_ram_used_mib = ram if self.peak_system_ram_used_mib is None else max(self.peak_system_ram_used_mib, ram)
            self.final_system_ram_used_mib = ram
        self.samples += 1

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="i57b-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._sample()
        return {
            "observation_scope": "device_wide_gpu0_and_system_wide_host_ram",
            "qualification_threshold": None,
            "sample_interval_s": self.interval_s,
            "sample_count": self.samples,
            "baseline_vram_mib": self.baseline_vram_mib,
            "peak_vram_mib": self.peak_vram_mib,
            "final_vram_mib": self.final_vram_mib,
            "baseline_system_ram_used_mib": self.baseline_system_ram_used_mib,
            "peak_system_ram_used_mib": self.peak_system_ram_used_mib,
            "final_system_ram_used_mib": self.final_system_ram_used_mib,
            "missing_metrics_preserved_as_null": True,
        }


def _explicit_selection(view_count: int, frame_id: str) -> tuple[DenseProviderSelectionRequest, DenseProviderSelectionDecision]:
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=frame_id,
        requested_provider_id="rolling-depth-temporal",
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id="rolling-depth-temporal",
        selected_provider_id="rolling-depth-temporal",
        selected_implementation=DenseProviderImplementation.ROLLING_DEPTH,
        considered_provider_ids=("rolling-depth-temporal",),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("i5_7b_frozen_explicit_rolling_depth_replay",),
    )
    return request, decision


def _load_and_validate_foundation(i57a_path: Path, i57a_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _verify_binding(i57a_path, i57a_sha, label="I5.7A admission")
    if i57a_sha.lower() != I57B_I57A_FROZEN_SHA256:
        raise RuntimeError("I5.7B requires the exact closed I5.7A admission SHA256.")
    admission = validate_i5_7a_for_i5_7b(_read_json(i57a_path))
    nested = admission["evidence_bindings"]["i5_6a_castle_sfm_admission"]
    i56a_path, _ = _binding(nested, label="nested I5.6A admission")
    i56a = validate_i5_6a_for_i5_6b(_read_json(i56a_path))
    analyses: dict[str, Any] = {}
    for sid in I57B_SCENARIO_IDS:
        node = i56a["scenario_matrix"][sid]
        report_path, report_sha = _binding(node["report_binding"], label=f"{sid} frozen SfM report")
        expected_input, expected_registered = _EXPECTED[sid]
        analysis = analyze_i5_6a_sfm_report(
            _read_json(report_path), scenario_id=sid,
            expected_input=expected_input, expected_registered=expected_registered,
        )
        analysis["report_binding"] = {"path": str(report_path), "sha256": report_sha}
        analyses[sid] = analysis
    return admission, i56a, analyses


def _runtime_preflight(python_path: Path, repository: Path, checkpoint: Path | None) -> tuple[Any, dict[str, Any]]:
    env = probe_rolling_depth_environment(
        python_executable=python_path, repository_root=repository, checkpoint_path=checkpoint,
    )
    descriptor = rolling_depth_dense_provider_descriptor(env)
    if not env.runtime_lock_valid or not env.checkpoint_ready or not env.cuda_available or not env.available:
        raise RuntimeError(f"RollingDepth runtime preflight failed: {env.error or env.as_dict()}")
    if not descriptor.available or descriptor.provider_id != "rolling-depth-temporal":
        raise RuntimeError("RollingDepth descriptor is unavailable or drifted.")
    payload = {
        "rolling_depth_runtime_ready": True,
        "provider_id": descriptor.provider_id,
        "implementation": descriptor.implementation.value,
        "source_revision": env.source_revision,
        "runtime_lock_valid": env.runtime_lock_valid,
        "checkpoint_ready": env.checkpoint_ready,
        "cuda_available": env.cuda_available,
        "device_names": list(env.device_names),
        "device": "cuda:0",
        "precision": "fp16",
        "production_config_unchanged": True,
        "production_config": RollingDepthProductionConfig().as_dict(),
        "environment": env.as_dict(),
        "descriptor": descriptor.as_dict(),
        "resource_metrics_are_observations_not_thresholds": True,
    }
    return env, payload


def _integrity_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    keys = ("registered_images_sha256", "selected_sparse_sha256", "registered_image_names", "selected_model_id")
    return all(before.get(k) == after.get(k) for k in keys)


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _verify_recovery_manifest(manifest_path: Path, expected_sha256: str, attempt_dir: Path) -> dict[Path, str]:
    _verify_binding(manifest_path, expected_sha256, label="I5.7B recovery SHA256 manifest")
    attempt_dir = attempt_dir.resolve()
    entries: dict[Path, str] = {}
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if len(line) < 67 or line[64:66] != "  ":
            raise RuntimeError(f"Malformed I5.7B recovery manifest line {line_no}.")
        expected = line[:64].lower()
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise RuntimeError(f"Malformed I5.7B recovery manifest SHA256 at line {line_no}.")
        path = Path(line[66:]).expanduser().resolve()
        if path != attempt_dir and attempt_dir not in path.parents:
            raise RuntimeError(f"Recovery manifest path escapes the preserved attempt: {path}")
        if path in entries:
            raise RuntimeError(f"Duplicate recovery manifest path: {path}")
        if not path.is_file():
            raise RuntimeError(f"Recovery manifest file is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"Recovery manifest SHA256 mismatch for {path}: expected={expected} actual={actual}"
            )
        entries[path] = expected
    if not entries:
        raise RuntimeError("I5.7B recovery manifest is empty.")
    return entries


def _manifest_binding(entries: Mapping[Path, str], path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    expected = entries.get(resolved)
    if expected is None:
        raise RuntimeError(f"{label} is not bound by the preserved recovery manifest: {resolved}")
    actual = _sha256(resolved)
    if actual != expected:
        raise RuntimeError(f"{label} changed after manifest verification: {resolved}")
    return {"path": str(resolved), "sha256": expected}


def _require_preserved_wrapper_failure(path: Path, *, scenario_level: bool) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("phase") != "I5.7B" or payload.get("attempt_preserved") is not True:
        raise RuntimeError(f"Preserved I5.7B failure identity is invalid: {path}")
    if payload.get("scientific_quality_terminal") is not False:
        raise RuntimeError(f"Recovery source was incorrectly classified as a scientific terminal result: {path}")
    if payload.get("exception_type") != "ReconstructionValidationError":
        raise RuntimeError(f"Recovery source exception type drifted: {path}")
    if payload.get("exception_message") != I57B_RECOVERY_EXCEPTION_MESSAGE:
        raise RuntimeError(f"Recovery source exception message is not the admitted wrapper-schema failure: {path}")
    if payload.get("retry_until_pass") is not False or payload.get("thresholds_modified") is not False:
        raise RuntimeError(f"Recovery source retry/threshold provenance drifted: {path}")
    if payload.get("production_default_modified") is not False:
        raise RuntimeError(f"Recovery source production-default provenance drifted: {path}")
    if scenario_level:
        if payload.get("scenario_id") != "castle_control":
            raise RuntimeError("Only the preserved castle_control wrapper failure may be recovered.")
        if payload.get("sfm_rerun") is not False or payload.get("mvs_rerun") is not False:
            raise RuntimeError("Recovery source cannot contain an SfM/MVS rerun.")
    return payload


def _collect_provider_artifacts(provider_dir: Path) -> dict[str, Any]:
    native_dir = provider_dir / "provider"
    candidates = {
        "provider_summary": provider_dir / "i4_3d2_rolling_depth_execution_summary.json",
        "provider_input_manifest": native_dir / "input_manifest.json",
        "worker_result_manifest": native_dir / "worker_output/rolling_depth_result_manifest.json",
        "worker_evidence_npz": native_dir / "worker_output/rolling_depth_evidence.npz",
        "canonical_npz": provider_dir / "canonical_rolling_depth_fused.npz",
        "canonical_ply": provider_dir / "canonical_rolling_depth_fused.ply",
    }
    return {name: binding for name, path in candidates.items() if (binding := _artifact(path)) is not None}


def _recover_castle_control(
    *,
    attempt_dir: Path,
    manifest_path: Path,
    manifest_sha256: str,
    console_log: Path,
    console_log_sha256: str,
    i56a: Mapping[str, Any],
    current_analysis: Mapping[str, Any],
    current_i57a_sha256: str,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    attempt_dir = attempt_dir.expanduser().resolve()
    if not attempt_dir.is_dir() or not attempt_dir.name.startswith("i5_7b_rolling_depth_production_native_"):
        raise RuntimeError(f"Invalid preserved I5.7B recovery attempt directory: {attempt_dir}")
    manifest_path = manifest_path.expanduser().resolve()
    entries = _verify_recovery_manifest(manifest_path, manifest_sha256, attempt_dir)
    console_log = console_log.expanduser().resolve()
    _verify_binding(console_log, console_log_sha256, label="I5.7B preserved console log")

    top_failure = attempt_dir / "i5_7b_attempt_failure.json"
    scenario_failure = attempt_dir / "01_castle_control/i5_7b_attempt_failure.json"
    _require_preserved_wrapper_failure(top_failure, scenario_level=False)
    _require_preserved_wrapper_failure(scenario_failure, scenario_level=True)

    original_preflight_path = attempt_dir / "i5_7b_runtime_preflight.json"
    original_preflight = _read_json(original_preflight_path)
    if original_preflight.get("mode") != "production-native":
        raise RuntimeError("Recovery source is not an original production-native I5.7B attempt.")
    if original_preflight.get("i5_7a_sha256") != current_i57a_sha256:
        raise RuntimeError("Recovery source I5.7A identity drifted.")
    if original_preflight.get("six_scenario_foundation_verified") is not True:
        raise RuntimeError("Recovery source did not verify the six-scenario frozen foundation.")
    old_foundation = original_preflight.get("scenario_foundation", {}).get("castle_control")
    if not isinstance(old_foundation, Mapping) or not _integrity_equal(old_foundation, current_analysis):
        raise RuntimeError("Recovery source castle_control frozen foundation no longer matches current evidence.")

    # Revalidate the authoritative SfM/source identity now; the original wrapper failed
    # before its post-provider revalidation line could execute.
    report_path, report_sha = _binding(
        i56a["scenario_matrix"]["castle_control"]["report_binding"],
        label="castle_control frozen SfM report",
    )
    report = _read_json(report_path)
    current_after = analyze_i5_6a_sfm_report(
        report, scenario_id="castle_control", expected_input=19, expected_registered=19,
    )
    if not _integrity_equal(current_analysis, current_after):
        raise RuntimeError("castle_control authoritative SfM/source identity drifted before recovery.")

    provider_dir = attempt_dir / "01_castle_control/provider"
    summary_path = provider_dir / "i4_3d2_rolling_depth_execution_summary.json"
    summary = _read_json(summary_path)
    artifact_paths = {
        "provider_summary": summary_path,
        "provider_input_manifest": provider_dir / "provider/input_manifest.json",
        "worker_result_manifest": provider_dir / "provider/worker_output/rolling_depth_result_manifest.json",
        "worker_evidence_npz": provider_dir / "provider/worker_output/rolling_depth_evidence.npz",
        "canonical_npz": provider_dir / "canonical_rolling_depth_fused.npz",
        "canonical_ply": provider_dir / "canonical_rolling_depth_fused.ply",
        "original_attempt_failure": top_failure,
        "original_scenario_failure": scenario_failure,
        "original_runtime_preflight": original_preflight_path,
    }
    artifacts = {
        name: _manifest_binding(entries, path, label=f"I5.7B recovery {name}")
        for name, path in artifact_paths.items()
    }
    artifacts["frozen_sfm_report"] = {"path": str(report_path), "sha256": report_sha}

    provenance = {
        "scenario_id": "castle_control",
        "original_attempt_path": str(attempt_dir),
        "original_manifest": {"path": str(manifest_path), "sha256": manifest_sha256.lower()},
        "original_console_log": {"path": str(console_log), "sha256": console_log_sha256.lower()},
        "original_wrapper_failure_after_provider_execution": True,
        "original_exception_message": I57B_RECOVERY_EXCEPTION_MESSAGE,
        "provider_execution_rerun": False,
        "retry_until_pass": False,
        "thresholds_modified": False,
        "production_default_modified": False,
        "sfm_rerun": False,
        "mvs_rerun": False,
        "manifest_verified": True,
        "console_log_verified": True,
        "current_frozen_sfm_revalidated": True,
        "missing_performance_fields": list(I57B_RECOVERY_MISSING_PERFORMANCE_FIELDS),
    }
    record = build_i5_7b_recovered_control_record(
        provider_summary=summary,
        artifact_bindings=artifacts,
        recovery_provenance=provenance,
    )
    recovery_dir = output / "01_castle_control_recovered"
    recovery_dir.mkdir(parents=False, exist_ok=False)
    _write_json(recovery_dir / "i5_7b_scenario_record.json", record)
    _write_json(recovery_dir / "i5_7b_recovery_provenance.json", provenance)
    return record, provenance


def _run_scenario(
    *, sid: str, i56a: Mapping[str, Any], before_analysis: Mapping[str, Any], env: Any,
    python_path: Path, repository: Path, output: Path,
) -> dict[str, Any]:
    scenario_dir = output / f"{I57B_SCENARIO_IDS.index(sid)+1:02d}_{sid}"
    scenario_dir.mkdir(parents=False, exist_ok=False)
    report_path, report_sha = _binding(i56a["scenario_matrix"][sid]["report_binding"], label=f"{sid} frozen SfM report")
    report = _read_json(report_path)
    expected_input, expected_registered = _EXPECTED[sid]
    # Revalidate immediately before provider execution.
    before = analyze_i5_6a_sfm_report(report, scenario_id=sid, expected_input=expected_input, expected_registered=expected_registered)
    if not _integrity_equal(before_analysis, before):
        raise RuntimeError(f"{sid} frozen SfM/source identity drifted before RollingDepth execution.")
    sfm = offline_sfm_result_from_dict(report["sfm_result"])
    frame_id = f"i5_7b:{sid}:pycolmap_sfm:{sfm.session_id}"
    provider_dir = scenario_dir / "provider"
    context = RollingDepthProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id=frame_id,
        output_dir=provider_dir,
        rolling_depth_python=python_path,
        rolling_depth_repository=repository,
        worker_script=ROOT / "scripts/reconstruction_rolling_depth_provider_worker.py",
        device="cuda:0", precision="fp16", config=RollingDepthProductionConfig(),
    )
    request, decision = _explicit_selection(expected_registered, frame_id)
    adapter = RollingDepthProductionProviderAdapter(context=context, environment=env)
    sampler = _SystemResourceSampler()
    started = time.monotonic()
    sampler.start()
    try:
        execution = adapter.execute(request, decision)
    finally:
        resources = sampler.stop()
    wall = time.monotonic() - started
    summary_path = provider_dir / "i4_3d2_rolling_depth_execution_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{sid} provider did not produce execution summary.")
    summary = _read_json(summary_path)
    analysis = analyze_i5_7b_provider_summary(summary, wall_time_s=wall, resource_observation=resources)
    # Re-read/re-hash the exact frozen SfM inputs after provider execution.
    after = analyze_i5_6a_sfm_report(report, scenario_id=sid, expected_input=expected_input, expected_registered=expected_registered)
    if not _integrity_equal(before, after):
        raise RuntimeError(f"{sid} authoritative SfM/source identity changed during RollingDepth execution.")
    artifacts: dict[str, Any] = {
        "frozen_sfm_report": {"path": str(report_path), "sha256": report_sha},
        **_collect_provider_artifacts(provider_dir),
    }
    record = {
        "scenario_id": sid,
        "runtime_status": analysis["runtime_status"],
        "provider_execution_terminal": True,
        "provider_execution_completed": analysis["provider_execution_completed"],
        "source_registered_views": expected_registered,
        "processed_views": analysis["native_metrics"]["processed_views"],
        "scientific_interpretation": _INTERPRETATION[sid],
        "native_metrics": analysis["native_metrics"],
        "resource_observation": resources,
        "artifact_bindings": artifacts,
        "failure": analysis["failure"],
        "execution_result": execution.as_dict(),
        "input_integrity_revalidated_after": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "per_frame_alignment_performed": False,
            "permanent_global_view_cap": None,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }
    _write_json(scenario_dir / "i5_7b_scenario_record.json", record)
    return record


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("preflight", "production-native", "recovery-production-native"),
        default="preflight",
    )
    p.add_argument("--i5-7a-admission", type=Path, required=True)
    p.add_argument("--i5-7a-sha256", default=I57B_I57A_FROZEN_SHA256)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--replay-receipt-sha256", required=True)
    p.add_argument("--rolling-depth-python", type=Path, required=True)
    p.add_argument("--rolling-depth-repository", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--recovery-attempt-dir", type=Path)
    p.add_argument("--recovery-manifest", type=Path)
    p.add_argument("--recovery-manifest-sha256")
    p.add_argument("--recovery-console-log", type=Path)
    p.add_argument("--recovery-console-log-sha256")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        admission, i56a, analyses = _load_and_validate_foundation(
            args.i5_7a_admission.expanduser().resolve(), args.i5_7a_sha256,
        )
        _verify_binding(args.replay_receipt.expanduser().resolve(), args.replay_receipt_sha256, label="I5.7B replay receipt")
        receipt = validate_i5_7b_receipt(_read_json(args.replay_receipt.expanduser().resolve()))
        env, preflight = _runtime_preflight(
            args.rolling_depth_python.expanduser().resolve(),
            args.rolling_depth_repository.expanduser().resolve(),
            None if args.checkpoint is None else args.checkpoint.expanduser().resolve(),
        )
        preflight.update({
            "mode": args.mode,
            "i5_7a_sha256": args.i5_7a_sha256,
            "six_scenario_foundation_verified": True,
            "scenario_foundation": analyses,
            "rolling_depth_inference_started": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "retry_until_pass": False,
        })
        _write_json(output / "i5_7b_runtime_preflight.json", preflight)
        if args.mode == "preflight":
            print("I5.7B RollingDepth runtime/frozen-foundation preflight: PASS")
            print("rolling_depth_inference=False sfm_rerun=False mvs_rerun=False retry_until_pass=False")
            print(f"evidence={output / 'i5_7b_runtime_preflight.json'}")
            return 0

        records: dict[str, Any] = {}
        recovery_provenance: dict[str, Any] | None = None
        scenario_ids = I57B_SCENARIO_IDS
        if args.mode == "recovery-production-native":
            required = {
                "--recovery-attempt-dir": args.recovery_attempt_dir,
                "--recovery-manifest": args.recovery_manifest,
                "--recovery-manifest-sha256": args.recovery_manifest_sha256,
                "--recovery-console-log": args.recovery_console_log,
                "--recovery-console-log-sha256": args.recovery_console_log_sha256,
            }
            missing = [name for name, value in required.items() if value is None or str(value).strip() == ""]
            if missing:
                raise RuntimeError(f"I5.7B recovery mode is missing required arguments: {', '.join(missing)}")
            print("===== I5.7B RECOVER PRESERVED CASTLE CONTROL — NO INFERENCE =====", flush=True)
            records["castle_control"], recovery_provenance = _recover_castle_control(
                attempt_dir=args.recovery_attempt_dir,
                manifest_path=args.recovery_manifest,
                manifest_sha256=str(args.recovery_manifest_sha256),
                console_log=args.recovery_console_log,
                console_log_sha256=str(args.recovery_console_log_sha256),
                i56a=i56a,
                current_analysis=analyses["castle_control"],
                current_i57a_sha256=args.i5_7a_sha256,
                output=output,
            )
            print(
                f"castle_control: {records['castle_control']['runtime_status'].upper()} "
                "(RECOVERED; provider_execution_rerun=False)",
                flush=True,
            )
            scenario_ids = I57B_SCENARIO_IDS[1:]
        elif any((
            args.recovery_attempt_dir, args.recovery_manifest, args.recovery_manifest_sha256,
            args.recovery_console_log, args.recovery_console_log_sha256,
        )):
            raise RuntimeError("Recovery arguments are only valid with --mode recovery-production-native.")

        for sid in scenario_ids:
            print(f"===== I5.7B REAL ROLLINGDEPTH: {sid} =====", flush=True)
            scenario_dir = output / f"{I57B_SCENARIO_IDS.index(sid)+1:02d}_{sid}"
            try:
                records[sid] = _run_scenario(
                    sid=sid, i56a=i56a, before_analysis=analyses[sid], env=env,
                    python_path=args.rolling_depth_python.expanduser().resolve(),
                    repository=args.rolling_depth_repository.expanduser().resolve(), output=output,
                )
            except Exception as exc:
                scenario_dir.mkdir(parents=True, exist_ok=True)
                _write_json(scenario_dir / "i5_7b_attempt_failure.json", {
                    "phase": "I5.7B",
                    "scenario_id": sid,
                    "attempt_preserved": True,
                    "provider_execution_terminal": False,
                    "scientific_quality_terminal": False,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "retry_until_pass": False,
                    "thresholds_modified": False,
                    "production_default_modified": False,
                    "sfm_rerun": False,
                    "mvs_rerun": False,
                })
                raise
            print(f"{sid}: {records[sid]['runtime_status'].upper()}", flush=True)

        final = build_i5_7b_rolling_depth_robustness_replay(
            i5_7a_admission=admission,
            i5_7a_binding=I51EvidenceBinding(str(args.i5_7a_admission.expanduser().resolve()), args.i5_7a_sha256),
            replay_receipt=receipt,
            replay_receipt_binding=I51EvidenceBinding(str(args.replay_receipt.expanduser().resolve()), args.replay_receipt_sha256),
            runtime_preflight=preflight,
            scenario_records=records,
        )
        if recovery_provenance is not None:
            final["scope"]["recovery_used"] = True
            final["scope"]["castle_control_provider_rerun"] = False
            final["scope"]["five_remaining_scenarios_executed_in_recovery_attempt"] = True
            final["recovery_provenance"] = recovery_provenance
        else:
            final["scope"]["recovery_used"] = False
            final["scope"]["castle_control_provider_rerun"] = False
        final_path = output / "i5_7b_rolling_depth_robustness_replay.json"
        _write_json(final_path, final)
        print("I5.7B Real Frozen-Foundation RollingDepth Robustness Replay: PASS")
        print(f"quality_status={final['quality_status']}")
        print("I5.7C RollingDepth robustness qualification + closure: AUTHORIZED")
        print(f"evidence={final_path}")
        return 0
    except Exception as exc:
        _write_json(output / "i5_7b_attempt_failure.json", {
            "phase": "I5.7B",
            "mode": args.mode,
            "attempt_preserved": True,
            "scientific_quality_terminal": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "retry_until_pass": False,
            "thresholds_modified": False,
            "production_default_modified": False,
        })
        print(f"I5.7B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
