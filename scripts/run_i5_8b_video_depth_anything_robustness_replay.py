#!/usr/bin/env python3
"""Run I5.8B real frozen-foundation Video Depth Anything robustness replay.

Preflight validates the frozen I5.8A/I5.6A evidence chain and the exact
hardware-resolved Video Depth Anything runtime/checkpoint without inference.
Production-native executes the six admitted Castle registered sequences once,
sequentially, through the existing I4.3D1 production VDA adapter.  It never
reruns SfM/MVS, retries a result, tunes thresholds, or promotes model metric
or intrinsic evidence to canonical authority.
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

from core.hardware.gpu_info import detect_hardware
from core.hardware.profiles import ComputeProfile, resolve_profile
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID
from core.models.registry import MODEL_REGISTRY
from core.models.runtime_config import ModelRuntimeContext
from core.reconstruction.dense_provider_architecture import (
    DenseProviderImplementation,
    DenseProviderSelectionDecision,
    DenseProviderSelectionMode,
    DenseProviderSelectionRequest,
)
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import analyze_i5_6a_sfm_report, validate_i5_6a_for_i5_6b
from core.reconstruction.i5_video_depth_anything_robustness_replay import (
    I58B_I58A_FROZEN_SHA256,
    I58B_SCENARIO_IDS,
    analyze_i5_8b_provider_summary,
    build_i5_8b_video_depth_anything_robustness_replay,
    validate_i5_8a_for_i5_8b,
    validate_i5_8b_receipt,
)
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict
from core.reconstruction.video_depth_anything_dense_provider import (
    VideoDepthAnythingProductionConfig,
    VideoDepthAnythingProductionProviderAdapter,
    VideoDepthAnythingProviderRunContext,
    probe_video_depth_anything_environment,
    video_depth_anything_dense_provider_descriptor,
)

_EXPECTED = {
    "castle_control": (19, 19),
    "castle_low_overlap": (7, 6),
    "castle_weak_parallax": (7, 7),
    "castle_overexposure": (19, 19),
    "castle_symmetry": (19, 12),
    "castle_dynamic_proxy": (19, 19),
}
_INTERPRETATION = {
    "castle_control": "streaming_cache_registered_sequence_control",
    "castle_low_overlap": "registered_keyframe_spacing_and_cache_continuity_stress_not_mvs_overlap_equivalence",
    "castle_weak_parallax": "narrow_baseline_streaming_temporal_stability_stress",
    "castle_overexposure": "photometric_overexposure_streaming_temporal_stress",
    "castle_symmetry": "registered_subset_repetitive_structure_streaming_temporal_stress",
    "castle_dynamic_proxy": "dynamic_motion_boundary_streaming_cache_stress",
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
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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
        vals: dict[str, float] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
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
    """Best-effort observations only; no observation is a qualification threshold."""

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
        gpu, ram = _gpu_used_mib(), _system_ram_used_mib()
        if self.samples == 0:
            self.baseline_vram_mib, self.baseline_system_ram_used_mib = gpu, ram
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
        self._thread = threading.Thread(target=self._loop, name="i58b-resource-sampler", daemon=True)
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


def _hardware_payload(hardware: Any) -> dict[str, Any]:
    return {
        "backend": getattr(getattr(hardware, "backend", None), "value", str(getattr(hardware, "backend", ""))),
        "device": hardware.device,
        "device_index": hardware.device_index,
        "name": hardware.name,
        "total_memory_bytes": hardware.total_memory_bytes,
        "free_memory_bytes": hardware.free_memory_bytes,
        "compute_capability": list(hardware.compute_capability) if hardware.compute_capability is not None else None,
        "supports_fp16": hardware.supports_fp16,
        "supports_bf16": hardware.supports_bf16,
        "torch_available": hardware.torch_available,
        "torch_cuda_available": hardware.torch_cuda_available,
        "torch_version": hardware.torch_version,
        "torch_cuda_version": hardware.torch_cuda_version,
        "compiled_architectures": list(hardware.compiled_architectures),
        "torch_build_compatible": hardware.torch_build_compatible,
        "cuda_ready": hardware.cuda_ready,
        "warning": hardware.warning,
    }


def _explicit_selection(view_count: int, frame_id: str) -> tuple[DenseProviderSelectionRequest, DenseProviderSelectionDecision]:
    request = DenseProviderSelectionRequest(
        mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        available_view_count=view_count,
        authoritative_sfm_available=True,
        authoritative_sfm_frame_id=frame_id,
        requested_provider_id="video-depth-anything-temporal",
        allow_fallback=False,
    )
    decision = DenseProviderSelectionDecision(
        request_mode=DenseProviderSelectionMode.EXPLICIT_PROVIDER,
        requested_provider_id="video-depth-anything-temporal",
        selected_provider_id="video-depth-anything-temporal",
        selected_implementation=DenseProviderImplementation.VIDEO_DEPTH_ANYTHING,
        considered_provider_ids=("video-depth-anything-temporal",),
        automatic=False,
        engineer_override_applied=False,
        fallback_used=False,
        reason_codes=("i5_8b_frozen_explicit_video_depth_anything_replay",),
    )
    return request, decision


def _integrity_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    keys = ("registered_images_sha256", "selected_sparse_sha256", "registered_image_names", "selected_model_id")
    return all(before.get(k) == after.get(k) for k in keys)


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _collect_provider_artifacts(provider_dir: Path) -> dict[str, Any]:
    native_dir = provider_dir / "provider"
    candidates = {
        "provider_summary": provider_dir / "i4_3d1_vda_execution_summary.json",
        "provider_input_manifest": native_dir / "input_manifest.json",
        "worker_result_manifest": native_dir / "worker_output/vda_result_manifest.json",
        "worker_evidence_npz": native_dir / "worker_output/vda_depth_evidence.npz",
        "canonical_npz": provider_dir / "canonical_vda_fused.npz",
        "canonical_ply": provider_dir / "canonical_vda_fused.ply",
    }
    return {name: binding for name, path in candidates.items() if (binding := _artifact(path)) is not None}


def _load_and_validate_foundation(i58a_path: Path, i58a_sha: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _verify_binding(i58a_path, i58a_sha, label="I5.8A admission")
    if i58a_sha.lower() != I58B_I58A_FROZEN_SHA256:
        raise RuntimeError("I5.8B requires the exact closed I5.8A admission SHA256.")
    admission = validate_i5_8a_for_i5_8b(_read_json(i58a_path))
    nested = admission["evidence_bindings"]["i5_6a_castle_sfm_admission"]
    i56a_path, _ = _binding(nested, label="nested I5.6A admission")
    i56a = validate_i5_6a_for_i5_6b(_read_json(i56a_path))
    analyses: dict[str, Any] = {}
    for sid in I58B_SCENARIO_IDS:
        report_path, report_sha = _binding(i56a["scenario_matrix"][sid]["report_binding"], label=f"{sid} frozen SfM report")
        expected_input, expected_registered = _EXPECTED[sid]
        analysis = analyze_i5_6a_sfm_report(
            _read_json(report_path), scenario_id=sid,
            expected_input=expected_input, expected_registered=expected_registered,
        )
        analysis["report_binding"] = {"path": str(report_path), "sha256": report_sha}
        analyses[sid] = analysis
    return admission, i56a, analyses


def _runtime_preflight(python_path: Path, project_root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    hardware = detect_hardware(0)
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    runtime_config = MODEL_REGISTRY.resolve_runtime_config(
        VIDEO_DEPTH_ANYTHING_MODEL_ID,
        ModelRuntimeContext(hardware_info=hardware, resolved_profile=resolved),
    )
    source = Path(str(runtime_config["source_dir"])).expanduser()
    if not source.is_absolute():
        source = (project_root / source).resolve()
    checkpoint = Path(str(runtime_config["checkpoint"])).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (project_root / checkpoint).resolve()
    runtime_config["source_dir"] = str(source)
    runtime_config["checkpoint"] = str(checkpoint)

    env = probe_video_depth_anything_environment(
        project_root,
        python_executable=python_path,
        checkpoint=checkpoint,
    )
    descriptor = video_depth_anything_dense_provider_descriptor(env)
    if not env.available or not env.cuda_available:
        raise RuntimeError(f"Video Depth Anything runtime preflight failed: {env.error or env.as_dict()}")
    if not descriptor.available or descriptor.provider_id != "video-depth-anything-temporal":
        raise RuntimeError("Video Depth Anything provider descriptor is unavailable or drifted.")
    if Path(env.source_directory).resolve() != source:
        raise RuntimeError("Video Depth Anything source directory drifted from resolved production runtime config.")
    if Path(env.checkpoint_path).resolve() != checkpoint:
        raise RuntimeError("Video Depth Anything checkpoint drifted from resolved production runtime config.")
    if not checkpoint.is_file():
        raise RuntimeError(f"Resolved Video Depth Anything checkpoint is missing: {checkpoint}")
    if not str(env.source_revision).strip():
        raise RuntimeError("Video Depth Anything source revision is missing.")
    profile = str(runtime_config.get("profile", "")).strip()
    if profile != str(getattr(resolved.effective, "value", resolved.effective)):
        raise RuntimeError("VDA registry runtime profile drifted from hardware resolver.")
    device = str(runtime_config.get("device", "")).strip()
    if not device.startswith("cuda"):
        raise RuntimeError("I5.8B production-native replay requires CUDA.")

    payload = {
        "video_depth_anything_runtime_ready": True,
        "provider_id": descriptor.provider_id,
        "implementation": descriptor.implementation.value,
        "source_ready": env.source_ready,
        "checkpoint_ready": env.checkpoint_ready,
        "cuda_available": env.cuda_available,
        "source_revision": env.source_revision,
        "resolved_profile": profile,
        "resolved_encoder": runtime_config.get("encoder"),
        "resolved_checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "device": device,
        "precision": str(runtime_config.get("precision", "")).strip().lower(),
        "production_config_unchanged": True,
        "production_config": VideoDepthAnythingProductionConfig().as_dict(),
        "runtime_config": runtime_config,
        "hardware": _hardware_payload(hardware),
        "resolved_profile_metadata": {
            "requested": str(getattr(resolved.requested, "value", resolved.requested)),
            "effective": str(getattr(resolved.effective, "value", resolved.effective)),
            "downgraded": resolved.downgraded,
            "memory_pressure": resolved.memory_pressure,
            "warning": resolved.warning,
        },
        "environment": env.as_dict(),
        "descriptor": descriptor.as_dict(),
        "resource_metrics_are_observations_not_thresholds": True,
        "frame_latency_metrics_are_observations_not_quality_thresholds": True,
    }
    return env, runtime_config, payload


def _run_scenario(
    *, sid: str, i56a: Mapping[str, Any], before_analysis: Mapping[str, Any], env: Any,
    python_path: Path, runtime_config: Mapping[str, Any], runtime_preflight: Mapping[str, Any], output: Path,
) -> dict[str, Any]:
    scenario_dir = output / f"{I58B_SCENARIO_IDS.index(sid)+1:02d}_{sid}"
    scenario_dir.mkdir(parents=False, exist_ok=False)
    report_path, report_sha = _binding(i56a["scenario_matrix"][sid]["report_binding"], label=f"{sid} frozen SfM report")
    report = _read_json(report_path)
    expected_input, expected_registered = _EXPECTED[sid]
    before = analyze_i5_6a_sfm_report(
        report, scenario_id=sid, expected_input=expected_input, expected_registered=expected_registered,
    )
    if not _integrity_equal(before_analysis, before):
        raise RuntimeError(f"{sid} frozen SfM/source identity drifted before Video Depth Anything execution.")

    sfm = offline_sfm_result_from_dict(report["sfm_result"])
    frame_id = f"i5_8b:{sid}:pycolmap_sfm:{sfm.session_id}"
    provider_dir = scenario_dir / "provider"
    context = VideoDepthAnythingProviderRunContext(
        sfm_result=sfm,
        reconstruction_frame_id=frame_id,
        output_dir=provider_dir,
        python_executable=python_path,
        worker_script=ROOT / "scripts/reconstruction_video_depth_anything_provider_worker.py",
        runtime_config=dict(runtime_config),
        device=str(runtime_preflight["device"]),
        precision=str(runtime_preflight["precision"]),
        config=VideoDepthAnythingProductionConfig(),
    )
    request, decision = _explicit_selection(expected_registered, frame_id)
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=env)
    sampler = _SystemResourceSampler()
    started = time.monotonic()
    sampler.start()
    try:
        execution = adapter.execute(request, decision)
    finally:
        resources = sampler.stop()
    wall = time.monotonic() - started

    summary_path = provider_dir / "i4_3d1_vda_execution_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{sid} provider did not produce I4.3D1 VDA execution summary.")
    summary = _read_json(summary_path)
    analysis = analyze_i5_8b_provider_summary(
        summary,
        wall_time_s=wall,
        resource_observation=resources,
        expected_runtime_config=runtime_config,
    )
    after = analyze_i5_6a_sfm_report(
        report, scenario_id=sid, expected_input=expected_input, expected_registered=expected_registered,
    )
    if not _integrity_equal(before, after):
        raise RuntimeError(f"{sid} authoritative SfM/source identity changed during Video Depth Anything execution.")

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
        "runtime_profile": {
            "resolved_profile": runtime_preflight["resolved_profile"],
            "resolved_encoder": runtime_preflight["resolved_encoder"],
            "resolved_checkpoint": runtime_preflight["resolved_checkpoint"],
            "checkpoint_sha256": runtime_preflight["checkpoint_sha256"],
            "source_revision": runtime_preflight["source_revision"],
            "device": runtime_preflight["device"],
            "precision": runtime_preflight["precision"],
        },
        "artifact_bindings": {
            "frozen_sfm_report": {"path": str(report_path), "sha256": report_sha},
            **_collect_provider_artifacts(provider_dir),
        },
        "failure": analysis["failure"],
        "degradation_reasons": analysis["degradation_reasons"],
        "execution_result": execution.as_dict(),
        "input_integrity_revalidated_after": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "source_rematerialized": False,
            "per_frame_alignment_performed": True,
            "sequence_global_alignment_performed": False,
            "native_estimated_intrinsics_are_canonical_authority": False,
            "native_metric_depth_establishes_physical_metric_scale": False,
            "permanent_global_view_cap": None,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "provider_promotion_performed": False,
        },
    }
    _write_json(scenario_dir / "i5_8b_scenario_record.json", record)
    return record


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "production-native"), default="preflight")
    p.add_argument("--i5-8a-admission", type=Path, required=True)
    p.add_argument("--i5-8a-sha256", default=I58B_I58A_FROZEN_SHA256)
    p.add_argument("--replay-receipt", type=Path, required=True)
    p.add_argument("--replay-receipt-sha256", required=True)
    p.add_argument("--vda-python", type=Path, default=Path(sys.executable))
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        project_root = args.project_root.expanduser().resolve()
        if project_root != ROOT:
            raise RuntimeError(f"I5.8B must execute against the current project root: expected={ROOT} actual={project_root}")
        python_path = args.vda_python.expanduser().resolve()
        if not python_path.is_file():
            raise RuntimeError(f"VDA Python executable is missing: {python_path}")

        admission, i56a, analyses = _load_and_validate_foundation(
            args.i5_8a_admission.expanduser().resolve(), args.i5_8a_sha256,
        )
        receipt_path = args.replay_receipt.expanduser().resolve()
        _verify_binding(receipt_path, args.replay_receipt_sha256, label="I5.8B replay receipt")
        receipt = validate_i5_8b_receipt(_read_json(receipt_path))
        env, runtime_config, preflight = _runtime_preflight(python_path, project_root)
        preflight.update({
            "mode": args.mode,
            "i5_8a_sha256": args.i5_8a_sha256,
            "six_scenario_foundation_verified": True,
            "scenario_foundation": analyses,
            "video_depth_anything_inference_started": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "retry_until_pass": False,
            "thresholds_modified": False,
            "production_default_modified": False,
        })
        preflight_path = output / "i5_8b_runtime_preflight.json"
        _write_json(preflight_path, preflight)
        if args.mode == "preflight":
            print("I5.8B Video Depth Anything runtime/frozen-foundation preflight: PASS")
            print(
                f"resolved_profile={preflight['resolved_profile']} "
                f"encoder={preflight['resolved_encoder']} device={preflight['device']} "
                f"precision={preflight['precision']}"
            )
            print(f"checkpoint={preflight['resolved_checkpoint']}")
            print(f"checkpoint_sha256={preflight['checkpoint_sha256']}")
            print(f"source_revision={preflight['source_revision']}")
            print("video_depth_anything_inference=False sfm_rerun=False mvs_rerun=False retry_until_pass=False")
            print(f"evidence={preflight_path}")
            return 0

        records: dict[str, Any] = {}
        for sid in I58B_SCENARIO_IDS:
            print(f"===== I5.8B REAL VIDEO DEPTH ANYTHING: {sid} =====", flush=True)
            scenario_dir = output / f"{I58B_SCENARIO_IDS.index(sid)+1:02d}_{sid}"
            try:
                records[sid] = _run_scenario(
                    sid=sid,
                    i56a=i56a,
                    before_analysis=analyses[sid],
                    env=env,
                    python_path=python_path,
                    runtime_config=runtime_config,
                    runtime_preflight=preflight,
                    output=output,
                )
            except Exception as exc:
                scenario_dir.mkdir(parents=True, exist_ok=True)
                _write_json(scenario_dir / "i5_8b_attempt_failure.json", {
                    "phase": "I5.8B",
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
                    "generic_exception_reclassified_as_scientific_rejection": False,
                })
                raise
            print(f"{sid}: {records[sid]['runtime_status'].upper()}", flush=True)

        final = build_i5_8b_video_depth_anything_robustness_replay(
            i5_8a_admission=admission,
            i5_8a_binding=I51EvidenceBinding(str(args.i5_8a_admission.expanduser().resolve()), args.i5_8a_sha256),
            replay_receipt=receipt,
            replay_receipt_binding=I51EvidenceBinding(str(receipt_path), args.replay_receipt_sha256),
            runtime_preflight=preflight,
            scenario_records=records,
        )
        final_path = output / "i5_8b_video_depth_anything_robustness_replay.json"
        _write_json(final_path, final)
        print("I5.8B Real Frozen-Foundation Video Depth Anything Robustness Replay: PASS")
        print(f"quality_status={final['quality_status']}")
        print("I5.8C Video Depth Anything robustness qualification + closure: AUTHORIZED")
        print(f"evidence={final_path}")
        return 0
    except Exception as exc:
        _write_json(output / "i5_8b_attempt_failure.json", {
            "phase": "I5.8B",
            "mode": args.mode,
            "attempt_preserved": True,
            "scientific_quality_terminal": False,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "retry_until_pass": False,
            "thresholds_modified": False,
            "production_default_modified": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
        })
        print(f"I5.8B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
