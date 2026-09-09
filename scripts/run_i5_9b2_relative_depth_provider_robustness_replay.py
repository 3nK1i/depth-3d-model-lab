#!/usr/bin/env python3
"""Run I5.9B2 relative-depth single-image provider robustness preflight/replay.

The preflight mode is execution-free: it validates the exact frozen I5.9A and
I5.6A evidence, re-hashes all six authoritative SfM reports and their registered
image/sparse inputs, probes the two existing B2 production runtimes, resolves
the existing hardware profile/runtime config, and resolves all 12 I4.3F adaptive
provider/scenario plans without starting inference.

The production-native mode performs the same preflight and then executes each of
the 12 admitted provider/scenario attempts exactly once, sequentially. Existing
provider PASS/DEGRADED/recognized-quality-REJECTED semantics are preserved.
Runtime/integrity errors fail the batch nonzero and are persisted; no retry,
retuning, threshold relaxation, provider comparison, SfM/MVS rerun, metric-scale
promotion, or fusion-capacity increase is authorized.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware.gpu_info import detect_hardware  # noqa: E402
from core.hardware.profiles import ComputeProfile, resolve_profile  # noqa: E402
from core.models.catalog import (  # noqa: E402
    DEPTH_ANYTHING_V2_MODEL_ID,
    MARIGOLD_MODEL_ID,
)
from core.models.registry import resolve_model_runtime_config  # noqa: E402
from core.models.runtime_config import ModelRuntimeContext  # noqa: E402
from core.reconstruction.adaptive_production_planning import (  # noqa: E402
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot  # noqa: E402
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus  # noqa: E402
from core.reconstruction.depth_anything_v2_dense_provider import (  # noqa: E402
    DepthAnythingV2ProductionProviderAdapter,
    DepthAnythingV2ProviderRunContext,
    depth_anything_v2_dense_provider_descriptor,
    probe_depth_anything_v2_environment,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding  # noqa: E402
from core.reconstruction.i5_mvs_robustness import (  # noqa: E402
    analyze_i5_6a_sfm_report,
    validate_i5_6a_for_i5_6b,
)
from core.reconstruction.i5_relative_depth_provider_robustness_replay import (  # noqa: E402
    I59B2_EXPECTED_RUN_COUNT,
    I59B2_I56A_FROZEN_SHA256,
    I59B2_I59A_FROZEN_SHA256,
    I59B2_PROVIDERS,
    I59B2_SCENARIO_IDS,
    I59B2_ALIGNMENT_SEMANTICS,
    I59B2_HISTORICAL_G4_LOW_TEXTURE,
    build_i5_9b2_replay,
    validate_i5_9a_for_i5_9b2,
    validate_i5_9b1_for_i5_9b2,
    validate_i5_9b2_receipt,
    validate_i5_9b2_runtime_preflight,
)
from core.reconstruction.marigold_dense_provider import (  # noqa: E402
    MarigoldProductionProviderAdapter,
    MarigoldProviderRunContext,
    marigold_dense_provider_descriptor,
    probe_marigold_environment,
)
from core.reconstruction.sfm_contracts import SfmStageStatistics, SfmWorkspaceManifest  # noqa: E402
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict  # noqa: E402
from core.reconstruction.single_image_provider_production_qualification import (  # noqa: E402
    recognized_quality_rejection,
    validate_single_image_production_plan,
)

OUTPUT_NAME = "i5_9b2_relative_depth_provider_robustness_replay.json"
PREFLIGHT_NAME = "i5_9b2_runtime_preflight.json"
RECORD_NAME = "i5_9b2_provider_scenario_record.json"
FAILURE_NAME = "i5_9b2_attempt_failure.json"
_SUMMARY_FILENAMES = {
    "depth_anything_v2": "i4_3e5_depth_anything_v2_execution_summary.json",
    "marigold": "i4_3e6_marigold_execution_summary.json",
}
_MODEL_IDS = {
    "depth_anything_v2": DEPTH_ANYTHING_V2_MODEL_ID,
    "marigold": MARIGOLD_MODEL_ID,
}
_WORKER_SCRIPTS = {
    "depth_anything_v2": "reconstruction_depth_anything_v2_provider_worker.py",
    "marigold": "reconstruction_marigold_provider_worker.py",
}
_WORKER_RESULT_MANIFESTS = {
    "depth_anything_v2": "depth_anything_v2_result_manifest.json",
    "marigold": "marigold_result_manifest.json",
}
_REJECTION_COUNTS = re.compile(r"accepted=(?P<accepted>\d+)\s+required=(?P<required>\d+)")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "production-native"), default="preflight")
    parser.add_argument("--i5-9a-admission", type=Path, required=True)
    parser.add_argument("--i5-9a-sha256", default=I59B2_I59A_FROZEN_SHA256)
    parser.add_argument("--i5-6a-admission", type=Path, required=True)
    parser.add_argument("--i5-6a-sha256", default=I59B2_I56A_FROZEN_SHA256)
    parser.add_argument("--i5-9b1-replay", type=Path, required=True)
    parser.add_argument("--i5-9b1-sha256", required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dav2-source-dir", type=Path)
    parser.add_argument("--dav2-checkpoint", type=Path)
    parser.add_argument("--dav2-variant", choices=("auto", "small", "base", "large"), default="auto")
    parser.add_argument("--marigold-checkpoint-dir", type=Path)
    parser.add_argument("--marigold-profile", choices=("auto", "efficient", "balanced", "ultra"), default="auto")
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be >= 1")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(path: Path, sha256: str, *, label: str, frozen_sha: str | None = None) -> I51EvidenceBinding:
    source = path.expanduser().resolve()
    expected = str(sha256).strip().lower()
    if not source.is_file():
        raise ReconstructionValidationError(f"I5.9B2 {label} is missing: {source}")
    if not _valid_sha(expected):
        raise ReconstructionValidationError(f"I5.9B2 {label} SHA256 is malformed.")
    if frozen_sha is not None and expected != frozen_sha:
        raise ReconstructionValidationError(
            f"I5.9B2 {label} expected SHA drifted from frozen admission: expected={frozen_sha} supplied={expected}"
        )
    actual = _sha256(source)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.9B2 {label} SHA256 mismatch: expected={expected} actual={actual} path={source}"
        )
    return I51EvidenceBinding(str(source), expected)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I5.9B2 could not read {label}: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.9B2 {label} root must be an object: {source}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return target


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I5.9B2 expected artifact is missing: {source}")
    return {"path": str(source), "sha256": _sha256(source), "size_bytes": int(source.stat().st_size)}


def _hash_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in paths:
        source = raw.expanduser().resolve()
        if str(source) in seen:
            continue
        seen.add(str(source))
        result.append(_artifact(source))
    return result


def _available_ram_bytes() -> int | None:
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return pages * page_size if pages > 0 and page_size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _adaptive_resources() -> tuple[Any, Any, AdaptiveSelectionResourceSnapshot]:
    hardware = detect_hardware(0)
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    resources = AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware,
        available_ram_bytes=_available_ram_bytes(),
    )
    return hardware, resolved, resources


class _NvidiaMemorySampler:
    def __init__(self, *, gpu_index: int = 0, interval_s: float = 0.20) -> None:
        self.gpu_index = int(gpu_index)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self._error = ""

    def _query(self) -> float | None:
        tool = shutil.which("nvidia-smi")
        if not tool:
            self._error = "nvidia_smi_missing"
            return None
        try:
            completed = subprocess.run(
                [tool, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(self.gpu_index)],
                text=True,
                capture_output=True,
                check=False,
                timeout=3.0,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._error = f"nvidia_smi_query_failed:{type(exc).__name__}:{exc}"
            return None
        if completed.returncode != 0:
            self._error = f"nvidia_smi_query_failed:{completed.stderr.strip() or completed.returncode}"
            return None
        try:
            value = float(completed.stdout.strip().splitlines()[0])
        except (ValueError, IndexError):
            self._error = "nvidia_smi_memory_parse_failed"
            return None
        return value if value >= 0.0 else None

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self._samples.append(value)
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        baseline = self._query()
        if baseline is not None:
            self._samples.append(baseline)
        self._thread = threading.Thread(target=self._loop, name="i59b1-nvidia-memory", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 5.0))
        final = self._query()
        if final is not None:
            self._samples.append(final)
        if not self._samples:
            raise ReconstructionValidationError(
                f"I5.9B2 could not measure required GPU memory evidence: {self._error or 'no_samples'}"
            )
        return {
            "measurement_scope": "device_wide_nvidia_smi_peak_used_memory",
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.interval_s,
            "sample_count": len(self._samples),
            "baseline_used_mib": float(self._samples[0]),
            "peak_used_mib": float(max(self._samples)),
            "final_used_mib": float(self._samples[-1]),
            "query_error": self._error,
        }


def _measured_call(call: Callable[[], Any]) -> tuple[Any | None, Exception | None, float, dict[str, Any]]:
    sampler = _NvidiaMemorySampler()
    sampler.start()
    started = time.perf_counter()
    result: Any | None = None
    error: Exception | None = None
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 - classified after evidence measurement
        error = exc
    finally:
        wall = time.perf_counter() - started
        resources = sampler.stop()
    return result, error, wall, resources


def _live_runtime_probes(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    return {
        "depth_anything_v2": probe_depth_anything_v2_environment(
            project_root=project,
            python_executable=python,
            variant=args.dav2_variant,
            source_directory=args.dav2_source_dir.expanduser().resolve() if args.dav2_source_dir else None,
            checkpoint_path=args.dav2_checkpoint.expanduser().resolve() if args.dav2_checkpoint else None,
        ),
        "marigold": probe_marigold_environment(
            project_root=project,
            python_executable=python,
            profile=args.marigold_profile,
            checkpoint_dir=args.marigold_checkpoint_dir.expanduser().resolve() if args.marigold_checkpoint_dir else None,
        ),
    }


def _descriptor(implementation: str, probe: Any):
    return {
        "depth_anything_v2": depth_anything_v2_dense_provider_descriptor,
        "marigold": marigold_dense_provider_descriptor,
    }[implementation](probe)


def _runtime_config(implementation: str, probe: Any, hardware: Any, resolved: Any) -> tuple[dict[str, object], str, str]:
    overrides: dict[str, object] = {}
    if implementation == "depth_anything_v2":
        encoder = {"small": "vits", "base": "vitb", "large": "vitl"}[probe.variant]
        overrides.update({"source_dir": probe.source_directory, "checkpoint": probe.checkpoint_path, "encoder": encoder})
    elif implementation == "marigold":
        overrides.update({"repository": probe.checkpoint_repository, "checkpoint_dir": probe.checkpoint_path, "profile": probe.profile})
    else:
        raise ReconstructionValidationError(f"I5.9B2 unsupported provider: {implementation}")
    config = resolve_model_runtime_config(
        _MODEL_IDS[implementation],
        ModelRuntimeContext(hardware_info=hardware, resolved_profile=resolved, overrides=overrides),
    )
    device = str(config.get("device", resolved.device))
    precision = str(config.get("precision", resolved.precision.value)).lower()
    if not device.startswith("cuda"):
        raise ReconstructionValidationError(f"I5.9B2 requires CUDA for {implementation}: device={device}")
    if precision not in {"fp32", "fp16"}:
        raise ReconstructionValidationError(f"I5.9B2 unsupported precision for {implementation}: {precision}")
    if implementation == "depth_anything_v2":
        configured = Path(str(config.get("checkpoint", ""))).expanduser()
        if not configured.is_absolute(): configured = (ROOT / configured).resolve()
        if configured != Path(probe.checkpoint_path).expanduser().resolve():
            raise ReconstructionValidationError("I5.9B2 Depth Anything V2 checkpoint drifted from probed identity.")
        if Path(str(config.get("source_dir", ""))).expanduser().resolve() != Path(probe.source_directory).expanduser().resolve():
            raise ReconstructionValidationError("I5.9B2 Depth Anything V2 source checkout drifted from probed identity.")
    if implementation == "marigold":
        configured = Path(str(config.get("checkpoint_dir", ""))).expanduser()
        if not configured.is_absolute(): configured = (ROOT / configured).resolve()
        if configured != Path(probe.checkpoint_path).expanduser().resolve() or str(config.get("repository")) != probe.checkpoint_repository:
            raise ReconstructionValidationError("I5.9B2 Marigold runtime config drifted from probed identity.")
    return config, device, precision


def _runtime_identity(implementation: str, probe: Any, config: Mapping[str, object], worker: Path, descriptor: Any) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "available": bool(descriptor.available), "cuda_ready": str(config.get("device", "")).startswith("cuda"),
        "provider_id": descriptor.provider_id, "worker_script": str(worker), "worker_script_sha256": _sha256(worker),
        "runtime_config": dict(config), "environment": probe.as_dict(),
        "alignment_semantics": I59B2_ALIGNMENT_SEMANTICS[implementation],
    }
    for attr in ("checkpoint_path", "source_directory"):
        raw = getattr(probe, attr, None)
        if raw:
            path = Path(str(raw)).expanduser().resolve(); identity[attr] = str(path)
            if path.is_file(): identity[f"{attr}_sha256"] = _sha256(path)
    for attr in ("source_revision", "source_repository", "checkpoint_repository", "checkpoint_declared_revision",
                 "checkpoint_resolved_revision", "checkpoint_requested_revision", "checkpoint_revision_pinned",
                 "checkpoint_marker_sha256", "model_index_sha256", "checkpoint_file_count", "checkpoint_size_bytes",
                 "variant", "encoder", "profile", "distribution_name", "distribution_version"):
        raw = getattr(probe, attr, None)
        if raw is not None: identity[attr] = raw
    return identity


def _adapter(
    implementation: str, *, sfm_result: Any, frame_id: str, output_dir: Path, probe: Any,
    runtime_config: Mapping[str, object], device: str, precision: str,
):
    common = {"sfm_result": sfm_result, "reconstruction_frame_id": frame_id, "output_dir": output_dir,
              "python_executable": Path(probe.python_executable), "worker_script": ROOT / "scripts" / _WORKER_SCRIPTS[implementation],
              "runtime_config": runtime_config, "device": device, "precision": precision}
    if implementation == "depth_anything_v2":
        return DepthAnythingV2ProductionProviderAdapter(context=DepthAnythingV2ProviderRunContext(**common), environment=probe)
    if implementation == "marigold":
        return MarigoldProductionProviderAdapter(context=MarigoldProviderRunContext(**common), environment=probe)
    raise ReconstructionValidationError(f"I5.9B2 unsupported provider: {implementation}")


def _registered_view_projection(sfm_result: Any) -> Any:
    registered = {camera.keyframe_id for camera in sfm_result.cameras}
    ordered_workspace = tuple(sorted(sfm_result.workspace.images, key=lambda item: item.order))
    selected_workspace = tuple(item for item in ordered_workspace if item.keyframe_id in registered)
    if len(selected_workspace) != sfm_result.statistics.registered_keyframes:
        raise ReconstructionValidationError("I5.9B2 could not project every registered SfM camera to one workspace image.")
    if len(selected_workspace) == len(ordered_workspace):
        return sfm_result
    projected_images = tuple(replace(item, order=index) for index, item in enumerate(selected_workspace))
    projected_workspace = SfmWorkspaceManifest(
        session_id=sfm_result.workspace.session_id,
        source=sfm_result.workspace.source,
        workspace_root=sfm_result.workspace.workspace_root,
        images_directory=sfm_result.workspace.images_directory,
        database_path=sfm_result.workspace.database_path,
        sparse_directory=sfm_result.workspace.sparse_directory,
        checkpoints_directory=sfm_result.workspace.checkpoints_directory,
        reports_directory=sfm_result.workspace.reports_directory,
        logs_directory=sfm_result.workspace.logs_directory,
        images=projected_images,
    )
    projected_landmarks = []
    for landmark in sfm_result.landmarks:
        observations = tuple(obs for obs in landmark.observations if obs.keyframe_id in registered)
        if len(observations) >= 2:
            projected_landmarks.append(replace(landmark, observations=observations))
    observation_count = sum(len(item.observations) for item in projected_landmarks)
    camera_ids = {camera.camera_id for camera in sfm_result.cameras}
    projected_camera_models = tuple(model for model in sfm_result.camera_models if model.camera_id in camera_ids)
    metadata = dict(sfm_result.metadata)
    metadata["i5_9b2_registered_view_projection"] = {
        "projection_only": True,
        "authoritative_sfm_modified": False,
        "original_input_keyframes": sfm_result.statistics.input_keyframes,
        "registered_keyframes": sfm_result.statistics.registered_keyframes,
        "camera_pose_authority_owner": "pycolmap_sfm",
    }
    return replace(
        sfm_result,
        workspace=projected_workspace,
        camera_models=projected_camera_models,
        landmarks=tuple(projected_landmarks),
        statistics=SfmStageStatistics(
            input_keyframes=len(projected_images),
            registered_keyframes=len(sfm_result.cameras),
            landmarks=len(projected_landmarks),
            observations=observation_count,
            elapsed_ns=sfm_result.statistics.elapsed_ns,
            mean_reprojection_error_px=sfm_result.statistics.mean_reprojection_error_px,
        ),
        metadata=metadata,
    )


def _scenario_report(admission: Mapping[str, Any], sid: str) -> tuple[Path, dict[str, Any], dict[str, Any], Any]:
    matrix = admission["scenario_matrix"]
    node = matrix[sid]
    rb = node["report_binding"]
    path = Path(str(rb["path"])).expanduser().resolve()
    expected = str(rb["sha256"]).strip().lower()
    _binding(path, expected, label=f"{sid} frozen SfM report")
    report = _read_json(path, label=f"{sid} frozen SfM report")
    analysis = analyze_i5_6a_sfm_report(
        report,
        scenario_id=sid,
        expected_input=int(node["input_images"]),
        expected_registered=int(node["registered_images"]),
    )
    sfm = offline_sfm_result_from_dict(report["sfm_result"])
    return path, report, analysis, sfm


def _verify_scenario_unchanged(admission: Mapping[str, Any], sid: str, before: Mapping[str, Any]) -> dict[str, Any]:
    _, _, after, _ = _scenario_report(admission, sid)
    for key in (
        "registered_images_sha256", "selected_sparse_sha256", "registered_image_names",
        "selected_model_id", "registered_images", "input_images",
    ):
        if before.get(key) != after.get(key):
            raise ReconstructionValidationError(f"I5.9B2 {sid} authoritative source/SfM drift after provider run: {key}")
    return after


def _canonical_point_count(path: Path) -> int:
    source = path.expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            points = np.asarray(archive["points_reconstruction"])
    except Exception as exc:  # noqa: BLE001
        raise ReconstructionValidationError(f"I5.9B2 could not inspect canonical point artifact {source}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3:
        raise ReconstructionValidationError("I5.9B2 canonical point artifact has invalid point shape.")
    return int(points.shape[0])


def _worker_artifacts(provider_out: Path, *, implementation: str) -> tuple[list[Path], int]:
    worker_manifest = provider_out / "provider" / "worker_output" / _WORKER_RESULT_MANIFESTS[implementation]
    input_manifest = provider_out / "provider" / "input_manifest.json"
    payload = _read_json(worker_manifest, label=f"{implementation} worker result")
    records = payload.get("frame_records")
    if not isinstance(records, list) or not records:
        raise ReconstructionValidationError(f"I5.9B2 {implementation} worker result lacks frame records.")
    paths = [input_manifest, worker_manifest]
    base = worker_manifest.parent
    for record in records:
        if not isinstance(record, Mapping):
            raise ReconstructionValidationError(f"I5.9B2 {implementation} worker frame record is malformed.")
        raw = str(record.get("evidence_npz_path", "")).strip()
        if not raw:
            raise ReconstructionValidationError(f"I5.9B2 {implementation} worker frame evidence path is missing.")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (base / path).resolve()
        paths.append(path)
    return paths, len(records)


def _alignment_summary(summary: Mapping[str, Any], *, selected: int) -> tuple[int, dict[str, Any]]:
    alignment = summary.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ReconstructionValidationError("I5.9B2 provider summary is missing alignment evidence.")
    accepted = int(alignment.get("accepted_views", 0))
    records = alignment.get("records")
    if accepted < 0 or accepted > selected or not isinstance(records, list) or len(records) != selected:
        raise ReconstructionValidationError("I5.9B2 provider alignment accounting is invalid.")
    residuals = [
        float(item["median_relative_residual"])
        for item in records
        if isinstance(item, Mapping) and item.get("status") == "accepted" and item.get("median_relative_residual") is not None
    ]
    return accepted, {
        "alignment_accepted_fraction": accepted / float(selected),
        "alignment_median_relative_residual": (float(np.median(residuals)) if residuals else None),
    }


def _fusion_capacity_drops(summary: Mapping[str, Any]) -> int:
    fusion = summary.get("fusion")
    if not isinstance(fusion, Mapping):
        return 0
    stats = fusion.get("statistics")
    if not isinstance(stats, Mapping):
        return 0
    return int(stats.get("dropped_new_voxels_capacity", 0) or 0)


def _attempt_dir(root: Path, sid_index: int, sid: str, provider_index: int, provider: str, attempt: int) -> Path:
    path = root / f"{sid_index:02d}_{sid}" / f"{provider_index:02d}_{provider}_attempt_{attempt:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _run_provider(
    *,
    admission: Mapping[str, Any],
    sid: str,
    sid_index: int,
    implementation: str,
    provider_index: int,
    base_sfm: Any,
    before_analysis: Mapping[str, Any],
    hardware: Any,
    resolved_profile: Any,
    resources: AdaptiveSelectionResourceSnapshot,
    probe: Any,
    root: Path,
    attempt: int,
) -> dict[str, Any]:
    frame_id = f"i5_9b2:{sid}:pycolmap_sfm:{base_sfm.session_id}"
    descriptor = _descriptor(implementation, probe)
    prep = prepare_adaptive_provider_execution(
        base_sfm,
        provider=descriptor,
        reconstruction_frame_id=frame_id,
        resources=resources,
        override=None,
    )
    validate_single_image_production_plan(prep.plan.as_dict(), implementation=implementation)
    attempt_dir = _attempt_dir(root, sid_index, sid, provider_index, implementation, attempt)
    plan_path = _write_json(attempt_dir / "adaptive_execution_plan.json", prep.plan.as_dict())
    runtime_config, device, precision = _runtime_config(implementation, probe, hardware, resolved_profile)
    provider_out = attempt_dir / "provider"
    adapter = _adapter(
        implementation,
        sfm_result=prep.provider_sfm_result,
        frame_id=frame_id,
        output_dir=provider_out,
        probe=probe,
        runtime_config=runtime_config,
        device=device,
        precision=precision,
    )
    selected = int(prep.plan.selected_view_count)
    local_result, error, wall, monitor = _measured_call(
        lambda: adapter.execute(prep.selection_request, prep.selection_decision)
    )
    rejection: dict[str, Any] | None = None
    summary_path: Path | None = None
    canonical_point_count: int | None = None
    accepted = 0
    processed = selected
    status = ""
    result_payload: dict[str, Any] | None = None
    native_artifacts: list[Path] = []
    alignment_metrics: dict[str, Any] = {}
    capacity_drops = 0

    if error is not None:
        message = str(error)
        if not isinstance(error, ReconstructionValidationError) or not recognized_quality_rejection(
            implementation=implementation, error_message=message
        ):
            failure = {
                "schema_version": 1,
                "phase": "I5.9B2",
                "status": "runtime_or_integrity_failure",
                "pass": False,
                "scenario_id": sid,
                "provider_implementation": implementation,
                "attempt": attempt,
                "error_type": type(error).__name__,
                "error": message,
                "retry_performed": False,
                "threshold_changed": False,
                "production_default_modified": False,
            }
            _write_json(attempt_dir / FAILURE_NAME, failure)
            raise error
        match = _REJECTION_COUNTS.search(message)
        if match is None:
            raise ReconstructionValidationError(
                f"I5.9B2 recognized {implementation} rejection lacks accepted/required counts: {message}"
            )
        accepted = int(match.group("accepted"))
        required = int(match.group("required"))
        native_artifacts, processed = _worker_artifacts(provider_out, implementation=implementation)
        if processed != selected or accepted < 0 or accepted >= required or accepted > selected:
            raise ReconstructionValidationError(
                f"I5.9B2 rejected {sid}/{implementation} accounting mismatch: "
                f"selected={selected} processed={processed} accepted={accepted} required={required}"
            )
        status = "rejected"
        rejection = {
            "terminal": True,
            "recognized_existing_quality_gate": True,
            "reason": message,
            "accepted_views": accepted,
            "required_accepted_views": required,
            "canonical_geometry_generated": False,
            "threshold_changed": False,
            "provider_execution_reran": False,
        }
    else:
        if local_result is None:
            raise ReconstructionValidationError(f"I5.9B2 {sid}/{implementation} returned no result and no error.")
        result = finalize_adaptive_provider_execution_result(local_result, prep)
        if result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
            raise ReconstructionValidationError(
                f"I5.9B2 {sid}/{implementation} returned non-terminal production status: "
                f"{result.status.value} {result.failure_reason}"
            )
        status = "degraded" if result.status is DenseProviderRunStatus.DEGRADED else "pass"
        result_payload = result.as_dict()
        summary_path = provider_out / _SUMMARY_FILENAMES[implementation]
        summary = _read_json(summary_path, label=f"{sid}/{implementation} execution summary")
        accepted, alignment_metrics = _alignment_summary(summary, selected=selected)
        capacity_drops = _fusion_capacity_drops(summary)
        canonical = Path(result.canonical_geometry_ref).expanduser().resolve()
        canonical_point_count = _canonical_point_count(canonical)
        native_artifacts = [summary_path, canonical, *[Path(value) for value in result.native_artifact_refs]]
        worker_artifacts, worker_count = _worker_artifacts(provider_out, implementation=implementation)
        if worker_count != selected:
            raise ReconstructionValidationError(f"I5.9B2 {sid}/{implementation} worker count drifted from adaptive plan.")
        processed = worker_count
        native_artifacts.extend(worker_artifacts)

    _verify_scenario_unchanged(admission, sid, before_analysis)
    record = {
        "schema_version": 1,
        "phase": "I5.9B2",
        "scenario_id": sid,
        "provider_implementation": implementation,
        "runtime_status": status,
        "source_registered_views": int(base_sfm.statistics.registered_keyframes),
        "selected_views": selected,
        "processed_views": processed,
        "accepted_views": accepted,
        "adaptive_selection": True,
        "plan_reason_codes": list(prep.plan.reason_codes),
        "input_integrity_revalidated_after": True,
        "fusion_capacity_drops": capacity_drops,
        "fusion_capacity_drops_change_status": False,
        "alignment_semantics": I59B2_ALIGNMENT_SEMANTICS[implementation],
        "runtime_config": dict(runtime_config),
        "native_metrics": {
            "wall_time_s": float(wall),
            "peak_vram_mib": float(monitor["peak_used_mib"]),
            "canonical_point_count": canonical_point_count,
            **alignment_metrics,
        },
        "resource_evidence": dict(monitor),
        "execution_result": result_payload,
        "artifact_bindings": {
            "adaptive_plan": _artifact(plan_path),
            "native_artifacts": _hash_artifacts(native_artifacts),
            **({"provider_summary": _artifact(summary_path)} if summary_path is not None else {}),
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "provider_promotion_performed": False,
            "permanent_global_view_cap": None,
        },
    }
    if rejection is not None:
        record["rejection"] = rejection
    record_path = _write_json(attempt_dir / RECORD_NAME, record)
    record["record_binding"] = _artifact(record_path)
    return record


def _build_preflight(
    *,
    admission: Mapping[str, Any],
    i59a_binding: I51EvidenceBinding,
    i56a_binding: I51EvidenceBinding,
    b1_binding: I51EvidenceBinding,
    receipt_binding: I51EvidenceBinding,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Any, Any, AdaptiveSelectionResourceSnapshot]:
    hardware, resolved_profile, resources = _adaptive_resources()
    probes = _live_runtime_probes(args)
    runtimes: dict[str, Any] = {}
    resolved_configs: dict[str, tuple[dict[str, object], str, str]] = {}
    for provider in I59B2_PROVIDERS:
        probe = probes[provider]
        descriptor = _descriptor(provider, probe)
        if not descriptor.available:
            raise ReconstructionValidationError(
                f"I5.9B2 runtime is not available for {provider}: {descriptor.unavailable_reason}; probe={probe.as_dict()}"
            )
        config, device, precision = _runtime_config(provider, probe, hardware, resolved_profile)
        worker = (ROOT / "scripts" / _WORKER_SCRIPTS[provider]).resolve()
        if not worker.is_file():
            raise ReconstructionValidationError(f"I5.9B2 worker script missing for {provider}: {worker}")
        identity = _runtime_identity(provider, probe, config, worker, descriptor)
        identity["device"] = device
        identity["precision"] = precision
        runtimes[provider] = identity
        resolved_configs[provider] = (config, device, precision)

    scenario_plans: dict[str, Any] = {}
    for sid in I59B2_SCENARIO_IDS:
        report_path, _, analysis, sfm = _scenario_report(admission, sid)
        base_sfm = _registered_view_projection(sfm)
        frame_id = f"i5_9b2:{sid}:pycolmap_sfm:{base_sfm.session_id}"
        provider_plans: dict[str, Any] = {}
        for provider in I59B2_PROVIDERS:
            descriptor = _descriptor(provider, probes[provider])
            prep = prepare_adaptive_provider_execution(
                base_sfm,
                provider=descriptor,
                reconstruction_frame_id=frame_id,
                resources=resources,
                override=None,
            )
            validate_single_image_production_plan(prep.plan.as_dict(), implementation=provider)
            provider_plans[provider] = {
                "provider_id": descriptor.provider_id,
                "source_view_count": int(prep.plan.source_view_count),
                "selected_view_count": int(prep.plan.selected_view_count),
                "selected_keyframe_ids": list(prep.plan.selected_keyframe_ids),
                "plan_reason_codes": list(prep.plan.reason_codes),
                "provider_inference_started": False,
            }
        scenario_plans[sid] = {
            "frozen_sfm_report": _artifact(report_path),
            "input_images": int(analysis["input_images"]),
            "registered_views": int(analysis["registered_images"]),
            "registered_images_sha256": analysis["registered_images_sha256"],
            "selected_sparse_sha256": analysis["selected_sparse_sha256"],
            "source_integrity_verified": True,
            "authoritative_sfm_reused_without_rerun": True,
            "registered_view_projection_used": base_sfm is not sfm,
            "providers": provider_plans,
        }
    preflight = {
        "schema_version": 1,
        "phase": "I5.9B2",
        "status": "runtime_source_preflight_pass",
        "pass": True,
        "providers": list(I59B2_PROVIDERS),
        "scenario_ids": list(I59B2_SCENARIO_IDS),
        "expected_provider_scenario_runs": I59B2_EXPECTED_RUN_COUNT,
        "historical_g4_low_texture_anchor": {**I59B2_HISTORICAL_G4_LOW_TEXTURE, "preserved": True},
        "runtime": runtimes,
        "scenario_plans": scenario_plans,
        "hardware": {
            "backend": str(getattr(getattr(hardware, "backend", None), "value", getattr(hardware, "backend", ""))),
            "device": str(getattr(hardware, "device", "")),
            "device_name": str(getattr(hardware, "name", "")),
            "total_memory_gib": getattr(hardware, "total_memory_gib", None),
            "free_memory_gib": getattr(hardware, "free_memory_gib", None),
            "resolved_profile": str(getattr(getattr(resolved_profile, "effective", None), "value", "")),
            "resolved_precision": str(getattr(getattr(resolved_profile, "precision", None), "value", "")),
            "profile_downgraded": bool(getattr(resolved_profile, "downgraded", False)),
            "profile_warning": getattr(resolved_profile, "warning", None),
        },
        "evidence_bindings": {
            "i5_9a_admission": i59a_binding.as_dict(),
            "i5_6a_admission": i56a_binding.as_dict(),
            "i5_9b1_replay": b1_binding.as_dict(),
            "replay_receipt": receipt_binding.as_dict(),
        },
        "scope": {
            "provider_inference_started": False,
            "sfm_rerun": False,
            "mvs_rerun": False,
            "provider_comparison_started": False,
            "result_driven_retuning": False,
            "retry_until_pass": False,
            "thresholds_modified": False,
            "fusion_capacity_increased": False,
            "production_default_modified": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }
    validate_i5_9b2_runtime_preflight(preflight)
    return preflight, probes, hardware, resolved_profile, resources


def run(args: argparse.Namespace) -> Path:
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise ReconstructionValidationError(f"I5.9B2 output directory must not already exist: {out}")
    out.mkdir(parents=True, exist_ok=False)

    i59a_binding = _binding(
        args.i5_9a_admission,
        args.i5_9a_sha256,
        label="I5.9A admission",
        frozen_sha=I59B2_I59A_FROZEN_SHA256,
    )
    i56a_binding = _binding(
        args.i5_6a_admission,
        args.i5_6a_sha256,
        label="I5.6A admission",
        frozen_sha=I59B2_I56A_FROZEN_SHA256,
    )
    b1_binding = _binding(args.i5_9b1_replay, args.i5_9b1_sha256, label="I5.9B1 replay")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="I5.9B2 replay receipt")
    admission = validate_i5_9a_for_i5_9b2(_read_json(Path(i59a_binding.path), label="I5.9A admission"))
    validate_i5_6a_for_i5_6b(_read_json(Path(i56a_binding.path), label="I5.6A admission"))
    b1_replay = validate_i5_9b1_for_i5_9b2(_read_json(Path(b1_binding.path), label="I5.9B1 replay"))
    receipt = validate_i5_9b2_receipt(_read_json(Path(receipt_binding.path), label="I5.9B2 replay receipt"))

    preflight, probes, hardware, resolved_profile, resources = _build_preflight(
        admission=admission,
        i59a_binding=i59a_binding,
        i56a_binding=i56a_binding,
        b1_binding=b1_binding,
        receipt_binding=receipt_binding,
        args=args,
    )
    preflight_path = _write_json(out / PREFLIGHT_NAME, preflight)
    preflight_binding = I51EvidenceBinding(str(preflight_path), _sha256(preflight_path))

    print("I5.9B2 relative-depth provider runtime/source preflight: PASS")
    print(
        f"providers={','.join(I59B2_PROVIDERS)} scenarios={len(I59B2_SCENARIO_IDS)} "
        f"provider_scenario_plans={I59B2_EXPECTED_RUN_COUNT}"
    )
    print(
        f"hardware={preflight['hardware']['device_name']} profile={preflight['hardware']['resolved_profile']} "
        f"precision={preflight['hardware']['resolved_precision']}"
    )
    for provider in I59B2_PROVIDERS:
        node = preflight["runtime"][provider]
        print(
            f"{provider} available={node['available']} device={node['device']} precision={node['precision']} "
            f"provider_id={node['provider_id']}"
        )
    print("provider_inference_started=False sfm_rerun=False mvs_rerun=False provider_comparison=False")
    print(f"preflight={preflight_path}")
    if args.mode == "preflight":
        return preflight_path

    records: dict[str, dict[str, Any]] = {}
    for sid_index, sid in enumerate(I59B2_SCENARIO_IDS, start=1):
        print(f"\n===== I5.9B2 SCENARIO {sid_index}/6: {sid} =====", flush=True)
        _, _, before, sfm = _scenario_report(admission, sid)
        base_sfm = _registered_view_projection(sfm)
        records[sid] = {}
        for provider_index, provider in enumerate(I59B2_PROVIDERS, start=1):
            print(f"[{sid_index}/6][{provider_index}/2] executing {provider} exactly once ...", flush=True)
            record = _run_provider(
                admission=admission,
                sid=sid,
                sid_index=sid_index,
                implementation=provider,
                provider_index=provider_index,
                base_sfm=base_sfm,
                before_analysis=before,
                hardware=hardware,
                resolved_profile=resolved_profile,
                resources=resources,
                probe=probes[provider],
                root=out,
                attempt=int(args.attempt),
            )
            records[sid][provider] = record
            print(
                f"[{sid_index}/6][{provider_index}/2] {provider}: {record['runtime_status']} "
                f"selected={record['selected_views']}/{record['source_registered_views']} "
                f"accepted={record['accepted_views']} capacity_drops={record['fusion_capacity_drops']} "
                f"wall_time_s={record['native_metrics']['wall_time_s']:.3f}",
                flush=True,
            )

    result = build_i5_9b2_replay(
        i5_9a_admission=admission,
        i5_9a_binding=i59a_binding,
        i5_6a_binding=i56a_binding,
        i5_9b1_replay=b1_replay,
        i5_9b1_binding=b1_binding,
        replay_receipt=receipt,
        replay_receipt_binding=receipt_binding,
        runtime_preflight=preflight,
        runtime_preflight_binding=preflight_binding,
        records=records,
    )
    evidence = _write_json(out / OUTPUT_NAME, result)
    counts = result["terminal_status_counts"]
    print("\nI5.9B2 Relative-Depth Single-Image Provider Robustness Replay: EVIDENCE_COMPLETE")
    print(
        f"terminal_runs={result['terminal_provider_scenario_runs']}/{I59B2_EXPECTED_RUN_COUNT} "
        f"pass={counts['pass']} degraded={counts['degraded']} rejected={counts['rejected']} "
        f"quality_status={result['quality_status']}"
    )
    print(
        "fusion_capacity_drops_change_single_image_status=False sfm_rerun=False mvs_rerun=False "
        "result_driven_retuning=False retry_until_pass=False threshold_relaxation=False "
        "production_default_modified=False provider_comparison=False metric_scale=False metrology=False "
        "permanent_global_view_cap=None"
    )
    print("I5.9C remaining depth-provider robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence}")
    return evidence


def main() -> int:
    args = _parser().parse_args()
    current: dict[str, Any] = {"scenario_id": "", "provider_implementation": ""}
    try:
        return_code_path = run(args)
        _ = return_code_path
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary preserves failure evidence
        failure = {
            "schema_version": 1,
            "phase": "I5.9B2",
            "status": "failed",
            "pass": False,
            "mode": str(getattr(args, "mode", "")),
            "attempt": int(getattr(args, "attempt", 1)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_performed": False,
            "threshold_changed": False,
            "fusion_capacity_increased": False,
            "production_default_modified": False,
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "authoritative_sfm_modified": False,
                "provider_selected": False,
                "provider_promoted": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "permanent_global_view_cap": None,
            },
            **current,
        }
        try:
            out = args.output_dir.expanduser().resolve()
            if out.exists():
                _write_json(out / FAILURE_NAME, failure)
        except Exception:
            pass
        print(f"I5.9B2 ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
