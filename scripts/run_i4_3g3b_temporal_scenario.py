#!/usr/bin/env python3
"""I4.3G3B production-native temporal-provider scenario runner.

Modes:
* preflight: verify the exact G3A snapshot, bound source, PyCOLMAP runtime, and
  the unchanged VDA/RollingDepth runtimes. No SfM or provider is started.
* production-native: build one authoritative PyCOLMAP SfM for the selected
  frozen temporal scenario, derive independent I4.3F provider plans from that
  same SfM, then execute RollingDepth and Video Depth Anything sequentially.

G3B does not perform matched-view diagnostics or manual provider ranking; those
remain I4.3G3C work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware.gpu_info import detect_hardware  # noqa: E402
from core.hardware.profiles import ComputeProfile, resolve_profile  # noqa: E402
from core.models.catalog import VIDEO_DEPTH_ANYTHING_MODEL_ID  # noqa: E402
from core.models.registry import MODEL_REGISTRY  # noqa: E402
from core.models.runtime_config import ModelRuntimeContext  # noqa: E402
from core.reconstruction import (  # noqa: E402
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    RecordedVideoReader,
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    SfmResumePolicy,
    materialize_sfm_workspace,
)
from core.reconstruction.adaptive_production_planning import (  # noqa: E402
    build_adaptive_pre_sfm_frame_plan,
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot  # noqa: E402
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus  # noqa: E402
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.rolling_depth_dense_provider import (  # noqa: E402
    RollingDepthProductionProviderAdapter,
    RollingDepthProviderRunContext,
    probe_rolling_depth_environment,
    rolling_depth_dense_provider_descriptor,
)
from core.reconstruction.sfm_environment import probe_pycolmap_environment  # noqa: E402
from core.reconstruction.temporal_provider_production_qualification import (  # noqa: E402
    build_g3b_runtime_preflight_snapshot,
    read_json_object,
    resolve_g3b_scenario_binding,
    sha256_file,
    validate_temporal_production_plan,
    validate_temporal_provider_summary_authority,
    validate_temporal_provider_run_evidence,
    verify_snapshot_sha256,
)
from core.reconstruction.temporal_provider_qualification import (  # noqa: E402
    build_temporal_video_identity,
    validate_i4_3g3_predecessors,
)
from core.reconstruction.video_depth_anything_dense_provider import (  # noqa: E402
    VideoDepthAnythingProductionProviderAdapter,
    VideoDepthAnythingProviderRunContext,
    probe_video_depth_anything_environment,
    video_depth_anything_dense_provider_descriptor,
)

_MIB = 1024**2


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"I4.3G3B expected artifact is missing: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "size_bytes": int(source.stat().st_size)}


def _hash_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        source = Path(path).expanduser().resolve()
        key = str(source)
        if key in seen:
            continue
        seen.add(key)
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


def _adaptive_resources() -> AdaptiveSelectionResourceSnapshot:
    hardware = detect_hardware(0)
    return AdaptiveSelectionResourceSnapshot.from_hardware(
        hardware,
        available_ram_bytes=_available_ram_bytes(),
    )


class _NvidiaMemorySampler:
    """Best-available device-wide peak used-memory sampler for G1 evidence."""

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
                [
                    tool,
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.gpu_index),
                ],
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
        self._thread = threading.Thread(target=self._loop, name="i43g3b-nvidia-memory", daemon=True)
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
                f"I4.3G3B could not measure required peak GPU memory: {self._error or 'no_samples'}"
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


def _timed_provider_run(call: Callable[[], Any]) -> tuple[Any, float, dict[str, Any]]:
    sampler = _NvidiaMemorySampler()
    sampler.start()
    started = time.perf_counter()
    try:
        result = call()
    finally:
        wall = time.perf_counter() - started
        resources = sampler.stop()
    return result, wall, resources


def _verify_g3a_predecessor_chain(g3a: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = g3a.get("predecessor_evidence")
    if not isinstance(predecessor, Mapping):
        raise ReconstructionValidationError("I4.3G3B requires G3A predecessor_evidence.")
    g1_path = Path(str(predecessor.get("i4_3g1_snapshot", ""))).expanduser().resolve()
    g2_path = Path(str(predecessor.get("i4_3g2_closure", ""))).expanduser().resolve()
    g1_sha = str(predecessor.get("i4_3g1_sha256", "")).strip().lower()
    g2_sha = str(predecessor.get("i4_3g2_sha256", "")).strip().lower()
    verify_snapshot_sha256(g1_path, g1_sha, label="G3A-bound I4.3G1 snapshot")
    verify_snapshot_sha256(g2_path, g2_sha, label="G3A-bound I4.3G2 closure")
    g1 = read_json_object(g1_path, label="G3A-bound I4.3G1 snapshot")
    g2 = read_json_object(g2_path, label="G3A-bound I4.3G2 closure")
    validate_i4_3g3_predecessors(i4_3g1_snapshot=g1, i4_3g2_closure=g2)
    return {
        "i4_3g1": {"path": str(g1_path), "sha256": g1_sha},
        "i4_3g2": {"path": str(g2_path), "sha256": g2_sha},
    }


def _current_temporal_probes(g3a: Mapping[str, Any]):
    runtime = g3a.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReconstructionValidationError("I4.3G3B G3A runtime section is missing.")
    bound_vda = runtime.get("video_depth_anything")
    bound_rolling = runtime.get("rolling_depth")
    if not isinstance(bound_vda, Mapping) or not isinstance(bound_rolling, Mapping):
        raise ReconstructionValidationError("I4.3G3B G3A temporal runtime bindings are missing.")

    vda_root = Path(str(bound_vda.get("project_root", ROOT))).expanduser().resolve()
    vda_python = Path(str(bound_vda.get("python_executable", sys.executable))).expanduser().resolve()
    vda_checkpoint = Path(str(bound_vda.get("checkpoint_path", ""))).expanduser().resolve()
    vda = probe_video_depth_anything_environment(
        vda_root,
        python_executable=vda_python,
        checkpoint=vda_checkpoint,
    )

    rolling_python = Path(str(bound_rolling.get("python_executable", ""))).expanduser().resolve()
    rolling_repo = Path(str(bound_rolling.get("repository_root", ""))).expanduser().resolve()
    rolling_checkpoint_text = str(bound_rolling.get("checkpoint_path", "")).strip()
    rolling_checkpoint = Path(rolling_checkpoint_text).expanduser().resolve() if rolling_checkpoint_text else None
    rolling = probe_rolling_depth_environment(
        python_executable=rolling_python,
        repository_root=rolling_repo,
        checkpoint_path=rolling_checkpoint,
    )
    return vda, rolling


def _validate_temporal_workspace_order(workspace) -> tuple[list[int], list[int]]:
    source_indices = [int(item.source_sequence_index) for item in workspace.images]
    timestamps = [int(item.timestamp_ns) for item in workspace.images]
    if source_indices != sorted(source_indices) or len(source_indices) != len(set(source_indices)):
        raise ReconstructionValidationError(
            "I4.3G3B SfM workspace did not preserve unique increasing temporal source order."
        )
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ReconstructionValidationError(
            "I4.3G3B SfM workspace did not preserve strictly increasing canonical timestamps."
        )
    return source_indices, timestamps


def _build_authoritative_sfm(
    *,
    binding,
    sfm_python: Path,
    output_dir: Path,
    resources: AdaptiveSelectionResourceSnapshot,
    timeout_minutes: float,
):
    source = RecordedVideoReader(binding.source_path)
    pre_plan = build_adaptive_pre_sfm_frame_plan(
        source_kind="recorded_video",
        estimated_total_frames=source.estimated_total_frames,
        expected_frame_rate_hz=source.metadata.expected_frame_rate_hz,
        resources=resources,
        engineer_max_keyframes_override=None,
    )
    target = pre_plan.target_sampled_frames
    total = source.estimated_total_frames
    reduced = target is not None and total is not None and target < total
    sampling = (
        FrameSamplingPolicy(
            mode=FrameSamplingMode.TARGET_FPS,
            target_fps_hz=float(source.frame_rate_hz),
            maximum_sampled_frames=target,
        )
        if reduced
        else FrameSamplingPolicy(mode=FrameSamplingMode.EVERY_FRAME, maximum_sampled_frames=None)
    )

    session_id = str(uuid.uuid4())
    events = ReconstructionEventRecorder(session_id)
    control = ReconstructionOperationControl()
    ingestion = ReconstructionIngestionPipeline(
        session_id=session_id,
        source=source,
        sampling_policy=sampling,
        keyframe_policy=KeyframeSelectionPolicy(
            minimum_spacing_ns=1,
            forced_spacing_ns=1,
            minimum_visual_change=0.0,
            minimum_quality_score=0.0,
            maximum_keyframes=pre_plan.target_sampled_frames,
        ),
        quality_policy=FrameQualityPolicy(),
        cache_policy=IngestionCachePolicy(maximum_cpu_bytes=512 * _MIB, maximum_frames=8, gpu_staging_bytes=0),
    ).run(control=control, events=events)
    if len(ingestion.keyframes) < 3:
        raise ReconstructionValidationError(
            f"I4.3G3B authoritative SfM requires at least three ingested temporal keyframes; got {len(ingestion.keyframes)}."
        )
    sfm_root = output_dir / "authoritative_sfm"
    workspace = materialize_sfm_workspace(ingestion, sfm_root / "workspace", overwrite=True)
    source_indices, timestamps = _validate_temporal_workspace_order(workspace)
    request = OfflineSfmRequest(
        session_id=session_id,
        profile=ReconstructionQualityProfile.BALANCED,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=SfmMatchingStrategy.SEQUENTIAL,
            use_gpu=True,
            maximum_image_size=None,
            sequential_overlap=None,
            resume_policy=SfmResumePolicy.REUSE_COMPLETED_STAGES,
        ),
    )
    backend = ExternalPycolmapSfmBackend(str(sfm_python.expanduser().resolve()), timeout_s=timeout_minutes * 60.0)
    started = time.perf_counter()
    try:
        result = backend.run_sparse_reconstruction(
            request,
            BackendOperationContext(
                session_id=session_id,
                stage=ReconstructionStage.SPARSE_RECONSTRUCTION,
                profile=ReconstructionQualityProfile.BALANCED,
                control=control,
                emit_event=lambda _event: None,
            ),
        )
    finally:
        backend.shutdown()
    elapsed = time.perf_counter() - started
    if result.statistics.registered_keyframes < 3 or result.statistics.landmarks <= 0:
        raise ReconstructionValidationError(
            "I4.3G3B authoritative temporal SfM did not produce a usable connected sparse reconstruction."
        )

    registered_images = [
        item for item in result.workspace.images
        if item.keyframe_id in {camera.keyframe_id for camera in result.cameras}
    ]
    registered_images.sort(key=lambda item: (item.source_sequence_index, item.timestamp_ns, item.order))
    registered_indices = [item.source_sequence_index for item in registered_images]
    registered_timestamps = [item.timestamp_ns for item in registered_images]
    if registered_indices != sorted(registered_indices) or len(registered_indices) != len(set(registered_indices)):
        raise ReconstructionValidationError("I4.3G3B registered SfM views lost temporal source order.")
    if any(b <= a for a, b in zip(registered_timestamps, registered_timestamps[1:])):
        raise ReconstructionValidationError("I4.3G3B registered SfM views lost increasing temporal timestamps.")

    pre_path = _write_json(sfm_root / "adaptive_pre_sfm_plan.json", pre_plan.as_dict())
    ingestion_path = _write_json(sfm_root / "ingestion_result.json", ingestion.as_dict())
    sfm_path = _write_json(sfm_root / "sfm_result.json", result.as_dict())
    frame_id = f"g3b:{binding.scenario_id}:pycolmap_sfm:{session_id}"
    return result, frame_id, {
        "session_id": session_id,
        "wall_time_s": float(elapsed),
        "matching_strategy": SfmMatchingStrategy.SEQUENTIAL.value,
        "source_reported_frame_count": binding.reported_frame_count,
        "ingested_keyframe_count": len(ingestion.keyframes),
        "registered_view_count": result.statistics.registered_keyframes,
        "registration_fraction": result.statistics.registered_keyframes / float(len(ingestion.keyframes)),
        "landmark_count": result.statistics.landmarks,
        "observation_count": result.statistics.observations,
        "mean_reprojection_error_px": result.statistics.mean_reprojection_error_px,
        "registered_source_sequence_indices": registered_indices,
        "registered_timestamps_ns": registered_timestamps,
        "pre_sfm_plan": _artifact(pre_path),
        "ingestion": _artifact(ingestion_path),
        "snapshot": _artifact(sfm_path),
        "frame_id": frame_id,
        "camera_pose_authority_owner": "pycolmap_sfm",
        "metric_scale_claimed": False,
    }


def _vda_runtime_config(vda_probe) -> tuple[dict[str, Any], str, str]:
    hardware = detect_hardware(0)
    resolved = resolve_profile(ComputeProfile.AUTO, hardware)
    config = MODEL_REGISTRY.resolve_runtime_config(
        VIDEO_DEPTH_ANYTHING_MODEL_ID,
        ModelRuntimeContext(hardware_info=hardware, resolved_profile=resolved),
    )
    configured_checkpoint = Path(str(config.get("checkpoint", ""))).expanduser()
    if not configured_checkpoint.is_absolute():
        configured_checkpoint = (ROOT / configured_checkpoint).resolve()
    bound_checkpoint = Path(vda_probe.checkpoint_path).expanduser().resolve()
    if configured_checkpoint != bound_checkpoint:
        raise ReconstructionValidationError(
            "I4.3G3B G3A-bound VDA checkpoint does not match the current hardware-resolved production profile: "
            f"bound={bound_checkpoint} resolved={configured_checkpoint}. Re-run the G3A binding with the production-native checkpoint; do not silently switch checkpoints in G3B."
        )
    config = dict(config)
    config["source_dir"] = vda_probe.source_directory
    config["checkpoint"] = vda_probe.checkpoint_path
    device = str(config.get("device", resolved.device))
    precision = str(config.get("precision", resolved.precision.value)).lower()
    return config, device, precision


def _provider_evidence_common(
    *,
    scenario_id: str,
    implementation: str,
    attempt: int,
    binding,
    sfm_info: Mapping[str, Any],
    plan_path: Path,
    plan,
    descriptor,
    wall_time_s: float,
    resource_evidence: Mapping[str, Any],
    processed_view_count: int,
    aligned_view_fraction: float,
    canonical_point_count: int | None,
    artifacts: list[dict[str, Any]],
    optional_metrics: Mapping[str, Any],
    optional_reasons: Mapping[str, str],
    provider_parameters: Mapping[str, Any],
    temporal_sequence: Mapping[str, Any],
    status: str,
    execution_completed: bool = True,
    rejection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "execution_completed": bool(execution_completed),
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": float(wall_time_s),
        "peak_vram_mib": float(resource_evidence["peak_used_mib"]),
        "selected_view_count": int(plan.selected_view_count),
        "processed_view_count": int(processed_view_count),
        "aligned_view_fraction": float(aligned_view_fraction),
    }
    if canonical_point_count is not None:
        metrics["canonical_point_count"] = int(canonical_point_count)
    metrics.update(dict(optional_metrics))
    reasons = dict(optional_reasons)
    all_optional = (
        "peak_host_memory_mib",
        "sparse_correspondence_coverage",
        "alignment_median_relative_residual",
        "alignment_p95_relative_residual",
        "inter_view_consistency_ratio",
        "inter_view_median_relative_residual",
        "registered_view_surface_coverage",
        "canonical_point_count",
        "detached_component_fraction",
        "unsupported_geometry_fraction",
    )
    for metric in all_optional:
        if metric not in metrics and metric not in reasons:
            reasons[metric] = "not_available_from_frozen_temporal_provider_adapter_without_adding_a_new_quality_policy"

    manifest = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "provider_implementation": implementation,
        "comparison_mode": "production_native",
        "attempt": int(attempt),
        "status": status,
        "source_identity": {
            "manifest_path": str(binding.source_manifest_path),
            "manifest_sha256": binding.source_manifest_sha256,
            "asset_set_sha256": binding.source_file_sha256,
            "source_path": str(binding.source_path),
            "source_file_sha256": binding.source_file_sha256,
        },
        "authoritative_sfm_identity": {
            "snapshot_path": sfm_info["snapshot"]["path"],
            "snapshot_sha256": sfm_info["snapshot"]["sha256"],
            "frame_id": sfm_info["frame_id"],
        },
        "adaptive_plan_identity": {
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
            "selected_view_ids": list(plan.selected_keyframe_ids),
        },
        "provider_provenance": {
            "provider_id": descriptor.provider_id,
            "implementation": implementation,
            "backend_version": descriptor.backend_version,
            "adapter_version": descriptor.adapter_version,
            "effective_profile": "production_native_i4_3f_adaptive",
            "checkpoint_or_model_identity": dict(descriptor.metadata),
            "parameters": dict(provider_parameters),
        },
        "runtime_resources": {
            "wall_time_s": float(wall_time_s),
            "peak_vram_mib": float(resource_evidence["peak_used_mib"]),
            "peak_host_memory_mib": None,
            **dict(resource_evidence),
        },
        "metrics": metrics,
        "optional_metric_reasons": reasons,
        "artifacts": artifacts,
        "temporal_sequence": {
            **dict(temporal_sequence),
            "source_order_preserved": True,
            "provider_specific_temporal_semantics_preserved": True,
            "matched_view_contiguous_subsequence_used": False,
            "matched_view_diagnostic_deferred_to": "I4.3G3C",
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
    }
    if rejection is not None:
        manifest["rejection"] = dict(rejection)
    validate_temporal_provider_run_evidence(manifest)
    return manifest


def _execute_rolling_depth(
    *, binding, sfm_result, frame_id: str, sfm_info: Mapping[str, Any], resources,
    rolling_probe, output_dir: Path, attempt: int,
) -> dict[str, Any]:
    descriptor = rolling_depth_dense_provider_descriptor(rolling_probe)
    prep = prepare_adaptive_provider_execution(
        sfm_result,
        provider=descriptor,
        reconstruction_frame_id=frame_id,
        resources=resources,
    )
    validate_temporal_production_plan(prep.plan.as_dict(), implementation="rolling_depth")
    plan_path = _write_json(output_dir / "adaptive_plans/rolling_depth.json", prep.plan.as_dict())

    before = build_temporal_video_identity(binding.source_path)
    if before.source_file_sha256 != binding.source_file_sha256:
        raise ReconstructionValidationError("I4.3G3B source changed immediately before RollingDepth execution.")

    provider_out = output_dir / "providers/rolling_depth"
    context = RollingDepthProviderRunContext(
        sfm_result=prep.provider_sfm_result,
        reconstruction_frame_id=frame_id,
        output_dir=provider_out,
        rolling_depth_python=Path(rolling_probe.python_executable),
        rolling_depth_repository=Path(rolling_probe.repository_root),
        worker_script=ROOT / "scripts/reconstruction_rolling_depth_provider_worker.py",
        device="cuda:0",
        precision="fp16",
    )
    adapter = RollingDepthProductionProviderAdapter(context=context, environment=rolling_probe)
    local_result, wall, monitor = _timed_provider_run(lambda: adapter.execute(prep.selection_request, prep.selection_decision))
    result = finalize_adaptive_provider_execution_result(local_result, prep)
    if result.status not in {
        DenseProviderRunStatus.COMPLETED,
        DenseProviderRunStatus.DEGRADED,
        DenseProviderRunStatus.FAILED,
    }:
        raise ReconstructionValidationError(f"I4.3G3B RollingDepth execution failed: {result.failure_reason}")

    summary_path = provider_out / "i4_3d2_rolling_depth_execution_summary.json"
    summary = read_json_object(summary_path, label="I4.3G3B RollingDepth execution summary")
    validate_temporal_provider_summary_authority(summary, implementation="rolling_depth")
    alignment = summary.get("alignment")
    fusion = summary.get("fusion")
    temporal = summary.get("temporal_sequence")
    if not isinstance(alignment, Mapping) or not isinstance(fusion, Mapping) or not isinstance(temporal, Mapping):
        raise ReconstructionValidationError("I4.3G3B RollingDepth summary evidence is incomplete.")

    rejected = result.status is DenseProviderRunStatus.FAILED
    if rejected:
        if alignment.get("accepted") is not False:
            raise ReconstructionValidationError("I4.3G3B rejected RollingDepth result must persist rejected alignment evidence.")
        if not str(result.failure_reason).startswith("global_sequence_alignment_rejected:"):
            raise ReconstructionValidationError(
                f"I4.3G3B RollingDepth hard execution failure is not a quality-terminal outcome: {result.failure_reason}"
            )
    elif alignment.get("accepted") is not True:
        raise ReconstructionValidationError("I4.3G3B successful RollingDepth result requires accepted global sequence alignment.")

    selected = prep.plan.selected_view_count
    accepted_views = int(fusion.get("accepted_views", 0)) if not rejected else 0
    point_cloud = fusion.get("point_cloud", {})
    point_count = int(point_cloud.get("point_count", 0)) if (not rejected and isinstance(point_cloud, Mapping)) else 0
    initial_corr = int(alignment.get("initial_correspondences", 0))
    retained_corr = int(alignment.get("retained_correspondences", 0))
    optional: dict[str, Any] = {
        "registered_view_surface_coverage": accepted_views / float(selected),
    }
    if initial_corr > 0:
        optional["sparse_correspondence_coverage"] = min(1.0, retained_corr / float(initial_corr))
    if alignment.get("median_relative_residual") is not None:
        optional["alignment_median_relative_residual"] = float(alignment["median_relative_residual"])
    evidence = _provider_evidence_common(
        scenario_id=binding.scenario_id,
        implementation="rolling_depth",
        attempt=attempt,
        binding=binding,
        sfm_info=sfm_info,
        plan_path=plan_path,
        plan=prep.plan,
        descriptor=descriptor,
        wall_time_s=wall,
        resource_evidence=monitor,
        processed_view_count=selected,
        aligned_view_fraction=accepted_views / float(selected),
        canonical_point_count=None if rejected else point_count,
        artifacts=_hash_artifacts(
            [summary_path, *[Path(value) for value in result.native_artifact_refs]]
            + ([] if rejected else [Path(result.canonical_geometry_ref)])
        ),
        optional_metrics=optional,
        optional_reasons={
            "peak_host_memory_mib": "not_measured_by_frozen_RollingDepth_worker_or_G3B_device_sampler",
            "alignment_p95_relative_residual": "frozen_RollingDepth_adapter_persists_sequence_p90_not_p95_no_posthoc_interpolation",
            "inter_view_consistency_ratio": "not_available_without_adding_a_new_G3C_temporal_consistency_diagnostic",
            "inter_view_median_relative_residual": "not_available_without_adding_a_new_G3C_temporal_consistency_diagnostic",
        },
        provider_parameters={
            "config": summary.get("config", {}),
            "plan_reason_codes": list(prep.plan.reason_codes),
            "sequence_alignment_scope": alignment.get("alignment_scope"),
            "per_frame_alignment_performed": False,
        },
        temporal_sequence={
            "ordered_keyframe_ids": temporal.get("ordered_keyframe_ids", []),
            "ordered_source_sequence_indices": temporal.get("ordered_source_sequence_indices", []),
            "ordered_timestamps_ns": temporal.get("ordered_timestamps_ns", []),
            "native_temporal_mode": "multi_dilation_snippet_coalignment",
            "sequence_level_alignment": "global_affine_to_authoritative_sfm",
            "per_frame_alignment_performed": False,
        },
        status=(
            "rejected"
            if rejected
            else ("degraded" if result.status is DenseProviderRunStatus.DEGRADED else "pass")
        ),
        execution_completed=not rejected,
        rejection=(
            {
                "terminal": True,
                "reason": result.failure_reason,
                "quality_gate": alignment.get("message"),
                "canonical_geometry_generated": False,
                "provider_execution_reran": False,
                "scenario_execution_continues_to_other_provider": True,
            }
            if rejected
            else None
        ),
    )
    evidence_path = _write_json(output_dir / "evidence/rolling_depth_production_native.json", evidence)

    after = build_temporal_video_identity(binding.source_path)
    if after.source_file_sha256 != binding.source_file_sha256:
        raise ReconstructionValidationError("I4.3G3B source changed during RollingDepth execution.")
    return {"evidence": _artifact(evidence_path), "result": result.as_dict(), "resource_evidence": monitor}


def _execute_vda(
    *, binding, sfm_result, frame_id: str, sfm_info: Mapping[str, Any], resources,
    vda_probe, output_dir: Path, attempt: int,
) -> dict[str, Any]:
    descriptor = video_depth_anything_dense_provider_descriptor(vda_probe)
    prep = prepare_adaptive_provider_execution(
        sfm_result,
        provider=descriptor,
        reconstruction_frame_id=frame_id,
        resources=resources,
    )
    validate_temporal_production_plan(prep.plan.as_dict(), implementation="video_depth_anything")
    plan_path = _write_json(output_dir / "adaptive_plans/video_depth_anything.json", prep.plan.as_dict())

    before = build_temporal_video_identity(binding.source_path)
    if before.source_file_sha256 != binding.source_file_sha256:
        raise ReconstructionValidationError("I4.3G3B source changed immediately before VDA execution.")

    runtime_config, device, precision = _vda_runtime_config(vda_probe)
    provider_out = output_dir / "providers/video_depth_anything"
    context = VideoDepthAnythingProviderRunContext(
        sfm_result=prep.provider_sfm_result,
        reconstruction_frame_id=frame_id,
        output_dir=provider_out,
        python_executable=Path(vda_probe.python_executable),
        worker_script=ROOT / "scripts/reconstruction_video_depth_anything_provider_worker.py",
        runtime_config=runtime_config,
        device=device,
        precision=precision,
    )
    adapter = VideoDepthAnythingProductionProviderAdapter(context=context, environment=vda_probe)
    local_result, wall, monitor = _timed_provider_run(lambda: adapter.execute(prep.selection_request, prep.selection_decision))
    result = finalize_adaptive_provider_execution_result(local_result, prep)
    if result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
        raise ReconstructionValidationError(f"I4.3G3B VDA execution failed: {result.failure_reason}")

    summary_path = provider_out / "i4_3d1_vda_execution_summary.json"
    summary = read_json_object(summary_path, label="I4.3G3B VDA execution summary")
    validate_temporal_provider_summary_authority(summary, implementation="video_depth_anything")
    alignment = summary.get("alignment")
    fusion = summary.get("fusion")
    temporal = summary.get("temporal_sequence")
    if not isinstance(alignment, Mapping) or not isinstance(fusion, Mapping) or not isinstance(temporal, Mapping):
        raise ReconstructionValidationError("I4.3G3B VDA summary evidence is incomplete.")
    records = alignment.get("records", [])
    if not isinstance(records, list):
        raise ReconstructionValidationError("I4.3G3B VDA alignment records are malformed.")
    accepted_records = [item for item in records if isinstance(item, Mapping) and item.get("status") == "accepted"]
    selected = prep.plan.selected_view_count
    accepted = int(alignment.get("accepted_views", 0))
    candidates = sum(int(item.get("candidate_observations", 0)) for item in records if isinstance(item, Mapping))
    valid_corr = sum(int(item.get("valid_correspondences", 0)) for item in records if isinstance(item, Mapping))
    medians = [float(item["median_relative_residual"]) for item in accepted_records if item.get("median_relative_residual") is not None]
    point_cloud = fusion.get("point_cloud", {})
    point_count = int(point_cloud.get("point_count", 0)) if isinstance(point_cloud, Mapping) else 0
    optional: dict[str, Any] = {
        "registered_view_surface_coverage": accepted / float(selected),
    }
    if candidates > 0:
        optional["sparse_correspondence_coverage"] = min(1.0, valid_corr / float(candidates))
    if medians:
        optional["alignment_median_relative_residual"] = float(median(medians))

    evidence = _provider_evidence_common(
        scenario_id=binding.scenario_id,
        implementation="video_depth_anything",
        attempt=attempt,
        binding=binding,
        sfm_info=sfm_info,
        plan_path=plan_path,
        plan=prep.plan,
        descriptor=descriptor,
        wall_time_s=wall,
        resource_evidence=monitor,
        processed_view_count=selected,
        aligned_view_fraction=accepted / float(selected),
        canonical_point_count=point_count,
        artifacts=_hash_artifacts([
            summary_path,
            *[Path(value) for value in result.native_artifact_refs],
            Path(result.canonical_geometry_ref),
        ]),
        optional_metrics=optional,
        optional_reasons={
            "peak_host_memory_mib": "not_measured_by_frozen_VDA_worker_or_G3B_device_sampler",
            "alignment_p95_relative_residual": "frozen_VDA_adapter_persists_per_view_p90_not_p95_no_posthoc_interpolation",
            "inter_view_consistency_ratio": "not_available_without_adding_a_new_G3C_temporal_consistency_diagnostic",
            "inter_view_median_relative_residual": "not_available_without_adding_a_new_G3C_temporal_consistency_diagnostic",
        },
        provider_parameters={
            "runtime_config": summary.get("runtime_config", {}),
            "config": summary.get("config", {}),
            "plan_reason_codes": list(prep.plan.reason_codes),
        },
        temporal_sequence={
            "ordered_keyframe_ids": temporal.get("ordered_keyframe_ids", []),
            "ordered_source_sequence_indices": temporal.get("ordered_source_sequence_indices", []),
            "ordered_timestamps_ns": temporal.get("ordered_timestamps_ns", []),
            "native_temporal_mode": "official_streaming_cache",
            "stateful_sequence_required": temporal.get("stateful_sequence_required") is True,
            "model_metric_depth_promoted_to_canonical_metric_scale": False,
        },
        status="degraded" if result.status is DenseProviderRunStatus.DEGRADED else "pass",
    )
    evidence_path = _write_json(output_dir / "evidence/video_depth_anything_production_native.json", evidence)

    after = build_temporal_video_identity(binding.source_path)
    if after.source_file_sha256 != binding.source_file_sha256:
        raise ReconstructionValidationError("I4.3G3B source changed during VDA execution.")
    return {"evidence": _artifact(evidence_path), "result": result.as_dict(), "resource_evidence": monitor}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "production-native"), default="preflight")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--i4-3g3a-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g3a-sha256", required=True)
    parser.add_argument("--sfm-python", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be >= 1")
    if args.sfm_timeout_minutes <= 0.0:
        parser.error("--sfm-timeout-minutes must be positive")
    return args


def main() -> int:
    args = _parse_args()
    out = args.output_dir.expanduser().resolve()
    try:
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"I4.3G3B output directory must be empty or absent: {out}")
        out.mkdir(parents=True, exist_ok=True)

        g3a_path = args.i4_3g3a_snapshot.expanduser().resolve()
        g3a_sha = verify_snapshot_sha256(g3a_path, args.i4_3g3a_sha256, label="I4.3G3A snapshot")
        g3a = read_json_object(g3a_path, label="I4.3G3A snapshot")
        predecessor_chain = _verify_g3a_predecessor_chain(g3a)
        binding = resolve_g3b_scenario_binding(
            i4_3g3a_snapshot=g3a,
            scenario_id=args.scenario,
            verify_files=True,
        )

        sfm_probe = probe_pycolmap_environment(str(args.sfm_python.expanduser().resolve()))
        vda_probe, rolling_probe = _current_temporal_probes(g3a)
        preflight = build_g3b_runtime_preflight_snapshot(
            i4_3g3a_snapshot=g3a,
            scenario_id=binding.scenario_id,
            sfm_runtime=sfm_probe.as_dict(),
            video_depth_anything_runtime=vda_probe.as_dict(),
            rolling_depth_runtime=rolling_probe.as_dict(),
            verify_files=True,
        )
        preflight["predecessors"] = {
            "i4_3g3a": {"path": str(g3a_path), "sha256": g3a_sha},
            **predecessor_chain,
        }
        preflight_path = _write_json(out / "i4_3g3b_runtime_preflight_snapshot.json", preflight)
        print(f"I4.3G3B runtime/source preflight: {'PASS' if preflight['pass'] else 'FAIL'}")
        print(
            f"scenario={binding.scenario_id} source_sha256={binding.source_file_sha256} "
            f"reported_frames={binding.reported_frame_count} fps={binding.expected_frame_rate_hz:.6g}"
        )
        print(
            "sfm_ready={sfm} rolling_ready={rolling} vda_ready={vda}".format(
                sfm=preflight["runtime"]["sfm_ready"],
                rolling=preflight["runtime"]["rolling_depth_ready_and_bound"],
                vda=preflight["runtime"]["video_depth_anything_ready_and_bound"],
            )
        )
        print("authoritative_sfm_started=False rolling_depth_started=False video_depth_anything_started=False matched_view_started=False")
        print(f"preflight_snapshot={preflight_path}")
        if not preflight["pass"]:
            return 3
        if args.mode == "preflight":
            return 0

        source_before = build_temporal_video_identity(binding.source_path)
        if source_before.source_file_sha256 != binding.source_file_sha256:
            raise ReconstructionValidationError("I4.3G3B source changed after preflight and before authoritative SfM.")

        resources = _adaptive_resources()
        sfm_result, frame_id, sfm_info = _build_authoritative_sfm(
            binding=binding,
            sfm_python=args.sfm_python,
            output_dir=out,
            resources=resources,
            timeout_minutes=float(args.sfm_timeout_minutes),
        )

        rolling = _execute_rolling_depth(
            binding=binding,
            sfm_result=sfm_result,
            frame_id=frame_id,
            sfm_info=sfm_info,
            resources=resources,
            rolling_probe=rolling_probe,
            output_dir=out,
            attempt=int(args.attempt),
        )
        vda = _execute_vda(
            binding=binding,
            sfm_result=sfm_result,
            frame_id=frame_id,
            sfm_info=sfm_info,
            resources=resources,
            vda_probe=vda_probe,
            output_dir=out,
            attempt=int(args.attempt),
        )

        source_after = build_temporal_video_identity(binding.source_path)
        if source_after.source_file_sha256 != binding.source_file_sha256:
            raise ReconstructionValidationError("I4.3G3B source identity changed during production-native execution.")

        rolling_evidence_payload = read_json_object(
            Path(rolling["evidence"]["path"]), label="I4.3G3B RollingDepth provider evidence"
        )
        vda_evidence_payload = read_json_object(
            Path(vda["evidence"]["path"]), label="I4.3G3B VDA provider evidence"
        )
        provider_quality_rejection_present = any(
            payload.get("status") == "rejected"
            for payload in (rolling_evidence_payload, vda_evidence_payload)
        )
        closure = {
            "schema_version": 1,
            "phase": "I4.3G3B",
            "status": (
                "production_native_complete_with_provider_rejection"
                if provider_quality_rejection_present
                else "production_native_complete"
            ),
            "pass": True,
            "scenario": binding.as_dict(),
            "preflight": _artifact(preflight_path),
            "authoritative_sfm": dict(sfm_info),
            "providers": {
                "rolling_depth": rolling,
                "video_depth_anything": vda,
                "same_authoritative_sfm_shared": True,
                "source_identity_preserved_after_both_providers": True,
            },
            "manual_review": {
                "required": True,
                "matched_view_diagnostic_started": False,
                "verdict_recorded": False,
                "deferred_to": "I4.3G3C",
            },
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "same_authoritative_sfm_shared_by_both_providers": True,
                "authoritative_sfm_modified": False,
                "provider_selected": False,
                "provider_promoted": False,
                "auto_policy_implemented": False,
                "hybrid_authorized": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "permanent_global_view_cap": None,
            },
            "decision": {
                "production_native_complete": True,
                "provider_terminal_outcomes_complete": True,
                "provider_quality_rejection_present": provider_quality_rejection_present,
                "matched_view_diagnostic_authorized_for_g3c": True,
                "scenario_verdict_pending_g3c_manual_review": True,
                "provider_selected": False,
                "provider_promoted": False,
                "universal_provider_winner_assumed": False,
                "global_scalar_score_forbidden": True,
            },
        }
        closure_path = _write_json(out / "i4_3g3b_production_native_snapshot.json", closure)
        print(
            "I4.3G3B production-native execution: PASS"
            + (" (provider quality rejection recorded)" if provider_quality_rejection_present else "")
        )
        print(
            f"registered_views={sfm_info['registered_view_count']} landmarks={sfm_info['landmark_count']} "
            "same_authoritative_sfm_shared=True"
        )
        print(
            "rolling_depth_status={rolling} video_depth_anything_status={vda}".format(
                rolling=rolling_evidence_payload.get("status"),
                vda=vda_evidence_payload.get("status"),
            )
        )
        print("provider_selected=False provider_promoted=False auto=False hybrid=False metric_scale=False metrology=False")
        print(f"production_snapshot={closure_path}")
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "phase": "I4.3G3B",
            "status": "failed",
            "pass": False,
            "scenario_id": str(getattr(args, "scenario", "")),
            "attempt": int(getattr(args, "attempt", 1)),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
                "provider_selected": False,
                "provider_promoted": False,
                "auto_policy_implemented": False,
                "hybrid_authorized": False,
                "metric_scale_claimed": False,
                "measurement_authoritative": False,
                "certified_metrology": False,
                "surface_mesh_generated": False,
                "permanent_global_view_cap": None,
            },
        }
        try:
            _write_json(out / "i4_3g3b_failure.json", failure)
        except Exception:
            pass
        print(f"I4.3G3B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
