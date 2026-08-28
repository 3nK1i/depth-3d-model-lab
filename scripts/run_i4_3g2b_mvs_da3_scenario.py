#!/usr/bin/env python3
"""I4.3G2B production-native MVS-vs-DA3 scenario runner.

The runner has two modes:

* preflight: verify frozen G1/G2A evidence, exact source identity, and isolated
  MVS/DA3 runtime readiness. No SfM or provider process is started.
* production-native: build one authoritative PyCOLMAP SfM, freeze it, derive
  independent I4.3F adaptive provider plans from that same SfM, then execute
  MVS and DA3 sequentially without changing authority or source identity.

House is historical_read_only and is deliberately refused by this runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
from statistics import median
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware.gpu_info import detect_hardware
from core.reconstruction import (
    BackendOperationContext,
    ExternalPycolmapSfmBackend,
    FrameQualityPolicy,
    FrameSamplingMode,
    FrameSamplingPolicy,
    ImageSequenceReader,
    IngestionCachePolicy,
    KeyframeSelectionPolicy,
    OfflineSfmConfiguration,
    OfflineSfmRequest,
    ReconstructionEventRecorder,
    ReconstructionIngestionPipeline,
    ReconstructionOperationControl,
    ReconstructionQualityProfile,
    ReconstructionStage,
    SfmMatchingStrategy,
    SfmResumePolicy,
    materialize_sfm_workspace,
)
from core.reconstruction.adaptive_production_planning import (
    build_adaptive_pre_sfm_frame_plan,
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.da3 import probe_da3_environment
from core.reconstruction.da3_dense_provider import (
    DA3ProductionProviderAdapter,
    DA3ProviderRunContext,
    da3_dense_provider_descriptor,
)
from core.reconstruction.da3_provider import DA3ProviderProfile
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_da3_production_comparison import (
    build_g2b_runtime_preflight_snapshot,
    read_json_object,
    resolve_g2b_scenario_binding,
    sha256_file,
    validate_provider_run_evidence,
    verify_snapshot_sha256,
)
from core.reconstruction.mvs_environment import probe_pycolmap_mvs_environment
from core.reconstruction.mvs_provider import (
    MvsProductionProviderAdapter,
    MvsProviderRunContext,
    colmap_mvs_provider_descriptor,
)
from core.reconstruction.mvs_da3_qualification import build_source_identity

_MIB = 1024**2
_G2A_EXPECTED_DEFAULT_SHA = "4d639119b903ec57481bbad8ae5144fce1a07cf977527f64328632947e24c7ec"
_G1_EXPECTED_DEFAULT_SHA = "d5a169366aaa328604833686d1d65cf648c2ed88fed7ceceb5834568e94e456f"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _artifact(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise ReconstructionValidationError(f"G2B expected artifact is missing: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "size_bytes": int(source.stat().st_size)}


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


def _pre_sfm_matching(scenario_id: str) -> SfmMatchingStrategy:
    # Complete-room is an ordered long sweep; bounded exhaustive matching on the
    # three smaller scenarios avoids making the matching heuristic itself the
    # failure mode. This policy is scenario-level and shared by both providers.
    if scenario_id == "g2_complete_room_mvs_da3":
        return SfmMatchingStrategy.SEQUENTIAL
    return SfmMatchingStrategy.EXHAUSTIVE


def _source_frame_rate(scenario_id: str) -> float:
    # Complete-room and low-texture sources were frozen from deterministic 1 FPS
    # video extraction. Other photo sets have no physical temporal cadence; 10 Hz
    # is metadata only and never becomes geometry authority.
    if scenario_id in {"g2_complete_room_mvs_da3", "g2_low_texture_mvs_da3"}:
        return 1.0
    return 10.0


def _build_authoritative_sfm(
    *,
    scenario_id: str,
    source_dir: Path,
    sfm_python: Path,
    output_dir: Path,
    resources: AdaptiveSelectionResourceSnapshot,
    timeout_minutes: float,
):
    source_fps = _source_frame_rate(scenario_id)
    source = ImageSequenceReader.from_directory(source_dir, recursive=False, frame_rate_hz=source_fps)
    pre_plan = build_adaptive_pre_sfm_frame_plan(
        source_kind="image_directory",
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
            target_fps_hz=source_fps,
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
        raise ReconstructionValidationError("G2B authoritative SfM requires at least three ingested keyframes.")

    sfm_root = output_dir / "authoritative_sfm"
    workspace = materialize_sfm_workspace(ingestion, sfm_root / "workspace", overwrite=True)
    matching = _pre_sfm_matching(scenario_id)
    request = OfflineSfmRequest(
        session_id=session_id,
        profile=ReconstructionQualityProfile.BALANCED,
        workspace=workspace,
        configuration=OfflineSfmConfiguration(
            matching_strategy=matching,
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
        raise ReconstructionValidationError("G2B authoritative SfM did not produce a usable connected sparse reconstruction.")

    pre_path = _write_json(sfm_root / "adaptive_pre_sfm_plan.json", pre_plan.as_dict())
    ingestion_path = _write_json(sfm_root / "ingestion_result.json", ingestion.as_dict())
    sfm_path = _write_json(sfm_root / "sfm_result.json", result.as_dict())
    frame_id = f"g2b:{scenario_id}:pycolmap_sfm:{session_id}"
    return result, frame_id, {
        "session_id": session_id,
        "wall_time_s": float(elapsed),
        "matching_strategy": matching.value,
        "input_view_count": len(workspace.images),
        "registered_view_count": result.statistics.registered_keyframes,
        "registration_fraction": result.statistics.registered_keyframes / float(len(workspace.images)),
        "landmark_count": result.statistics.landmarks,
        "observation_count": result.statistics.observations,
        "mean_reprojection_error_px": result.statistics.mean_reprojection_error_px,
        "pre_sfm_plan": _artifact(pre_path),
        "ingestion": _artifact(ingestion_path),
        "snapshot": _artifact(sfm_path),
        "frame_id": frame_id,
    }


def _weak_overlap_connectivity(sfm_result) -> dict[str, Any]:
    registered = tuple(camera.keyframe_id for camera in sfm_result.cameras)
    if not registered:
        raise ReconstructionValidationError("G2B weak-overlap SfM has no registered cameras.")
    index = {key: i for i, key in enumerate(registered)}
    adjacency = {key: set() for key in registered}
    shared_counts: dict[tuple[str, str], int] = {}
    for landmark in sfm_result.landmarks:
        keys = sorted({obs.keyframe_id for obs in landmark.observations if obs.keyframe_id in index}, key=index.__getitem__)
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                pair = (left, right)
                shared_counts[pair] = shared_counts.get(pair, 0) + 1
                adjacency[left].add(right)
                adjacency[right].add(left)
    unseen = set(registered)
    components: list[set[str]] = []
    while unseen:
        root = next(iter(unseen))
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(adjacency[node] - component)
        components.append(component)
    largest = max(len(item) for item in components)
    consecutive = []
    for left, right in zip(registered, registered[1:]):
        pair = (left, right) if (left, right) in shared_counts else (right, left)
        consecutive.append(shared_counts.get(pair, 0))
    connected = len(components) == 1 and len(registered) >= 3
    return {
        "registered_views": len(registered),
        "co_visible_pair_count": len(shared_counts),
        "connected_component_count": len(components),
        "largest_component_fraction": largest / float(len(registered)),
        "minimum_consecutive_shared_landmarks": min(consecutive) if consecutive else 0,
        "median_consecutive_shared_landmarks": float(median(consecutive)) if consecutive else 0.0,
        "authoritative_sfm_connectivity_qualified": connected,
    }


def _max_mvs_gpu_mib(worker: Mapping[str, Any]) -> float:
    snapshots = worker.get("resource_snapshots", {})
    values: list[float] = []
    if isinstance(snapshots, Mapping):
        for snapshot in snapshots.values():
            if not isinstance(snapshot, Mapping):
                continue
            gpus = snapshot.get("gpus", [])
            if isinstance(gpus, list):
                for item in gpus:
                    if isinstance(item, Mapping):
                        values.append(max(0.0, float(item.get("used_vram_gib", 0.0))) * 1024.0)
    if not values:
        raise ReconstructionValidationError("G2B MVS run lacks measured VRAM evidence.")
    return max(values)


def _hash_provider_artifacts(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        if path.is_file():
            records.append(_artifact(path))
    if not records:
        raise ReconstructionValidationError("G2B provider run produced no hashable artifact evidence.")
    return records


def _provider_evidence_common(
    *, scenario_id: str, implementation: str, attempt: int, source_manifest: Path,
    source_asset_sha: str, sfm_info: Mapping[str, Any], plan_path: Path, plan, descriptor,
    wall_time_s: float, peak_vram_mib: float, peak_host_memory_mib: float | None,
    processed_view_count: int, aligned_view_fraction: float, canonical_point_count: int | None,
    artifacts: list[dict[str, Any]], optional_metrics: Mapping[str, Any], optional_reasons: Mapping[str, str],
    status: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "execution_completed": True,
        "source_input_identity_preserved": True,
        "authoritative_sfm_preserved": True,
        "adaptive_plan_bound": True,
        "provider_provenance_complete": True,
        "forbidden_authority_claims_absent": True,
        "wall_time_s": float(wall_time_s),
        "peak_vram_mib": float(peak_vram_mib),
        "selected_view_count": int(plan.selected_view_count),
        "processed_view_count": int(processed_view_count),
        "aligned_view_fraction": float(aligned_view_fraction),
    }
    if peak_host_memory_mib is not None:
        metrics["peak_host_memory_mib"] = float(peak_host_memory_mib)
    if canonical_point_count is not None:
        metrics["canonical_point_count"] = int(canonical_point_count)
    metrics.update(dict(optional_metrics))
    reasons = dict(optional_reasons)
    all_optional = (
        "peak_host_memory_mib", "sparse_correspondence_coverage", "alignment_median_relative_residual",
        "alignment_p95_relative_residual", "inter_view_consistency_ratio", "inter_view_median_relative_residual",
        "registered_view_surface_coverage", "canonical_point_count", "detached_component_fraction",
        "unsupported_geometry_fraction",
    )
    for metric in all_optional:
        if metric not in metrics and metric not in reasons:
            reasons[metric] = "not_available_from_frozen_provider_adapter_without_adding_a_new_quality_policy"
    manifest = {
        "schema_version": 1,
        "scenario_id": scenario_id,
        "provider_implementation": implementation,
        "comparison_mode": "production_native",
        "attempt": int(attempt),
        "status": status,
        "source_identity": {
            "manifest_path": str(source_manifest.resolve()),
            "manifest_sha256": sha256_file(source_manifest),
            "asset_set_sha256": source_asset_sha,
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
            "effective_profile": "production_native_adaptive",
            "checkpoint_or_model_identity": dict(descriptor.metadata),
            "parameters": {"plan_reason_codes": list(plan.reason_codes)},
        },
        "runtime_resources": {
            "wall_time_s": float(wall_time_s),
            "peak_vram_mib": float(peak_vram_mib),
            "peak_host_memory_mib": peak_host_memory_mib,
            "measurement_scope": (
                "provider_worker_process_peak_cuda_allocated" if implementation == "da3"
                else "device_wide_nvidia_smi_stage_snapshots_best_available"
            ),
        },
        "metrics": metrics,
        "optional_metric_reasons": reasons,
        "artifacts": artifacts,
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
    validate_provider_run_evidence(manifest)
    return manifest


def _execute_providers(
    *, scenario_id: str, binding, sfm_result, frame_id: str, sfm_info: Mapping[str, Any],
    resources: AdaptiveSelectionResourceSnapshot, mvs_probe, da3_probe, sfm_python: Path,
    da3_python: Path, da3_repo: Path, output_dir: Path, source_manifest: Path, attempt: int,
) -> dict[str, Any]:
    mvs_desc = colmap_mvs_provider_descriptor(mvs_probe)
    da3_desc = da3_dense_provider_descriptor(da3_probe)
    mvs_prep = prepare_adaptive_provider_execution(
        sfm_result, provider=mvs_desc, reconstruction_frame_id=frame_id, resources=resources
    )
    da3_prep = prepare_adaptive_provider_execution(
        sfm_result, provider=da3_desc, reconstruction_frame_id=frame_id, resources=resources
    )
    plans = output_dir / "adaptive_plans"
    mvs_plan_path = _write_json(plans / "colmap_mvs.json", mvs_prep.plan.as_dict())
    da3_plan_path = _write_json(plans / "da3.json", da3_prep.plan.as_dict())

    source_before = build_source_identity(binding.source_directory, minimum_images=3)
    if source_before.asset_set_sha256 != binding.source_asset_set_sha256:
        raise ReconstructionValidationError("G2B source changed immediately before provider execution.")

    mvs_output = output_dir / "providers" / "colmap_mvs"
    mvs_context = MvsProviderRunContext(
        sfm_result=mvs_prep.provider_sfm_result,
        reconstruction_frame_id=frame_id,
        output_dir=mvs_output,
        pycolmap_python=sfm_python,
        worker_script=ROOT / "scripts/reconstruction_pycolmap_dense_provider_mvs_worker.py",
    )
    mvs_adapter = MvsProductionProviderAdapter(mvs_context, environment_probe=mvs_probe)
    mvs_started = time.perf_counter()
    mvs_local_result = mvs_adapter.execute(mvs_prep.selection_request, mvs_prep.selection_decision)
    mvs_wall = time.perf_counter() - mvs_started
    mvs_result = finalize_adaptive_provider_execution_result(mvs_local_result, mvs_prep)
    if mvs_result.status is not DenseProviderRunStatus.COMPLETED:
        raise ReconstructionValidationError(f"G2B MVS production-native execution failed: {mvs_result.failure_reason}")
    mvs_summary_path = mvs_output / "i4_3b_summary.json"
    mvs_summary = read_json_object(mvs_summary_path, label="G2B MVS summary")
    worker = mvs_summary.get("worker", {})
    if not isinstance(worker, Mapping):
        raise ReconstructionValidationError("G2B MVS worker evidence is missing.")
    mvs_peak = _max_mvs_gpu_mib(worker)
    mvs_host = float(worker.get("process_peak_rss_gib", 0.0)) * 1024.0 if worker.get("process_peak_rss_gib") is not None else None
    fusion = worker.get("fusion", {})
    if not isinstance(fusion, Mapping):
        raise ReconstructionValidationError("G2B MVS fusion evidence is missing.")
    mvs_points = int(fusion.get("ply_vertex_count", 0))
    mvs_evidence = _provider_evidence_common(
        scenario_id=scenario_id, implementation="colmap_mvs", attempt=attempt,
        source_manifest=source_manifest, source_asset_sha=binding.source_asset_set_sha256,
        sfm_info=sfm_info, plan_path=mvs_plan_path, plan=mvs_prep.plan, descriptor=mvs_desc,
        wall_time_s=mvs_wall, peak_vram_mib=mvs_peak, peak_host_memory_mib=mvs_host,
        processed_view_count=mvs_prep.plan.selected_view_count, aligned_view_fraction=1.0,
        canonical_point_count=mvs_points,
        artifacts=_hash_provider_artifacts([
            mvs_summary_path,
            Path(str(fusion.get("raw_fused_ply", ""))),
            Path(str(fusion.get("native_visibility", ""))),
        ]),
        optional_metrics={"registered_view_surface_coverage": 1.0},
        optional_reasons={
            "sparse_correspondence_coverage": "not_applicable_direct_authoritative_sfm_frame_mvs_geometry",
            "alignment_median_relative_residual": "not_applicable_direct_authoritative_sfm_frame_mvs_geometry",
            "alignment_p95_relative_residual": "not_applicable_direct_authoritative_sfm_frame_mvs_geometry",
        },
        status="pass",
    )
    mvs_ev_path = _write_json(output_dir / "evidence/colmap_mvs_production_native.json", mvs_evidence)

    # Source identity is checked again before DA3 so MVS can never alter scenario input.
    mid_source = build_source_identity(binding.source_directory, minimum_images=3)
    if mid_source.asset_set_sha256 != binding.source_asset_set_sha256:
        raise ReconstructionValidationError("G2B source changed during MVS execution.")

    da3_output = output_dir / "providers" / "da3"
    da3_profile = DA3ProviderProfile(process_res=392)
    da3_context = DA3ProviderRunContext(
        sfm_result=da3_prep.provider_sfm_result,
        reconstruction_frame_id=frame_id,
        output_dir=da3_output,
        da3_python=da3_python,
        da3_repository=da3_repo,
        worker_script=ROOT / "scripts/da3_workers/da3_worker.py",
        profile=da3_profile,
    )
    da3_adapter = DA3ProductionProviderAdapter(da3_context, environment_probe=da3_probe)
    da3_started = time.perf_counter()
    da3_local_result = da3_adapter.execute(da3_prep.selection_request, da3_prep.selection_decision)
    da3_wall = time.perf_counter() - da3_started
    da3_result = finalize_adaptive_provider_execution_result(da3_local_result, da3_prep)
    if da3_result.status not in {DenseProviderRunStatus.COMPLETED, DenseProviderRunStatus.DEGRADED}:
        raise ReconstructionValidationError(f"G2B DA3 production-native execution failed: {da3_result.failure_reason}")
    da3_summary_path = da3_output / "i4_3c_summary.json"
    da3_summary = read_json_object(da3_summary_path, label="G2B DA3 summary")
    da3_manifest = da3_summary.get("da3_manifest", {})
    alignment = da3_summary.get("alignment", {})
    da3_fusion = da3_summary.get("fusion", {})
    if not all(isinstance(v, Mapping) for v in (da3_manifest, alignment, da3_fusion)):
        raise ReconstructionValidationError("G2B DA3 summary evidence is incomplete.")
    selected = da3_prep.plan.selected_view_count
    accepted = int(alignment.get("accepted_views", 0))
    records = alignment.get("records", [])
    accepted_records = [r for r in records if isinstance(r, Mapping) and r.get("status") == "accepted"] if isinstance(records, list) else []
    medians = [float(r["median_relative_residual"]) for r in accepted_records if r.get("median_relative_residual") is not None]
    candidates = sum(int(r.get("candidate_observations", 0)) for r in records if isinstance(r, Mapping)) if isinstance(records, list) else 0
    valid_corr = sum(int(r.get("valid_correspondences", 0)) for r in records if isinstance(r, Mapping)) if isinstance(records, list) else 0
    point_cloud = da3_fusion.get("point_cloud", {})
    da3_points = int(point_cloud.get("point_count", 0)) if isinstance(point_cloud, Mapping) else 0
    da3_optional: dict[str, Any] = {
        "registered_view_surface_coverage": accepted / float(selected),
    }
    if candidates > 0:
        da3_optional["sparse_correspondence_coverage"] = min(1.0, valid_corr / float(candidates))
    if medians:
        da3_optional["alignment_median_relative_residual"] = float(median(medians))
    da3_evidence = _provider_evidence_common(
        scenario_id=scenario_id, implementation="da3", attempt=attempt,
        source_manifest=source_manifest, source_asset_sha=binding.source_asset_set_sha256,
        sfm_info=sfm_info, plan_path=da3_plan_path, plan=da3_prep.plan, descriptor=da3_desc,
        wall_time_s=da3_wall, peak_vram_mib=float(da3_manifest.get("peak_vram_mib", 0.0)),
        peak_host_memory_mib=None, processed_view_count=selected,
        aligned_view_fraction=accepted / float(selected), canonical_point_count=da3_points,
        artifacts=_hash_provider_artifacts([
            da3_summary_path,
            Path(str(da3_manifest.get("output_npz_path", ""))),
            Path(str(da3_fusion.get("canonical_ply", ""))),
            Path(str(da3_fusion.get("canonical_npz", ""))),
        ]),
        optional_metrics=da3_optional,
        optional_reasons={
            "alignment_p95_relative_residual": "frozen_DA3_adapter_persists_per_view_p90_not_p95_no_posthoc_interpolation",
        },
        status="degraded" if da3_result.status is DenseProviderRunStatus.DEGRADED else "pass",
    )
    da3_ev_path = _write_json(output_dir / "evidence/da3_production_native.json", da3_evidence)

    final_source = build_source_identity(binding.source_directory, minimum_images=3)
    if final_source.asset_set_sha256 != binding.source_asset_set_sha256:
        raise ReconstructionValidationError("G2B source changed during DA3 execution.")

    return {
        "colmap_mvs": {"evidence": _artifact(mvs_ev_path), "result": mvs_result.as_dict()},
        "da3": {"evidence": _artifact(da3_ev_path), "result": da3_result.as_dict()},
        "source_identity_preserved_after_both_providers": True,
        "same_authoritative_sfm_shared": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "production-native"), default="preflight")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--i4-3g1-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g1-sha256", default=_G1_EXPECTED_DEFAULT_SHA)
    parser.add_argument("--i4-3g2a-snapshot", type=Path, required=True)
    parser.add_argument("--i4-3g2a-sha256", default=_G2A_EXPECTED_DEFAULT_SHA)
    parser.add_argument("--sfm-python", type=Path, required=True)
    parser.add_argument("--da3-python", type=Path, required=True)
    parser.add_argument("--da3-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--sfm-timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("--attempt must be >= 1")
    if args.sfm_timeout_minutes <= 0:
        parser.error("--sfm-timeout-minutes must be positive")
    return args


def main() -> int:
    args = _parse_args()
    out = args.output_dir.expanduser().resolve()
    try:
        if out.exists() and any(out.iterdir()):
            raise ReconstructionValidationError(f"G2B output directory must be empty or absent: {out}")
        out.mkdir(parents=True, exist_ok=True)
        g1_path = args.i4_3g1_snapshot.expanduser().resolve()
        g2a_path = args.i4_3g2a_snapshot.expanduser().resolve()
        g1_sha = verify_snapshot_sha256(g1_path, args.i4_3g1_sha256, label="I4.3G1 snapshot")
        g2a_sha = verify_snapshot_sha256(g2a_path, args.i4_3g2a_sha256, label="final I4.3G2A snapshot")
        g1 = read_json_object(g1_path, label="I4.3G1 snapshot")
        g2a = read_json_object(g2a_path, label="I4.3G2A snapshot")
        binding = resolve_g2b_scenario_binding(
            i4_3g1_snapshot=g1, i4_3g2a_snapshot=g2a, scenario_id=args.scenario, verify_files=True
        )
        mvs_probe = probe_pycolmap_mvs_environment(str(args.sfm_python.expanduser().resolve()))
        da3_probe = probe_da3_environment(args.da3_python.expanduser().resolve(), args.da3_repo.expanduser().resolve())
        preflight = build_g2b_runtime_preflight_snapshot(
            i4_3g1_snapshot=g1,
            i4_3g2a_snapshot=g2a,
            scenario_id=args.scenario,
            mvs_runtime=mvs_probe.as_dict(),
            da3_runtime=da3_probe.as_dict(),
            verify_files=True,
        )
        preflight["predecessors"] = {
            "i4_3g1": {"path": str(g1_path), "sha256": g1_sha},
            "i4_3g2a": {"path": str(g2a_path), "sha256": g2a_sha},
        }
        preflight_path = _write_json(out / "i4_3g2b_runtime_preflight_snapshot.json", preflight)
        print(f"I4.3G2B runtime/source preflight: {'PASS' if preflight['pass'] else 'FAIL'}")
        print(f"scenario={binding.scenario_id} source_images={binding.source_image_count} source_sha256={binding.source_asset_set_sha256}")
        print(f"mvs_ready={preflight['runtime']['mvs_ready']} da3_ready={preflight['runtime']['da3_ready']}")
        print("authoritative_sfm_started=False mvs_started=False da3_started=False")
        print(f"preflight_snapshot={preflight_path}")
        if not preflight["pass"]:
            return 3
        if args.mode == "preflight":
            return 0

        source_identity = build_source_identity(binding.source_directory, minimum_images=3)
        source_manifest = _write_json(out / "source_identity.json", source_identity.as_dict())
        resources = _adaptive_resources()
        sfm_result, frame_id, sfm_info = _build_authoritative_sfm(
            scenario_id=binding.scenario_id,
            source_dir=binding.source_directory,
            sfm_python=args.sfm_python,
            output_dir=out,
            resources=resources,
            timeout_minutes=float(args.sfm_timeout_minutes),
        )
        weak_overlap = None
        if binding.scenario_id == "g2_weak_overlap_mvs_da3":
            weak_overlap = _weak_overlap_connectivity(sfm_result)
            _write_json(out / "authoritative_sfm/weak_overlap_connectivity.json", weak_overlap)
            if weak_overlap["authoritative_sfm_connectivity_qualified"] is not True:
                raise ReconstructionValidationError(
                    "G2B weak-overlap authoritative SfM connectivity did not qualify; preserving scenario outcome without provider retuning."
                )

        provider_results = _execute_providers(
            scenario_id=binding.scenario_id,
            binding=binding,
            sfm_result=sfm_result,
            frame_id=frame_id,
            sfm_info=sfm_info,
            resources=resources,
            mvs_probe=mvs_probe,
            da3_probe=da3_probe,
            sfm_python=args.sfm_python.expanduser().resolve(),
            da3_python=args.da3_python.expanduser().resolve(),
            da3_repo=args.da3_repo.expanduser().resolve(),
            output_dir=out,
            source_manifest=source_manifest,
            attempt=int(args.attempt),
        )
        closure = {
            "schema_version": 1,
            "phase": "I4.3G2B",
            "status": "production_native_complete",
            "pass": True,
            "scenario": binding.as_dict(),
            "preflight": _artifact(preflight_path),
            "source_identity": _artifact(source_manifest),
            "authoritative_sfm": dict(sfm_info),
            "weak_overlap_connectivity": weak_overlap,
            "providers": provider_results,
            "manual_review": {
                "required": True,
                "matched_view_diagnostic_started": False,
                "verdict_recorded": False,
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
                "matched_view_diagnostic_authorized": True,
                "scenario_verdict_pending_manual_review": True,
                "universal_provider_winner_assumed": False,
                "global_scalar_score_forbidden": True,
            },
        }
        closure_path = _write_json(out / "i4_3g2b_production_native_snapshot.json", closure)
        print("I4.3G2B production-native execution: PASS")
        print(f"registered_views={sfm_info['registered_view_count']} landmarks={sfm_info['landmark_count']}")
        print(f"same_authoritative_sfm_shared=True")
        print(f"production_snapshot={closure_path}")
        return 0
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "phase": "I4.3G2B",
            "status": "failed",
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authority": {
                "camera_pose_authority_owner": "pycolmap_sfm",
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
            _write_json(out / "i4_3g2b_failure.json", failure)
        except Exception:
            pass
        print(f"I4.3G2B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
