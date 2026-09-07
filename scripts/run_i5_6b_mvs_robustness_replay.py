#!/usr/bin/env python3
"""Run I5.6B production-native MVS robustness replay on frozen authoritative SfM evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.adaptive_production_planning import (
    finalize_adaptive_provider_execution_result,
    prepare_adaptive_provider_execution,
)
from core.reconstruction.adaptive_view_selection import AdaptiveSelectionResourceSnapshot
from core.reconstruction.dense_provider_architecture import DenseProviderRunStatus
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.i5_difficult_handheld_robustness import I51EvidenceBinding
from core.reconstruction.i5_mvs_robustness import (
    I56BReplayPolicy,
    I56B_SCENARIO_IDS,
    analyze_i5_6a_sfm_report,
    analyze_i5_6b_provider_summary,
    build_i5_6b_mvs_robustness_replay,
    validate_i5_6a_for_i5_6b,
    validate_i5_6b_receipt,
)
from core.reconstruction.mvs_environment import probe_mvs_host_resources, probe_pycolmap_mvs_environment
from core.reconstruction.mvs_provider import (
    MvsProductionConfig,
    MvsProductionProviderAdapter,
    MvsProviderRunContext,
    colmap_mvs_provider_descriptor,
)
from core.reconstruction.sfm_result_serialization import offline_sfm_result_from_dict

OUTPUT_NAME = "i5_6b_mvs_robustness_replay.json"
SCENARIO_RECORD_NAME = "i5_6b_scenario_record.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i5-6a-admission", type=Path, required=True)
    parser.add_argument("--i5-6a-sha256", required=True)
    parser.add_argument("--replay-receipt", type=Path, required=True)
    parser.add_argument("--replay-receipt-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mvs-python", required=True)
    parser.add_argument("--scenario-timeout-minutes", type=float, default=240.0)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _binding(path: Path, sha256: str, *, label: str) -> I51EvidenceBinding:
    resolved = path.expanduser().resolve()
    expected = str(sha256).strip().lower()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6B {label} file missing: {resolved}")
    if not _valid_sha256(expected):
        raise ReconstructionValidationError(f"I5.6B {label} SHA256 is malformed.")
    actual = _sha256(resolved)
    if actual != expected:
        raise ReconstructionValidationError(
            f"I5.6B {label} SHA256 mismatch: expected {expected}, found {actual}"
        )
    return I51EvidenceBinding(str(resolved), expected)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I5.6B could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I5.6B JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def _path_binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ReconstructionValidationError(f"I5.6B expected artifact is missing: {resolved}")
    return {"path": str(resolved), "sha256": _sha256(resolved), "size_bytes": resolved.stat().st_size}


def _resource_snapshot(host: Any, *, cuda_ready: bool) -> AdaptiveSelectionResourceSnapshot:
    gpu = next((item for item in host.gpus if int(item.index) == 0), None)
    if gpu is None:
        raise ReconstructionValidationError(
            "I5.6B formal replay requires visible GPU index 0 for the frozen production MVS configuration."
        )
    gib = 1024 ** 3
    return AdaptiveSelectionResourceSnapshot(
        gpu_name=str(gpu.name),
        total_vram_bytes=int(round(float(gpu.total_vram_gib) * gib)),
        free_vram_bytes=int(round(float(gpu.free_vram_gib) * gib)),
        available_ram_bytes=int(round(float(host.ram_available_gib) * gib)),
        cuda_ready=bool(cuda_ready),
    )


class _ConsoleProgress:
    def __init__(self, scenario_id: str) -> None:
        self._scenario_id = scenario_id
        self._last: tuple[str, str, int] | None = None

    def publish(self, event: Any) -> None:
        stage = getattr(getattr(event, "stage", None), "value", str(getattr(event, "stage", "")))
        detail = str(getattr(event, "stage_detail", ""))
        percent = int(round(float(getattr(event, "progress_fraction", 0.0)) * 100.0))
        signature = (stage, detail, percent)
        if signature != self._last:
            print(f"I5.6B {self._scenario_id}: [{percent:3d}%] {stage} {detail}", flush=True)
            self._last = signature


def _report_analysis(admission_node: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    report_binding = admission_node.get("report_binding")
    if not isinstance(report_binding, Mapping):
        raise ReconstructionValidationError("I5.6B scenario report binding is missing.")
    report_path = Path(str(report_binding.get("path", ""))).expanduser().resolve()
    expected_sha = str(report_binding.get("sha256", "")).strip().lower()
    _binding(report_path, expected_sha, label=f"{admission_node.get('scenario_id')} frozen SfM report")
    report = _read_json(report_path)
    analysis = analyze_i5_6a_sfm_report(
        report,
        scenario_id=str(admission_node["scenario_id"]),
        expected_input=int(admission_node["input_images"]),
        expected_registered=int(admission_node["registered_images"]),
    )
    return report_path, report, analysis


def _run_scenario(
    *,
    scenario_id: str,
    admission_node: Mapping[str, Any],
    scenario_dir: Path,
    environment: Any,
    descriptor: Any,
    mvs_python: Path,
    policy: I56BReplayPolicy,
    timeout_s: float,
) -> dict[str, Any]:
    report_path, report, before = _report_analysis(admission_node)
    sfm = offline_sfm_result_from_dict(report["sfm_result"])
    frame_id = f"i5_6b:{scenario_id}:pycolmap_sfm:{sfm.session_id}"

    host_before = probe_mvs_host_resources(scenario_dir)
    adaptive_resources = _resource_snapshot(host_before, cuda_ready=bool(environment.patchmatch_cuda_ready))
    preparation = prepare_adaptive_provider_execution(
        sfm,
        provider=descriptor,
        reconstruction_frame_id=frame_id,
        resources=adaptive_resources,
        override=None,
    )
    adaptive_path = _write_json(scenario_dir / "adaptive_execution_plan.json", preparation.as_dict())

    config = MvsProductionConfig(
        source_neighbors=policy.source_neighbors,
        patch_match_cache_gib=policy.patch_match_cache_gib,
        stereo_fusion_cache_gib=policy.stereo_fusion_cache_gib,
        gpu_index="0",
        max_image_size=policy.max_image_size,
        geometric_consistency=policy.geometric_consistency,
    )
    provider_output = scenario_dir / "providers" / "colmap_mvs"
    context = MvsProviderRunContext(
        sfm_result=preparation.provider_sfm_result,
        reconstruction_frame_id=frame_id,
        output_dir=provider_output,
        pycolmap_python=mvs_python,
        worker_script=ROOT / "scripts" / "reconstruction_pycolmap_dense_provider_mvs_worker.py",
        config=config,
        worker_timeout_s=timeout_s,
    )
    adapter = MvsProductionProviderAdapter(
        context,
        progress_sink=_ConsoleProgress(scenario_id),
        environment_probe=environment,
        resource_probe=host_before,
    )

    print(
        f"I5.6B scenario={scenario_id} source_registered={sfm.statistics.registered_keyframes} "
        f"adaptive_selected={preparation.plan.selected_view_count} "
        f"reasons={','.join(preparation.plan.reason_codes)}",
        flush=True,
    )
    started = time.monotonic()
    raw_result = adapter.execute(preparation.selection_request, preparation.selection_decision)
    wall_time_s = time.monotonic() - started
    final_result = finalize_adaptive_provider_execution_result(raw_result, preparation)

    summary_path = provider_output / "i4_3b_summary.json"
    summary = _read_json(summary_path)
    native = analyze_i5_6b_provider_summary(summary, wall_time_s=wall_time_s)
    if int(native["native_metrics"]["selected_view_count"]) != int(preparation.plan.selected_view_count):
        raise ReconstructionValidationError(f"I5.6B {scenario_id} selected-view count drifted between plan and provider.")

    # Re-run the admission-grade integrity analysis after MVS. This hashes the
    # original registered images and selected sparse model again; provider work
    # must not change them.
    _, _, after = _report_analysis(admission_node)
    for key in ("registered_images_sha256", "selected_sparse_sha256", "registered_image_names", "selected_model_id"):
        if before.get(key) != after.get(key):
            raise ReconstructionValidationError(f"I5.6B {scenario_id} authoritative SfM input drifted after MVS: {key}")

    passed = native["runtime_status"] == "pass"
    if passed and final_result.status is not DenseProviderRunStatus.COMPLETED:
        raise ReconstructionValidationError(f"I5.6B {scenario_id} summary/result completion status disagree.")
    if not passed and final_result.status not in {DenseProviderRunStatus.FAILED, DenseProviderRunStatus.CANCELLED}:
        raise ReconstructionValidationError(f"I5.6B {scenario_id} rejected summary/result status disagree.")

    artifact_bindings: dict[str, Any] = {
        "frozen_sfm_report": _path_binding(report_path),
        "adaptive_execution_plan": _path_binding(adaptive_path),
        "provider_summary": _path_binding(summary_path),
    }
    worker_result = provider_output / "worker" / "result.json"
    worker_state = provider_output / "worker" / "execution_state.json"
    if worker_result.is_file():
        artifact_bindings["worker_result"] = _path_binding(worker_result)
    if worker_state.is_file():
        artifact_bindings["worker_state"] = _path_binding(worker_state)
    fused = provider_output / "dense" / "fused.ply"
    vis = Path(f"{fused}.vis")
    if passed:
        artifact_bindings["fused_ply"] = _path_binding(fused)
        artifact_bindings["native_visibility"] = _path_binding(vis)

    host_after = probe_mvs_host_resources(scenario_dir)
    record = {
        "scenario_id": scenario_id,
        "runtime_status": native["runtime_status"],
        "provider_execution_completed": passed,
        "source_registered_views": int(sfm.statistics.registered_keyframes),
        "selected_views": int(preparation.plan.selected_view_count),
        "selection_fraction": preparation.plan.selected_view_count / float(max(1, sfm.statistics.registered_keyframes)),
        "adaptive_selection": True,
        "plan_reason_codes": list(preparation.plan.reason_codes),
        "native_metrics": native["native_metrics"],
        "failure": native["failure"],
        "execution_result": final_result.as_dict(),
        "artifact_bindings": artifact_bindings,
        "resource_probe_before": host_before.as_dict(),
        "resource_probe_after": host_after.as_dict(),
        "input_integrity_revalidated_after": True,
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "single_authoritative_sfm_frame_preserved": True,
            "sfm_rerun": False,
            "bundle_adjustment_rerun": False,
            "pose_refinement": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "production_filter_policy_changed": False,
            "provider_promotion_performed": False,
            "permanent_global_view_cap": None,
        },
    }
    _write_json(scenario_dir / SCENARIO_RECORD_NAME, record)
    return record


def run(args: argparse.Namespace) -> Path:
    if float(args.scenario_timeout_minutes) <= 0.0:
        raise ReconstructionValidationError("I5.6B --scenario-timeout-minutes must be positive.")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    admission_binding = _binding(args.i5_6a_admission, args.i5_6a_sha256, label="I5.6A admission")
    receipt_binding = _binding(args.replay_receipt, args.replay_receipt_sha256, label="replay receipt")
    admission = validate_i5_6a_for_i5_6b(_read_json(Path(admission_binding.path)))
    receipt = validate_i5_6b_receipt(_read_json(Path(receipt_binding.path)))
    policy = I56BReplayPolicy()

    mvs_python = Path(args.mvs_python).expanduser().resolve()
    if not mvs_python.is_file():
        raise ReconstructionValidationError(f"I5.6B MVS Python executable missing: {mvs_python}")
    environment = probe_pycolmap_mvs_environment(str(mvs_python))
    descriptor = colmap_mvs_provider_descriptor(environment)
    if not descriptor.available:
        raise ReconstructionValidationError(
            f"I5.6B production MVS runtime is not qualified: {descriptor.unavailable_reason}; probe={environment.as_dict()}"
        )
    preflight = environment.as_dict()
    preflight["mvs_runtime_ready"] = True
    preflight["provider_id"] = descriptor.provider_id
    preflight["provider_available"] = descriptor.available
    _write_json(output / "runtime_preflight.json", preflight)

    matrix = admission["scenario_matrix"]
    records: dict[str, Any] = {}
    timeout_s = float(args.scenario_timeout_minutes) * 60.0
    for index, scenario_id in enumerate(I56B_SCENARIO_IDS, start=1):
        print(f"\n===== I5.6B SCENARIO {index}/6: {scenario_id} =====", flush=True)
        scenario_dir = output / f"{index:02d}_{scenario_id}"
        scenario_dir.mkdir(parents=True, exist_ok=False)
        records[scenario_id] = _run_scenario(
            scenario_id=scenario_id,
            admission_node=matrix[scenario_id],
            scenario_dir=scenario_dir,
            environment=environment,
            descriptor=descriptor,
            mvs_python=mvs_python,
            policy=policy,
            timeout_s=timeout_s,
        )
        node = records[scenario_id]
        metrics = node["native_metrics"]
        print(
            f"I5.6B scenario_result={scenario_id} status={node['runtime_status']} "
            f"selected={node['selected_views']}/{node['source_registered_views']} "
            f"fused_points={metrics.get('fused_point_count')} "
            f"support_mean={metrics.get('visibility_support_mean')} "
            f"wall_time_s={metrics.get('wall_time_s')}",
            flush=True,
        )

    result = build_i5_6b_mvs_robustness_replay(
        i5_6a_admission=admission,
        i5_6a_binding=admission_binding,
        replay_receipt=receipt,
        replay_receipt_binding=receipt_binding,
        runtime_preflight=preflight,
        scenario_records=records,
    )
    evidence_path = _write_json(output / OUTPUT_NAME, result)

    print("\nI5.6B Current Frozen-Foundation MVS Robustness Replay: EVIDENCE_COMPLETE")
    print(f"quality_status={result['quality_status']}")
    for sid in I56B_SCENARIO_IDS:
        node = records[sid]
        metrics = node["native_metrics"]
        print(
            f"{sid} status={node['runtime_status']} source={node['source_registered_views']} selected={node['selected_views']} "
            f"depth_maps={metrics.get('geometric_depth_maps')} normal_maps={metrics.get('geometric_normal_maps')} "
            f"fused_points={metrics.get('fused_point_count')} visibility_support_mean={metrics.get('visibility_support_mean')} "
            f"wall_time_s={metrics.get('wall_time_s')} peak_vram_mib={metrics.get('peak_vram_mib')}"
        )
    print(
        "sfm_rerun=False bundle_adjustment_rerun=False result_driven_retuning=False retry_until_pass=False "
        "geometric_consistency=True production_default_modified=False da3_rerun=False mp_sfm=False auto=False hybrid=False "
        "metric_scale=False metrology=False permanent_global_view_cap=None"
    )
    print(
        "cross_run_absolute_geometry_comparison=False classification_threshold_used=None global_scalar_score_created=False"
    )
    print("I5.6C MVS robustness qualification and closure: AUTHORIZED")
    print(f"evidence={evidence_path}")
    return evidence_path


def main() -> int:
    try:
        run(_parser().parse_args())
    except (ReconstructionValidationError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"I5.6B ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
