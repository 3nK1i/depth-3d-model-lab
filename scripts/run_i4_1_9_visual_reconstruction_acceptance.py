#!/usr/bin/env python3
"""I4.1.9 visual reconstruction acceptance with geometry authority separated.

Consumes an existing I4.1.7/I4.1.6B campaign.  It does NOT rerun PyCOLMAP,
DA3, localization, or dense pair tests.  Instead it selects a broad set of
already alignment-qualified observations, applies the preserved fusion cap,
and runs the unchanged fusion -> F filtering -> G persistence -> I2 visual /
spatial quality chain.

Strict I4.1.7 dense pair-consistency and orbit gates remain unchanged and are
reported separately.  This visual path does not claim metric scale, geometry
accuracy, measurement authority, certified metrology, or mesh authority.
"""
from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    DenseFusionPolicy,
    DenseSceneProfile,
    SfmScaleStatus,
    arbitrary_scale_reference,
    camera_geometry_from_estimated_camera,
    fuse_dense_frames,
    reconstruction_camera_from_sfm_model,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_visual_acceptance import (  # noqa: E402
    I4_1_9_SCHEMA_VERSION,
    build_visual_acceptance_decision,
    build_visual_binding,
    choose_visual_plan,
    validate_i4_1_7_visual_source,
)
from scripts.run_d1_three_view_canonical_accumulation import _all_sparse_landmarks, _sfm_result_from_payload  # noqa: E402
from scripts.run_e3_qualified_scene_fusion_expansion import _robust_diagonal  # noqa: E402
from scripts.run_i1_offline_dense_campaign import _DENSE_THRESHOLDS  # noqa: E402
from scripts.run_i4_1_8_production_binding import (  # noqa: E402
    _build_e3_compatibility,
    _load_bound_frames,
    _selected_rows,
)

_DEFAULT_I4_1_7 = Path("/tmp/depth3d_i4_1_7_bonsai_clean_static_orbit_03")
_DEFAULT_OUTPUT = Path("/tmp/depth3d_i4_1_9_visual_acceptance")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--i4-1-7-dir", type=Path, default=_DEFAULT_I4_1_7)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    p.add_argument("--minimum-alignment-fraction", type=float, default=0.80)
    p.add_argument("--minimum-aligned-views", type=int, default=10)
    p.add_argument("--minimum-aligned-source-span-fraction", type=float, default=0.80)
    p.add_argument("--visual-fusion-cap", type=int, default=None)
    p.add_argument("--voxel-scene-fraction", type=float, default=0.004)
    p.add_argument("--max-voxels", type=int, default=1_500_000)
    p.add_argument("--minimum-supported-views", type=int, default=2)
    p.add_argument("--high-support-views", type=int, default=3)
    p.add_argument("--minimum-supported-voxels", type=int, default=1_000)
    p.add_argument("--sfm-focal-relative-uncertainty", type=float, default=0.20)
    p.add_argument("--skip-i2-view-attribution", action="store_true")
    a = p.parse_args()
    if not 0.0 < float(a.minimum_alignment_fraction) <= 1.0:
        p.error("--minimum-alignment-fraction must be in (0,1]")
    if int(a.minimum_aligned_views) < 10:
        p.error("--minimum-aligned-views must be >=10")
    if not 0.0 < float(a.minimum_aligned_source_span_fraction) <= 1.0:
        p.error("--minimum-aligned-source-span-fraction must be in (0,1]")
    if a.visual_fusion_cap is not None and int(a.visual_fusion_cap) < 10:
        p.error("--visual-fusion-cap must be >=10")
    if not 0.0 < float(a.voxel_scene_fraction) < 0.1:
        p.error("--voxel-scene-fraction must be in (0,0.1)")
    if int(a.max_voxels) <= 0 or int(a.minimum_supported_voxels) <= 0:
        p.error("voxel limits must be positive")
    return a


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"I4.1.9 cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"I4.1.9 JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _run(command: list[str], *, log_path: Path) -> int:
    print("I4.1.9 command: " + " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return int(proc.wait())


def _plan_by_name(dense_ab: Mapping[str, Any], name: str) -> dict[str, Any]:
    plans = dense_ab.get("plans")
    if not isinstance(plans, list):
        raise ReconstructionValidationError("I4.1.9 dense evidence has no plans.")
    for value in plans:
        if isinstance(value, Mapping) and str(value.get("plan_name")) == name:
            return dict(value)
    raise ReconstructionValidationError(f"I4.1.9 cannot resolve visual plan {name!r}.")


def _write_visual_i1_adapter(
    *,
    adapter_dir: Path,
    source_i1: Mapping[str, Any],
    sfm_result: Any,
    binding: Any,
    plan_assessment: Mapping[str, Any],
    e3_summary: Mapping[str, Any],
    f_summary: Mapping[str, Any],
    g_summary: Mapping[str, Any],
) -> Path:
    selected_count = int(binding.selected_view_count)
    selected_orders = list(range(selected_count))
    planned_count = int(plan_assessment["planned_candidates"])
    aligned_count = int(plan_assessment["alignment_accepted"])
    final = {
        "phase": "I1",
        "schema_version": int(source_i1.get("schema_version", 1)),
        "status": "pass",
        "pipeline_execution_complete": True,
        "quality_campaign_status": "pending_I2_visual_spatial_review",
        "campaign_adapter": {
            "source_phase": "I4.1.9",
            "visual_reconstruction_binding": True,
            "plan_name": binding.plan_name,
            "planned_candidate_count": planned_count,
            "alignment_accepted_count": aligned_count,
            "alignment_fraction": float(plan_assessment["alignment_fraction"]),
            "aligned_source_span_fraction": float(plan_assessment["aligned_source_span_fraction"]),
            "bound_visual_view_count": selected_count,
            "visual_fusion_cap": int(binding.visual_fusion_cap),
            "strict_pair_consistency_required_for_visual_path": False,
            "strict_pair_consistency_gate_changed": False,
            "strict_orbit_gate_changed": False,
            "pycolmap_rerun": False,
            "da3_rerun": False,
            "localization_rerun": False,
            "dense_pair_rerun": False,
        },
        "source": dict(source_i1.get("source", {})),
        "sfm": {
            "session_id": str(sfm_result.session_id),
            "registered_view_count": len(sfm_result.cameras),
            "registered_orders": [
                int(item.order)
                for item in sfm_result.workspace.images
                if item.keyframe_id in {camera.keyframe_id for camera in sfm_result.cameras}
            ],
            "candidate_dense_orders": list(range(planned_count)),
            "scale_status": "arbitrary",
        },
        "dense_qualification": {
            "alignment_accepted_views": aligned_count,
            "visual_selected_path": {
                "orders": selected_orders,
                "view_count": selected_count,
                "span": selected_count - 1,
                "selection_basis": "alignment_qualified_visual_view_set_not_strict_pair_consistency",
            },
            # Compatibility field consumed by the unchanged I2 quality kernel.
            "selected_pass_path": {
                "orders": selected_orders,
                "view_count": selected_count,
                "span": selected_count - 1,
                "selection_basis": "I4.1.9_visual_alignment_qualified_compatibility_alias",
            },
            "thresholds": dict(_DENSE_THRESHOLDS),
            "threshold_relaxation": False,
        },
        "fusion": dict(e3_summary["fusion"]),
        "filtering": {
            "core_voxels": int(f_summary["classification"]["core_voxels"]),
            "extended_voxels": int(f_summary["classification"]["extended_voxels"]),
            "rejected_supported_voxels": int(f_summary["classification"]["rejected_supported_voxels"]),
        },
        "persistent_scene": {
            "scene_id": str(g_summary["scene_id"]),
            "fingerprint": str(g_summary["scene_fingerprint_sha256"]),
            "scene_dir": str((adapter_dir / "06_persistent_scene").resolve()),
            "extended_points": int(g_summary["geometry"]["primary_point_count"]),
            "core_points": int(g_summary["geometry"]["core_point_count"]),
            "self_contained": bool(g_summary["persistence"]["bundle_self_contained"]),
        },
        "stages": {
            "sfm": {"reused": True, "source_dir": str((adapter_dir / "01_sfm").resolve())},
            "visual_voxel_fusion": {"status": str(e3_summary["status"])},
            "formal_filtering": {"status": "pass" if bool(f_summary.get("pass")) else "degraded"},
            "canonical_persistence": {"status": "pass" if bool(g_summary.get("pass")) else "degraded"},
        },
        "authority": {
            "canonical_geometry_authoritative": True,
            "canonical_persistence_promoted": True,
            "geometry_accuracy_qualified": False,
            "geometry_accuracy_scope": "canonical coordinates for rendering/picking; metric accuracy not claimed",
            "sfm_scale_status": "arbitrary",
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "decision": {
            "authorizes_i2_current_full_video_validation": True,
            "next_milestone": "I4.1.9 visual-scene I2 quality qualification",
            "note": "I4.1.9 reuses unchanged I2 visual/spatial quality kernels over a broad alignment-qualified visual scene.",
        },
    }
    return _write_json(adapter_dir / "i1_campaign_summary.json", final)


def _degraded_summary(
    *,
    source_summary: Mapping[str, Any],
    selection: Mapping[str, Any],
    output: Path,
    stage: str,
    reason: str,
    binding: Any | None = None,
    fusion_pass: bool = False,
    filtering_pass: bool = False,
    persistence_pass: bool = False,
    i2_pass: bool = False,
) -> dict[str, Any]:
    result = build_visual_acceptance_decision(
        source_summary=source_summary,
        visual_plan_selection=selection,
        binding=binding,
        fusion_pass=fusion_pass,
        filtering_pass=filtering_pass,
        persistence_pass=persistence_pass,
        i2_automated_pass=i2_pass,
    )
    result["failed_stage"] = stage
    result["reason"] = reason
    _write_json(output / "i4_1_9_summary.json", result)
    return result


def main() -> int:
    args = _args()
    i4_dir = args.i4_1_7_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.9 output directory must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_summary = _read_json(i4_dir / "i4_1_7_summary.json")
    source_guardrails = validate_i4_1_7_visual_source(source_summary)
    dense_ab = _read_json(i4_dir / "03_real_dense_ab" / "i4_1_6b_summary.json")
    dense_decision = dense_ab.get("decision")
    if not isinstance(dense_decision, Mapping):
        raise ReconstructionValidationError("I4.1.9 dense A/B decision is missing.")
    recorded_cap = int(dense_decision.get("final_fusion_cap_preserved", 0))
    visual_cap = int(args.visual_fusion_cap) if args.visual_fusion_cap is not None else recorded_cap
    if visual_cap < 10 or (recorded_cap > 0 and visual_cap > recorded_cap):
        raise ReconstructionValidationError("I4.1.9 visual fusion cap must be >=10 and may not exceed the preserved production cap.")

    selection = choose_visual_plan(
        dense_ab,
        minimum_alignment_fraction=float(args.minimum_alignment_fraction),
        minimum_aligned_views=int(args.minimum_aligned_views),
        minimum_aligned_source_span_fraction=float(args.minimum_aligned_source_span_fraction),
        visual_fusion_cap=visual_cap,
    )
    if not bool(selection.get("visual_plan_available")):
        _degraded_summary(
            source_summary=source_summary,
            selection=selection,
            output=output,
            stage="visual_plan_selection",
            reason="No dense plan satisfied the visual alignment/count/source-span gates; strict pair thresholds were not changed.",
        )
        print("I4.1.9 Visual Reconstruction Acceptance: DEGRADED", flush=True)
        print("failed_stage=visual_plan_selection strict_pair_consistency_gate_changed=False", flush=True)
        print(f"summary={output / 'i4_1_9_summary.json'}", flush=True)
        return 3

    binding = build_visual_binding(selection, visual_fusion_cap=visual_cap)
    plan = _plan_by_name(dense_ab, binding.plan_name)
    rows = _selected_rows(plan, binding.selected_source_sequence_indices)

    source_i1_dir = i4_dir / "01_i1_baseline"
    source_i1 = _read_json(source_i1_dir / "i1_campaign_summary.json")
    sfm_report = _read_json(source_i1_dir / "01_sfm" / "real_sfm_report.json")
    sfm_result = _sfm_result_from_payload(sfm_report["sfm_result"])
    if sfm_result.gauge.scale_status is not SfmScaleStatus.ARBITRARY or sfm_result.gauge.unit_to_metre is not None:
        raise ReconstructionValidationError("I4.1.9 requires arbitrary-scale SfM with no unit-to-metre promotion.")
    if len(sfm_result.camera_models) != 1:
        raise ReconstructionValidationError("I4.1.9 currently requires one shared SfM camera model.")
    frames, frame_rows, selected_poses = _load_bound_frames(rows=rows, i4_dir=i4_dir, sfm_result=sfm_result)

    binding_manifest = {
        "phase": "I4.1.9",
        "schema_version": I4_1_9_SCHEMA_VERSION,
        "binding": binding.as_dict(),
        "visual_plan_selection": selection,
        "views": frame_rows,
        "source_i4_1_7": str(i4_dir),
        "source_i4_1_7_strict_status": str(source_summary.get("status")),
        "source_guardrails": source_guardrails,
        "single_authoritative_sfm_frame": True,
        "strict_pair_consistency_required_for_visual_path": False,
        "strict_pair_consistency_gate_changed": False,
        "strict_orbit_gate_changed": False,
        "dense_consistency_thresholds_changed": False,
        "production_default_changed": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
    }
    _write_json(output / "00_visual_binding" / "i4_1_9_visual_binding.json", binding_manifest)
    np.savez_compressed(
        output / "00_visual_binding" / "visual_camera_poses.npz",
        selected_camera_matrices=selected_poses,
        selected_source_sequence_indices=np.asarray(binding.selected_source_sequence_indices, dtype=np.int64),
        schema_version=np.asarray(I4_1_9_SCHEMA_VERSION, dtype=np.int32),
    )

    sparse_points, sparse_colors = _all_sparse_landmarks(sfm_result)
    scene_scale = _robust_diagonal(sparse_points)
    if not isfinite(scene_scale) or scene_scale <= 0.0:
        raise ReconstructionValidationError("I4.1.9 robust sparse scene scale is invalid.")
    voxel_size = float(scene_scale * float(args.voxel_scene_fraction))
    model = sfm_result.camera_models[0]
    reconstruction_camera = reconstruction_camera_from_sfm_model(
        model,
        keyframe_id=str(frame_rows[0]["keyframe_id"]),
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    camera_profile = camera_geometry_from_estimated_camera(
        profile_id=f"i4-1-9-{model.camera_id}-sfm-estimated",
        camera=reconstruction_camera,
        focal_length_relative_uncertainty=float(args.sfm_focal_relative_uncertainty),
        notes="I4.1.9 visual reconstruction camera profile from authoritative global SfM; arbitrary gauge only.",
    )
    scale_profile = arbitrary_scale_reference(
        profile_id=f"i4-1-9-{sfm_result.session_id[:12]}-arbitrary-sfm-gauge",
        notes="I4.1.9 preserves arbitrary offline-SfM gauge; metric accuracy is deliberately not claimed.",
    )
    policy = DenseFusionPolicy(
        scene_profile=DenseSceneProfile.SPATIAL_DIGITAL_TWIN,
        voxel_size_reconstruction_units=voxel_size,
        pixel_stride=1,
        minimum_view_observations=1,
        max_voxels=int(args.max_voxels),
    )
    print(
        f"I4.1.9 visual binding: plan={binding.plan_name} aligned={binding.alignment_accepted_count}/{binding.planned_candidate_count} "
        f"bound={binding.selected_view_count}/{binding.visual_fusion_cap} source_span={binding.aligned_source_span_fraction:.3f} "
        "strict_pair_consistency_required=False",
        flush=True,
    )

    adapter = output / "01_visual_i1"
    fusion_dir = adapter / "04_qualified_fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fusion_result = fuse_dense_frames(
        frames,
        session_id=f"{sfm_result.session_id}:i4-1-9-visual-fusion",
        camera_profile=camera_profile,
        scale_profile=scale_profile,
        policy=policy,
        reconstruction_frame_id="offline-sfm-reconstruction",
    )
    fusion_duration = time.perf_counter() - started
    e3_summary = _build_e3_compatibility(
        output_dir=fusion_dir,
        sfm_result=sfm_result,
        frame_rows=frame_rows,
        fusion_result=fusion_result,
        voxel_size=voxel_size,
        sparse_points=sparse_points,
        sparse_colors=sparse_colors,
        minimum_supported_views=int(args.minimum_supported_views),
        high_support_views=int(args.high_support_views),
        minimum_supported_voxels=int(args.minimum_supported_voxels),
        plan_name=binding.plan_name,
        qualified_view_count=binding.alignment_accepted_count,
    )
    e3_summary["campaign_adapter"].update({
        "source_phase": "I4.1.9",
        "purpose": "visual reconstruction from a broad alignment-qualified observation pool",
        "selection_policy": "deterministic uniform cap over alignment-qualified visual observations",
        "strict_pair_consistency_required": False,
        "strict_pair_consistency_gate_changed": False,
        "strict_orbit_gate_changed": False,
        "threshold_relaxation": False,
    })
    e3_summary["campaign_path"].update({
        "source": "I4.1.9 broad alignment-qualified visual binding; not the strict pair-PASS path",
    })
    _write_json(fusion_dir / "e3_summary.json", e3_summary)
    if not bool(e3_summary.get("pass")):
        _degraded_summary(
            source_summary=source_summary, selection=selection, output=output, binding=binding,
            stage="visual_fusion", reason="Broad visual fusion did not pass unchanged E3/F prerequisites.",
        )
        print("I4.1.9 Visual Reconstruction Acceptance: DEGRADED\nfailed_stage=visual_fusion", flush=True)
        return 3

    sfm_adapter_dir = adapter / "01_sfm"
    sfm_adapter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_i1_dir / "01_sfm" / "real_sfm_summary.json", sfm_adapter_dir / "real_sfm_summary.json")
    shutil.copy2(source_i1_dir / "01_sfm" / "real_sfm_report.json", sfm_adapter_dir / "real_sfm_report.json")

    filter_dir = adapter / "05_filtering"
    code = _run(
        [sys.executable, str(ROOT / "scripts/run_f_confidence_artifact_filtering.py"), "--e3-dir", str(fusion_dir), "--output-dir", str(filter_dir)],
        log_path=output / "logs" / "02_formal_filtering.log",
    )
    f_summary = _read_json(filter_dir / "f_summary.json") if (filter_dir / "f_summary.json").is_file() else {"pass": False}
    filter_pass = code == 0 and bool(f_summary.get("pass"))
    if not filter_pass:
        _degraded_summary(
            source_summary=source_summary, selection=selection, output=output, binding=binding,
            stage="formal_filtering", reason="Visual scene failed unchanged F filtering.", fusion_pass=True,
        )
        print("I4.1.9 Visual Reconstruction Acceptance: DEGRADED\nfailed_stage=formal_filtering", flush=True)
        return 3

    persistent_dir = adapter / "06_persistent_scene"
    code = _run(
        [sys.executable, str(ROOT / "scripts/run_g_persistent_dense_rgb_geometry.py"), "--f-dir", str(filter_dir), "--output-dir", str(persistent_dir)],
        log_path=output / "logs" / "03_canonical_persistence.log",
    )
    g_summary = _read_json(persistent_dir / "g_summary.json") if (persistent_dir / "g_summary.json").is_file() else {"pass": False}
    persistence_pass = code == 0 and bool(g_summary.get("pass"))
    if not persistence_pass:
        _degraded_summary(
            source_summary=source_summary, selection=selection, output=output, binding=binding,
            stage="canonical_persistence", reason="Visual scene failed unchanged G persistence.",
            fusion_pass=True, filtering_pass=True,
        )
        print("I4.1.9 Visual Reconstruction Acceptance: DEGRADED\nfailed_stage=canonical_persistence", flush=True)
        return 3

    plan_assessment = selection["selected_plan_assessment"]
    i1_adapter_path = _write_visual_i1_adapter(
        adapter_dir=adapter,
        source_i1=source_i1,
        sfm_result=sfm_result,
        binding=binding,
        plan_assessment=plan_assessment,
        e3_summary=e3_summary,
        f_summary=f_summary,
        g_summary=g_summary,
    )

    i2_dir = output / "02_visual_i2_quality"
    i2_command = [
        sys.executable,
        str(ROOT / "scripts/run_i2_full_video_quality_qualification.py"),
        "--i1-dir", str(adapter),
        "--output-dir", str(i2_dir),
    ]
    if args.skip_i2_view_attribution:
        i2_command.append("--skip-view-attribution")
    code = _run(i2_command, log_path=output / "logs" / "04_i2_visual_quality.log")
    i2_summary_path = i2_dir / "i2_summary.json"
    i2_summary = _read_json(i2_summary_path) if i2_summary_path.is_file() else {"phase": "I2", "automated_pass": False}
    i2_pass = code == 0 and bool(i2_summary.get("automated_pass"))

    result = build_visual_acceptance_decision(
        source_summary=source_summary,
        visual_plan_selection=selection,
        binding=binding,
        fusion_pass=True,
        filtering_pass=True,
        persistence_pass=True,
        i2_automated_pass=i2_pass,
    )
    result["source"] = {
        "i4_1_7_dir": str(i4_dir),
        "i4_1_7_status": str(source_summary.get("status")),
        "strict_geometry_qualification_passed": bool(source_guardrails["strict_geometry_qualification_passed"]),
    }
    result["execution"] = {
        "pycolmap_rerun": False,
        "da3_rerun": False,
        "localization_rerun": False,
        "dense_pair_rerun": False,
        "visual_binding_executed": True,
        "bounded_visual_fusion_executed": True,
        "formal_filtering_executed": True,
        "canonical_persistence_executed": True,
        "i2_visual_quality_executed": True,
        "fusion_duration_s": float(fusion_duration),
    }
    result["artifacts"] = {
        "visual_binding_manifest": str((output / "00_visual_binding" / "i4_1_9_visual_binding.json").resolve()),
        "visual_camera_poses": str((output / "00_visual_binding" / "visual_camera_poses.npz").resolve()),
        "i1_adapter_summary": str(i1_adapter_path.resolve()),
        "e3_summary": str((fusion_dir / "e3_summary.json").resolve()),
        "f_summary": str((filter_dir / "f_summary.json").resolve()),
        "g_summary": str((persistent_dir / "g_summary.json").resolve()),
        "i2_summary": str(i2_summary_path.resolve()),
        "persistent_scene": str(persistent_dir.resolve()),
    }
    summary_path = _write_json(output / "i4_1_9_summary.json", result)

    print("\nI4.1.9 Visual Reconstruction Acceptance: " + ("READY_FOR_MANUAL_VISUAL_REVIEW" if result["automated_visual_reconstruction_pass"] else "DEGRADED"), flush=True)
    print(
        f"plan={binding.plan_name} planned={binding.planned_candidate_count} aligned={binding.alignment_accepted_count} "
        f"alignment_fraction={binding.alignment_fraction:.3f} bound_visual_views={binding.selected_view_count}/{binding.visual_fusion_cap} "
        f"aligned_source_span={binding.aligned_source_span_fraction:.3f}", flush=True,
    )
    print(
        f"visual_scene fusion_pass=True filtering_pass=True persistence_pass=True i2_automated_pass={i2_pass}", flush=True,
    )
    print(
        f"strict_geometry_qualification_passed={source_guardrails['strict_geometry_qualification_passed']} "
        "strict_pair_consistency_gate_changed=False strict_orbit_gate_changed=False dense_thresholds_changed=False",
        flush=True,
    )
    print("metric_scale_claimed=False measurement_authoritative=False certified_metrology=False surface_mesh_generated=False", flush=True)
    print(f"persistent_scene={persistent_dir}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if bool(result["automated_visual_reconstruction_pass"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
