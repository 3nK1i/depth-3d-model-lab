#!/usr/bin/env python3
"""Phase G persistent canonical Dense RGB Points scene bundle.

G consumes a passing Phase-F filtered dense scene and materializes a portable,
self-contained canonical Dense RGB Points representation.  It copies the formal
Core/Extended geometry, export PLY files, filtering diagnostics and F/E3
provenance into one integrity-bound bundle, then reloads both canonical geometry
representations from the persisted files.

G promotes canonical geometry persistence/authority for Dense RGB Points only.
The scene remains in the arbitrary SfM gauge: no metric scale, measurement
certification, UI integration or surface mesh is introduced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction.dense_scene_persistence import (  # noqa: E402
    DenseSceneFileRole,
    persist_filtered_dense_rgb_scene,
    read_persistent_dense_rgb_scene,
    reference_for_role,
    resolve_dense_scene_reference,
    restore_persistent_dense_geometry,
    verify_persistent_dense_scene_files,
)
from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402

_DEFAULT_F_DIR = Path("/tmp/eia_f_confidence_artifact_filtering")
_DEFAULT_OUTPUT_DIR = Path("/tmp/eia_g_persistent_dense_rgb_geometry")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-dir", type=Path, default=_DEFAULT_F_DIR)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    args = _parse_args()
    f_dir = args.f_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    f_summary = f_dir / "f_summary.json"

    print("G persistent canonical Dense RGB Points:", flush=True)
    print(f"  source_f={f_summary}", flush=True)
    print(f"  output_dir={output_dir}", flush=True)
    print("  persistence_policy=self-contained relative references + SHA-256 integrity", flush=True)
    print("  canonical_geometry=dense_rgb_points | metric_scale=NOT claimed | mesh=NOT generated", flush=True)

    result = persist_filtered_dense_rgb_scene(f_summary, output_dir)
    scene = read_persistent_dense_rgb_scene(result.manifest_path, verify_files=True)
    references = verify_persistent_dense_scene_files(scene, result.manifest_path)
    extended = restore_persistent_dense_geometry(result.manifest_path, representation="extended")
    core = restore_persistent_dense_geometry(result.manifest_path, representation="core")

    extended_export = resolve_dense_scene_reference(
        result.manifest_path, scene, DenseSceneFileRole.PRIMARY_EXTENDED_PLY
    )
    core_export = resolve_dense_scene_reference(
        result.manifest_path, scene, DenseSceneFileRole.HIGH_CONFIDENCE_CORE_PLY
    )
    external_references = [
        item.relative_path for item in references if Path(item.relative_path).is_absolute()
    ]
    pass_conditions = {
        "manifest_round_trip_identity": scene.scene_id == result.scene.scene_id,
        "all_file_references_integrity_verified": len(references) == 10,
        "extended_restore_exact_count": extended.point_count == scene.primary_point_count,
        "core_restore_exact_count": core.point_count == scene.core_point_count,
        "selected_view_provenance_restored": (
            extended.selected_workspace_orders == scene.selected_workspace_orders
            and core.selected_workspace_orders == scene.selected_workspace_orders
            and extended.selected_keyframe_ids == scene.selected_keyframe_ids
            and core.selected_keyframe_ids == scene.selected_keyframe_ids
        ),
        "portable_relative_references": result.portable_relative_references and not external_references,
        "bundle_self_contained": result.source_independent,
        "canonical_geometry_authority_promoted": scene.canonical_geometry_authoritative,
        "canonical_persistence_promoted": scene.canonical_persistence_promoted,
        "metric_scale_not_claimed": not scene.metric_scale_claimed,
        "measurement_authority_not_promoted": not scene.measurement_authoritative,
        "surface_mesh_not_generated": not scene.surface_mesh_generated,
        "ui_not_enabled_before_h": not scene.reconstruction_ui_enabled,
    }
    passed = all(pass_conditions.values())
    summary = {
        "phase": "G",
        "pass": passed,
        "status": "pass" if passed else "degraded",
        "scene_id": scene.scene_id,
        "scene_fingerprint_sha256": scene.scene_fingerprint_sha256,
        "source_f_dir": str(f_dir),
        "source_f_summary_sha256": scene.source_f_summary_sha256,
        "source_e3_summary_sha256": scene.source_e3_summary_sha256,
        "source_sfm_session_id": scene.source_session_id,
        "reconstruction_frame_id": scene.reconstruction_frame_id,
        "selected_view_count": scene.selected_view_count,
        "selected_workspace_orders": list(scene.selected_workspace_orders),
        "selected_keyframe_ids": list(scene.selected_keyframe_ids),
        "voxel_size_reconstruction_units": scene.voxel_size_reconstruction_units,
        "geometry": {
            "primary_representation": "extended",
            "primary_point_count": scene.primary_point_count,
            "high_confidence_representation": "core",
            "core_point_count": scene.core_point_count,
            "source_supported_point_count": scene.source_supported_point_count,
            "source_all_voxel_count": scene.source_all_voxel_count,
            "primary_npz_sha256": reference_for_role(
                scene, DenseSceneFileRole.PRIMARY_EXTENDED_NPZ
            ).sha256,
            "core_npz_sha256": reference_for_role(
                scene, DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ
            ).sha256,
        },
        "persistence": {
            "manifest": str(result.manifest_path),
            "file_reference_count": len(references),
            "all_references_relative": result.portable_relative_references,
            "external_file_reference_count": len(external_references),
            "bundle_self_contained": result.source_independent,
            "integrity_algorithm": "sha256",
            "atomic_directory_publish": True,
            "restore_requires_source_f_or_e3": False,
        },
        "exports": {
            "extended_ply": str(extended_export),
            "core_ply": str(core_export),
        },
        "pass_conditions": pass_conditions,
        "decision": {
            "authorizes_next_dense_rgb_ui_integration_milestone": passed,
            "next_milestone": "H - Dense RGB Reconstruction UI Integration" if passed else None,
            "primary_persistent_representation": "dense_rgb_extended",
            "high_confidence_persistent_representation": "dense_rgb_core",
        },
        "authority": {
            "canonical_geometry_authoritative": scene.canonical_geometry_authoritative,
            "canonical_persistence_promoted": scene.canonical_persistence_promoted,
            "geometry_authority_scope": ["geometry", "picking", "spatial_relationships"],
            "sfm_scale_status": "arbitrary",
            "sfm_world_alignment_status": "arbitrary",
            "metric_scale_claimed": scene.metric_scale_claimed,
            "measurement_authoritative": scene.measurement_authoritative,
            "certified_metrology": scene.certified_metrology,
            "surface_mesh_generated": scene.surface_mesh_generated,
            "reconstruction_ui_enabled": scene.reconstruction_ui_enabled,
            "note": (
                "G persists filtered Dense RGB Points as canonical geometry in the arbitrary SfM gauge. "
                "Metric/measurement authority is not promoted; H owns UI integration and J owns mesh reconstruction."
            ),
        },
        "artifacts": {
            "persistent_scene_manifest": str(result.manifest_path),
            "extended_npz": str(resolve_dense_scene_reference(
                result.manifest_path, scene, DenseSceneFileRole.PRIMARY_EXTENDED_NPZ
            )),
            "core_npz": str(resolve_dense_scene_reference(
                result.manifest_path, scene, DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ
            )),
            "extended_ply": str(extended_export),
            "core_ply": str(core_export),
        },
    }
    summary_path = _write_json(output_dir / "g_summary.json", summary)

    print(f"\nG persistent canonical Dense RGB geometry: {summary['status'].upper()}")
    print(f"scene_id={scene.scene_id} fingerprint={scene.scene_fingerprint_sha256}")
    print(
        f"views={scene.selected_view_count} extended_points={extended.point_count} "
        f"core_points={core.point_count} voxel_size={scene.voxel_size_reconstruction_units:.6f}"
    )
    print(
        f"references={len(references)} relative={result.portable_relative_references} "
        f"self_contained={result.source_independent} external_references={len(external_references)}"
    )
    print(
        "canonical_geometry_authoritative=True canonical_persistence_promoted=True "
        "metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False ui_enabled=False"
    )
    print(f"manifest={result.manifest_path}")
    print(f"summary={summary_path}")
    if passed:
        print("next=H - Dense RGB Reconstruction UI Integration")
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReconstructionValidationError as exc:
        print(f"G persistence validation error: {exc}", file=sys.stderr)
        raise
