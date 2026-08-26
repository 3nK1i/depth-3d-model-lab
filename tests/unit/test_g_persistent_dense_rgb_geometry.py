from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from core.reconstruction.dense_scene_persistence import (
    DENSE_RGB_SCENE_MANIFEST,
    DenseSceneFileReference,
    DenseSceneFileRole,
    PersistentDenseRgbScene,
    persist_filtered_dense_rgb_scene,
    read_persistent_dense_rgb_scene,
    reference_for_role,
    resolve_dense_scene_reference,
    restore_persistent_dense_geometry,
)
from core.reconstruction.errors import ReconstructionValidationError
from scripts.run_e2_consistency_aware_fusion_refinement import _write_binary_ply
from scripts.run_g_persistent_dense_rgb_geometry import main as g_main
from scripts.run_g_persistent_dense_rgb_restore import main as g_restore_main

_SESSION_ID = "123e4567-e89b-12d3-a456-426614174000"


def _save_filtered_npz(
    path: Path,
    points: np.ndarray,
    *,
    voxel_size: float,
    orders: list[int],
    keyframes: list[str],
    representation: str,
) -> None:
    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    colors = np.tile(np.asarray([[70, 120, 180]], dtype=np.uint8), (count, 1))
    class_code = np.full(count, 3 if representation == "core" else 2, dtype=np.uint8)
    if representation == "extended" and count:
        class_code[: min(2, count)] = 3
    np.savez_compressed(
        path,
        points_reconstruction=points,
        colors_rgb=colors,
        view_observation_counts=np.full(count, 4, dtype=np.uint32),
        filter_confidence=np.full(count, 0.7, dtype=np.float32),
        median_dispersion_voxels=np.full(count, 0.2, dtype=np.float32),
        p90_dispersion_voxels=np.full(count, 0.3, dtype=np.float32),
        maximum_dispersion_voxels=np.full(count, 0.4, dtype=np.float32),
        neighbor26_count=np.full(count, 12, dtype=np.int32),
        radius2_neighbor_count=np.full(count, 40, dtype=np.int32),
        high_support_neighbor_count=np.full(count, 5, dtype=np.int32),
        component_size=np.full(count, count, dtype=np.int32),
        class_code=class_code,
        artifact_reason_code=np.zeros(count, dtype=np.uint16),
        voxel_size_reconstruction_units=np.asarray(voxel_size, dtype=np.float64),
        selected_workspace_orders=np.asarray(orders, dtype=np.int32),
        selected_keyframe_ids=np.asarray(keyframes, dtype="U160"),
        sparse_landmarks_reconstruction=np.empty((0, 3), dtype=np.float64),
        sparse_landmark_colors_rgb=np.empty((0, 3), dtype=np.uint8),
        schema_version=np.asarray(1, dtype=np.int32),
    )


def _fixture(tmp_path: Path, *, core_outside_extended: bool = False, ply_count_mismatch: bool = False):
    e3_dir = tmp_path / "e3"
    f_dir = tmp_path / "f"
    e3_dir.mkdir()
    f_dir.mkdir()
    voxel = 0.1
    orders = [0, 1, 2, 3, 4, 5]
    keyframes = [f"{_SESSION_ID}:keyframe:{order:08d}" for order in orders]
    extended_points = np.asarray(
        [[(x + 0.2) * voxel, (y + 0.2) * voxel, (10 + z + 0.2) * voxel]
         for x in range(3) for y in range(2) for z in range(2)],
        dtype=np.float64,
    )
    core_points = extended_points[:8].copy()
    if core_outside_extended:
        core_points[-1] = np.asarray([9.2 * voxel, 9.2 * voxel, 19.2 * voxel])

    extended_npz = f_dir / "f_filtered_extended.npz"
    core_npz = f_dir / "f_filtered_core.npz"
    _save_filtered_npz(extended_npz, extended_points, voxel_size=voxel, orders=orders, keyframes=keyframes, representation="extended")
    _save_filtered_npz(core_npz, core_points, voxel_size=voxel, orders=orders, keyframes=keyframes, representation="core")
    extended_ply = _write_binary_ply(f_dir / "f_filtered_extended.ply", extended_points, np.tile([[70, 120, 180]], (len(extended_points), 1)).astype(np.uint8))
    core_ply_points = core_points[:-1] if ply_count_mismatch else core_points
    core_ply = _write_binary_ply(f_dir / "f_filtered_core.ply", core_ply_points, np.tile([[70, 120, 180]], (len(core_ply_points), 1)).astype(np.uint8))

    diagnostics = {}
    for key, name in (
        ("rejected_supported_npz", "f_rejected_supported_artifacts.npz"),
        ("attached_single_view_fringe_npz", "f_attached_single_view_fringe_diagnostic.npz"),
        ("detached_single_view_artifacts_npz", "f_detached_single_view_artifacts_diagnostic.npz"),
        ("coverage_candidate_npz", "f_coverage_candidate_diagnostic.npz"),
    ):
        path = f_dir / name
        np.savez_compressed(path, points=np.zeros((1, 3), dtype=np.float32))
        diagnostics[key] = str(path)

    e3_summary = {
        "phase": "E3",
        "pass": True,
        "status": "pass",
        "source_sfm_session_id": _SESSION_ID,
        "selected_view_count": len(orders),
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": keyframes,
    }
    (e3_dir / "e3_summary.json").write_text(json.dumps(e3_summary), encoding="utf-8")

    f_summary = {
        "phase": "F",
        "pass": True,
        "status": "pass",
        "source_e3_dir": str(e3_dir),
        "source_e3_selected_view_count": len(orders),
        "selected_workspace_orders": orders,
        "selected_keyframe_ids": keyframes,
        "voxel_size_reconstruction_units": voxel,
        "classification": {
            "source_all_voxels": 30,
            "source_supported_voxels": len(extended_points) + 4,
            "core_voxels": len(core_points),
            "extended_voxels": len(extended_points),
            "rejected_supported_voxels": 4,
            "attached_single_view_fringe_voxels": 3,
            "detached_single_view_artifact_voxels": 12,
            "artifact_reason_histogram": {"weak_high_support_anchor": 4},
        },
        "confidence": {"extended_quantiles": {"p50": 0.7}},
        "dispersion": {"p90_dispersion_voxel_quantiles": {"p50": 0.3}},
        "topology": {"largest_supported_component_fraction": 0.9},
        "policy": {"maximum_views": len(orders), "minimum_component_size": 1},
        "pass_conditions": {"exact_observation_accounting": True},
        "decision": {
            "authorizes_next_canonical_dense_persistence_milestone": True,
            "next_milestone": "G - Persistent Canonical Dense RGB Geometry",
        },
        "authority": {
            "filtered_dense_geometry_qualified": True,
            "metric_scale_claimed": False,
            "canonical_persistence_promoted": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
        },
        "artifacts": {
            "extended_npz": str(extended_npz),
            "extended_ply": str(extended_ply),
            "core_npz": str(core_npz),
            "core_ply": str(core_ply),
            **diagnostics,
        },
    }
    f_summary_path = f_dir / "f_summary.json"
    f_summary_path.write_text(json.dumps(f_summary), encoding="utf-8")
    return f_dir, e3_dir, f_summary_path, orders, keyframes, extended_points, core_points


def test_dense_file_reference_rejects_absolute_and_parent_paths():
    with pytest.raises(ReconstructionValidationError):
        DenseSceneFileReference(
            role=DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
            relative_path="/tmp/x.npz",
            size_bytes=1,
            sha256="0" * 64,
            media_type="application/x-npz",
            canonical_geometry=True,
            diagnostic_only=False,
        )
    with pytest.raises(ReconstructionValidationError):
        DenseSceneFileReference(
            role=DenseSceneFileRole.PRIMARY_EXTENDED_NPZ,
            relative_path="../x.npz",
            size_bytes=1,
            sha256="0" * 64,
            media_type="application/x-npz",
            canonical_geometry=True,
            diagnostic_only=False,
        )


def test_g_persistence_round_trip_is_self_contained_and_restores_without_sources(tmp_path: Path):
    f_dir, e3_dir, f_summary, orders, keyframes, extended_points, core_points = _fixture(tmp_path)
    out = tmp_path / "persistent"
    result = persist_filtered_dense_rgb_scene(f_summary, out, created_at_utc="2026-08-18T13:00:00+00:00")
    assert result.scene.primary_point_count == len(extended_points)
    assert result.scene.core_point_count == len(core_points)
    assert result.source_independent is True
    assert result.portable_relative_references is True
    shutil.rmtree(f_dir)
    shutil.rmtree(e3_dir)
    extended = restore_persistent_dense_geometry(out, representation="extended")
    core = restore_persistent_dense_geometry(out, representation="core")
    assert extended.point_count == len(extended_points)
    assert core.point_count == len(core_points)
    assert extended.selected_workspace_orders == tuple(orders)
    assert extended.selected_keyframe_ids == tuple(keyframes)


def test_g_bundle_can_be_relocated_without_changing_scene_identity(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    original = tmp_path / "persistent"
    result = persist_filtered_dense_rgb_scene(f_summary, original, created_at_utc="2026-08-18T13:00:00+00:00")
    relocated = tmp_path / "relocated" / "scene"
    relocated.parent.mkdir()
    shutil.copytree(original, relocated)
    scene = read_persistent_dense_rgb_scene(relocated, verify_files=True)
    restored = restore_persistent_dense_geometry(relocated, representation="extended")
    assert scene.scene_id == result.scene.scene_id
    assert scene.scene_fingerprint_sha256 == result.scene.scene_fingerprint_sha256
    assert restored.scene_id == result.scene.scene_id


def test_g_manifest_promotes_only_canonical_persistence_not_metric_measurement_ui_or_mesh(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    result = persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")
    scene = result.scene
    assert scene.canonical_geometry_authoritative is True
    assert scene.canonical_persistence_promoted is True
    assert scene.metric_scale_claimed is False
    assert scene.measurement_authoritative is False
    assert scene.certified_metrology is False
    assert scene.surface_mesh_generated is False
    assert scene.reconstruction_ui_enabled is False
    payload = json.loads((result.manifest_path).read_text(encoding="utf-8"))
    assert payload["authority"]["geometry_authority_scope"] == ["geometry", "picking", "spatial_relationships"]


def test_g_manifest_contains_complete_portable_reference_set(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    result = persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")
    scene = result.scene
    assert {item.role for item in scene.file_references} == set(DenseSceneFileRole)
    assert len(scene.file_references) == 10
    assert all(not Path(item.relative_path).is_absolute() for item in scene.file_references)
    assert all(".." not in Path(item.relative_path).parts for item in scene.file_references)
    assert reference_for_role(scene, DenseSceneFileRole.PRIMARY_EXTENDED_NPZ).canonical_geometry is True
    assert reference_for_role(scene, DenseSceneFileRole.REJECTED_SUPPORTED_DIAGNOSTIC).diagnostic_only is True


def test_g_restore_detects_referenced_file_tampering(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    result = persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")
    core_path = resolve_dense_scene_reference(result.manifest_path, result.scene, DenseSceneFileRole.HIGH_CONFIDENCE_CORE_NPZ)
    with core_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ReconstructionValidationError):
        read_persistent_dense_rgb_scene(result.manifest_path, verify_files=True)


def test_g_restore_detects_manifest_tampering(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    result = persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")
    manifest = result.manifest_path
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["primary_point_count"] += 1
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError):
        read_persistent_dense_rgb_scene(manifest, verify_files=False)


def test_g_refuses_core_geometry_that_is_not_subset_of_extended(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path, core_outside_extended=True)
    with pytest.raises(ReconstructionValidationError, match="Core geometry.*subset"):
        persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")


def test_g_refuses_ply_export_count_drift(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path, ply_count_mismatch=True)
    with pytest.raises(ReconstructionValidationError, match="Core PLY vertex count"):
        persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")


def test_g_refuses_premature_or_unqualified_f_source(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    payload = json.loads(f_summary.read_text(encoding="utf-8"))
    payload["authority"]["metric_scale_claimed"] = True
    f_summary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError):
        persist_filtered_dense_rgb_scene(f_summary, tmp_path / "persistent")


def test_g_refuses_overwrite_of_existing_nonempty_scene_directory(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    output = tmp_path / "persistent"
    output.mkdir()
    (output / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refuses to overwrite"):
        persist_filtered_dense_rgb_scene(f_summary, output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_g_script_authorizes_h_and_preserves_phase_boundaries():
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "run_g_persistent_dense_rgb_geometry.py").read_text(encoding="utf-8")
    assert '"authorizes_next_dense_rgb_ui_integration_milestone": passed' in script
    assert '"next_milestone": "H - Dense RGB Reconstruction UI Integration" if passed else None' in script
    assert '"metric_scale_claimed": scene.metric_scale_claimed' in script
    assert '"measurement_authoritative": scene.measurement_authoritative' in script
    assert '"surface_mesh_generated": scene.surface_mesh_generated' in script
    assert '"reconstruction_ui_enabled": scene.reconstruction_ui_enabled' in script


def test_g_restore_script_requires_only_persistent_bundle():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts" / "run_g_persistent_dense_rgb_restore.py").read_text(encoding="utf-8")
    assert "restore_persistent_dense_geometry" in source
    assert "--scene-dir" in source
    assert "--f-dir" not in source
    assert "DA3=False PyCOLMAP=False" in source


def test_invalid_restore_representation_is_rejected(tmp_path: Path):
    _, _, f_summary, *_ = _fixture(tmp_path)
    output = tmp_path / "persistent"
    persist_filtered_dense_rgb_scene(f_summary, output)
    with pytest.raises(ReconstructionValidationError):
        restore_persistent_dense_geometry(output, representation="mesh")


def test_g_scripts_end_to_end_persist_and_bundle_only_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, _, f_summary, *_ = _fixture(tmp_path)
    output = tmp_path / "persistent"
    monkeypatch.setattr(sys, "argv", [
        "run_g_persistent_dense_rgb_geometry.py",
        "--f-dir", str(f_summary.parent),
        "--output-dir", str(output),
    ])
    assert g_main() == 0
    summary = json.loads((output / "g_summary.json").read_text(encoding="utf-8"))
    assert summary["pass"] is True
    assert summary["persistence"]["bundle_self_contained"] is True
    assert summary["decision"]["authorizes_next_dense_rgb_ui_integration_milestone"] is True
    monkeypatch.setattr(sys, "argv", [
        "run_g_persistent_dense_rgb_restore.py",
        "--scene-dir", str(output),
    ])
    assert g_restore_main() == 0
