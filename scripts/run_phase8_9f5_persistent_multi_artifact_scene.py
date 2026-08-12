#!/usr/bin/env python3
"""Phase 8.9F5 persistent multi-artifact real-scene qualification.

F5 starts no heavy reconstruction backend. It reuses the already-qualified real
F2 canonical geometry, real F3 Gaussian appearance artifact, and real F4
Nerfacto artifact, materializes bounded self-contained copies, and exercises the
existing PersistentReconstructionScene contract end to end.

The persistent-scene schema still requires semantic/QC stage records. F5 binds
those two stages as explicit contract-only context; it does not claim that
South Building semantics or inspection/QC were requalified in F5.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reconstruction_gaussian_visualization import build_gaussian_visualization_presentation
from app.reconstruction_nerf_visualization import build_nerf_visualization_presentation
from core.reconstruction import (
    PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM,
    persistent_scene_from_pipeline_payload,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)
from core.reconstruction.errors import ReconstructionValidationError

PHASE = "8.9F5"
EXPECTED_BRANCH = "phase8-reconstruction"
EXPECTED_HEAD = "101537cd118f8b752a1b440bc88aa3b069ed75fc"
DEFAULT_F2_ROOT = Path("/tmp/phase8_9f2_real_geometry")
DEFAULT_F3_ROOT = Path("/tmp/phase8_9f3_real_gaussian")
DEFAULT_F4_ROOT = Path("/tmp/phase8_9f4_real_nerf")
DEFAULT_OUTPUT = Path("/tmp/phase8_9f5_persistent_scene")

F5_REPO_FILES = {
    "docs/phase_history/PHASE8_9F5_VALIDATION.md",
    "scripts/run_phase8_9f5_persistent_multi_artifact_scene.py",
    "scripts/run_phase8_9f5_tests.sh",
    "tests/integration/test_phase8_9f5_persistent_real_scene_contract.py",
    "tests/unit/test_phase8_9f5_persistent_multi_artifact_scene.py",
}

HEAVY_MODULES = (
    "depth_anything_3",
    "diff_gaussian_rasterization",
    "gsplat",
    "nerfstudio",
    "pycolmap",
    "simple_knn",
    "tinycudann",
)


def _git(command: list[str], *, cwd: Path = ROOT) -> str:
    completed = subprocess.run(
        ["git", *command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git command failed")
    return completed.stdout.strip()


def _repository_state(expected_head: str) -> dict[str, Any]:
    branch = _git(["branch", "--show-current"])
    head = _git(["rev-parse", "HEAD"])
    origin = _git(["rev-parse", "origin/phase8-reconstruction"])
    status_lines = _git(["status", "--short", "--untracked-files=all"]).splitlines()
    overlay: list[str] = []
    unexpected: list[str] = []
    for line in status_lines:
        path = line[3:].strip() if len(line) >= 4 else line.strip()
        if path in F5_REPO_FILES:
            overlay.append(path)
        else:
            unexpected.append(line)
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "expected_head": expected_head,
        "head_matches_origin": head == origin,
        "head_matches_expected": not expected_head or head == expected_head,
        "f5_overlay_files_present": sorted(overlay),
        "unexpected_worktree_changes": unexpected,
        "pass": bool(
            branch == EXPECTED_BRANCH
            and head == origin
            and (not expected_head or head == expected_head)
            and not unexpected
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid or missing JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence is not an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, target: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError(f"required source artifact is missing/empty: {source}")
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if _sha256_file(source) != _sha256_file(target):
        raise RuntimeError(f"artifact copy checksum mismatch: {source} -> {target}")
    return target


def _first_render_from_manifest(path: Path) -> Path:
    payload = _load_json(path)
    renders = payload.get("renders")
    if not isinstance(renders, list) or not renders or not isinstance(renders[0], dict):
        raise RuntimeError(f"held-out render manifest has no render entries: {path}")
    rendered = str(renders[0].get("rendered_image_path", "")).strip()
    if not rendered:
        raise RuntimeError(f"held-out render manifest omits rendered_image_path: {path}")
    candidate = Path(rendered).expanduser().resolve()
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise RuntimeError(f"held-out rendered image is missing: {candidate}")
    return candidate


def _find_heldout_manifest(root: Path) -> Path:
    candidates = sorted(root.rglob("heldout_render_manifest.json"))
    if not candidates:
        raise RuntimeError(f"no held_out render manifest found below {root}")
    # Prefer the E3/quality worker output when several manifests exist.
    for candidate in candidates:
        if "quality" in candidate.parts or "nerfstudio-nerfacto-e3" in candidate.parts:
            return candidate.resolve()
    return candidates[0].resolve()


def _heavy_runtime_importability() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in HEAVY_MODULES}


def _validate_source_chain(f2_root: Path, f3_root: Path, f4_root: Path) -> dict[str, Any]:
    f2 = _load_json(f2_root / "summary.json")
    f3 = _load_json(f3_root / "summary.json")
    f4 = _load_json(f4_root / "summary.json")

    f2_e = f2.get("evidence") if isinstance(f2.get("evidence"), dict) else {}
    f3_e = f3.get("evidence") if isinstance(f3.get("evidence"), dict) else {}
    f4_e = f4.get("evidence") if isinstance(f4.get("evidence"), dict) else {}
    f2_auth = f2_e.get("authority") if isinstance(f2_e.get("authority"), dict) else {}
    f3_auth = f3_e.get("authority") if isinstance(f3_e.get("authority"), dict) else {}
    f4_auth = f4_e.get("authority") if isinstance(f4_e.get("authority"), dict) else {}

    checks = {
        "f2_pass": bool(f2.get("pass") is True),
        "f2_canonical_authoritative": f2_auth.get("canonical_geometry") == "authoritative",
        "f2_not_metrology": f2_auth.get("certified_metrology") is False,
        "f3_pass": bool(f3.get("pass") is True),
        "f3_gaussian_appearance_only": f3_auth.get("gaussian") == "appearance_only",
        "f3_not_metrology": f3_auth.get("certified_metrology") is False,
        "f4_pass": bool(f4.get("pass") is True),
        "f4_nerf_appearance_only": f4_auth.get("nerf") == "appearance_only",
        "f4_canonical_authoritative": f4_auth.get("canonical_geometry") == "authoritative",
        "f4_not_metrology": f4_auth.get("certified_metrology") is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"F2/F3/F4 source-chain qualification failed: {failed}")

    return {
        "checks": checks,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "source_summary_sha256": {
            "f2": _sha256_file((f2_root / "summary.json").resolve()),
            "f3": _sha256_file((f3_root / "summary.json").resolve()),
            "f4": _sha256_file((f4_root / "summary.json").resolve()),
        },
    }


def _materialize_real_artifacts(
    *,
    source: dict[str, Any],
    f2_root: Path,
    f3_root: Path,
    f4_root: Path,
    output: Path,
) -> dict[str, Any]:
    f2 = source["f2"]
    f3 = source["f3"]
    f4 = source["f4"]
    f2_e = f2["evidence"]
    f3_e = f3["evidence"]
    f4_e = f4["evidence"]

    canonical_source = Path(str(f2_e["canonical"]["surface_obj"]))
    gaussian_scene_source = Path(str(f3_e["gaussian"]["scene_path"]))
    gaussian_render_manifest = Path(str(f3_e["gaussian"]["render_manifest"]))
    gaussian_preview_source = _first_render_from_manifest(gaussian_render_manifest)
    nerf_checkpoint_source = Path(str(f4_e["nerf"]["final_checkpoint"]))
    nerf_config_source = Path(str(f4_e["nerf"]["final_config"]))
    nerf_render_manifest = _find_heldout_manifest((f4_root / "quality").resolve())
    nerf_preview_source = _first_render_from_manifest(nerf_render_manifest)

    artifact_root = (output / "artifacts").resolve()
    canonical_dir = artifact_root / "canonical"
    gaussian_dir = artifact_root / "gaussian"
    nerf_dir = artifact_root / "nerf"

    canonical_obj = _copy_file(canonical_source, canonical_dir / "da3_canonical_surface.obj")
    gaussian_scene = _copy_file(gaussian_scene_source, gaussian_dir / "scene.pt")
    gaussian_preview = _copy_file(gaussian_preview_source, gaussian_dir / "preview.png")
    nerf_checkpoint = _copy_file(nerf_checkpoint_source, nerf_dir / "model.ckpt")
    nerf_config = _copy_file(nerf_config_source, nerf_dir / "config.yml")
    nerf_preview = _copy_file(nerf_preview_source, nerf_dir / "preview.png")

    return {
        "artifact_root": str(artifact_root),
        "canonical": {
            "artifact_dir": str(canonical_dir.resolve()),
            "surface_obj_path": str(canonical_obj),
            "surface_obj_sha256": _sha256_file(canonical_obj),
            "surface_vertex_count": int(f2_e["canonical"]["surface_vertex_count"]),
            "surface_face_count": int(f2_e["canonical"]["surface_face_count"]),
            "metric_geometry_status": str(f2_e["canonical"]["metric_geometry_status"]),
            "unit_to_metre": f2_e["canonical"].get("unit_to_metre"),
        },
        "gaussian": {
            "artifact_dir": str(gaussian_dir.resolve()),
            "scene_path": str(gaussian_scene),
            "preview_path": str(gaussian_preview),
            "scene_sha256": _sha256_file(gaussian_scene),
            "preview_sha256": _sha256_file(gaussian_preview),
            "backend_id": str(f3_e["gaussian"].get("backend_kind", "gsplat_external")),
            "backend_version": "",
            "render_capable": True,
            "scene_role": "appearance_only",
            "metric_geometry_status": str(f2_e["canonical"]["metric_geometry_status"]),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
        "nerf": {
            "artifact_dir": str(nerf_dir.resolve()),
            "checkpoint_path": str(nerf_checkpoint),
            "config_path": str(nerf_config),
            "preview_path": str(nerf_preview),
            "checkpoint_sha256": _sha256_file(nerf_checkpoint),
            "config_sha256": _sha256_file(nerf_config),
            "preview_sha256": _sha256_file(nerf_preview),
            "backend_id": str(f4_e["nerf"].get("backend_kind", "nerfstudio_nerfacto_external")),
            "backend_version": str(f4_e["nerf"].get("backend_version", "")),
            "method_name": str(f4_e["nerf"].get("method_name", "nerfacto")),
            "input_frame_count": int(f4_e["bridge"].get("selected_frame_count", 0)),
            "training_steps": int(f4_e["nerf"].get("training_steps", 0)),
            "render_capable": True,
            "scene_role": "appearance_only",
            "metric_geometry_status": str(f2_e["canonical"]["metric_geometry_status"]),
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    }


def _completed_stage(
    *,
    request_id: str,
    session_id: str,
    stage: str,
    kind: str,
    authority: str,
    provider_id: str,
    payload_ref: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "state": "completed",
        "required": True,
        "provider_attempts": [provider_id],
        "provider_id": provider_id,
        "fallback_used": False,
        "reason": "",
        "artifact": {
            "schema_version": 1,
            "artifact_id": f"{request_id}:{stage}:{provider_id}",
            "session_id": session_id,
            "stage": stage,
            "kind": kind,
            "authority": authority,
            "provider_id": provider_id,
            "payload_ref": payload_ref,
            "canonical_geometry_authoritative": authority == "canonical_geometry",
            "certified_metrology": False,
            "metadata": metadata,
        },
    }


def _build_persistence_payload(
    *,
    f2_pipeline: dict[str, Any],
    materialized: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    f2_stages = f2_pipeline.get("stages")
    if not isinstance(f2_stages, list):
        raise RuntimeError("F2 canonical pipeline_result.json has no stage list")
    expected_core = ["source_ingestion", "camera_pose", "depth_evidence", "dense_geometry", "surface_mesh"]
    core = [copy.deepcopy(item) for item in f2_stages if isinstance(item, dict) and item.get("stage") in expected_core]
    if [item.get("stage") for item in core] != expected_core:
        raise RuntimeError("F2 canonical pipeline result does not contain the expected five real geometry stages in order")
    if not all(item.get("state") == "completed" for item in core):
        raise RuntimeError("F2 canonical pipeline result contains a non-completed real geometry stage")

    request_id = "phase8-9f5-persistent-real-scene"
    session_id = str(f2_pipeline.get("session_id", ""))
    if not session_id:
        raise RuntimeError("F2 canonical pipeline result omits session_id")

    # Preserve the real F2 geometry-stage records verbatim except for F5-owned
    # file-backed persistence snapshot references supplied separately below.
    semantic = _completed_stage(
        request_id=request_id,
        session_id=session_id,
        stage="semantic_geometry",
        kind="semantic_geometry",
        authority="semantic_context",
        provider_id="phase8-9f5-semantic-contract-context",
        payload_ref="memory://phase8-9f5/semantic-contract-context",
        metadata={
            "validation_context_only": True,
            "real_south_building_semantic_requalification_performed": False,
            "source_contract": "existing_phase8_7f3_semantic_stage_contract",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "metric_geometry_status": materialized["canonical"]["metric_geometry_status"],
        },
    )
    inspection = _completed_stage(
        request_id=request_id,
        session_id=session_id,
        stage="inspection_qc",
        kind="inspection_report",
        authority="inspection_decision",
        provider_id="phase8-9f5-inspection-contract-context",
        payload_ref="memory://phase8-9f5/inspection-contract-context",
        metadata={
            "validation_context_only": True,
            "real_south_building_inspection_requalification_performed": False,
            "source_contract": "existing_phase8_7f3_inspection_stage_contract",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )
    gaussian = _completed_stage(
        request_id=request_id,
        session_id=session_id,
        stage="gaussian_appearance",
        kind="gaussian_scene",
        authority="appearance_only",
        provider_id="phase8-9f3-real-gsplat-evidence",
        payload_ref=f"file://{materialized['gaussian']['scene_path']}",
        metadata={
            "source_phase": "8.9F3",
            "source_summary_sha256": source["source_summary_sha256"]["f3"],
            "backend_id": materialized["gaussian"]["backend_id"],
            "render_capable": True,
            "scene_role": "appearance_only",
            "metric_geometry_status": materialized["gaussian"]["metric_geometry_status"],
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )
    nerf = _completed_stage(
        request_id=request_id,
        session_id=session_id,
        stage="nerf_appearance",
        kind="nerf_scene",
        authority="appearance_only",
        provider_id="phase8-9f4-real-nerfacto-evidence",
        payload_ref=f"file://{materialized['nerf']['checkpoint_path']}",
        metadata={
            "source_phase": "8.9F4",
            "source_summary_sha256": source["source_summary_sha256"]["f4"],
            "backend_id": materialized["nerf"]["backend_id"],
            "backend_version": materialized["nerf"]["backend_version"],
            "method_name": materialized["nerf"]["method_name"],
            "render_capable": True,
            "scene_role": "appearance_only",
            "metric_geometry_status": materialized["nerf"]["metric_geometry_status"],
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
        },
    )

    provenance = copy.deepcopy(f2_pipeline.get("provider_provenance", []))
    if not isinstance(provenance, list):
        provenance = []
    provenance.extend(
        [
            {
                "provider_id": "phase8-9f5-semantic-contract-context",
                "provider_version": "phase8.9f5-contract-context",
                "execution": "in_process",
                "stages": ["semantic_geometry"],
            },
            {
                "provider_id": "phase8-9f5-inspection-contract-context",
                "provider_version": "phase8.9f5-contract-context",
                "execution": "in_process",
                "stages": ["inspection_qc"],
            },
            {
                "provider_id": "phase8-9f3-real-gsplat-evidence",
                "provider_version": "phase8.9f3-real-qualified",
                "execution": "external_process",
                "stages": ["gaussian_appearance"],
            },
            {
                "provider_id": "phase8-9f4-real-nerfacto-evidence",
                "provider_version": str(materialized["nerf"]["backend_version"] or "phase8.9f4-real-qualified"),
                "execution": "external_process",
                "stages": ["nerf_appearance"],
            },
        ]
    )

    canonical_id = str(f2_pipeline.get("canonical_geometry_artifact_id", ""))
    if not canonical_id:
        raise RuntimeError("F2 canonical pipeline result omits canonical_geometry_artifact_id")

    payload = {
        "schema_version": 1,
        "request_id": request_id,
        "session_id": session_id,
        "status": "completed",
        "execution_scope": "mixed",
        "stages": core + [semantic, inspection, gaussian, nerf],
        "provider_provenance": provenance,
        "canonical_geometry_artifact_id": canonical_id,
        "canonical_geometry_authoritative": True,
        "development_mock_used": False,
        "production_execution_claimed": False,
        "certified_metrology": False,
        "reconstruction_ui_enabled": False,
    }

    snapshots: dict[str, dict[str, Any]] = {
        "surface_mesh": {
            "artifact_dir": materialized["canonical"]["artifact_dir"],
            "surface_obj_path": materialized["canonical"]["surface_obj_path"],
            "metric_geometry_status": materialized["canonical"]["metric_geometry_status"],
            "unit_to_metre": materialized["canonical"]["unit_to_metre"],
            "surface_vertex_count": materialized["canonical"]["surface_vertex_count"],
            "surface_face_count": materialized["canonical"]["surface_face_count"],
            "canonical_geometry_authoritative": True,
            "certified_metrology": False,
        },
        "gaussian_scene": dict(materialized["gaussian"]),
        "nerf_scene": dict(materialized["nerf"]),
    }
    return payload, snapshots


def _tamper_rejection(scene, manifest: Path) -> dict[str, bool]:
    refs = verify_persistent_scene_file_references(scene)
    if not refs:
        raise RuntimeError("persistent scene has no referenced files")
    smallest = min(refs, key=lambda item: item.size_bytes)
    probe = Path(smallest.path)
    original = probe.read_bytes()

    referenced_tamper_rejected = False
    if original:
        mutated = bytes([original[0] ^ 0x01]) + original[1:]
    else:
        mutated = b"x"
    probe.write_bytes(mutated)
    try:
        verify_persistent_scene_file_references(scene)
    except ReconstructionValidationError:
        referenced_tamper_rejected = True
    finally:
        probe.write_bytes(original)
    verify_persistent_scene_file_references(scene)

    referenced_missing_rejected = False
    missing_probe = probe.with_name(probe.name + ".phase8_9f5_missing_probe")
    if missing_probe.exists():
        missing_probe.unlink()
    probe.rename(missing_probe)
    try:
        verify_persistent_scene_file_references(scene)
    except ReconstructionValidationError:
        referenced_missing_rejected = True
    finally:
        missing_probe.rename(probe)
    verify_persistent_scene_file_references(scene)

    manifest_tamper_rejected = False
    original_manifest = manifest.read_bytes()
    manifest.write_bytes(original_manifest + b" ")
    try:
        read_persistent_reconstruction_scene(manifest)
    except ReconstructionValidationError:
        manifest_tamper_rejected = True
    finally:
        manifest.write_bytes(original_manifest)
    read_persistent_reconstruction_scene(manifest)

    return {
        "referenced_tamper_rejected": referenced_tamper_rejected,
        "referenced_missing_rejected": referenced_missing_rejected,
        "manifest_tamper_rejected": manifest_tamper_rejected,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f2-root", type=Path, default=DEFAULT_F2_ROOT)
    parser.add_argument("--f3-root", type=Path, default=DEFAULT_F3_ROOT)
    parser.add_argument("--f4-root", type=Path, default=DEFAULT_F4_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-head", default=EXPECTED_HEAD)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    f2_root = args.f2_root.expanduser().resolve()
    f3_root = args.f3_root.expanduser().resolve()
    f4_root = args.f4_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    repository = _repository_state(str(args.expected_head).strip())
    if not repository["pass"]:
        raise SystemExit(f"ERROR: repository precondition failed: {repository}")

    source = _validate_source_chain(f2_root, f3_root, f4_root)
    f2_pipeline = _load_json(f2_root / "canonical" / "pipeline_result.json")

    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)

    print("[8.9F5] materializing self-contained real F2/F3/F4 artifacts")
    materialized = _materialize_real_artifacts(
        source=source,
        f2_root=f2_root,
        f3_root=f3_root,
        f4_root=f4_root,
        output=output,
    )
    print("[8.9F5] binding existing persistent-scene contract")
    payload, snapshots = _build_persistence_payload(
        f2_pipeline=f2_pipeline,
        materialized=materialized,
        source=source,
    )

    scene = persistent_scene_from_pipeline_payload(
        payload,
        payload_snapshots_by_kind=snapshots,
        created_at_utc="2026-08-12T14:30:00+00:00",
    )
    scene_a = write_persistent_reconstruction_scene(scene, (output / "scene-a").resolve())
    scene_b = write_persistent_reconstruction_scene(scene, (output / "scene-b").resolve())
    loaded_a = read_persistent_reconstruction_scene(scene_a)
    loaded_b = read_persistent_reconstruction_scene(scene_b)
    refs = verify_persistent_scene_file_references(loaded_a)
    verify_persistent_scene_file_references(loaded_b)

    byte_reproducible = scene_a.read_bytes() == scene_b.read_bytes()
    checksum_a = (scene_a.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).read_bytes()
    checksum_b = (scene_b.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM).read_bytes()
    checksum_reproducible = checksum_a == checksum_b
    identity_reproducible = bool(
        scene.scene_id == loaded_a.scene_id == loaded_b.scene_id
        and scene.scene_fingerprint_sha256
        == loaded_a.scene_fingerprint_sha256
        == loaded_b.scene_fingerprint_sha256
    )

    tamper = _tamper_rejection(loaded_a, scene_a)
    gaussian_presentation = build_gaussian_visualization_presentation(loaded_a)
    nerf_presentation = build_nerf_visualization_presentation(loaded_a)
    heavy = _heavy_runtime_importability()

    artifacts = {item.kind.value: item for item in loaded_a.artifacts}
    semantic = artifacts.get("semantic_geometry")
    inspection = artifacts.get("inspection_report")
    semantic_snapshot = dict(semantic.metadata) if semantic is not None else {}
    inspection_snapshot = dict(inspection.metadata) if inspection is not None else {}

    artifact_root_path = Path(materialized["artifact_root"]).resolve()
    references_self_contained = all(
        Path(item.path).resolve().is_relative_to(artifact_root_path) for item in refs
    )

    checks = {
        "source_chain_pass": all(source["checks"].values()),
        "scene_ready": loaded_a.state.value == "ready",
        "pipeline_completed": loaded_a.pipeline_status.value == "completed",
        "artifact_count_nine": len(loaded_a.artifacts) == 9,
        "canonical_geometry_authoritative": loaded_a.canonical_geometry_authoritative,
        "gaussian_present": loaded_a.gaussian_artifact is not None,
        "nerf_present": loaded_a.nerf_artifact is not None,
        "gaussian_appearance_only": bool(loaded_a.gaussian_artifact and loaded_a.gaussian_artifact.authority.value == "appearance_only"),
        "nerf_appearance_only": bool(loaded_a.nerf_artifact and loaded_a.nerf_artifact.authority.value == "appearance_only"),
        "not_certified_metrology": loaded_a.certified_metrology is False,
        "file_references_cover_real_assets": len(refs) >= 6,
        "file_references_self_contained_under_f5_artifacts": references_self_contained,
        "byte_reproducible": byte_reproducible,
        "checksum_reproducible": checksum_reproducible,
        "identity_reproducible": identity_reproducible,
        "referenced_tamper_rejected": tamper["referenced_tamper_rejected"],
        "referenced_missing_rejected": tamper["referenced_missing_rejected"],
        "manifest_tamper_rejected": tamper["manifest_tamper_rejected"],
        "gaussian_persisted_preview_ready": bool(gaussian_presentation and gaussian_presentation.ready),
        "nerf_persisted_preview_ready": bool(nerf_presentation and nerf_presentation.ready),
        "production_isolation": not any(heavy.values()),
        "semantic_context_explicitly_not_requalified": bool(semantic_snapshot.get("validation_context_only") is True and semantic_snapshot.get("real_south_building_semantic_requalification_performed") is False),
        "inspection_context_explicitly_not_requalified": bool(inspection_snapshot.get("validation_context_only") is True and inspection_snapshot.get("real_south_building_inspection_requalification_performed") is False),
    }

    blockers = sorted(key for key, value in checks.items() if not value)
    passed = not blockers and repository["pass"]

    summary = {
        "phase": PHASE,
        "pass": passed,
        "blockers": blockers,
        "production_code_changed_by_8_9f5": False,
        "new_persistence_format_created": False,
        "heavy_backends_started": False,
        "real_execution": {
            "f2_real_canonical_reused": True,
            "f3_real_gaussian_reused": True,
            "f4_real_nerfacto_reused": True,
            "pycolmap_started": False,
            "da3_started": False,
            "gaussian_training_started": False,
            "nerfacto_training_started": False,
        },
        "authority": {
            "canonical_geometry": "authoritative",
            "gaussian": "appearance_only",
            "nerf": "appearance_only",
            "semantic_context": "contract_context_only_not_requalified_in_f5",
            "inspection_qc": "contract_context_only_not_requalified_in_f5",
            "certified_metrology": False,
        },
        "source_chain": {
            "checks": source["checks"],
            "source_summary_sha256": source["source_summary_sha256"],
            "f2_canonical_surface_vertex_count": materialized["canonical"]["surface_vertex_count"],
            "f2_canonical_surface_face_count": materialized["canonical"]["surface_face_count"],
            "metric_geometry_status": materialized["canonical"]["metric_geometry_status"],
            "unit_to_metre": materialized["canonical"]["unit_to_metre"],
        },
        "scene": {
            "manifest": str(scene_a),
            "checksum": str(scene_a.parent / PERSISTENT_RECONSTRUCTION_SCENE_CHECKSUM),
            "scene_id": loaded_a.scene_id,
            "scene_fingerprint_sha256": loaded_a.scene_fingerprint_sha256,
            "state": loaded_a.state.value,
            "pipeline_status": loaded_a.pipeline_status.value,
            "artifact_count": len(loaded_a.artifacts),
            "provider_count": len(loaded_a.provider_provenance),
            "file_reference_count": len(refs),
            "file_references": [item.as_dict() for item in refs],
            "all_file_references_under_f5_artifact_root": references_self_contained,
            "canonical_geometry_artifact_id": loaded_a.canonical_geometry_artifact_id,
            "canonical_geometry_authoritative": loaded_a.canonical_geometry_authoritative,
            "gaussian_artifact_id": loaded_a.gaussian_artifact.artifact_id if loaded_a.gaussian_artifact else "",
            "gaussian_authority": loaded_a.gaussian_artifact.authority.value if loaded_a.gaussian_artifact else "missing",
            "nerf_artifact_id": loaded_a.nerf_artifact.artifact_id if loaded_a.nerf_artifact else "",
            "nerf_authority": loaded_a.nerf_artifact.authority.value if loaded_a.nerf_artifact else "missing",
            "certified_metrology": loaded_a.certified_metrology,
        },
        "materialized_artifacts": materialized,
        "integrity": {
            "byte_reproducible": byte_reproducible,
            "checksum_reproducible": checksum_reproducible,
            "identity_reproducible": identity_reproducible,
            **tamper,
        },
        "presentation_readiness": {
            "gaussian": gaussian_presentation.__dict__ if gaussian_presentation is not None and hasattr(gaussian_presentation, "__dict__") else {
                "ready": bool(gaussian_presentation and gaussian_presentation.ready),
                "preview_path": gaussian_presentation.preview_path if gaussian_presentation else "",
                "state_text": gaussian_presentation.state_text if gaussian_presentation else "",
            },
            "nerf": nerf_presentation.__dict__ if nerf_presentation is not None and hasattr(nerf_presentation, "__dict__") else {
                "ready": bool(nerf_presentation and nerf_presentation.ready),
                "preview_path": nerf_presentation.preview_path if nerf_presentation else "",
                "state_text": nerf_presentation.state_text if nerf_presentation else "",
            },
        },
        "isolation": {
            "heavy_module_importability_in_depth": heavy,
            "production_heavy_runtime_leak": any(heavy.values()),
        },
        "truthful_scope": {
            "persistent_scene_contains_real_f2_canonical_artifact": True,
            "persistent_scene_contains_real_f3_gaussian_artifact": True,
            "persistent_scene_contains_real_f4_nerfacto_artifact": True,
            "semantic_qc_requalified_on_south_building_in_f5": False,
            "semantic_qc_records_are_contract_context_only": True,
            "gaussian_or_nerf_promoted_to_geometry_authority": False,
            "new_reconstruction_algorithm_created": False,
            "statement": (
                "F5 persisted self-contained copies of the already-qualified real F2 canonical surface, "
                "real F3 gsplat scene, and real F4 Nerfacto checkpoint/config through the existing "
                "PersistentReconstructionScene contract. The schema-required semantic/QC stages are "
                "explicit contract-only context and are not claimed as new real South Building semantic/QC qualification."
            ),
        },
        "checks": checks,
        "repository": repository,
        "next_step_if_pass": "8.9F6_actual_application_end_to_end_acceptance",
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    print("Phase 8.9F5 persistent multi-artifact scene qualification: " + ("PASS" if passed else "FAIL"))
    print(
        f"scene={loaded_a.scene_id} artifacts={len(loaded_a.artifacts)} refs={len(refs)} "
        f"fingerprint={loaded_a.scene_fingerprint_sha256[:16]}..."
    )
    print(
        f"canonical_authoritative={loaded_a.canonical_geometry_authoritative} "
        f"gaussian_authority={loaded_a.gaussian_artifact.authority.value if loaded_a.gaussian_artifact else 'missing'} "
        f"nerf_authority={loaded_a.nerf_artifact.authority.value if loaded_a.nerf_artifact else 'missing'} "
        f"certified_metrology={loaded_a.certified_metrology}"
    )
    print(
        f"reproducible={byte_reproducible and checksum_reproducible and identity_reproducible} "
        f"tamper_rejected={tamper['referenced_tamper_rejected']} "
        f"missing_rejected={tamper['referenced_missing_rejected']} "
        f"manifest_tamper_rejected={tamper['manifest_tamper_rejected']}"
    )
    print(
        f"gaussian_preview_ready={bool(gaussian_presentation and gaussian_presentation.ready)} "
        f"nerf_preview_ready={bool(nerf_presentation and nerf_presentation.ready)} "
        f"heavy_backends_started=False"
    )
    print("semantic_qc_requalified_on_south_building_in_f5=False contract_context_only=True")
    print(f"manifest={scene_a}")
    print(f"summary={summary_path}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
