from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from uuid import uuid4

import pytest

from core.reconstruction import (
    persistent_scene_from_pipeline_payload,
    read_persistent_reconstruction_scene,
    verify_persistent_scene_file_references,
    write_persistent_reconstruction_scene,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_phase8_9f5_persistent_multi_artifact_scene.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase8_9f5_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage(session_id: str, stage: str, kind: str, authority: str, provider: str) -> dict:
    return {
        "stage": stage,
        "state": "completed",
        "required": True,
        "provider_attempts": [provider],
        "provider_id": provider,
        "fallback_used": False,
        "reason": "",
        "artifact": {
            "schema_version": 1,
            "artifact_id": f"f2:{stage}:{provider}",
            "session_id": session_id,
            "stage": stage,
            "kind": kind,
            "authority": authority,
            "provider_id": provider,
            "payload_ref": f"memory://f2/{stage}",
            "canonical_geometry_authoritative": authority == "canonical_geometry",
            "certified_metrology": False,
            "metadata": {
                "real_f2_evidence": True,
                "canonical_geometry_authoritative": authority == "canonical_geometry",
                "certified_metrology": False,
            },
        },
    }


def _f2_pipeline() -> dict:
    session_id = str(uuid4())
    definitions = [
        ("source_ingestion", "ingestion_result", "input_evidence", "f2-input"),
        ("camera_pose", "camera_trajectory", "supporting_geometry", "f2-pycolmap"),
        ("depth_evidence", "depth_evidence", "supporting_geometry", "f2-da3"),
        ("dense_geometry", "dense_geometry", "canonical_geometry", "f2-dense"),
        ("surface_mesh", "surface_mesh", "canonical_geometry", "f2-surface"),
    ]
    stages = [_stage(session_id, *definition) for definition in definitions]
    return {
        "schema_version": 1,
        "request_id": "phase8-9f2-real",
        "session_id": session_id,
        "status": "completed",
        "execution_scope": "real_providers",
        "stages": stages,
        "provider_provenance": [
            {
                "provider_id": provider,
                "provider_version": "1",
                "execution": "external_process" if provider in {"f2-pycolmap", "f2-da3"} else "in_process",
                "stages": [stage],
            }
            for stage, _kind, _authority, provider in definitions
        ],
        "canonical_geometry_artifact_id": stages[-1]["artifact"]["artifact_id"],
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }


def _write_render_manifest(path: Path, image: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"renders": [{"rendered_image_path": str(image.resolve())}]}, indent=2),
        encoding="utf-8",
    )


def _source_tree(tmp_path: Path):
    f2 = tmp_path / "f2"
    f3 = tmp_path / "f3"
    f4 = tmp_path / "f4"
    (f2 / "canonical").mkdir(parents=True)
    (f3 / "benchmark/gsplat/worker-output/gsplat").mkdir(parents=True)
    (f4 / "quality/worker-output/nerfstudio-nerfacto-e3").mkdir(parents=True)
    (f4 / "performance/worker-output/nerfstudio-nerfacto-e4").mkdir(parents=True)

    surface = f2 / "canonical/da3_canonical_surface.obj"
    surface.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    (f2 / "canonical/pipeline_result.json").write_text(json.dumps(_f2_pipeline()), encoding="utf-8")

    g_dir = f3 / "benchmark/gsplat/worker-output/gsplat"
    g_scene = g_dir / "scene.pt"
    g_preview = g_dir / "render.png"
    g_scene.write_bytes(b"real-gsplat-scene")
    g_preview.write_bytes(b"png-gsplat-render")
    g_manifest = g_dir / "held_out_render_manifest.json"
    _write_render_manifest(g_manifest, g_preview)

    n_quality = f4 / "quality/worker-output/nerfstudio-nerfacto-e3"
    n_preview = n_quality / "render.png"
    n_preview.write_bytes(b"png-nerf-render")
    _write_render_manifest(n_quality / "heldout_render_manifest.json", n_preview)
    n_perf = f4 / "performance/worker-output/nerfstudio-nerfacto-e4"
    n_ckpt = n_perf / "step-000000999.ckpt"
    n_cfg = n_perf / "config.yml"
    n_ckpt.write_bytes(b"nerfacto-checkpoint")
    n_cfg.write_text("method: nerfacto\n", encoding="utf-8")

    f2_summary = {
        "phase": "8.9F2", "pass": True,
        "evidence": {
            "authority": {"canonical_geometry": "authoritative", "certified_metrology": False},
            "canonical": {
                "surface_obj": str(surface.resolve()),
                "surface_vertex_count": 3,
                "surface_face_count": 1,
                "metric_geometry_status": "arbitrary",
                "unit_to_metre": None,
            },
        },
    }
    f3_summary = {
        "phase": "8.9F3", "pass": True,
        "evidence": {
            "authority": {"canonical_geometry": "authoritative", "gaussian": "appearance_only", "certified_metrology": False},
            "gaussian": {
                "backend_kind": "gsplat_external",
                "scene_path": str(g_scene.resolve()),
                "render_manifest": str(g_manifest.resolve()),
                "scene_role": "appearance_only",
            },
        },
    }
    f4_summary = {
        "phase": "8.9F4", "pass": True,
        "evidence": {
            "authority": {
                "canonical_geometry": "authoritative",
                "gaussian": "appearance_only",
                "nerf": "appearance_only",
                "certified_metrology": False,
            },
            "bridge": {"selected_frame_count": 24},
            "nerf": {
                "backend_kind": "nerfstudio_nerfacto_external",
                "backend_version": "1.1.5",
                "method_name": "nerfacto",
                "training_steps": 1000,
                "final_checkpoint": str(n_ckpt.resolve()),
                "final_config": str(n_cfg.resolve()),
            },
        },
    }
    (f2 / "summary.json").write_text(json.dumps(f2_summary), encoding="utf-8")
    (f3 / "summary.json").write_text(json.dumps(f3_summary), encoding="utf-8")
    (f4 / "summary.json").write_text(json.dumps(f4_summary), encoding="utf-8")
    return f2, f3, f4


def test_f5_scope_is_validation_only() -> None:
    module = _load()
    assert len(module.F5_REPO_FILES) == 5
    assert not any(path.startswith("core/") or path.startswith("app/") for path in module.F5_REPO_FILES)
    assert module.EXPECTED_HEAD == "101537cd118f8b752a1b440bc88aa3b069ed75fc"


def test_f5_starts_no_heavy_backend_by_design() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"heavy_backends_started": False' in text
    assert '"gaussian_training_started": False' in text
    assert '"nerfacto_training_started": False' in text
    assert "Popen(" not in text


def test_source_chain_requires_locked_authority(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    result = module._validate_source_chain(f2, f3, f4)
    assert all(result["checks"].values())

    payload = json.loads((f4 / "summary.json").read_text())
    payload["evidence"]["authority"]["nerf"] = "canonical_geometry"
    (f4 / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-chain qualification failed"):
        module._validate_source_chain(f2, f3, f4)


def test_materialization_copies_real_artifacts_and_previews(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=tmp_path / "out"
    )
    assert Path(materialized["canonical"]["surface_obj_path"]).is_file()
    assert Path(materialized["gaussian"]["scene_path"]).is_file()
    assert Path(materialized["gaussian"]["preview_path"]).is_file()
    assert Path(materialized["nerf"]["checkpoint_path"]).is_file()
    assert Path(materialized["nerf"]["config_path"]).is_file()
    assert Path(materialized["nerf"]["preview_path"]).is_file()


def test_persistence_payload_keeps_real_f2_core_and_contract_only_semantic_qc(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=tmp_path / "out"
    )
    f2_pipeline = json.loads((f2 / "canonical/pipeline_result.json").read_text())
    payload, snapshots = module._build_persistence_payload(
        f2_pipeline=f2_pipeline, materialized=materialized, source=source
    )
    assert [stage["stage"] for stage in payload["stages"]] == [
        "source_ingestion", "camera_pose", "depth_evidence", "dense_geometry", "surface_mesh",
        "semantic_geometry", "inspection_qc", "gaussian_appearance", "nerf_appearance",
    ]
    assert payload["stages"][0]["artifact"]["metadata"]["real_f2_evidence"] is True
    assert payload["stages"][5]["artifact"]["metadata"]["validation_context_only"] is True
    assert payload["stages"][5]["artifact"]["metadata"]["real_south_building_semantic_requalification_performed"] is False
    assert payload["stages"][6]["artifact"]["metadata"]["real_south_building_inspection_requalification_performed"] is False
    assert payload["stages"][7]["artifact"]["authority"] == "appearance_only"
    assert payload["stages"][8]["artifact"]["authority"] == "appearance_only"
    assert snapshots["surface_mesh"]["unit_to_metre"] is None


def test_full_synthetic_f5_roundtrip_has_six_file_refs_and_preview_readiness(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    out = tmp_path / "out"
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=out
    )
    payload, snapshots = module._build_persistence_payload(
        f2_pipeline=json.loads((f2 / "canonical/pipeline_result.json").read_text()),
        materialized=materialized,
        source=source,
    )
    scene = persistent_scene_from_pipeline_payload(
        payload, payload_snapshots_by_kind=snapshots, created_at_utc="2026-08-12T14:30:00+00:00"
    )
    manifest = write_persistent_reconstruction_scene(scene, (out / "scene").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    refs = verify_persistent_scene_file_references(loaded)
    assert len(refs) == 6
    artifact_root = (out / "artifacts").resolve()
    assert all(Path(item.path).resolve().is_relative_to(artifact_root) for item in refs)
    assert loaded.canonical_geometry_authoritative is True
    assert loaded.gaussian_artifact is not None and loaded.gaussian_artifact.authority.value == "appearance_only"
    assert loaded.nerf_artifact is not None and loaded.nerf_artifact.authority.value == "appearance_only"
    assert loaded.certified_metrology is False
    assert module.build_gaussian_visualization_presentation(loaded).ready is True
    assert module.build_nerf_visualization_presentation(loaded).ready is True


def test_tamper_and_missing_reference_fail_closed(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    out = tmp_path / "out"
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=out
    )
    payload, snapshots = module._build_persistence_payload(
        f2_pipeline=json.loads((f2 / "canonical/pipeline_result.json").read_text()),
        materialized=materialized,
        source=source,
    )
    scene = persistent_scene_from_pipeline_payload(payload, payload_snapshots_by_kind=snapshots)
    manifest = write_persistent_reconstruction_scene(scene, (out / "scene").resolve())
    loaded = read_persistent_reconstruction_scene(manifest)
    result = module._tamper_rejection(loaded, manifest)
    assert all(result.values())
    verify_persistent_scene_file_references(loaded)
    read_persistent_reconstruction_scene(manifest)


def test_f5_truthful_scope_does_not_claim_real_semantic_qc() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"semantic_qc_requalified_on_south_building_in_f5": False' in text
    assert '"semantic_qc_records_are_contract_context_only": True' in text
    assert '"production_execution_claimed": False' in text
    assert '"new_persistence_format_created": False' in text


def test_f5_authority_is_never_promoted_by_appearance_snapshots(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=tmp_path / "out"
    )
    payload, snapshots = module._build_persistence_payload(
        f2_pipeline=json.loads((f2 / "canonical/pipeline_result.json").read_text()),
        materialized=materialized,
        source=source,
    )
    snapshots["gaussian_scene"]["canonical_geometry_authoritative"] = True
    snapshots["nerf_scene"]["canonical_geometry_authoritative"] = True
    scene = persistent_scene_from_pipeline_payload(payload, payload_snapshots_by_kind=snapshots)
    assert scene.gaussian_artifact is not None and scene.gaussian_artifact.canonical_geometry_authoritative is False
    assert scene.nerf_artifact is not None and scene.nerf_artifact.canonical_geometry_authoritative is False


def test_f5_isolation_probe_covers_all_heavy_backends() -> None:
    module = _load()
    assert {"depth_anything_3", "pycolmap", "gsplat", "nerfstudio", "tinycudann"}.issubset(module.HEAVY_MODULES)


def test_f5_requires_exact_real_f2_core_stage_order(tmp_path: Path) -> None:
    module = _load()
    f2, f3, f4 = _source_tree(tmp_path)
    source = module._validate_source_chain(f2, f3, f4)
    materialized = module._materialize_real_artifacts(
        source=source, f2_root=f2, f3_root=f3, f4_root=f4, output=tmp_path / "out"
    )
    pipeline = json.loads((f2 / "canonical/pipeline_result.json").read_text())
    pipeline["stages"] = pipeline["stages"][1:]
    with pytest.raises(RuntimeError, match="expected five real geometry stages"):
        module._build_persistence_payload(f2_pipeline=pipeline, materialized=materialized, source=source)
