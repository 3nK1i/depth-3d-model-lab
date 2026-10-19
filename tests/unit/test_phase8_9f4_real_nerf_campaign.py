from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_phase8_9f4_real_nerf_campaign.py"


def _load():
    spec = importlib.util.spec_from_file_location("phase8_9f4_campaign", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_f4_scope_is_validation_only() -> None:
    module = _load()
    assert len(module.F4_REPO_FILES) == 5
    assert not any(path.startswith("core/") or path.startswith("app/") for path in module.F4_REPO_FILES)
    assert module.EXPECTED_HEAD == "0db3b0d92b84a538985e7932bfec9f65890cf6a0"


def test_f4_uses_qualified_nerfstudio_and_tcnn_pins() -> None:
    module = _load()
    assert module.EXPECTED_NERFSTUDIO_VERSION == "1.1.5"
    assert module.EXPECTED_NERFSTUDIO_REVISION == "6b60855003011b2ca23c2fe3f8e2ca6314c69924"
    assert module.EXPECTED_TCNN_REVISION == "b3473c81396fe927293bdfd5a6be32df8769927c"
    assert "depth3d-nerf-nerfstudio" in str(module.DEFAULT_NERF_PYTHON)


def test_even_selection_is_bounded_and_spans_real_trajectory() -> None:
    module = _load()
    indices = module._even_indices(80, 24)
    assert len(indices) == 24
    assert indices[0] == 0
    assert indices[-1] == 79
    assert tuple(sorted(indices)) == indices


def test_even_selection_rejects_too_few_registered_cameras() -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="at least 12"):
        module._even_indices(10, 24)


def test_simple_radial_intrinsics_are_derived_not_hardcoded() -> None:
    module = _load()
    fx, fy, cx, cy, distortion = module._camera_parameters({
        "backend_model_name": "SIMPLE_RADIAL",
        "parameters": [1200.0, 800.0, 600.0, -0.012],
    })
    assert (fx, fy, cx, cy) == (1200.0, 1200.0, 800.0, 600.0)
    assert np.allclose(distortion, [-0.012, 0.0, 0.0, 0.0, 0.0])


def test_unsupported_camera_model_fails_closed() -> None:
    module = _load()
    with pytest.raises(RuntimeError, match="does not silently reinterpret"):
        module._camera_parameters({"backend_model_name": "FISHEYE_UNKNOWN", "parameters": [1, 2, 3]})


def test_quality_command_reuses_existing_e3_runner_and_is_bounded(tmp_path: Path) -> None:
    module = _load()
    args = type("Args", (), {
        "training_steps": 1000,
        "nerf_python": Path("/isolated/nerf/python"),
        "metrics_python": Path("/isolated/metrics/python"),
        "nerfstudio_root": Path("/isolated/nerfstudio"),
        "tcnn_root": Path("/isolated/tcnn"),
        "timeout_s": 3600.0,
    })()
    bridge = {"dataset_manifest": str(tmp_path / "dataset.json"), "split_manifest": str(tmp_path / "split.json")}
    command = module._quality_command(args, tmp_path, bridge)
    joined = " ".join(command)
    assert "run_nerf_real_scene_validation.py" in joined
    assert "--training-steps 1000" in joined
    assert "--profile efficient" in joined
    assert "/isolated/nerf/python" in joined


def test_performance_command_reuses_existing_e4_runner_and_real_2060_profile(tmp_path: Path) -> None:
    module = _load()
    args = type("Args", (), {
        "training_steps": 1000,
        "render_repeats": 8,
        "nerf_python": Path("/isolated/nerf/python"),
        "nerfstudio_root": Path("/isolated/nerfstudio"),
        "tcnn_root": Path("/isolated/tcnn"),
        "timeout_s": 3600.0,
    })()
    bridge = {"dataset_manifest": str(tmp_path / "dataset.json"), "split_manifest": str(tmp_path / "split.json")}
    command = module._performance_command(args, tmp_path, bridge)
    joined = " ".join(command)
    assert "run_nerf_performance_hardening.py" in joined
    assert "--training-steps 1000" in joined
    assert "--render-repeats 8" in joined
    assert "--hardware-profile rtx_2060_6gb" in joined


def test_f4_authority_and_truthful_chain_are_explicit() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert '"nerf": "appearance_only"' in text
    assert '"canonical_geometry": "authoritative"' in text
    assert '"certified_metrology": False' in text
    assert '"gaussian_used_as_nerf_geometry_input": False' in text
    assert '"canonical_surface_used_as_nerf_geometry_input": False' in text
    assert '"new_quality_floor_invented": False' in text


def test_f4_isolation_probe_covers_nerf_and_other_heavy_backends() -> None:
    module = _load()
    assert "nerfstudio" in module.HEAVY_MODULES
    assert "tinycudann" in module.HEAVY_MODULES
    assert "gsplat" in module.HEAVY_MODULES
    assert "depth_anything_3" in module.HEAVY_MODULES
    assert "pycolmap" in module.HEAVY_MODULES


def test_build_sfm_bridge_materializes_registered_views_without_metric_claim(tmp_path: Path) -> None:
    module = _load()
    import cv2
    import json

    f2_root = tmp_path / "f2"
    f3_root = tmp_path / "f3"
    workspace = f2_root / "sfm/workspace"
    images = workspace / "images"
    images.mkdir(parents=True)
    f3_root.mkdir(parents=True)
    scene = f3_root / "scene.pt"
    scene.write_bytes(b"scene")

    cameras = []
    workspace_images = []
    for index in range(12):
        name = f"frame-{index:03d}.png"
        image = np.full((96, 128, 3), 32 + index, dtype=np.uint8)
        assert cv2.imwrite(str(images / name), image)
        keyframe_id = f"kf-{index:03d}"
        frame_id = f"frame-{index:03d}"
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = index * 0.05
        cameras.append({
            "camera_id": "pycolmap-camera-1",
            "keyframe_id": keyframe_id,
            "frame_id": frame_id,
            "T_reconstruction_from_camera": pose.tolist(),
        })
        workspace_images.append({
            "order": index,
            "keyframe_id": keyframe_id,
            "frame_id": frame_id,
            "timestamp_ns": index * 100_000_000,
            "staged_image_path": f"images/{name}",
        })

    f2_summary = {
        "phase": "8.9F2",
        "pass": True,
        "production_code_changed_by_8_9f2": False,
        "evidence": {
            "authority": {"canonical_geometry": "authoritative", "certified_metrology": False},
            "sfm": {"checks": {"real_sfm_executed": True}},
        },
    }
    (f2_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (f2_root / "summary.json").write_text(json.dumps(f2_summary), encoding="utf-8")
    report = {
        "sfm_result": {
            "session_id": "real-f2-session",
            "gauge": {"scale_status": "arbitrary", "unit_to_metre": None},
            "camera_models": [{
                "camera_id": "pycolmap-camera-1",
                "backend_model_name": "SIMPLE_RADIAL",
                "image_size": {"width": 128, "height": 96},
                "parameters": [100.0, 64.0, 48.0, 0.01],
            }],
            "cameras": cameras,
            "workspace": {"workspace_root": str(workspace), "images": workspace_images},
        }
    }
    report_path = f2_root / "sfm/real_sfm_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    f3_summary = {
        "phase": "8.9F3",
        "pass": True,
        "production_code_changed_by_8_9f3": False,
        "evidence": {
            "authority": {"gaussian": "appearance_only", "certified_metrology": False},
            "gaussian": {"scene_role": "appearance_only", "scene_path": str(scene)},
        },
    }
    (f3_root / "summary.json").write_text(json.dumps(f3_summary), encoding="utf-8")

    bridge = module._build_sfm_appearance_bridge(
        f2_root,
        f3_root,
        tmp_path / "out",
        selected_frames=12,
        heldout_frames=3,
        max_long_edge=720,
    )
    assert bridge["selected_frame_count"] == 12
    assert bridge["training_frame_count"] == 9
    assert bridge["validation_frame_count"] == 3
    assert bridge["metric_geometry_status"] == "arbitrary"
    assert bridge["unit_to_metre"] is None
    assert bridge["nerf_scene_role"] == "appearance_only"
    assert bridge["gaussian_used_as_nerf_geometry_input"] is False
    assert Path(bridge["dataset_manifest"]).is_file()
    assert Path(bridge["split_manifest"]).is_file()


def test_qualify_accepts_complete_existing_e3_e4_evidence(tmp_path: Path) -> None:
    module = _load()
    import json

    f2_root = tmp_path / "f2"
    f3_root = tmp_path / "f3"
    out = tmp_path / "out"
    (f2_root / "sfm").mkdir(parents=True)
    f3_root.mkdir(parents=True)
    (out / "quality").mkdir(parents=True)
    (out / "performance").mkdir(parents=True)
    scene = f3_root / "scene.pt"
    scene.write_bytes(b"gaussian")
    (f2_root / "sfm/real_sfm_report.json").write_text("{}", encoding="utf-8")
    (f2_root / "summary.json").write_text(json.dumps({
        "phase": "8.9F2", "pass": True, "production_code_changed_by_8_9f2": False,
        "evidence": {
            "authority": {"canonical_geometry": "authoritative", "certified_metrology": False},
            "sfm": {"checks": {"real_sfm_executed": True}},
        },
    }), encoding="utf-8")
    (f3_root / "summary.json").write_text(json.dumps({
        "phase": "8.9F3", "pass": True, "production_code_changed_by_8_9f3": False,
        "evidence": {
            "authority": {"gaussian": "appearance_only", "certified_metrology": False},
            "gaussian": {"scene_role": "appearance_only", "scene_path": str(scene)},
        },
    }), encoding="utf-8")
    (out / "bridge_summary.json").write_text(json.dumps({
        "sfm_pose_source": "real_phase8_9f2_pycolmap",
        "selected_frame_count": 24,
        "training_frame_count": 20,
        "validation_frame_count": 4,
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
    }), encoding="utf-8")
    contact = out / "quality/heldout_contact_sheet.png"
    contact.write_bytes(b"png")
    q_ckpt = out / "quality/q.ckpt"
    q_cfg = out / "quality/q.yml"
    q_ckpt.write_bytes(b"checkpoint")
    q_cfg.write_text("config", encoding="utf-8")
    (out / "quality/summary.json").write_text(json.dumps({
        "pass": True,
        "quality_sanity_pass": True,
        "training_frame_count": 20,
        "validation_frame_count": 4,
        "training_steps": 1000,
        "quality_policy": {"scope": "nondegenerate_real_scene_sanity_only"},
        "quality": {"aggregate": {"psnr_db": 10.0, "ssim": 0.2, "lpips": 0.8}},
        "artifact": {
            "scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "checkpoint_path": str(q_ckpt),
            "config_path": str(q_cfg),
            "backend_revision": module.EXPECTED_NERFSTUDIO_REVISION,
            "backend_version": module.EXPECTED_NERFSTUDIO_VERSION,
        },
        "contact_sheet_path": str(contact),
    }), encoding="utf-8")
    final_ckpt = out / "performance/final.ckpt"
    final_cfg = out / "performance/config.yml"
    final_ckpt.write_bytes(b"checkpoint")
    final_cfg.write_text("config", encoding="utf-8")
    (out / "performance/summary.json").write_text(json.dumps({
        "pass": True,
        "performance_metrics_complete": True,
        "artifact_hardening_complete": True,
        "report": {
            "hardware": {"measured": True, "gpu_name": "NVIDIA GeForce RTX 2060"},
            "performance": {
                "training_peak_vram_mib": 600.0,
                "render_peak_vram_mib": 700.0,
                "render_fps": 0.7,
            },
            "artifact_integrity": {
                "passed": True,
                "checkpoint_path": str(final_ckpt),
                "config_path": str(final_cfg),
                "checkpoint_reload_s": 2.0,
                "render_probe_passed": True,
                "missing_artifact_rejected": True,
                "malformed_checkpoint_rejected": True,
            },
        },
    }), encoding="utf-8")
    evidence = module._qualify(
        out,
        f2_root,
        f3_root,
        {"nerfstudio": {"ready": True}, "tiny_cuda_nn": {"ready": True}},
    )
    assert evidence["pass"] is True
    assert evidence["authority"]["nerf"] == "appearance_only"
    assert evidence["truthful_chain"]["gaussian_used_as_nerf_geometry_input"] is False


def test_qualify_respects_explicit_campaign_view_counts(tmp_path: Path) -> None:
    module = _load()
    import json

    f2_root = tmp_path / "f2"
    f3_root = tmp_path / "f3"
    out = tmp_path / "out"
    (f2_root / "sfm").mkdir(parents=True)
    f3_root.mkdir(parents=True)
    (out / "quality").mkdir(parents=True)
    (out / "performance").mkdir(parents=True)
    scene = f3_root / "scene.pt"
    scene.write_bytes(b"gaussian")
    (f2_root / "sfm/real_sfm_report.json").write_text("{}", encoding="utf-8")
    (f2_root / "summary.json").write_text(json.dumps({
        "phase": "8.9F2", "pass": True, "production_code_changed_by_8_9f2": False,
        "evidence": {
            "authority": {"canonical_geometry": "authoritative", "certified_metrology": False},
            "sfm": {"checks": {"real_sfm_executed": True}},
        },
    }), encoding="utf-8")
    (f3_root / "summary.json").write_text(json.dumps({
        "phase": "8.9F3", "pass": True, "production_code_changed_by_8_9f3": False,
        "evidence": {
            "authority": {"gaussian": "appearance_only", "certified_metrology": False},
            "gaussian": {"scene_role": "appearance_only", "scene_path": str(scene)},
        },
    }), encoding="utf-8")
    (out / "bridge_summary.json").write_text(json.dumps({
        "sfm_pose_source": "real_phase8_9f2_pycolmap",
        "selected_frame_count": 12,
        "training_frame_count": 9,
        "validation_frame_count": 3,
        "metric_geometry_status": "arbitrary",
        "unit_to_metre": None,
    }), encoding="utf-8")
    contact = out / "quality/heldout_contact_sheet.png"
    contact.write_bytes(b"png")
    q_ckpt = out / "quality/q.ckpt"
    q_cfg = out / "quality/q.yml"
    q_ckpt.write_bytes(b"checkpoint")
    q_cfg.write_text("config", encoding="utf-8")
    (out / "quality/summary.json").write_text(json.dumps({
        "pass": True,
        "quality_sanity_pass": True,
        "training_frame_count": 9,
        "validation_frame_count": 3,
        "training_steps": 1000,
        "quality_policy": {"scope": "nondegenerate_real_scene_sanity_only"},
        "quality": {"aggregate": {"psnr_db": 10.0, "ssim": 0.2, "lpips": 0.8}},
        "artifact": {
            "scene_role": "appearance_only",
            "canonical_geometry_authoritative": False,
            "certified_metrology": False,
            "checkpoint_path": str(q_ckpt),
            "config_path": str(q_cfg),
            "backend_revision": module.EXPECTED_NERFSTUDIO_REVISION,
            "backend_version": module.EXPECTED_NERFSTUDIO_VERSION,
        },
        "contact_sheet_path": str(contact),
    }), encoding="utf-8")
    final_ckpt = out / "performance/final.ckpt"
    final_cfg = out / "performance/config.yml"
    final_ckpt.write_bytes(b"checkpoint")
    final_cfg.write_text("config", encoding="utf-8")
    (out / "performance/summary.json").write_text(json.dumps({
        "pass": True,
        "performance_metrics_complete": True,
        "artifact_hardening_complete": True,
        "report": {
            "hardware": {"measured": True, "gpu_name": "NVIDIA GeForce RTX 2060"},
            "performance": {
                "training_peak_vram_mib": 600.0,
                "render_peak_vram_mib": 700.0,
                "render_fps": 0.7,
            },
            "artifact_integrity": {
                "passed": True,
                "checkpoint_path": str(final_ckpt),
                "config_path": str(final_cfg),
                "checkpoint_reload_s": 2.0,
                "render_probe_passed": True,
                "missing_artifact_rejected": True,
                "malformed_checkpoint_rejected": True,
            },
        },
    }), encoding="utf-8")
    evidence = module._qualify(
        out,
        f2_root,
        f3_root,
        {"nerfstudio": {"ready": True}, "tiny_cuda_nn": {"ready": True}},
        expected_selected_frames=12,
        expected_heldout_frames=3,
    )
    assert evidence["pass"] is True
    assert evidence["checks"]["bridge_selected_view_count"] is True
    assert evidence["checks"]["quality_counts"] is True
