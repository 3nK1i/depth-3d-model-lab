from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_phase8_9f2_real_geometry_campaign.py"
    spec = importlib.util.spec_from_file_location("phase8_9f2_campaign", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=tmp_path / "images",
        sfm_python=Path("/isolated/pycolmap/bin/python"),
        max_keyframes=80,
        sfm_timeout_minutes=60.0,
        da3_max_images=3,
        da3_process_res=392,
        pixel_stride=4,
    )


def test_f2_scope_is_qualification_only() -> None:
    module = _load_module()
    assert len(module.F2_REPO_FILES) == 5
    assert all(not path.startswith("core/") and not path.startswith("app/") for path in module.F2_REPO_FILES)
    assert module.EXPECTED_HEAD == "57041c27a20c17a225b35ac50fed8528e8081b71"



def test_f2_da3_overlap_window_is_contiguous_and_centered(tmp_path: Path) -> None:
    module = _load_module()
    images = [tmp_path / f"frame_{index:03d}.jpg" for index in range(9)]
    selected = module._select_da3_overlap_window(images, 3)
    assert selected == images[3:6]


def test_f2_da3_overlap_window_materializes_only_selected_source_links(tmp_path: Path) -> None:
    module = _load_module()
    images = []
    for index in range(8):
        image = tmp_path / "images" / f"frame_{index:03d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"frame")
        images.append(image.resolve())
    out = tmp_path / "out"
    subset = module._materialize_da3_overlap_window(images, out, 3)
    links = sorted(subset.iterdir())
    assert len(links) == 3
    assert all(link.is_symlink() for link in links)
    assert [link.resolve() for link in links] == images[2:5]


def test_f2_commands_use_existing_real_backends_without_pose_injection(tmp_path: Path) -> None:
    module = _load_module()
    args = _args(tmp_path)
    out = tmp_path / "out"
    sfm = module._stage_command_sfm(args, out)
    overlap = tmp_path / "da3-overlap"
    da3 = module._stage_command_da3(args, out, image_dir=overlap)
    canonical = module._stage_command_canonical(args, out)

    assert "run_reconstruction_pycolmap_smoke.py" in " ".join(sfm)
    assert "--matching" in sfm and sfm[sfm.index("--matching") + 1] == "sequential"
    assert "--max-keyframes" in sfm and sfm[sfm.index("--max-keyframes") + 1] == "80"
    assert "run_da3_real_qualification.py" in " ".join(da3)
    assert da3[da3.index("--image-dir") + 1] == str(overlap)
    assert da3[da3.index("--max-images") + 1] == "3"
    assert da3[da3.index("--process-res") + 1] == "392"
    assert "run_da3_canonical_geometry_integration.py" in " ".join(canonical)
    joined = " ".join(da3 + canonical)
    assert "sfm_pose" not in joined
    assert "external_sfm" not in joined


def _write_fixture(root: Path, dataset: Path, *, metric=False, registration_ratio=1.0, landmarks=5000) -> None:
    (root / "sfm").mkdir(parents=True)
    (root / "da3/worker-output").mkdir(parents=True)
    (root / "canonical").mkdir(parents=True)
    dataset.mkdir(parents=True, exist_ok=True)
    selected = []
    for index in range(8):
        image = dataset / f"frame_{index:03d}.jpg"
        image.write_bytes(b"image")
        if index in (0, 3, 7):
            selected.append(str(image.resolve()))

    (root / "sfm/real_sparse_arbitrary_scale.ply").write_text("ply\n", encoding="ascii")
    (root / "sfm/real_sfm_summary.json").write_text(json.dumps({
        "real_sfm_executed": True,
        "input_keyframes": 80,
        "registered_keyframes": int(80 * registration_ratio),
        "registration_ratio": registration_ratio,
        "camera_models": 1,
        "landmarks": landmarks,
        "observations": 12000,
        "mean_reprojection_error_px": 0.7,
        "gauge": {"scale_status": "metric" if metric else "arbitrary", "unit_to_metre": 1.0 if metric else None},
        "pycolmap_version": "4.1.1",
    }), encoding="utf-8")
    (root / "da3/input_manifest.json").write_text(json.dumps({"request_id": "req", "image_paths": selected}), encoding="utf-8")
    (root / "da3/worker-output/result_manifest.json").write_text(json.dumps({"request_id": "req"}), encoding="utf-8")
    (root / "da3/summary.json").write_text(json.dumps({
        "pass": True,
        "model_id": "depth-anything/DA3-SMALL",
        "model_revision": "model-rev",
        "source_revision": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        "frame_count": 3,
        "inference_duration_s": 1.0,
        "peak_vram_mib": 4000.0,
        "gpu_name": "fixture",
        "pose_count": 3,
        "geometry_scale": "relative_arbitrary_scale",
        "canonical_geometry_authoritative": False,
        "certified_metrology": False,
    }), encoding="utf-8")
    surface = root / "canonical/da3_canonical_surface.obj"
    surface.write_text("v 0 0 0\n", encoding="ascii")
    (root / "canonical/summary.json").write_text(json.dumps({
        "pass": True,
        "pipeline_status": "completed",
        "canonical_geometry_authoritative": True,
        "da3_supporting_evidence_authoritative": False,
        "metric_geometry_status": "metric" if metric else "arbitrary",
        "unit_to_metre": 1.0 if metric else None,
        "certified_metrology": False,
        "surface_vertex_count": 300,
        "surface_face_count": 600,
        "surface_obj": str(surface),
        "canonical_geometry_artifact_id": "fixture:surface_mesh",
    }), encoding="utf-8")


def test_f2_evidence_qualification_preserves_authority_and_scale_truth(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    dataset = tmp_path / "images"
    out = tmp_path / "out"
    _write_fixture(out, dataset)
    monkeypatch.setattr(module, "_heavy_runtime_importability", lambda: {name: False for name in module.HEAVY_MODULES})
    payload = module._qualify_evidence(out, dataset, min_registration_ratio=0.75, min_landmarks=1000)
    assert payload["pass"] is True
    assert payload["authority"]["canonical_geometry"] == "authoritative"
    assert payload["authority"]["da3"] == "supporting_relative_geometry_until_canonicalized"
    assert payload["authority"]["certified_metrology"] is False
    assert payload["truthful_chain"]["literal_sfm_pose_to_da3_feed_supported"] is False
    assert payload["truthful_chain"]["sfm_fused_into_canonical_surface_in_f2"] is False


def test_f2_rejects_invented_metric_scale(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    dataset = tmp_path / "images"
    out = tmp_path / "out"
    _write_fixture(out, dataset, metric=True)
    monkeypatch.setattr(module, "_heavy_runtime_importability", lambda: {name: False for name in module.HEAVY_MODULES})
    payload = module._qualify_evidence(out, dataset, min_registration_ratio=0.75, min_landmarks=1000)
    assert payload["pass"] is False
    assert payload["sfm"]["checks"]["arbitrary_scale"] is False
    assert payload["canonical"]["checks"]["unit_to_metre_absent"] is False


def test_f2_rejects_weak_sfm_scene(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    dataset = tmp_path / "images"
    out = tmp_path / "out"
    _write_fixture(out, dataset, registration_ratio=0.50, landmarks=200)
    monkeypatch.setattr(module, "_heavy_runtime_importability", lambda: {name: False for name in module.HEAVY_MODULES})
    payload = module._qualify_evidence(out, dataset, min_registration_ratio=0.75, min_landmarks=1000)
    assert payload["pass"] is False
    assert payload["sfm"]["checks"]["registration_ratio"] is False
    assert payload["sfm"]["checks"]["landmarks"] is False
