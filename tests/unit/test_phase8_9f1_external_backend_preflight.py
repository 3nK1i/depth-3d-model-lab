from pathlib import Path
import importlib.util


def _load_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_phase8_9f1_external_backend_preflight.py"
    spec = importlib.util.spec_from_file_location("phase8_9f1_preflight", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_f1_repo_overlay_filter_only_accepts_known_f1_paths(tmp_path: Path) -> None:
    module = _load_module()
    assert "scripts/run_phase8_9f1_external_backend_preflight.py" in module.F1_REPO_FILES
    assert "core/reconstruction/da3.py" not in module.F1_REPO_FILES


def test_f1_source_contracts_preserve_authority_and_reveal_validation_boundary() -> None:
    module = _load_module()
    contracts = module._source_contracts(Path(__file__).resolve().parents[2])
    assert contracts["da3_accepts_external_sfm_poses"] is False
    assert contracts["da3_worker_generates_pose_output"] is True
    assert contracts["da3_supporting_geometry_authority"] is True
    assert contracts["da3_canonical_bridge_exists"] is True
    assert contracts["appearance_dataset_materialized_from_ingestion_and_trajectory"] is True
    assert contracts["current_real_f4_script_uses_deterministic_fixture"] is True
    assert contracts["persistent_scene_tracks_file_references"] is True
    assert contracts["ui_switching_is_persisted_preview_only"] is True


def test_f1_direct_heavy_import_scan_is_clean() -> None:
    module = _load_module()
    findings = module._direct_heavy_imports(Path(__file__).resolve().parents[2])
    assert findings == []


def test_f1_dataset_inventory_is_read_only_and_counts_supported_images(tmp_path: Path) -> None:
    module = _load_module()
    for name in ("a.jpg", "b.png", "c.txt"):
        (tmp_path / name).write_bytes(b"x")
    result = module._dataset_inventory(tmp_path)
    assert result["image_count"] == 2
    assert result["da3_ready"] is True
    assert result["sfm_ready"] is False


def test_f1_checkout_allows_first_porcelain_line_after_runner_whitespace_trim(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()

    monkeypatch.setattr(module, "_git_revision", lambda _path: "abc123")
    monkeypatch.setattr(
        module,
        "_git_status",
        lambda _path: [
            "M src/depth_anything_3/api.py",
            "?? .depth3d_da3_runtime_lock.json",
        ],
    )

    result = module._checkout(
        checkout,
        expected_revision="abc123",
        allow_status_prefixes=(
            " M src/depth_anything_3/api.py",
            "?? .depth3d_da3_runtime_lock.json",
        ),
    )

    assert result["revision_match"] is True
    assert result["unexpected_status"] == []


def test_f1_checkout_still_rejects_unrelated_drift_after_normalization(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").mkdir()

    monkeypatch.setattr(module, "_git_revision", lambda _path: "abc123")
    monkeypatch.setattr(
        module,
        "_git_status",
        lambda _path: [
            "M src/depth_anything_3/api.py",
            "?? unexpected.txt",
        ],
    )

    result = module._checkout(
        checkout,
        expected_revision="abc123",
        allow_status_prefixes=(" M src/depth_anything_3/api.py",),
    )

    assert result["unexpected_status"] == ["?? unexpected.txt"]
