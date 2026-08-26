from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core.reconstruction.errors import ReconstructionValidationError
import core.reconstruction.mvs_camera_bridge as bridge
from core.reconstruction.mvs_camera_bridge import (
    CameraBridgeInputs,
    CameraBridgeView,
    _pose_fingerprint,
    _require_clean_output,
    _write_derived_text_model,
    load_house_camera_bridge_inputs,
    run_pycolmap_bridge_worker,
)


def _pose(x: float = 0.0) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x
    return result


def _tiny_inputs(tmp_path: Path, *, point_id: int = 7) -> tuple[CameraBridgeInputs, Path]:
    native = tmp_path / "native"
    native.mkdir()
    (native / "cameras.txt").write_text(
        "# cameras\n1 SIMPLE_RADIAL 100 80 90 50 40 0.01\n",
        encoding="utf-8",
    )
    (native / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 global0.jpg\n10 20 7\n"
        "2 1 0 0 0 -1 0 0 1 global1.jpg\n11 21 7\n",
        encoding="utf-8",
    )
    (native / "points3D.txt").write_text(
        "# points\n7 0 0 3 1 2 3 0.5 1 0 2 0\n",
        encoding="utf-8",
    )
    global0 = tmp_path / "global0.jpg"
    global0.write_bytes(b"g0")
    global1 = tmp_path / "global1.jpg"
    global1.write_bytes(b"g1")
    extra = tmp_path / "candidate_00000002.jpg"
    extra.write_bytes(b"e")
    inputs = CameraBridgeInputs(
        i4_1_7_dir=tmp_path,
        i4_1_7_summary=tmp_path / "summary.json",
        orbit_diagnostics=tmp_path / "diag.npz",
        sfm_report=tmp_path / "sfm.json",
        native_sparse_model=native,
        localization_dir=tmp_path,
        selected_model_id=0,
        shared_camera_id=1,
        views=(
            CameraBridgeView(
                0,
                "global_sfm_anchor",
                "global0.jpg",
                str(global0),
                1,
                _pose(0.0),
                backend_image_id=1,
            ),
            CameraBridgeView(
                1,
                "global_sfm_anchor",
                "global1.jpg",
                str(global1),
                1,
                _pose(1.0),
                backend_image_id=2,
            ),
            CameraBridgeView(
                2,
                "localized_extra_observation",
                extra.name,
                str(extra),
                1,
                _pose(2.0),
                matched_point3d_ids=(point_id,),
                matched_feature_xy=((12.0, 22.0),),
            ),
        ),
        camera_model_payload={"backend_camera_id": 1, "model_name": "SIMPLE_RADIAL"},
        native_workspace_root=tmp_path,
    )
    return inputs, native


def test_missing_i4_1_7_workspace_fails_closed_without_reconstruction(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError, match="fails closed instead of re-running SfM"):
        load_house_camera_bridge_inputs(tmp_path / "missing")


def test_non_empty_output_workspace_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "a1"
    output.mkdir()
    marker = output / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="refusing to overwrite evidence"):
        _require_clean_output(output)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_non_finite_or_non_rigid_localized_pose_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "extra.jpg"
    image.write_bytes(b"x")
    bad = _pose()
    bad[0, 3] = np.nan
    with pytest.raises(ReconstructionValidationError):
        CameraBridgeView(
            40,
            "localized_extra_observation",
            image.name,
            str(image),
            1,
            bad,
            matched_point3d_ids=(7,),
            matched_feature_xy=((1.0, 2.0),),
        )


def test_duplicate_localized_landmark_observation_is_rejected(tmp_path: Path) -> None:
    image = tmp_path / "extra.jpg"
    image.write_bytes(b"x")
    with pytest.raises(ReconstructionValidationError, match="same 3D point twice"):
        CameraBridgeView(
            40,
            "localized_extra_observation",
            image.name,
            str(image),
            1,
            _pose(),
            matched_point3d_ids=(7, 7),
            matched_feature_xy=((1.0, 2.0), (3.0, 4.0)),
        )


def test_missing_native_landmark_reference_fails_before_model_is_accepted(tmp_path: Path) -> None:
    inputs, native = _tiny_inputs(tmp_path, point_id=999)
    old = bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS
    bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS = 2
    try:
        with pytest.raises(ReconstructionValidationError, match="missing native point3D 999"):
            _write_derived_text_model(
                native_text_dir=native,
                output_sparse_dir=tmp_path / "derived",
                inputs=inputs,
            )
    finally:
        bridge.MVS_CAMERA_BRIDGE_EXPECTED_GLOBAL_VIEWS = old


def test_pose_authority_fingerprint_is_deterministic_and_pose_sensitive(tmp_path: Path) -> None:
    inputs, _ = _tiny_inputs(tmp_path)
    first = _pose_fingerprint(inputs.views)
    second = _pose_fingerprint(tuple(reversed(inputs.views)))
    assert first == second

    changed = list(inputs.views)
    view = changed[-1]
    changed[-1] = CameraBridgeView(
        view.source_sequence_index,
        view.provenance,
        view.image_name,
        view.image_path,
        view.camera_id,
        _pose(2.001),
        backend_image_id=view.backend_image_id,
        matched_point3d_ids=view.matched_point3d_ids,
        matched_feature_xy=view.matched_feature_xy,
    )
    assert _pose_fingerprint(changed) != first


def test_external_worker_nonzero_exit_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stderr="worker rejected request", stdout=""),
    )
    with pytest.raises(ReconstructionValidationError, match="failed with code 2"):
        run_pycolmap_bridge_worker(
            python,
            worker,
            {"action": "validate"},
            request_path=tmp_path / "request.json",
            result_path=tmp_path / "result.json",
        )


def test_external_worker_timeout_is_normalized_to_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["worker"], timeout=0.01)

    monkeypatch.setattr(bridge.subprocess, "run", timeout)
    with pytest.raises(ReconstructionValidationError, match="timed out"):
        run_pycolmap_bridge_worker(
            python,
            worker,
            {"action": "validate"},
            request_path=tmp_path / "request.json",
            result_path=tmp_path / "result.json",
            timeout_s=0.01,
        )


def test_external_worker_success_without_result_json_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stderr="", stdout=""),
    )
    with pytest.raises(ReconstructionValidationError, match="did not emit its result JSON"):
        run_pycolmap_bridge_worker(
            python,
            worker,
            {"action": "validate"},
            request_path=tmp_path / "request.json",
            result_path=tmp_path / "result.json",
        )
