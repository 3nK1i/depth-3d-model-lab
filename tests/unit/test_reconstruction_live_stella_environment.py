from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from core.reconstruction import (
    CameraDistortionModel,
    LiveSlamRequest,
    ReconstructionValidationError,
    STELLA_LIVE_BACKEND_ID,
    STELLA_QUALIFIED_COMMIT,
    STELLA_QUALIFIED_FBOW_COMMIT,
    probe_stella_live_environment,
    stella_monocular_config_text,
)
from tests.reconstruction_live_helpers import make_calibrated_live_camera, make_live_request


def _make_probe_layout(tmp_path: Path) -> tuple[Path, Path, Path]:
    prefix = tmp_path / "stella"
    (prefix / "lib").mkdir(parents=True)
    (prefix / "lib" / "libstella_vslam.so").write_bytes(b"probe")
    (prefix / "DEPTH3D_STELLA_PROBE_INFO.txt").write_text(
        "\n".join(
            [
                "probe_status=PASS",
                "stella_vslam_version=0.3.13",
                f"stella_vslam_commit={STELLA_QUALIFIED_COMMIT}",
                f"stella_FBoW_submodule_commit={STELLA_QUALIFIED_FBOW_COMMIT}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker.chmod(0o755)
    vocab = tmp_path / "orb_vocab.fbow"
    vocab.write_bytes(b"vocab")
    return prefix, worker, vocab


def test_stella_environment_probe_accepts_exact_qualified_isolated_layout(tmp_path: Path) -> None:
    prefix, worker, vocab = _make_probe_layout(tmp_path)
    probe = probe_stella_live_environment(prefix, worker, vocab)
    assert probe.available is True
    assert probe.stella_version == "0.3.13"
    assert probe.stella_commit == STELLA_QUALIFIED_COMMIT
    assert probe.fbow_commit == STELLA_QUALIFIED_FBOW_COMMIT


def test_stella_environment_probe_rejects_unqualified_commit(tmp_path: Path) -> None:
    prefix, worker, vocab = _make_probe_layout(tmp_path)
    metadata = prefix / "DEPTH3D_STELLA_PROBE_INFO.txt"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace(STELLA_QUALIFIED_COMMIT, "bad-commit"),
        encoding="utf-8",
    )
    probe = probe_stella_live_environment(prefix, worker, vocab)
    assert probe.available is False
    assert "commit" in probe.reason


def test_generated_stella_config_preserves_calibration_and_disables_loop_detector() -> None:
    request = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    text = stella_monocular_config_text(request)
    assert "setup: monocular" in text
    assert "model: perspective" in text
    assert "fx: 250" in text
    assert "fy: 251" in text
    assert "cols: 320" in text
    assert "rows: 240" in text
    assert "color_order: BGR" in text
    assert "LoopDetector:\n  enabled: false" in text
    assert "unit_to_metre" not in text


def test_generated_stella_config_rejects_silent_high_order_brown_truncation() -> None:
    camera = replace(
        make_calibrated_live_camera(),
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        distortion_coefficients=(0.1, -0.1, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0),
    )
    request = replace(
        make_live_request(backend_id=STELLA_LIVE_BACKEND_ID),
        camera=camera,
    )
    with pytest.raises(ReconstructionValidationError, match="4 or 5 coefficients"):
        stella_monocular_config_text(request)


def test_generated_stella_config_requires_explicit_resolved_fps() -> None:
    base = make_live_request(backend_id=STELLA_LIVE_BACKEND_ID)
    source = replace(base.source, expected_frame_rate_hz=None)
    request = LiveSlamRequest(
        session_id=base.session_id,
        source=source,
        camera=base.camera,
        profile=base.profile,
        backend_id=base.backend_id,
    )
    with pytest.raises(ReconstructionValidationError, match="resolved frame rate"):
        stella_monocular_config_text(request)
