from __future__ import annotations

import inspect
from pathlib import Path

from core.reconstruction import (
    CAMERA_GEOMETRY_SCHEMA_VERSION,
    DENSE_FUSION_SCHEMA_VERSION,
    LIVE_TRAJECTORY_SCHEMA_VERSION,
    SCALE_REFERENCE_SCHEMA_VERSION,
    CameraIntrinsicsPolicy,
    DenseFusionBackend,
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
)


def test_prior_phase_schema_versions_remain_unchanged() -> None:
    assert CAMERA_GEOMETRY_SCHEMA_VERSION == 1
    assert SCALE_REFERENCE_SCHEMA_VERSION == 1
    assert LIVE_TRAJECTORY_SCHEMA_VERSION == 2
    assert DENSE_FUSION_SCHEMA_VERSION == 1


def test_raw_reconstruction_gauge_default_remains_arbitrary() -> None:
    gauge = ReconstructionGauge()
    assert gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert gauge.alignment_status is SfmWorldAlignmentStatus.ARBITRARY
    assert gauge.unit_to_metre is None


def test_existing_live_slam_intrinsics_policy_default_semantics_remain_strict() -> None:
    assert CameraIntrinsicsPolicy.CALIBRATED_ONLY.value == "calibrated_only"
    assert CameraIntrinsicsPolicy.ALLOW_ESTIMATED.value == "allow_estimated"


def test_existing_dense_backend_protocol_signature_is_not_rewritten() -> None:
    signature = inspect.signature(DenseFusionBackend.integrate_keyframe)
    assert tuple(signature.parameters) == ("self", "keyframe", "context")


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_6c_does_not_import_native_stella_or_pycolmap() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source
