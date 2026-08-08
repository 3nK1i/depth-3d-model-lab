from __future__ import annotations

import inspect
from pathlib import Path

from core.reconstruction import (
    ReconstructionGauge,
    SfmScaleStatus,
    SfmWorldAlignmentStatus,
    arbitrary_scale_reference,
)


def test_reconstruction_gauge_existing_default_semantics_are_unchanged() -> None:
    gauge = ReconstructionGauge()
    assert gauge.scale_status is SfmScaleStatus.ARBITRARY
    assert gauge.alignment_status is SfmWorldAlignmentStatus.ARBITRARY
    assert gauge.unit_to_metre is None
    assert gauge.scale_source == ""
    assert gauge.relative_scale_uncertainty is None


def test_existing_reconstruction_gauge_signature_is_unchanged() -> None:
    assert tuple(inspect.signature(ReconstructionGauge).parameters) == (
        "scale_status",
        "alignment_status",
        "unit_to_metre",
        "scale_source",
        "relative_scale_uncertainty",
    )


def test_phase8_6b_arbitrary_profile_preserves_existing_raw_gauge_contract() -> None:
    profile = arbitrary_scale_reference(profile_id="raw")
    assert profile.gauge.as_dict() == ReconstructionGauge().as_dict()


def test_reconstruction_ui_remains_disabled() -> None:
    text = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    enabled_block = text.split("_ENABLED_MODES", 1)[1].split("}", 1)[0]
    assert "RECONSTRUCTION" not in enabled_block


def test_phase8_6b_does_not_import_native_stella_or_pycolmap() -> None:
    for path in Path("core/reconstruction").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import pycolmap" not in source
        assert "from pycolmap" not in source
        assert "import stella_vslam" not in source
        assert "from stella_vslam" not in source


def test_phase8_6b_does_not_change_camera_geometry_schema_version() -> None:
    from core.reconstruction import CAMERA_GEOMETRY_SCHEMA_VERSION

    assert CAMERA_GEOMETRY_SCHEMA_VERSION == 1
