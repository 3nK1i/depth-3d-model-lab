from __future__ import annotations

import json
from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_export_creates_professional_package(tmp_path: Path) -> None:
    bundle = write_inspection_bundle(tmp_path / "bundle", manual=True)
    summary, arrays = load_inspection_bundle(bundle)
    result = export_inspection_bundle(
        summary,
        arrays,
        InspectionExportOptions(
            output_dir=tmp_path / "export",
            observation_ids=("object-0",),
            locked_ids=("object-0",),
        ),
    )
    assert result.object_count == 1
    for relative in (
        "report.html", "report.pdf", "measurements.json", "measurements.csv",
        "manifest.json", "annotated_rgb.png", "depth_preview.png",
        "objects/object_001_box/automatic_mask.png",
        "objects/object_001_box/approved_mask.png",
        "objects/object_001_box/raw_points.ply",
        "objects/object_001_box/filtered_points.ply",
        "objects/object_001_box/boxes.json",
    ):
        assert (result.export_dir / relative).is_file(), relative
    payload = json.loads((result.export_dir / "measurements.json").read_text())
    assert payload["certified_metrology"] is False
    assert payload["objects"][0]["locked_in_session"] is True
    assert payload["objects"][0]["mask_source"] == "manual_correction"
