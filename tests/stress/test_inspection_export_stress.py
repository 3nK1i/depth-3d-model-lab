from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_repeated_exports_are_isolated(tmp_path: Path) -> None:
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    for index in range(5):
        result = export_inspection_bundle(summary, arrays, InspectionExportOptions(tmp_path / f"export_{index}"))
        assert result.manifest_path.is_file()
