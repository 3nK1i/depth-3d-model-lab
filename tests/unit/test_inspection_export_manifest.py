from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.inspection import InspectionExportOptions, export_inspection_bundle, load_inspection_bundle
from tests.inspection_helpers import write_inspection_bundle


def test_manifest_hashes_every_exported_file(tmp_path: Path) -> None:
    summary, arrays = load_inspection_bundle(write_inspection_bundle(tmp_path / "bundle"))
    result = export_inspection_bundle(summary, arrays, InspectionExportOptions(tmp_path / "export"))
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["object_count"] == 1
    assert manifest["files"]
    for item in manifest["files"]:
        path = result.export_dir / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert all(item["path"] != "manifest.json" for item in manifest["files"])
