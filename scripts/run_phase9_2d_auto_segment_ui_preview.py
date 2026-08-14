#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt6.QtWidgets import QApplication

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from app.settings import APP_NAME, ORG_NAME
from app.theme import apply_theme
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project


def _image() -> np.ndarray:
    image = np.zeros((540, 900, 3), dtype=np.uint8)
    image[:] = (28, 31, 34)
    image[80:430, 80:410] = (70, 90, 105)
    image[120:380, 500:820] = (86, 68, 52)
    image[250:500, 330:690] = (50, 70, 60)
    return image


def _result(root: Path) -> Path:
    h, w = 540, 900
    masks = []
    m0 = np.zeros((h, w), dtype=bool); m0[90:420, 95:395] = True; masks.append(m0)
    m1 = np.zeros((h, w), dtype=bool); m1[130:365, 520:805] = True; masks.append(m1)
    m2 = np.zeros((h, w), dtype=bool); m2[275:490, 355:665] = True; masks.append(m2)
    np.savez_compressed(root / "masks.npz", **{f"mask_{i:04d}": value for i, value in enumerate(masks)})
    manifest = {
        "model_id": "ultralytics-sam2-prompted",
        "status": "success",
        "proposals": [
            {"object_id": f"proposal:{i}", "mask_id": f"preview-mask-{i}", "mask_key": f"mask_{i:04d}", "confidence": None, "confidence_kind": "none"}
            for i in range(len(masks))
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def main() -> int:
    QApplication.setOrganizationName(ORG_NAME)
    QApplication.setApplicationName(APP_NAME)
    app = QApplication(sys.argv)
    apply_theme(app)
    temp = Path(tempfile.mkdtemp(prefix="eia_phase9_2d_preview_"))
    store = SegmentationProjectStore(temp / "projects")
    project = create_segmentation_project("Factory Semantics", project_id="phase9-2d-preview")
    project, _ = add_project_class(project, "robot_arm")
    project, _ = add_project_class(project, "conveyor")
    project, _ = add_project_class(project, "pallet_station")
    store.save(project)
    dialog = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=_image(), initial_source_label="Preview image")
    result_dir = temp / "result"
    result_dir.mkdir()
    dialog.load_job_result(_result(result_dir))
    dialog.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
