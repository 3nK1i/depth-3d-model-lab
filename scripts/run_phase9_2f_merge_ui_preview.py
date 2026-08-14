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

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app.dialogs.auto_segment_workspace_dialog import AutoSegmentWorkspaceDialog
from app.settings import APP_NAME, ORG_NAME
from app.theme import apply_theme
from core.perception.semantic_project import SegmentationProjectStore, add_project_class, create_segmentation_project


def main() -> int:
    QApplication.setOrganizationName(ORG_NAME)
    QApplication.setApplicationName(APP_NAME)
    app = QApplication(sys.argv)
    apply_theme(app)
    temp = Path(tempfile.mkdtemp(prefix="eia_phase9_2f_preview_"))
    store = SegmentationProjectStore(temp / "projects")
    project = create_segmentation_project("Factory Semantics", project_id="phase9-2f-preview")
    project, _ = add_project_class(project, "plant")
    project, _ = add_project_class(project, "table")
    store.save(project)

    image = np.zeros((540, 900, 3), dtype=np.uint8)
    image[:] = (30, 31, 34)
    image[80:330, 100:300] = (70, 115, 75)   # plant/leaves region
    image[310:470, 145:275] = (80, 90, 120)  # pot region
    image[180:430, 420:760] = (75, 95, 110)  # table region
    dialog = AutoSegmentWorkspaceDialog(project_store=store, initial_bgr=image, initial_source_label="Plant + pot merge preview")

    result = temp / "result"; result.mkdir()
    masks = []
    m0 = np.zeros((540, 900), dtype=bool); m0[70:350, 90:315] = True; masks.append(m0)
    m1 = np.zeros((540, 900), dtype=bool); m1[305:485, 135:285] = True; masks.append(m1)
    m2 = np.zeros((540, 900), dtype=bool); m2[175:440, 410:770] = True; masks.append(m2)
    np.savez_compressed(result / "masks.npz", **{f"mask_{i:04d}": mask for i, mask in enumerate(masks)})
    (result / "manifest.json").write_text(json.dumps({
        "model_id": "development-perception-mock",
        "proposals": [
            {"object_id": f"proposal:{i}", "mask_id": f"preview-mask-{i}", "mask_key": f"mask_{i:04d}", "confidence": None, "confidence_kind": "none"}
            for i in range(3)
        ],
    }), encoding="utf-8")
    dialog.load_job_result(result)
    dialog.select_object("proposal:0")
    plant_index = next(i for i in range(dialog._class_combo.count()) if dialog._class_combo.itemText(i) == "plant")
    dialog._class_combo.setCurrentIndex(plant_index)
    dialog._assign.click()
    for object_id in ("proposal:0", "proposal:1"):
        for row in range(dialog._object_list.count()):
            item = dialog._object_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == object_id:
                item.setCheckState(Qt.CheckState.Checked)
                break
    dialog.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
