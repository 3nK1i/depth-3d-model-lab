#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.perception import (
    SegmentationProjectStore,
    add_project_class,
    create_segmentation_project_from_model_names,
    rename_project_class,
)


def main() -> int:
    project = create_segmentation_project_from_model_names(
        "Factory Digital Twin",
        {0: "person", 1: "vehicle"},
        source_model_id="ultralytics-yolo-seg",
        project_id="phase9-2b-smoke-project",
    )
    project, robot = add_project_class(project, "robot_arm")
    project, pallet = add_project_class(project, "pallet")
    project = rename_project_class(project, pallet.class_id, "pallet_station")
    instance_name = project.suggest_instance_name(robot.class_id, existing_names=("robot_arm_001",))

    with tempfile.TemporaryDirectory(prefix="phase9_2b_") as temp_dir:
        store = SegmentationProjectStore(Path(temp_dir) / "projects")
        store.save(project)
        restored = store.load(project.project_id)
        payload = {
            "status": "PASS" if restored.as_dict() == project.as_dict() else "FAIL",
            "project_id": restored.project_id,
            "project_name": restored.project_name,
            "class_count": len(restored.class_catalog.classes),
            "robot_arm_class_id": robot.class_id,
            "suggested_instance_name": instance_name,
            "renamed_class": restored.class_by_id(pallet.class_id).class_name,
            "geometry_authority": restored.is_geometry_authority,
            "metric_authority": restored.is_metric_authority,
            "measurement_authority": restored.is_measurement_authority,
            "external_database_required": False,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
