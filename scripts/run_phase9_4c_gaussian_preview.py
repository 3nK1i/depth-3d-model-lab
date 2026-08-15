#!/usr/bin/env python3
"""Create/show a backend-free reference-vs-gsplat preview for Phase 9.4C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import read_neural_scene_dataset_manifest

DEFAULT_SUMMARY = Path("/tmp/phase9_4c_real_gaussian_optimization/phase9_4c_real_optimization_summary.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    summary_path = args.summary.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise SystemExit("Phase 9.4C preview requires a PASS optimization summary.")
    dataset = read_neural_scene_dataset_manifest(Path(str(payload["dataset_manifest"])))
    reference = cv2.imread(dataset.frames[0].image_path, cv2.IMREAD_COLOR)
    rendered = cv2.imread(str(payload["preview_path"]), cv2.IMREAD_COLOR)
    if reference is None or rendered is None:
        raise SystemExit("Reference or gsplat preview image could not be decoded.")
    if rendered.shape[:2] != reference.shape[:2]:
        rendered = cv2.resize(rendered, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)
    gap = np.full((reference.shape[0], 16, 3), 24, dtype=np.uint8)
    combined = np.concatenate([reference, gap, rendered], axis=1)
    cv2.putText(combined, "Reference", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    offset = reference.shape[1] + gap.shape[1] + 8
    cv2.putText(combined, "gsplat optimized preview", (offset, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    out = Path("/tmp/phase9_4c_gaussian_preview.png").resolve()
    cv2.imwrite(str(out), combined)
    print(f"Phase 9.4C Gaussian preview: {out}")
    print(
        f"steps={payload['training_steps']} gaussians={payload['gaussian_count']} "
        f"peak_vram_mib={float(payload['peak_vram_mib']):.1f} device={payload['gpu_name']}"
    )
    print("Gaussian is appearance-only; canonical geometry/metric/measurement/QC authority is unchanged.")
    print("No Gaussian backend was started by this preview.")
    if args.show:
        from PyQt6.QtGui import QPixmap
        from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea
        app = QApplication.instance() or QApplication(sys.argv)
        label = QLabel()
        label.setPixmap(QPixmap(str(out)))
        area = QScrollArea()
        area.setWindowTitle("Phase 9.4C — Reference vs gsplat optimized preview")
        area.setWidget(label)
        area.setWidgetResizable(False)
        area.resize(min(combined.shape[1] + 40, 1300), min(combined.shape[0] + 60, 760))
        area.show()
        app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
