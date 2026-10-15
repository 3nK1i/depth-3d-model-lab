#!/usr/bin/env python3
"""Dependency-light Step 8 inspection-bundle and rendering smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.inspection import load_inspection_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/phase7_5_step8_inspection.png")
    parser.add_argument("--bundle-dir", default="/tmp/phase7_5_step8_inspection_bundle")
    return parser


def _synthetic_frame(width: int = 640, height: int = 420) -> np.ndarray:
    image = np.full((height, width, 3), (34, 38, 44), dtype=np.uint8)
    cv2.rectangle(image, (0, int(height * 0.72)), (width, height), (70, 75, 80), -1)
    cv2.rectangle(image, (int(width * 0.25), int(height * 0.22)), (int(width * 0.75), int(height * 0.72)), (55, 145, 205), -1)
    cv2.rectangle(image, (int(width * 0.31), int(height * 0.31)), (int(width * 0.69), int(height * 0.67)), (40, 95, 165), -1)
    cv2.putText(image, "DEVELOPMENT MOCK INSPECTION", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return image


def _panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _render(bundle_dir: Path, output: Path) -> tuple[str, int]:
    summary, arrays = load_inspection_bundle(bundle_dir)
    rgb = cv2.cvtColor(arrays["aligned_rgb"], cv2.COLOR_RGB2BGR)
    depth = cv2.cvtColor(arrays["depth_preview"], cv2.COLOR_RGB2BGR)
    overlay = rgb.copy()
    if summary.objects:
        obj = summary.objects[0]
        mask = arrays[obj.mask_key]
        tint = np.zeros_like(overlay)
        tint[:] = (60, 200, 80)
        overlay[mask] = cv2.addWeighted(overlay[mask], 0.45, tint[mask], 0.55, 0)
        if obj.detection_box_xyxy:
            x0, y0, x1, y1 = (int(round(v)) for v in obj.detection_box_xyxy)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (60, 230, 100), 2)
    panel_w, panel_h = 560, 360
    canvas = np.full((650, 1180, 3), (13, 15, 18), dtype=np.uint8)
    canvas[70:70 + panel_h, 30:30 + panel_w] = _panel(overlay, panel_w, panel_h)
    canvas[70:70 + panel_h, 610:610 + panel_w] = _panel(depth, panel_w, panel_h)
    cv2.putText(canvas, "Phase 7.5 Step 8 - Measure & Inspect", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Aligned RGB + authoritative mask", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 185, 195), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Depth preview", (610, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 185, 195), 1, cv2.LINE_AA)
    lines = [
        f"status={summary.status.value} objects={len(summary.objects)}",
        f"depth={summary.depth_model_name} perception={summary.perception_model_name}",
        f"intrinsics={summary.intrinsics_source} calibration={summary.calibration_status}",
        "Estimated Uncertainty - Not Calibrated",
    ]
    if summary.objects:
        obj = summary.objects[0]
        lines += [
            f"selected={obj.class_name} mask={obj.mask_source} revision={obj.mask_revision}",
            f"extraction={obj.extraction_status} fitting={obj.fitting_status} measurements={obj.measurement_status}",
        ]
    y = 475
    for line in lines:
        cv2.putText(canvas, line, (32, y), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (215, 218, 224), 1, cv2.LINE_AA)
        y += 28
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write smoke visualization: {output}")
    return summary.status.value, len(summary.objects)


def main() -> int:
    args = build_parser().parse_args()
    output = Path(args.output).expanduser().resolve()
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="depth3d_step8_smoke_") as temporary:
        temp = Path(temporary)
        frame = temp / "frame.npy"
        config = temp / "depth.json"
        np.save(frame, _synthetic_frame())
        config.write_text(json.dumps({"device": "cpu", "precision": "fp32", "latency_ms": 0.0}), encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "scripts" / "run_measurement_inspection_job.py"),
            "--input", str(frame), "--output", str(bundle_dir),
            "--depth-model", "development-mock", "--depth-config", str(config),
            "--perception-model", "development-perception-mock", "--max-objects", "1",
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=60)
        if completed.returncode != 0:
            print(completed.stdout, end="")
            print(completed.stderr, end="", file=sys.stderr)
            return completed.returncode
    status, object_count = _render(bundle_dir, output)
    print(f"status=success bundle_status={status} objects={object_count} bundle={bundle_dir} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
