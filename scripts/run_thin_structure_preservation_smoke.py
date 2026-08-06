#!/usr/bin/env python3
"""Visual smoke test for thin-structure support preservation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.object_geometry import ObjectPointExtractionRequest, extract_object_point_clouds
from tests.object_geometry_helpers import make_bundle


def _scene() -> tuple[np.ndarray, np.ndarray]:
    height = width = 180
    mask = np.zeros((height, width), dtype=bool)
    mask[18:78, 48:132] = True
    mask[78:104, 58:122] = True
    mask[104:166, 60:70] = True
    mask[104:166, 110:120] = True
    mask[62:118, 38:50] = True
    mask[62:118, 130:142] = True

    depth = np.full((height, width), 5.0, dtype=np.float32)
    depth[18:78, 48:132] = 3.0
    depth[78:104, 58:122] = 2.75
    depth[104:166, 60:70] = 2.80
    depth[104:166, 110:120] = 2.85
    depth[62:118, 38:50] = 2.90
    depth[62:118, 130:142] = 2.92
    depth[84:96, 84:96] = 4.6
    return mask, depth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/phase7_5_thin_structure_preservation.png"),
    )
    args = parser.parse_args()

    mask, depth = _scene()
    bundle = make_bundle(width=180, height=180, masks=(mask,), depth=depth)
    item = extract_object_point_clouds(ObjectPointExtractionRequest(bundle)).object_results[0]
    if item.raw_cloud is None or item.filtered_cloud is None:
        raise RuntimeError("Synthetic extraction did not produce point clouds.")

    height, width = mask.shape
    panel_mask = np.zeros((height, width, 3), dtype=np.uint8)
    panel_mask[mask] = (80, 180, 80)
    panel_raw = np.zeros_like(panel_mask)
    raw_pixels = item.raw_cloud.source_pixel_indices_xy
    panel_raw[raw_pixels[:, 1], raw_pixels[:, 0]] = (210, 150, 70)
    panel_filtered = np.zeros_like(panel_mask)
    filtered_pixels = item.filtered_cloud.source_pixel_indices_xy
    panel_filtered[filtered_pixels[:, 1], filtered_pixels[:, 0]] = (80, 220, 120)

    for panel, title in (
        (panel_mask, "SEGMENTATION MASK"),
        (panel_raw, "RAW DEPTH SUPPORT"),
        (panel_filtered, "FILTERED + PRESERVED"),
    ):
        cv2.putText(panel, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)

    canvas = np.hstack((panel_mask, panel_raw, panel_filtered))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Could not write {args.output}")

    contamination = (
        (filtered_pixels[:, 0] >= 84)
        & (filtered_pixels[:, 0] < 96)
        & (filtered_pixels[:, 1] >= 84)
        & (filtered_pixels[:, 1] < 96)
    )
    print(
        "status=success "
        f"raw={len(raw_pixels)} filtered={len(filtered_pixels)} "
        f"vertical_min={int(np.min(filtered_pixels[:, 1]))} "
        f"vertical_max={int(np.max(filtered_pixels[:, 1]))} "
        f"contamination={int(np.count_nonzero(contamination))} "
        f"warnings={','.join(item.warnings) or 'none'} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
