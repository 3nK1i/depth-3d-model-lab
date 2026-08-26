#!/usr/bin/env python3
"""Quick I4.1.7 clean-capture preflight and contact-sheet generator."""
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

from core.reconstruction.errors import ReconstructionValidationError  # noqa: E402
from core.reconstruction.offline_dense_i4_decoupled_sampling_integration import probe_video_metadata  # noqa: E402


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/tmp/eia_i4_1_7_capture_preflight"))
    p.add_argument("--frames", type=int, default=12)
    args = p.parse_args()
    if args.frames < 6 or args.frames > 24:
        p.error("--frames must be in [6,24]")
    return args


def main() -> int:
    args = _args()
    video = args.video.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not video.is_file():
        raise ReconstructionValidationError(f"I4.1.7 capture video missing: {video}")
    if output.exists() and any(output.iterdir()):
        raise ReconstructionValidationError(f"I4.1.7 preflight output must be empty or absent: {output}")
    output.mkdir(parents=True, exist_ok=True)

    total, fps, duration = probe_video_metadata(video)
    indices = np.linspace(0, max(0, total - 1), num=int(args.frames), dtype=np.int64)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ReconstructionValidationError(f"Could not open video: {video}")
    tiles: list[np.ndarray] = []
    source_w = source_h = 0
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ReconstructionValidationError(f"Could not decode frame {int(index)}")
            source_h, source_w = frame.shape[:2]
            target_w = 320
            target_h = max(1, int(round(source_h * target_w / max(1, source_w))))
            tile = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"{int(index)}  {int(index)/fps:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            tiles.append(tile)
    finally:
        capture.release()
    columns = 4
    rows = int(np.ceil(len(tiles) / columns))
    th, tw = tiles[0].shape[:2]
    sheet = np.zeros((rows * th, columns * tw, 3), dtype=np.uint8)
    for ordinal, tile in enumerate(tiles):
        row, col = divmod(ordinal, columns)
        sheet[row*th:(row+1)*th, col*tw:(col+1)*tw] = tile
    sheet_path = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise ReconstructionValidationError("Could not write I4.1.7 contact sheet.")

    advisory = {
        "duration_20_to_90_seconds": bool(20.0 <= duration <= 90.0),
        "resolution_at_least_720p": bool(source_w >= 1280 and source_h >= 720),
        "fps_at_least_20": bool(fps >= 20.0),
    }
    summary = {
        "phase": "I4.1.7-capture-preflight",
        "status": "ready_for_manual_review",
        "video": str(video),
        "source_total_frames": int(total),
        "source_fps": float(fps),
        "source_duration_seconds": float(duration),
        "source_width": int(source_w),
        "source_height": int(source_h),
        "contact_sheet": str(sheet_path.resolve()),
        "sampled_source_indices": [int(v) for v in indices],
        "advisory": advisory,
        "manual_review": {
            "object_static": "required",
            "camera_moves_around_object": "required",
            "no_cuts": "required",
            "minimal_motion_blur": "required",
            "stable_exposure": "recommended",
            "object_remains_in_frame": "required",
        },
        "authority": {
            "automated_static_scene_certification": False,
            "production_authorization": False,
            "metric_scale_claimed": False,
            "mesh_authorization": False,
        },
    }
    summary_path = output / "capture_preflight.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print("I4.1.7 capture preflight: READY_FOR_MANUAL_REVIEW")
    print(f"duration={duration:.3f}s frames={total} fps={fps:.3f} resolution={source_w}x{source_h}")
    print(f"advisory_duration={advisory['duration_20_to_90_seconds']} advisory_720p={advisory['resolution_at_least_720p']} advisory_fps={advisory['fps_at_least_20']}")
    print("manual_review_required=object_static,camera_orbit,no_cuts,minimal_motion_blur,object_in_frame")
    print(f"contact_sheet={sheet_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
