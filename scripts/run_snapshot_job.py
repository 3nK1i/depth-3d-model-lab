#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.registry import MODEL_REGISTRY
from core.runtime.messages import FrameRequest
from core.snapshot.export import export_snapshot_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    image = np.load(args.input, allow_pickle=False)
    model = MODEL_REGISTRY.build(args.model)
    try:
        model.load(str(config.get("device", "cpu")), str(config.get("precision", "fp32")), config)
        model.warmup()
        result = model.infer(FrameRequest(1, 1, image, point_budget=config.get("point_budget"), submitted_at=time.monotonic()))
        export_snapshot_package(args.output, result)
        np.savez_compressed(
            Path(args.output) / "result.npz",
            rgb=result.rgb,
            depth=np.array([]) if result.depth is None else result.depth,
            points=np.array([]) if result.points is None else result.points,
            colors=np.array([]) if result.colors is None else result.colors,
            intrinsics=np.array([]) if result.intrinsics is None else result.intrinsics,
            valid_mask=np.array([]) if result.valid_mask is None else result.valid_mask,
            point_stride=np.asarray([int(result.point_stride)], dtype=np.int32),
        )
        print(json.dumps({"status": "passed", "output": str(Path(args.output).resolve())}), flush=True)
        return 0
    finally:
        model.unload()

if __name__ == "__main__":
    raise SystemExit(main())
