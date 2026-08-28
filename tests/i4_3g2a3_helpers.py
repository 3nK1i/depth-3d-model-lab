from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def make_multiview_fixture(root: Path, *, count: int = 36, low_start: int = 12, low_end: int = 21) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    height, width = 240, 320
    rng = np.random.default_rng(431)
    base = np.full((height, width), 78, dtype=np.uint8)
    for y in range(15, height, 24):
        cv2.line(base, (0, y), (width - 1, y), 180, 1)
    for x in range(15, width, 24):
        cv2.line(base, (x, 0), (x, height - 1), 180, 1)
    for _ in range(350):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 4))
        cv2.circle(base, (x, y), radius, int(rng.integers(120, 255)), -1)

    for index in range(count):
        transform = np.float32([[1.0, 0.0, index * 1.5], [0.0, 1.0, 0.0]])
        gray = cv2.warpAffine(base, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
        if low_start <= index <= low_end:
            gray = cv2.GaussianBlur(gray, (21, 21), 7)
            gray = np.clip(110 + (gray.astype(np.float32) - 110) * 0.22, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert cv2.imwrite(str(root / f"frame_{index:03d}.png"), image)
    return root
