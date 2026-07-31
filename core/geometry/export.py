"""Point-cloud export to PLY (spec §10 "Export PLY").

A minimal, dependency-free binary-little-endian PLY writer — Open3D
stays optional (spec §22 "Open3D only for optional filtering and
point-cloud export, not the primary renderer"), so a basic export must
work without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a flat (N, 3) point/color pair to a binary PLY file.

    ``colors`` must be uint8 RGB, ``points`` float32/float64 XYZ. Raises
    ValueError on a shape mismatch rather than writing a corrupt file.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors shape {colors.shape} must match points shape {points.shape}")

    count = points.shape[0]
    points_f32 = points.astype("<f4", copy=False)
    colors_u8 = colors.astype(np.uint8, copy=False)

    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertex_data = np.empty(count, dtype=vertex_dtype)
    vertex_data["x"] = points_f32[:, 0]
    vertex_data["y"] = points_f32[:, 1]
    vertex_data["z"] = points_f32[:, 2]
    vertex_data["red"] = colors_u8[:, 0]
    vertex_data["green"] = colors_u8[:, 1]
    vertex_data["blue"] = colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    path = Path(path)
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())


def read_ply_for_verification(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read back a PLY written by :func:`export_ply`.

    This exists primarily so the writer can be round-trip tested
    without depending on a third-party PLY reader; it is not part of
    the application's normal runtime path.
    """
    path = Path(path)
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        vertex_count = next(
            int(line.split()[-1]) for line in header_lines if line.startswith("element vertex")
        )
        vertex_dtype = np.dtype(
            [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )
        data = np.frombuffer(f.read(), dtype=vertex_dtype, count=vertex_count)

    points = np.stack([data["x"], data["y"], data["z"]], axis=-1)
    colors = np.stack([data["red"], data["green"], data["blue"]], axis=-1)
    return points, colors