from __future__ import annotations

import math

from core.geometry.orientation_overlay import project_turntable_axes


def test_projected_axes_are_finite_and_bounded() -> None:
    projected = project_turntable_axes(45.0, 25.0, 0.0)
    assert set(projected) == {"x", "y", "z"}
    for screen_x, screen_y, depth in projected.values():
        assert math.isfinite(screen_x)
        assert math.isfinite(screen_y)
        assert math.isfinite(depth)
        assert -1.0 <= screen_x <= 1.0
        assert -1.0 <= screen_y <= 1.0
        assert -1.0 <= depth <= 1.0


def test_overlay_orientation_changes_with_turntable_camera() -> None:
    first = project_turntable_axes(0.0, 20.0, 0.0)
    second = project_turntable_axes(90.0, 20.0, 0.0)
    assert first["x"] != second["x"]
    assert first["y"] != second["y"]


def test_roll_rotates_screen_projection_without_changing_depth() -> None:
    base = project_turntable_axes(35.0, 20.0, 0.0)
    rolled = project_turntable_axes(35.0, 20.0, 90.0)
    for axis in ("x", "y", "z"):
        bx, by, bd = base[axis]
        rx, ry, rd = rolled[axis]
        assert math.isclose(rd, bd, abs_tol=1e-9)
        assert math.isclose(rx, -by, abs_tol=1e-9)
        assert math.isclose(ry, bx, abs_tol=1e-9)


def test_projection_remains_stable_at_camera_poles() -> None:
    for elevation in (-90.0, 90.0):
        projected = project_turntable_axes(0.0, elevation, 0.0)
        assert all(math.isfinite(value) for values in projected.values() for value in values)


def test_inspection_canvas_uses_screen_overlay_not_scene_axis() -> None:
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "app"
        / "widgets"
        / "inspection_3d_canvas.py"
    ).read_text(encoding="utf-8")
    assert "scene.visuals.XYZAxis" not in source
    assert "_OrientationGizmo" in source
    assert "AlignLeft | Qt.AlignmentFlag.AlignBottom" in source
    assert "self._native_host" in source
