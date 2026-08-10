"""Shared pytest configuration for the Depth & 3D Model Lab test suite.

Automated Qt tests run without a real display server.  A real VisPy
``SceneCanvas`` creates a ``QOpenGLWidget`` and may enter native OpenGL code
while pytest-qt processes events.  The Qt ``offscreen`` plugin does not provide
an OpenGL framebuffer, and some Mesa/driver combinations terminate the Python
process inside ``glDrawArrays`` instead of raising a Python exception.

UI tests therefore replace only the VisPy canvas wrapper with a small Qt test
double.  This keeps controller/signal/layout behaviour testable while leaving
real OpenGL rendering for normal desktop/manual validation.  Production code is
not modified.
"""

from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_OPENGL", "software")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("DEPTH3D_TEST_FAKE_VISPY", "1")


def _install_fake_vispy_canvas() -> None:
    """Install a Qt-only PointCloudCanvas test double before app collection."""
    if os.environ.get("DEPTH3D_TEST_FAKE_VISPY", "1") != "1":
        return

    module_name = "app.widgets.viewport.vispy_canvas"
    if module_name in sys.modules:
        return

    try:
        import numpy as np
        from PyQt6.QtCore import QObject, Qt, pyqtSignal
        from PyQt6.QtWidgets import QWidget
    except ModuleNotFoundError:
        # Pure unit tests may run in a minimal environment without Qt.
        return

    valid_modes = {"select", "orbit", "pan", "zoom"}
    valid_backgrounds = {"dark", "black", "white", "transparent"}

    class _NativeWidget(QWidget):
        def __init__(self, owner: "PointCloudCanvas") -> None:
            super().__init__()
            self._owner = owner
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt override
            if event.key() == Qt.Key.Key_Escape:
                self._owner.escape_pressed.emit()
                event.accept()
                return
            super().keyPressEvent(event)

    class PointCloudCanvas(QObject):
        """API-compatible, non-OpenGL test double for UI/controller tests."""

        interaction_started = pyqtSignal()
        interaction_ended = pyqtSignal()
        escape_pressed = pyqtSignal()

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = _NativeWidget(self)
            self._interaction_mode = "orbit"
            self._point_size = 4.0
            self._smoothing = 0.5
            self._background_name = "dark"
            self._grid_visible = True
            self._has_points = False
            self._is_interacting = False
            self._last_points = None
            self._last_colors = None
            self.fit_calls = 0
            self.reset_calls = 0

        @property
        def native(self):
            return self._native

        def update_points(self, points, colors) -> None:
            self._last_points = np.asarray(points).copy()
            self._last_colors = np.asarray(colors).copy()
            self._has_points = bool(self._last_points.shape[0])

        def clear_points(self) -> None:
            self._last_points = None
            self._last_colors = None
            self._has_points = False

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1
            self.fit_scene()

        def set_interaction_mode(self, mode: str) -> None:
            normalized = str(mode).strip().lower()
            if normalized not in valid_modes:
                raise ValueError(f"Unsupported viewport interaction mode: {mode!r}")
            self._interaction_mode = normalized

        @property
        def interaction_mode(self) -> str:
            return self._interaction_mode

        def set_point_size(self, size: float) -> None:
            self._point_size = max(0.5, float(size))

        @property
        def point_size(self) -> float:
            return self._point_size

        def set_smoothing(self, value: float) -> None:
            self._smoothing = float(np.clip(value, 0.0, 1.0))

        @property
        def smoothing(self) -> float:
            return self._smoothing

        def set_background(self, background: str) -> None:
            normalized = str(background).strip().lower()
            if normalized not in valid_backgrounds:
                raise ValueError(f"Unsupported viewport background: {background!r}")
            self._background_name = normalized

        @property
        def background_name(self) -> str:
            return self._background_name

        def set_grid_visible(self, visible: bool) -> None:
            self._grid_visible = bool(visible)

        @property
        def grid_visible(self) -> bool:
            return self._grid_visible

        @property
        def is_interacting(self) -> bool:
            return self._is_interacting

        @property
        def has_points(self) -> bool:
            return self._has_points

        def close(self) -> None:
            self._native.close()

    class ReconstructionCanvas(QObject):
        """Qt-only test double for the Phase 8.8C reconstruction canvas."""

        interaction_started = pyqtSignal()
        interaction_ended = pyqtSignal()
        escape_pressed = pyqtSignal()

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = _NativeWidget(self)
            self._interaction_mode = "orbit"
            self._grid_visible = True
            self._axes_visible = True
            self._trajectory_visible = False
            self._trajectory_positions = []
            self._current_camera_position = None
            self._has_pose = False
            self._is_interacting = False
            self._intrinsics_matrix = None
            self._image_size = None
            self.fit_calls = 0
            self.focus_calls = 0
            self.reset_calls = 0
            self.closed = False

        @property
        def native(self):
            return self._native

        def configure_camera_geometry(self, *, intrinsics_matrix, image_size) -> None:
            self._intrinsics_matrix = None if intrinsics_matrix is None else np.asarray(intrinsics_matrix).copy()
            self._image_size = image_size

        def update_live_pose(self, T_reconstruction_from_camera) -> None:
            from app.reconstruction_visualization import camera_position_in_viewer

            position = camera_position_in_viewer(T_reconstruction_from_camera).astype(np.float32)
            self._current_camera_position = position
            self._trajectory_positions.append(position.copy())
            self._has_pose = True

        def clear_live_state(self) -> None:
            self._trajectory_positions.clear()
            self._current_camera_position = None
            self._has_pose = False

        def set_trajectory_visible(self, visible: bool) -> None:
            self._trajectory_visible = bool(visible)

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def focus_current_camera(self) -> None:
            self.focus_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1

        def set_interaction_mode(self, mode: str) -> None:
            normalized = str(mode).strip().lower()
            if normalized not in valid_modes:
                raise ValueError(f"Unsupported viewport interaction mode: {mode!r}")
            self._interaction_mode = normalized

        def set_grid_visible(self, visible: bool) -> None:
            self._grid_visible = bool(visible)

        def set_axes_visible(self, visible: bool) -> None:
            self._axes_visible = bool(visible)

        @property
        def interaction_mode(self) -> str:
            return self._interaction_mode

        @property
        def trajectory_count(self) -> int:
            return len(self._trajectory_positions)

        @property
        def trajectory_visible(self) -> bool:
            return self._trajectory_visible

        @property
        def current_camera_position(self):
            return None if self._current_camera_position is None else self._current_camera_position.copy()

        @property
        def has_pose(self) -> bool:
            return self._has_pose

        @property
        def grid_visible(self) -> bool:
            return self._grid_visible

        @property
        def axes_visible(self) -> bool:
            return self._axes_visible

        @property
        def is_interacting(self) -> bool:
            return self._is_interacting

        def close(self) -> None:
            self.closed = True
            self._native.close()

    fake_module = types.ModuleType(module_name)
    fake_module.PointCloudCanvas = PointCloudCanvas
    fake_module.ReconstructionCanvas = ReconstructionCanvas
    fake_module.DEFAULT_INTERACTION_MODE = "orbit"
    fake_module.DEFAULT_POINT_SIZE = 4.0
    sys.modules[module_name] = fake_module


_install_fake_vispy_canvas()

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_qsettings(tmp_path_factory):
    """Redirect QSettings to a temporary INI directory for the test process."""
    try:
        from PyQt6.QtCore import QSettings
    except ModuleNotFoundError:
        yield
        return

    settings_dir = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(settings_dir),
    )
    settings = QSettings("EIA Tech", "Depth3DModelLab")
    settings.clear()
    settings.sync()
    yield
    settings.clear()
    settings.sync()


def _install_fake_inspection_canvas() -> None:
    """Install a Qt-only inspection 3D canvas before Step 8 UI collection."""
    if os.environ.get("DEPTH3D_TEST_FAKE_VISPY", "1") != "1":
        return
    module_name = "app.widgets.inspection_3d_canvas"
    if module_name in sys.modules:
        return
    try:
        import numpy as np
        from PyQt6.QtCore import QObject, pyqtSignal
        from PyQt6.QtWidgets import QWidget
    except ModuleNotFoundError:
        return

    class Inspection3DCanvas(QObject):
        escape_pressed = pyqtSignal()

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._native = QWidget()
            self.last_object = None
            self.last_arrays = {}
            self.point_mode = None
            self.box_method = None
            self.show_boxes = True
            self.fit_calls = 0
            self.reset_calls = 0
            self.closed = False

        @property
        def native(self):
            return self._native

        def close(self) -> None:
            self.closed = True
            self._native.close()

        def clear(self) -> None:
            self.last_object = None
            self.last_arrays = {}

        def set_object(self, obj, arrays, *, point_mode, box_method, show_boxes) -> None:
            self.last_object = obj
            self.last_arrays = {key: np.asarray(value) for key, value in arrays.items()}
            self.point_mode = point_mode
            self.box_method = box_method
            self.show_boxes = bool(show_boxes)

        def refresh_style(self, *, point_size=None, show_boxes=None) -> None:
            if show_boxes is not None:
                self.show_boxes = bool(show_boxes)

        def fit_scene(self) -> None:
            self.fit_calls += 1

        def reset_view(self) -> None:
            self.reset_calls += 1
            self.fit_scene()

    fake_module = types.ModuleType(module_name)
    fake_module.Inspection3DCanvas = Inspection3DCanvas
    sys.modules[module_name] = fake_module


_install_fake_inspection_canvas()
