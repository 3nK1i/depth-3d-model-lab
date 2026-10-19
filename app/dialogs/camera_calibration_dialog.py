"""Professional Phase 9.1F camera-calibration workspace.

The dialog is application presentation/glue only.  It reuses the Phase 9.1A-E
calibration contracts and never invents metric authority.  Automatic calibration,
manual entry, legacy/native import, profile persistence, and safe profile selection
all remain delegated to the existing core APIs.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QStandardPaths
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.reconstruction.automatic_calibration import (
    AutomaticCalibrationQualityPolicy,
    CharucoCalibrationTarget,
    CheckerboardCalibrationTarget,
    calibrate_charuco_image_set,
    calibrate_checkerboard_image_set,
    inspect_opencv_calibration_capabilities,
    load_calibration_image_set,
)
from core.reconstruction.calibration_management import (
    CameraCalibrationProfileStore,
    create_manual_camera_calibration_profile,
    import_camera_calibration_file,
)
from core.reconstruction.calibration_profile import (
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
)
from core.reconstruction.camera import CameraDistortionModel
from core.reconstruction.errors import ReconstructionValidationError


MODE_AUTOMATIC = "automatic"
MODE_MANUAL = "manual"
MODE_IMPORT = "import"


def default_camera_calibration_profile_root() -> Path:
    """Return a platform-appropriate profile library root without creating it."""

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not str(base).strip():
        base = str(Path.home() / ".local" / "share" / "depth3d-model-lab")
    return Path(base).expanduser().resolve() / "camera_calibration_profiles"


class CameraCalibrationDialog(QDialog):
    """Modal calibration workspace with progressive disclosure and safe authority UX."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        profile_store: CameraCalibrationProfileStore | None = None,
        source_label: str = "No live source selected",
        target_resolution_text: str = "Resolution unavailable",
        metric_status_text: str = "Not qualified",
        initial_profile: CameraCalibrationProfile | None = None,
        allow_test_mode: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cameraCalibrationDialog")
        self.setWindowTitle("Camera Calibration")
        self.setModal(True)
        self.setMinimumSize(760, 560)
        self.resize(860, 620)

        self._store = profile_store or CameraCalibrationProfileStore(
            default_camera_calibration_profile_root()
        )
        self._source_label_text = str(source_label).strip() or "No live source selected"
        self._target_resolution_text = (
            str(target_resolution_text).strip() or "Resolution unavailable"
        )
        self._metric_status_text = str(metric_status_text).strip() or "Not qualified"
        self._current_profile: CameraCalibrationProfile | None = initial_profile
        self._automatic_image_paths: tuple[Path, ...] = ()
        self._test_mode_requested = False
        self._mode_buttons: dict[str, QPushButton] = {}

        self._build_ui(allow_test_mode=allow_test_mode)
        self._refresh_saved_profiles(select_profile_id=(initial_profile.profile_id if initial_profile else ""))
        self._refresh_profile_presentation()
        self._set_mode(MODE_AUTOMATIC)

    @property
    def selected_profile(self) -> CameraCalibrationProfile | None:
        return self._current_profile

    @property
    def test_mode_requested(self) -> bool:
        return self._test_mode_requested

    @property
    def profile_store(self) -> CameraCalibrationProfileStore:
        return self._store

    def _build_ui(self, *, allow_test_mode: bool) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(12)

        header = QFrame()
        header.setObjectName("cameraCalibrationHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 11, 14, 11)
        header_layout.setSpacing(3)
        title = QLabel("Camera Calibration")
        title.setObjectName("cameraCalibrationTitle")
        subtitle = QLabel(
            "Create, import, validate, save, and reuse camera intrinsics without promoting scene metric authority."
        )
        subtitle.setObjectName("cameraCalibrationSubtitle")
        subtitle.setWordWrap(True)
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        outer.addWidget(header)

        state = QFrame()
        state.setObjectName("cameraCalibrationStateCard")
        state_layout = QGridLayout(state)
        state_layout.setContentsMargins(12, 9, 12, 9)
        state_layout.setHorizontalSpacing(14)
        state_layout.setVerticalSpacing(4)
        self._source_value = self._state_value(self._source_label_text)
        self._resolution_value = self._state_value(self._target_resolution_text)
        self._calibration_value = self._state_value("No profile selected")
        self._metric_value = self._state_value(self._metric_status_text)
        self._metric_value.setProperty("statusKind", "warning")
        for column, (label, value) in enumerate(
            (
                ("SOURCE", self._source_value),
                ("TARGET RESOLUTION", self._resolution_value),
                ("CAMERA CALIBRATION", self._calibration_value),
                ("METRIC SCALE", self._metric_value),
            )
        ):
            caption = QLabel(label)
            caption.setObjectName("cameraCalibrationStateLabel")
            state_layout.addWidget(caption, 0, column)
            state_layout.addWidget(value, 1, column)
        outer.addWidget(state)

        body = QHBoxLayout()
        body.setSpacing(12)
        outer.addLayout(body, stretch=1)

        sidebar = QFrame()
        sidebar.setObjectName("cameraCalibrationSidebar")
        sidebar.setMinimumWidth(190)
        sidebar.setMaximumWidth(220)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(8)

        saved_title = QLabel("SAVED PROFILES")
        saved_title.setObjectName("cameraCalibrationSectionTitle")
        side_layout.addWidget(saved_title)
        self._saved_profiles_combo = QComboBox()
        self._saved_profiles_combo.setObjectName("cameraCalibrationCombo")
        self._saved_profiles_combo.setMinimumContentsLength(10)
        self._saved_profiles_combo.currentIndexChanged.connect(self._on_saved_profile_selected)
        side_layout.addWidget(self._saved_profiles_combo)
        refresh = QPushButton("Refresh Profiles")
        refresh.setObjectName("cameraCalibrationAction")
        refresh.clicked.connect(lambda: self._refresh_saved_profiles())
        side_layout.addWidget(refresh)

        mode_title = QLabel("CALIBRATION SOURCE")
        mode_title.setObjectName("cameraCalibrationSectionTitle")
        side_layout.addSpacing(4)
        side_layout.addWidget(mode_title)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        for mode, label in (
            (MODE_AUTOMATIC, "Automatic"),
            (MODE_MANUAL, "Manual"),
            (MODE_IMPORT, "Import Profile"),
        ):
            button = QPushButton(label)
            button.setObjectName("cameraCalibrationModeButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, value=mode: self._set_mode(value))
            mode_group.addButton(button)
            self._mode_buttons[mode] = button
            side_layout.addWidget(button)

        side_layout.addSpacing(4)
        profile_title = QLabel("CURRENT PROFILE")
        profile_title.setObjectName("cameraCalibrationSectionTitle")
        side_layout.addWidget(profile_title)
        self._profile_summary = QLabel("No profile selected")
        self._profile_summary.setObjectName("cameraCalibrationProfileSummary")
        self._profile_summary.setWordWrap(True)
        side_layout.addWidget(self._profile_summary)
        side_layout.addStretch(1)
        body.addWidget(sidebar)

        self._stack = QStackedWidget()
        self._stack.setObjectName("cameraCalibrationStack")
        self._stack.addWidget(self._build_automatic_page())
        self._stack.addWidget(self._build_manual_page())
        self._stack.addWidget(self._build_import_page())
        body.addWidget(self._stack, stretch=1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self._status_message = QLabel("Camera calibration does not qualify reconstruction metric scale.")
        self._status_message.setObjectName("cameraCalibrationStatusMessage")
        self._status_message.setWordWrap(True)
        footer.addWidget(self._status_message, stretch=1)

        if allow_test_mode:
            self._test_mode_button = QPushButton("Use Test Mode")
            self._test_mode_button.setObjectName("cameraCalibrationAction")
            self._test_mode_button.setToolTip(
                "Use explicit provisional estimated intrinsics for functional testing only. "
                "Scale stays arbitrary and metrology is not certified."
            )
            self._test_mode_button.clicked.connect(self._accept_test_mode)
            footer.addWidget(self._test_mode_button)
        else:
            self._test_mode_button = None

        self._save_button = QPushButton("Save Profile")
        self._save_button.setObjectName("cameraCalibrationAction")
        self._save_button.clicked.connect(self._save_current_profile)
        footer.addWidget(self._save_button)

        self._use_button = QPushButton("Use Profile")
        self._use_button.setObjectName("cameraCalibrationAction")
        self._use_button.setProperty("actionRole", "primary")
        self._use_button.clicked.connect(self._accept_profile)
        footer.addWidget(self._use_button)

        cancel = QPushButton("Cancel")
        cancel.setObjectName("cameraCalibrationAction")
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        outer.addLayout(footer)

    def _build_automatic_page(self) -> QWidget:
        page, layout = self._page_host("AUTOMATIC CALIBRATION")

        intro = QLabel(
            "Import a checkerboard or ChArUco image set, then evaluate explicit reprojection-quality thresholds before accepting the profile."
        )
        intro.setObjectName("cameraCalibrationHint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self._automatic_target_combo = QComboBox()
        self._automatic_target_combo.setObjectName("cameraCalibrationCombo")
        self._automatic_target_combo.addItem("Checkerboard", "checkerboard")
        self._automatic_target_combo.addItem("ChArUco", "charuco")
        self._automatic_target_combo.currentIndexChanged.connect(self._refresh_automatic_target_fields)
        self._add_form_row(form, 0, "Target", self._automatic_target_combo)
        self._automatic_camera_id = QLineEdit(self._source_label_text)
        self._automatic_camera_id.setObjectName("cameraCalibrationLineEdit")
        self._automatic_camera_id.setPlaceholderText("Camera / source identity")
        self._add_form_row(form, 1, "Camera / source ID", self._automatic_camera_id)

        image_row = QWidget()
        image_layout = QHBoxLayout(image_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(6)
        self._automatic_images_value = QLabel("No image set selected")
        self._automatic_images_value.setObjectName("cameraCalibrationFieldValue")
        self._automatic_images_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        choose_images = QPushButton("Choose Images")
        choose_images.setObjectName("cameraCalibrationAction")
        choose_images.clicked.connect(self._choose_automatic_images)
        image_layout.addWidget(self._automatic_images_value, stretch=1)
        image_layout.addWidget(choose_images)
        self._add_form_row(form, 2, "Image set", image_row)

        self._board_x = self._int_field(9, 2, 100)
        self._board_y = self._int_field(6, 2, 100)
        self._square_length = self._float_field(0.025, 0.000001, 1_000_000.0, decimals=6)
        self._marker_length = self._float_field(0.018, 0.000001, 1_000_000.0, decimals=6)
        self._dictionary_combo = QComboBox()
        self._dictionary_combo.setObjectName("cameraCalibrationCombo")
        self._dictionary_combo.addItems(
            ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100", "DICT_6X6_50"]
        )
        self._add_form_row(form, 3, "Corners / squares X", self._board_x)
        self._add_form_row(form, 4, "Corners / squares Y", self._board_y)
        self._add_form_row(form, 5, "Square length", self._square_length)
        self._marker_length_label = QLabel("Marker length")
        self._marker_length_label.setObjectName("cameraCalibrationFieldLabel")
        form.addWidget(self._marker_length_label, 6, 0)
        form.addWidget(self._marker_length, 6, 1)
        self._dictionary_label = QLabel("Dictionary")
        self._dictionary_label.setObjectName("cameraCalibrationFieldLabel")
        form.addWidget(self._dictionary_label, 7, 0)
        form.addWidget(self._dictionary_combo, 7, 1)
        layout.addLayout(form)

        policy_title = QLabel("QUALITY POLICY")
        policy_title.setObjectName("cameraCalibrationSectionTitle")
        layout.addWidget(policy_title)
        policy = QGridLayout()
        policy.setHorizontalSpacing(10)
        policy.setVerticalSpacing(8)
        self._minimum_views = self._int_field(6, 3, 1000)
        self._max_rms = self._float_field(1.0, 0.0001, 100.0, decimals=4)
        self._max_mean = self._float_field(1.0, 0.0001, 100.0, decimals=4)
        self._max_view = self._float_field(2.0, 0.0001, 100.0, decimals=4)
        self._add_form_row(policy, 0, "Minimum accepted views", self._minimum_views)
        self._add_form_row(policy, 1, "Maximum RMS error (px)", self._max_rms)
        self._add_form_row(policy, 2, "Maximum mean error (px)", self._max_mean)
        self._add_form_row(policy, 3, "Maximum per-view error (px)", self._max_view)
        layout.addLayout(policy)

        caution = QLabel(
            "Target dimensions are calibration-board geometry only. They do not qualify scene metric scale."
        )
        caution.setObjectName("cameraCalibrationCaution")
        caution.setWordWrap(True)
        layout.addWidget(caution)

        run = QPushButton("Run Calibration")
        run.setObjectName("cameraCalibrationAction")
        run.setProperty("actionRole", "primary")
        run.clicked.connect(self._run_automatic_calibration)
        layout.addWidget(run, alignment=Qt.AlignmentFlag.AlignRight)

        self._automatic_result = QLabel("No automatic calibration result yet.")
        self._automatic_result.setObjectName("cameraCalibrationResult")
        self._automatic_result.setWordWrap(True)
        layout.addWidget(self._automatic_result)
        layout.addStretch(1)
        self._refresh_automatic_target_fields()
        return page

    def _build_manual_page(self) -> QWidget:
        page, layout = self._page_host("MANUAL CALIBRATION ENTRY")
        intro = QLabel(
            "Enter known camera parameters directly. Structural validation is immediate; manual entry remains unvalidated calibration evidence until separately qualified."
        )
        intro.setObjectName("cameraCalibrationHint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        self._manual_camera_id = QLineEdit(self._source_label_text)
        self._manual_camera_id.setObjectName("cameraCalibrationLineEdit")
        self._manual_camera_id.setPlaceholderText("Camera / source identity")
        self._manual_width = self._int_field(1920, 1, 32768)
        self._manual_height = self._int_field(1080, 1, 32768)
        self._manual_fx = self._float_field(1000.0, 0.000001, 1_000_000.0, decimals=6)
        self._manual_fy = self._float_field(1000.0, 0.000001, 1_000_000.0, decimals=6)
        self._manual_cx = self._float_field(959.5, 0.0, 32767.999999, decimals=6)
        self._manual_cy = self._float_field(539.5, 0.0, 32767.999999, decimals=6)
        self._manual_distortion = QComboBox()
        self._manual_distortion.setObjectName("cameraCalibrationCombo")
        self._manual_distortion.addItem("None", CameraDistortionModel.NONE)
        self._manual_distortion.addItem("Brown-Conrady", CameraDistortionModel.BROWN_CONRADY)
        self._manual_distortion.currentIndexChanged.connect(self._refresh_manual_distortion_fields)
        self._manual_k1 = self._float_field(0.0, -100.0, 100.0, decimals=9)
        self._manual_k2 = self._float_field(0.0, -100.0, 100.0, decimals=9)
        self._manual_p1 = self._float_field(0.0, -100.0, 100.0, decimals=9)
        self._manual_p2 = self._float_field(0.0, -100.0, 100.0, decimals=9)
        self._manual_k3 = self._float_field(0.0, -100.0, 100.0, decimals=9)
        rows = (
            ("Camera / source ID", self._manual_camera_id),
            ("Image width", self._manual_width),
            ("Image height", self._manual_height),
            ("fx", self._manual_fx),
            ("fy", self._manual_fy),
            ("cx", self._manual_cx),
            ("cy", self._manual_cy),
            ("Distortion model", self._manual_distortion),
            ("k1", self._manual_k1),
            ("k2", self._manual_k2),
            ("p1", self._manual_p1),
            ("p2", self._manual_p2),
            ("k3", self._manual_k3),
        )
        self._manual_distortion_widgets: list[QWidget] = []
        for index, (label, widget) in enumerate(rows):
            caption = QLabel(label)
            caption.setObjectName("cameraCalibrationFieldLabel")
            form.addWidget(caption, index, 0)
            form.addWidget(widget, index, 1)
            if index >= 8:
                self._manual_distortion_widgets.extend((caption, widget))
        layout.addLayout(form)

        advanced = QLabel(
            "Advanced • Additional distortion coefficients/models should be imported from a qualified profile unless a later workflow explicitly requires manual editing."
        )
        advanced.setObjectName("cameraCalibrationHint")
        advanced.setWordWrap(True)
        layout.addWidget(advanced)

        build = QPushButton("Build Manual Profile")
        build.setObjectName("cameraCalibrationAction")
        build.setProperty("actionRole", "primary")
        build.clicked.connect(self._build_manual_profile)
        layout.addWidget(build, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addStretch(1)
        self._refresh_manual_distortion_fields()
        return page

    def _build_import_page(self) -> QWidget:
        page, layout = self._page_host("IMPORT CALIBRATION PROFILE")
        intro = QLabel(
            "Import a native Phase 9 profile or a legacy Phase 8 live-calibration JSON. Legacy records stay legacy-compatible and are not upgraded to new validation evidence."
        )
        intro.setObjectName("cameraCalibrationHint")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._import_path = QLineEdit()
        self._import_path.setObjectName("cameraCalibrationLineEdit")
        self._import_path.setPlaceholderText("Select a calibration JSON profile…")
        browse = QPushButton("Browse")
        browse.setObjectName("cameraCalibrationAction")
        browse.clicked.connect(self._choose_import_profile)
        row.addWidget(self._import_path, stretch=1)
        row.addWidget(browse)
        layout.addLayout(row)

        import_button = QPushButton("Import Profile")
        import_button.setObjectName("cameraCalibrationAction")
        import_button.setProperty("actionRole", "primary")
        import_button.clicked.connect(self._import_profile)
        layout.addWidget(import_button, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addStretch(1)
        return page

    @staticmethod
    def _page_host(title_text: str) -> tuple[QWidget, QVBoxLayout]:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setObjectName("cameraCalibrationScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("cameraCalibrationPageBody")
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 12, 18, 12)
        layout.setSpacing(10)
        title = QLabel(title_text)
        title.setObjectName("cameraCalibrationPageTitle")
        layout.addWidget(title)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        return host, layout

    @staticmethod
    def _state_value(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("cameraCalibrationStateValue")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _int_field(value: int, minimum: int, maximum: int) -> QSpinBox:
        field = QSpinBox()
        field.setObjectName("cameraCalibrationSpin")
        field.setRange(minimum, maximum)
        field.setValue(value)
        field.setKeyboardTracking(False)
        return field

    @staticmethod
    def _float_field(
        value: float,
        minimum: float,
        maximum: float,
        *,
        decimals: int,
    ) -> QDoubleSpinBox:
        field = QDoubleSpinBox()
        field.setObjectName("cameraCalibrationDoubleSpin")
        field.setDecimals(decimals)
        field.setRange(minimum, maximum)
        field.setValue(value)
        field.setKeyboardTracking(False)
        return field

    @staticmethod
    def _add_form_row(layout: QGridLayout, row: int, label_text: str, widget: QWidget) -> None:
        label = QLabel(label_text)
        label.setObjectName("cameraCalibrationFieldLabel")
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        layout.setColumnStretch(1, 1)

    def _set_mode(self, mode: str) -> None:
        mapping = {MODE_AUTOMATIC: 0, MODE_MANUAL: 1, MODE_IMPORT: 2}
        if mode not in mapping:
            raise ValueError(f"Unsupported calibration UI mode: {mode}")
        self._stack.setCurrentIndex(mapping[mode])
        self._mode_buttons[mode].setChecked(True)

    def _refresh_automatic_target_fields(self) -> None:
        charuco = self._automatic_target_combo.currentData() == "charuco"
        self._marker_length.setVisible(charuco)
        self._marker_length_label.setVisible(charuco)
        self._dictionary_combo.setVisible(charuco)
        self._dictionary_label.setVisible(charuco)

    def _refresh_manual_distortion_fields(self) -> None:
        enabled = self._manual_distortion.currentData() is CameraDistortionModel.BROWN_CONRADY
        for widget in self._manual_distortion_widgets:
            widget.setVisible(enabled)

    def _choose_automatic_images(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Camera Calibration Images",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*)",
        )
        if not selected:
            return
        self._automatic_image_paths = tuple(Path(value).expanduser().resolve() for value in selected)
        self._automatic_images_value.setText(f"{len(self._automatic_image_paths)} images selected")
        self._automatic_images_value.setToolTip("\n".join(str(path) for path in self._automatic_image_paths))

    def _run_automatic_calibration(self) -> None:
        try:
            if not self._automatic_image_paths:
                raise ReconstructionValidationError("Choose a calibration image set first.")
            policy = AutomaticCalibrationQualityPolicy(
                minimum_accepted_views=self._minimum_views.value(),
                maximum_rms_reprojection_error_px=self._max_rms.value(),
                maximum_mean_reprojection_error_px=self._max_mean.value(),
                maximum_view_reprojection_error_px=self._max_view.value(),
            )
            images = load_calibration_image_set(self._automatic_image_paths)
            if self._automatic_target_combo.currentData() == "checkerboard":
                target = CheckerboardCalibrationTarget(
                    inner_corners_x=self._board_x.value(),
                    inner_corners_y=self._board_y.value(),
                    square_length=self._square_length.value(),
                )
                result = calibrate_checkerboard_image_set(
                    images,
                    target,
                    quality_policy=policy,
                    camera_id=self._automatic_camera_id.text().strip() or None,
                )
            else:
                target = CharucoCalibrationTarget(
                    squares_x=max(3, self._board_x.value()),
                    squares_y=max(3, self._board_y.value()),
                    square_length=self._square_length.value(),
                    marker_length=self._marker_length.value(),
                    dictionary_name=self._dictionary_combo.currentText(),
                )
                result = calibrate_charuco_image_set(
                    images,
                    target,
                    quality_policy=policy,
                    camera_id=self._automatic_camera_id.text().strip() or None,
                )
            self._current_profile = result.profile
            quality = result.profile.quality
            assert quality is not None
            self._automatic_result.setText(
                f"{result.profile.validation_state.value.replace('_', ' ').title()} • "
                f"{result.accepted_view_count} accepted / {result.rejected_view_count} rejected • "
                f"RMS {quality.rms_reprojection_error_px:.3f} px • "
                f"mean {quality.mean_reprojection_error_px:.3f} px • "
                f"max {quality.max_reprojection_error_px:.3f} px"
            )
            self._set_status("Automatic calibration completed. Review the validation state, then save/use the profile.", "ready")
            self._refresh_profile_presentation()
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            QMessageBox.warning(self, "Camera Calibration", str(exc))

    def _build_manual_profile(self) -> None:
        try:
            distortion_model = self._manual_distortion.currentData()
            coefficients: tuple[float, ...]
            if distortion_model is CameraDistortionModel.NONE:
                coefficients = ()
            else:
                coefficients = (
                    self._manual_k1.value(),
                    self._manual_k2.value(),
                    self._manual_p1.value(),
                    self._manual_p2.value(),
                    self._manual_k3.value(),
                )
            self._current_profile = create_manual_camera_calibration_profile(
                width=self._manual_width.value(),
                height=self._manual_height.value(),
                fx=self._manual_fx.value(),
                fy=self._manual_fy.value(),
                cx=self._manual_cx.value(),
                cy=self._manual_cy.value(),
                distortion_model=distortion_model,
                distortion_coefficients=coefficients,
                camera_id=self._manual_camera_id.text().strip() or None,
            )
            self._set_status(
                "Manual profile created as UNVALIDATED. It may supply explicit intrinsics but does not qualify metric scale.",
                "warning",
            )
            self._refresh_profile_presentation()
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            QMessageBox.warning(self, "Camera Calibration", str(exc))

    def _choose_import_profile(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import Camera Calibration Profile",
            str(Path.home()),
            "Camera calibration (*.json);;JSON files (*.json);;All files (*)",
        )
        if selected:
            self._import_path.setText(str(Path(selected).expanduser().resolve()))

    def _import_profile(self) -> None:
        try:
            path = Path(self._import_path.text()).expanduser().resolve()
            imported = import_camera_calibration_file(path)
            self._current_profile = imported.profile
            self._set_status(
                f"Imported {imported.file_format.value.replace('_', ' ')} profile. Original calibration evidence was preserved.",
                "ready" if imported.profile.has_phase9_validation_evidence else "warning",
            )
            self._refresh_profile_presentation()
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            QMessageBox.warning(self, "Camera Calibration", str(exc))

    def _save_current_profile(self) -> None:
        if self._current_profile is None:
            self._set_status("Create, import, or select a calibration profile first.", "warning")
            return
        try:
            overwrite = self._store.contains(self._current_profile.profile_id)
            self._store.save(self._current_profile, overwrite=overwrite)
            self._refresh_saved_profiles(select_profile_id=self._current_profile.profile_id)
            self._set_status("Calibration profile saved to the reusable profile library.", "ready")
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            QMessageBox.warning(self, "Camera Calibration", str(exc))

    def _refresh_saved_profiles(self, *, select_profile_id: str = "") -> None:
        try:
            profiles = self._store.list_profiles()
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            profiles = ()
        self._saved_profiles_combo.blockSignals(True)
        self._saved_profiles_combo.clear()
        self._saved_profiles_combo.addItem("Select saved profile…", None)
        selected_index = 0
        for profile in profiles:
            label = (
                f"{profile.camera.camera_id} • {profile.calibration_image_size.width}×"
                f"{profile.calibration_image_size.height} • {profile.method.value}"
            )
            self._saved_profiles_combo.addItem(label, profile.profile_id)
            if select_profile_id and profile.profile_id == select_profile_id:
                selected_index = self._saved_profiles_combo.count() - 1
        self._saved_profiles_combo.setCurrentIndex(selected_index)
        self._saved_profiles_combo.blockSignals(False)

    def _on_saved_profile_selected(self, index: int) -> None:
        profile_id = self._saved_profiles_combo.itemData(index)
        if not profile_id:
            return
        try:
            self._current_profile = self._store.load(str(profile_id))
            self._set_status("Loaded reusable calibration profile.", "ready")
            self._refresh_profile_presentation()
        except (OSError, RuntimeError, TypeError, ValueError, ReconstructionValidationError) as exc:
            self._set_status(str(exc), "error")
            QMessageBox.warning(self, "Camera Calibration", str(exc))

    def _refresh_profile_presentation(self) -> None:
        profile = self._current_profile
        self._save_button.setEnabled(profile is not None)
        self._use_button.setEnabled(
            profile is not None
            and profile.validation_state is not CameraCalibrationValidationState.REJECTED
        )
        if profile is None:
            self._profile_summary.setText("No profile selected")
            self._calibration_value.setText("No profile selected")
            self._calibration_value.setProperty("statusKind", "neutral")
            self._refresh_style(self._calibration_value)
            return
        quality = profile.quality
        quality_text = ""
        if quality is not None:
            quality_text = f"\nRMS {quality.rms_reprojection_error_px:.3f} px"
        self._profile_summary.setText(
            f"{profile.camera.camera_id}\n"
            f"{profile.calibration_image_size.width}×{profile.calibration_image_size.height}\n"
            f"{profile.method.value.replace('_', ' ').title()} • "
            f"{profile.validation_state.value.replace('_', ' ').title()}"
            f"{quality_text}"
        )
        self._calibration_value.setText(
            profile.validation_state.value.replace("_", " ").title()
        )
        kind = (
            "ready"
            if profile.validation_state is CameraCalibrationValidationState.VALIDATED
            else "warning"
        )
        self._calibration_value.setProperty("statusKind", kind)
        self._refresh_style(self._calibration_value)

    def _accept_profile(self) -> None:
        if self._current_profile is None:
            self._set_status("Select or create a calibration profile first.", "warning")
            return
        if self._current_profile.validation_state is CameraCalibrationValidationState.REJECTED:
            self._set_status(
                "Rejected calibration evidence may be saved for audit but cannot be used as the active camera profile.",
                "error",
            )
            return
        self._test_mode_requested = False
        self.accept()

    def _accept_test_mode(self) -> None:
        self._test_mode_requested = True
        self.accept()

    def _set_status(self, text: str, kind: str = "neutral") -> None:
        self._status_message.setText(str(text))
        self._status_message.setProperty("statusKind", kind)
        self._refresh_style(self._status_message)

    @staticmethod
    def _refresh_style(widget: QWidget) -> None:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
