#!/usr/bin/env python3
"""Launch only the Phase 9.1F camera-calibration workspace for visual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from app.dialogs.camera_calibration_dialog import CameraCalibrationDialog
from app.theme import apply_theme
from core.reconstruction.calibration_management import CameraCalibrationProfileStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/tmp/depth3d_phase9_1f_calibration_profiles"),
        help="Profile library used by the preview (default: /tmp, so production profiles are untouched).",
    )
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    dialog = CameraCalibrationDialog(
        profile_store=CameraCalibrationProfileStore(args.profile_dir),
        source_label="Preview camera",
        target_resolution_text="1280 × 720",
        metric_status_text="Not qualified • scale evidence required",
        allow_test_mode=True,
    )
    dialog.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
