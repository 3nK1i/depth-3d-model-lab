from __future__ import annotations

from pathlib import Path


def test_ui4_is_qualification_only_and_composes_locked_ui2_ui3_gates() -> None:
    script = Path("scripts/run_phase8_9_pre_h_ui4_tests.sh").read_text(encoding="utf-8")

    assert "run_phase8_9_pre_h_ui2_tests.sh affected" in script
    assert "run_phase8_9_pre_h_ui3_tests.sh affected" in script
    assert "test_phase8_9_pre_h_ui4_combined_ui_qualification.py" in script
    assert "test_main_window_shell.py" in script
    assert "test_phase7_mode_switching.py" in script
    assert "test_shutdown_lifecycle.py" in script
    assert "test_phase8_9_pre_h_rtsp_fps_fix.py" in script


def test_ui4_combined_ui_covers_compact_shared_viewports_and_dual_fullscreen() -> None:
    source = Path(
        "tests/ui/test_phase8_9_pre_h_ui4_combined_ui_qualification.py"
    ).read_text(encoding="utf-8")

    assert "AppMode.LIVE_3D" in source
    assert "AppMode.SNAPSHOT" in source
    assert "AppMode.STABLE_VIDEO" in source
    assert "AppMode.INSPECTION" in source
    assert "AppMode.RECONSTRUCTION" in source
    assert 'window._focus_target == "reconstruction"' in source
    assert 'window._focus_target == "viewport"' in source
    assert "reconstruction_native.setFocus" in source
    assert "Qt.Key.Key_Escape" in source
    assert "viewport._canvas.minimumHeight() == 180" in source


def test_ui4_does_not_introduce_product_code_replacements() -> None:
    doc = Path(
        "docs/phase_history/PHASE8_9_PRE_H_UI4_COMBINED_UI_QUALIFICATION.md"
    ).read_text(encoding="utf-8")
    assert "qualification-only" in doc.lower()
    assert "No production application files are changed" in doc
