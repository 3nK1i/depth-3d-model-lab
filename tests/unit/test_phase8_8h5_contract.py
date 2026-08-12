from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_phase8_8h5_real_ui_acceptance.py"
SHELL = ROOT / "scripts/run_phase8_8h5_tests.sh"
DOC = ROOT / "docs/phase_history/PHASE8_8H5_VALIDATION.md"


def test_h5_is_real_application_acceptance_not_feature_work() -> None:
    doc = DOC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "adds no Reconstruction feature" in doc
    assert '"production_behavior_changed_by_h5": False' in runner
    assert '"new_feature_work_added_by_h5": False' in runner
    assert '"release_tag_created_by_h5": False' in runner


def test_h5_requires_real_image_or_video_source_and_external_pycolmap() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'source.add_argument("--images"' in text
    assert 'source.add_argument("--video"' in text
    assert '"EIA_RECONSTRUCTION_PYCOLMAP_PYTHON"' in text
    assert '"pycolmap" not in sys.modules' in text
    assert 'result.get("tests_fixture_used") is False' in text


def test_h5_uses_real_main_window_reconstruction_ui() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "window = MainWindow()" in text
    assert "window.resize(1366, 768)" in text
    assert "window._mode_tabs.set_active_mode(AppMode.RECONSTRUCTION)" in text
    assert "page._start_button.click()" in text
    assert "page._offline_focus_camera_button.click()" in text


def test_h5_does_not_use_fake_or_software_vispy_path() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ.pop("DEPTH3D_TEST_FAKE_VISPY", None)' in text
    assert 'os.environ.pop("QT_QPA_PLATFORM", None)' in text
    assert 'os.environ.pop("QT_OPENGL", None)' in text
    assert 'os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)' in text
    assert 'os.environ["DEPTH3D_TEST_FAKE_VISPY"]' not in text


def test_h5_checks_real_sparse_scene_camera_and_landmark_presentation() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    for token in (
        "canvas.offline_visualization_loaded",
        "canvas.offline_registered_camera_count",
        "canvas.selected_offline_camera_visible",
        "canvas.sparse_landmark_count",
        "page._viewport_placeholder.isHidden()",
    ):
        assert token in text


def test_h5_checks_1366_sidebar_without_horizontal_overflow() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert '"horizontal_maximum": hbar.maximum()' in text
    assert '"horizontal_visible": hbar.isVisible()' in text
    assert "ScrollBarAlwaysOff" in text
    assert '"responsive_1366x768": responsive_ok' in text


def test_h5_preserves_sparse_authority_and_metrology_boundaries() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "ReconstructionSceneAvailability.PARTIAL" in text
    assert 'result.get("canonical_geometry_authoritative") is False' in text
    assert 'result.get("certified_metrology") is False' in text
    assert '"Not certified" in page._metrology_value.text()' in text
    assert '"heavy_gaussian_or_nerf_executed_by_h5": False' in text


def test_h5_supporting_gate_reuses_integrated_session_and_recovery_acceptance() -> None:
    text = SHELL.read_text(encoding="utf-8")
    assert "test_phase8_8h1_whole_reconstruction_workflow.py" in text
    assert "test_phase8_8h3_failure_recovery_integration.py" in text
    assert "run_phase8_8h1_whole_workflow_smoke.py" in text
    assert "run_phase8_8h3_failure_recovery_smoke.py" in text
