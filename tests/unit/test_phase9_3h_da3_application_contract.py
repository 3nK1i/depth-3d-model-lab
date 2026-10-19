from __future__ import annotations

from pathlib import Path

from app.da3_canonical_integration_controller import (
    default_da3_python_path,
    default_da3_repository_path,
)


def _text(relative: str) -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / relative).read_text(encoding="utf-8")


def test_3h_da3_worker_boundary_keeps_external_runtime_isolated() -> None:
    controller = _text("app/da3_canonical_integration_controller.py")
    worker = _text("scripts/run_phase9_3h_da3_canonical_job.py")
    assert "DEPTH3D_RECONSTRUCTION_DA3_PYTHON" in controller
    assert "DEPTH3D_RECONSTRUCTION_DA3_REPOSITORY" in controller
    assert '"--da3-python"' in controller
    assert '"--da3-repo"' in controller
    assert "DA3ExternalWorkerRunner" in worker
    assert "probe_da3_environment" in worker
    assert "_qualified_da3_import_source" in worker
    assert "inspect.getfile(DepthAnything3)" in worker
    assert "imported.relative_to(repository)" in worker
    assert "pip install" not in worker
    assert "conda install" not in worker


def test_3h_da3_cancellation_targets_the_worker_process_group() -> None:
    source = _text("app/da3_canonical_integration_controller.py")
    assert "setChildProcessModifier(os.setsid)" in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "os.killpg(pid, signal.SIGKILL)" in source
    assert "self._terminate_process_group()" in source


def test_3h_application_resource_interlock_tracks_da3_activity() -> None:
    controller = _text("app/reconstruction_controller.py")
    window = _text("app/main_window.py")
    assert "or self._da3_canonical_controller.is_active" in controller
    assert "da3_canonical_state_changed.connect" in window
    assert "da3_canonical_run_finished.connect" in window


def test_3h_ui_keeps_provider_selection_separate_from_explicit_execution() -> None:
    page = _text("app/widgets/reconstruction_page.py")
    assert "Provider selection is inspection-only" in page
    assert 'QPushButton("Integrate DA3")' in page
    assert "start_selected_da3_canonical()" in page
    assert "calibration is optional" in page.lower()
