from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_e1_qprocess_supervisor_is_shell_free_group_isolated_and_bounded() -> None:
    source = (ROOT / "app/reconstruction_processing_job_controller.py").read_text(encoding="utf-8")
    assert "QProcess(" in source
    assert "setChildProcessModifier(os.setsid)" in source
    assert "os.killpg(pid, signal.SIGTERM)" in source
    assert "os.killpg(pid, signal.SIGKILL)" in source
    assert "shell=True" not in source
    assert "subprocess.Popen" not in source
    assert "A Digital Twin processing job is already active." in source
    assert "EIA_DIGITAL_TWIN_JOB_ID" in source
    assert "rejected stale/mismatched job_id" in source
    assert "outside declared output_dir" in source


def test_e1_depth_side_foundation_does_not_import_heavy_backends() -> None:
    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "app/reconstruction_processing_jobs.py",
            "app/reconstruction_processing_job_controller.py",
        )
    ).lower()
    for forbidden in (
        "import torch",
        "import nerfstudio",
        "import gsplat",
        "import depth_anything_3",
        "from nerfstudio",
        "from gsplat",
    ):
        assert forbidden not in combined


def test_e1_page_owns_shutdown_and_adapter_seam_without_build_backend_logic() -> None:
    source = (ROOT / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")
    assert "DigitalTwinProcessingJobController(self)" in source
    assert "def start_digital_twin_processing_job(" in source
    assert "def cancel_digital_twin_processing_job(" in source
    assert "self._digital_twin_processing_controller.shutdown()" in source
    assert 'self._settings_group("BUILD & PROCESSING")' in source
    # E1 establishes orchestration only; concrete product build actions are E2/E3.
    assert 'QPushButton("Build NeRF")' not in source
    assert 'QPushButton("Build Dense Geometry")' not in source
