from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "app/main_window.py"
INFERENCE = ROOT / "app/inference_controller.py"
OFFLINE = ROOT / "app/offline_reconstruction_controller.py"


def _function_block(text: str, name: str, next_name: str | None = None) -> str:
    start = text.index(f"    def {name}(")
    if next_name is None:
        return text[start:]
    end = text.index(f"    def {next_name}(", start + 1)
    return text[start:end]


def test_h2_reuses_a5_reconstruction_start_guards_and_model_release_boundary() -> None:
    text = MAIN.read_text(encoding="utf-8")
    assert "configure_start_guards(" in text
    assert "live=self._reconstruction_live_start_guard" in text
    assert "offline=self._reconstruction_offline_start_guard" in text
    offline = _function_block(text, "_reconstruction_offline_start_guard", "_on_reconstruction_activity_changed")
    assert "release_model_for_external_job()" in offline


def test_h2_adds_one_reciprocal_application_admission_helper() -> None:
    text = MAIN.read_text(encoding="utf-8")
    block = _function_block(text, "_reconstruction_conflict_for_application_job", "_reconstruction_live_start_guard")
    assert "self._reconstruction_controller.is_active" in block
    assert "Reconstruction is active; stop it before starting" in block
    assert "Persistent scenes" in block


def test_h2_reciprocal_guard_covers_every_existing_long_running_application_entry() -> None:
    text = MAIN.read_text(encoding="utf-8")
    boundaries = (
        ("_inspection_jobs_conflict", "_start_inspection_analysis"),
        ("_export_inspection", "_on_inspection_export_state_changed"),
        ("_start_stable_video", "_pause_resume_stable_video"),
        ("_run_snapshot", "_on_snapshot_state_changed"),
        ("_on_benchmark_run_requested", "_on_benchmark_state_changed"),
    )
    for name, next_name in boundaries:
        block = _function_block(text, name, next_name)
        assert "_reconstruction_conflict_for_application_job(" in block, name


def test_h2_readiness_surfaces_mirror_the_same_resource_owner() -> None:
    text = MAIN.read_text(encoding="utf-8")
    stable = _function_block(text, "_refresh_stable_video_context", "_start_stable_video")
    benchmark = _function_block(text, "_refresh_benchmark_safety", "_toggle_panel_focus")
    activity = _function_block(text, "_on_reconstruction_activity_changed", "_on_mode_changed")
    assert "_reconstruction_conflict_for_application_job(" in stable
    assert "_reconstruction_conflict_for_application_job(" in benchmark
    assert "_refresh_stable_video_context()" in activity
    assert "_refresh_benchmark_safety()" in activity


def test_h2_live_sharing_and_offline_exclusive_inference_semantics_stay_distinct() -> None:
    text = MAIN.read_text(encoding="utf-8")
    live = _function_block(text, "_reconstruction_live_start_guard", "_reconstruction_offline_start_guard")
    offline = _function_block(text, "_reconstruction_offline_start_guard", "_on_reconstruction_activity_changed")
    assert "release_model_for_external_job" not in live
    assert "release_model_for_external_job" in offline
    assert "ApplicationState.RUNNING" in offline


def test_h2_does_not_add_device_specific_vram_policy_or_heavy_backend_linkage() -> None:
    text = MAIN.read_text(encoding="utf-8")
    helper = _function_block(text, "_reconstruction_conflict_for_application_job", "_reconstruction_live_start_guard").lower()
    for forbidden in ("rtx 2060", "rtx 4090", "rtx 5090", "nvidia-smi", "torch.cuda", "6 gb", "32 gb"):
        assert forbidden not in helper
    lowered = text.lower()
    for forbidden_import in (
        "import pycolmap",
        "from pycolmap",
        "import stella_vslam",
        "from stella_vslam",
        "import gsplat",
        "from gsplat",
        "import nerfstudio",
        "from nerfstudio",
        "import gaussian_renderer",
        "from gaussian_renderer",
    ):
        assert forbidden_import not in lowered


def test_h2_reuses_existing_inference_release_implementation() -> None:
    text = INFERENCE.read_text(encoding="utf-8")
    assert "def release_model_for_external_job(" in text
    block = text[text.index("    def release_model_for_external_job("):text.index("    # -- Runtime toolbar handlers", text.index("    def release_model_for_external_job("))]
    assert "self._process_manager.shutdown()" in block
    assert "self._model_lifecycle.reset()" in block
    assert "SOURCE_READY" in block
    assert "DISCONNECTED" in text


def test_h2_does_not_rewrite_offline_process_cleanup_or_active_ownership() -> None:
    text = OFFLINE.read_text(encoding="utf-8")
    assert "def is_active(" in text
    assert "return self.is_running or not self._run_finalized" in text
    assert "def cancel(" in text
    assert "def shutdown(" in text
    assert "QProcess" in text


def test_h2_keeps_f5_appearance_resources_advisory_and_backend_free() -> None:
    appearance = ROOT / "app/reconstruction_appearance_resources.py"
    assert appearance.is_file()
    text = appearance.read_text(encoding="utf-8").lower()
    assert "static previews remain supported" in text
    assert "does not import, probe," in text
    assert "or launch an external gaussian/nerf runtime" in text
    assert "cross_representation_fallback" in text
    assert "canonical geometry" in text
    for forbidden_import in (
        "import gsplat",
        "from gsplat",
        "import nerfstudio",
        "from nerfstudio",
        "import gaussian_renderer",
        "from gaussian_renderer",
    ):
        assert forbidden_import not in text
