from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIVE_RUNTIME = ROOT / "core/reconstruction/live_runtime.py"
MAIN = ROOT / "app/main_window.py"
CONTROLLER = ROOT / "app/reconstruction_controller.py"
OFFLINE = ROOT / "app/offline_reconstruction_controller.py"
RECOVERY = ROOT / "app/reconstruction_failure_recovery.py"
PAGE = ROOT / "app/widgets/reconstruction_page.py"


def test_h3_closes_live_failure_resource_release_gap_without_new_recovery_engine() -> None:
    text = LIVE_RUNTIME.read_text(encoding="utf-8")
    assert "terminal_failure = False" in text
    assert "terminal_failure = True" in text
    assert "self._set_state(LiveCaptureRuntimeState.STOPPING)" in text
    assert "if terminal_failure" in text
    assert "LiveCaptureRuntimeState.FAILED" in text


def test_h3_keeps_existing_bounded_live_recovery_budget() -> None:
    text = LIVE_RUNTIME.read_text(encoding="utf-8")
    assert "def _recover_backend(" in text
    assert "policy.max_worker_restarts" in text
    assert "Live SLAM worker recovery budget exhausted" in text


def test_h3_offline_controller_reserves_resources_until_run_finalization() -> None:
    text = OFFLINE.read_text(encoding="utf-8")
    assert "return self.is_running or not self._run_finalized" in text
    assert "self._set_state(state)" in text
    assert "self._run_finalized = True" in text
    assert "self.run_finished.emit(" in text


def test_h3_application_observes_offline_run_finished_for_resource_unlock() -> None:
    text = MAIN.read_text(encoding="utf-8")
    assert "self._reconstruction_controller.offline_run_finished.connect(" in text
    assert "self._on_reconstruction_activity_changed" in text
    assert "self._reconstruction_controller.is_active" in text


def test_h3_shutdown_order_stops_reconstruction_before_inference_and_capture() -> None:
    text = MAIN.read_text(encoding="utf-8")
    group = text[text.index("def closeEvent("):]
    reconstruction = group.index("self._reconstruction_controller.shutdown()")
    viewport = group.index("self._reconstruction_page.shutdown_viewport()")
    inference = group.index("self._inference_controller.shutdown()")
    capture = group.index("self._controller.shutdown()")
    assert reconstruction < viewport < inference < capture


def test_h3_controller_shutdown_reuses_existing_live_offline_and_persistence_owners() -> None:
    text = CONTROLLER.read_text(encoding="utf-8")
    group = text[text.index("def shutdown("):]
    assert "self._offline_controller.shutdown()" in group
    assert "self._live_slam_controller.shutdown()" in group
    assert "self._persistence_controller.shutdown()" in group


def test_h3_recovery_keeps_fresh_retry_and_checkpoint_resume_separate() -> None:
    recovery = RECOVERY.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "retry_starts_new_session=True" in recovery
    assert "Prepare Resume" in page
    assert "Retry From Source" in page
    assert "not checkpoint resume" in page


def test_h3_preserved_evidence_blocks_ambiguous_fresh_retry() -> None:
    recovery = RECOVERY.read_text(encoding="utf-8")
    assert "if has_scene or has_checkpoint:" in recovery
    assert "can_retry_from_source=False" in recovery
    assert "PRESERVED_EVIDENCE" in recovery


def test_h3_changes_no_authority_or_metrology_policy() -> None:
    recovery = RECOVERY.read_text(encoding="utf-8")
    assert "certified_metrology=False" in recovery
    assert "canonical_geometry_authoritative" in recovery
    text = LIVE_RUNTIME.read_text(encoding="utf-8").lower()
    for forbidden in ("certified_metrology", "gaussian", "nerf", "semantic_geometry"):
        assert forbidden not in text


def test_h3_adds_no_device_specific_resource_thresholds() -> None:
    text = LIVE_RUNTIME.read_text(encoding="utf-8").lower()
    for forbidden in ("rtx 2060", "rtx2060", "rtx 4090", "rtx4090", "rtx 5090", "rtx5090", "vram"):
        assert forbidden not in text
