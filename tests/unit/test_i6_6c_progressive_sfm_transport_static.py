from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRANSPORT = ROOT / "core/reconstruction/reconstruction_progressive_sfm_transport.py"
BACKEND = ROOT / "core/reconstruction/sfm_pycolmap.py"
OFFLINE_WORKER = ROOT / "scripts/run_reconstruction_offline_job.py"
OFFLINE_CONTROLLER = ROOT / "app/offline_reconstruction_controller.py"
CONTROLLER = ROOT / "app/reconstruction_controller.py"
PAGE = ROOT / "app/widgets/reconstruction_page.py"
WORKER = ROOT / "scripts/reconstruction_pycolmap_worker.py"
WORKER_PROTOCOL = ROOT / "core/reconstruction/sfm_worker_protocol.py"


def test_i66c_adds_validated_dedicated_transport_without_touching_worker_protocol() -> None:
    transport = TRANSPORT.read_text(encoding="utf-8")
    backend = BACKEND.read_text(encoding="utf-8")
    offline_worker = OFFLINE_WORKER.read_text(encoding="utf-8")
    protocol = WORKER_PROTOCOL.read_text(encoding="utf-8")

    assert 'PROGRESSIVE_SFM_EVENT_PREFIX = "DEPTH3D_SFM_PROGRESSIVE "' in transport
    assert "ProgressiveSfmSnapshot.from_payload" in transport
    assert "parse_progressive_sfm_worker_line" in backend
    assert "progressive_snapshot_callback" in backend
    assert "expected_request_fingerprint" in backend
    assert "validate_progressive_sfm_snapshot_transition" in backend
    assert "progressive_sfm_transport_line" in offline_worker
    assert "progressive_snapshot_callback=on_progressive_sfm_snapshot" in offline_worker
    assert "DEPTH3D_SFM_PROGRESSIVE " not in protocol
    assert 'SFM_WORKER_EVENT_PREFIX = "DEPTH3D_SFM_EVENT "' in protocol


def test_i66c_controller_uses_latest_only_backpressure_and_session_isolation() -> None:
    offline = OFFLINE_CONTROLLER.read_text(encoding="utf-8")
    controller = CONTROLLER.read_text(encoding="utf-8")

    assert "PROGRESSIVE_SFM_COALESCE_INTERVAL_MS = 50" in offline
    assert "ProgressiveSfmSnapshotBuffer" in offline
    assert "progressive_sparse_snapshot_received = pyqtSignal(object)" in offline
    assert "_progressive_sfm_buffer.reset(sid)" in offline
    assert "_progressive_sfm_timer.isActive()" in offline
    assert "ProgressiveSfmStage.FINALIZING" in offline

    assert "progressive_sparse_snapshot_changed = pyqtSignal(object)" in controller
    assert "snapshot.session_id != active_session_id" in controller
    assert "_validate_progressive_sparse_transition(previous, snapshot)" in controller
    assert "self._clear_progressive_sparse_snapshot()" in controller
    assert "The admitted terminal result supersedes every provisional I6.6 snapshot" in controller


def test_i66c_does_not_add_viewport_or_ui_consumption() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "progressive_sparse_snapshot_changed.connect" not in page
    assert "DEPTH3D_SFM_PROGRESSIVE" not in page
    assert "vispy" not in TRANSPORT.read_text(encoding="utf-8").lower()
    assert "persistent_canonical_scene = True" not in controller


def test_i66b_worker_remains_frozen_at_sanitized_equivalent_hash() -> None:
    import hashlib

    # Open-source namespace migration updates the worker protocol prefix to
    # the public DEPTH3D contract without claiming a new runtime qualification.
    source = WORKER.read_text(encoding="utf-8")
    assert source.count('description="Isolated PyCOLMAP SfM worker"') == 1

    digest = hashlib.sha256(WORKER.read_bytes()).hexdigest()
    assert digest == "53440a45daf19db70bb32c97f16ff9e596120e2244f13f1ea6d16b0544f7d68d"
