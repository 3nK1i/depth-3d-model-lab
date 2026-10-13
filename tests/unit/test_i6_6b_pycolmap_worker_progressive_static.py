from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "reconstruction_pycolmap_worker.py"
PROTOCOL = ROOT / "core" / "reconstruction" / "sfm_worker_protocol.py"
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"


def test_worker_enables_native_incremental_snapshots_without_changing_sfm_algorithms() -> None:
    text = WORKER.read_text(encoding="utf-8")
    assert "mapping.snapshot_path" in text
    assert "mapping.snapshot_frames_freq" in text
    assert "mapping_snapshot_frame_interval" in text
    assert "pycolmap.incremental_mapping(" in text
    assert "multiple_models = True" in text
    assert "max_num_models = 4" in text
    assert "min_model_size = 3" in text


def test_worker_progressive_transport_is_separate_from_existing_event_protocol() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "DEPTH3D_SFM_PROGRESSIVE " in worker or "progressive_event_line" in worker
    assert "DEPTH3D_SFM_PROGRESSIVE " not in protocol
    assert 'SFM_WORKER_EVENT_PREFIX = "EIA_SFM_EVENT "' in protocol


def test_i66b_does_not_wire_controller_or_ui_consumption() -> None:
    controller = CONTROLLER.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    assert "DEPTH3D_SFM_PROGRESSIVE" not in controller
    assert "progressive_sfm_snapshot" not in controller.lower()
    assert "DEPTH3D_SFM_PROGRESSIVE" not in page


def test_worker_does_not_import_application_core_or_ml_gui_dependencies() -> None:
    text = WORKER.read_text(encoding="utf-8")
    for forbidden in ("from app", "import app", "from core", "import core", "PyQt6", "torch", "cv2"):
        assert forbidden not in text
