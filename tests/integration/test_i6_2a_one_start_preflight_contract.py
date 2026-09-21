from __future__ import annotations

from pathlib import Path

from app.reconstruction_one_start_preflight import (
    ONE_START_MIN_FREE_DISK_BYTES,
    observe_offline_one_start_disk_free_bytes,
)
from core.reconstruction.mvs_environment import MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB


def test_i6_2a_reuses_existing_mvs_disk_safety_floor_exactly() -> None:
    assert ONE_START_MIN_FREE_DISK_BYTES == int(
        MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB * 1024**3
    )
    assert MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB == 8.0


def test_i6_2a_disk_observation_is_read_only_and_does_not_create_output_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "future" / "reconstruction" / "output"
    assert not root.exists()

    free = observe_offline_one_start_disk_free_bytes(root)

    assert free > 0
    assert not root.exists()


def test_i6_2a_contract_remains_dependency_light_when_later_controller_wiring_exists() -> None:
    project = Path(__file__).resolve().parents[2]
    controller = (project / "app/reconstruction_controller.py").read_text(encoding="utf-8")
    page = (project / "app/widgets/reconstruction_page.py").read_text(encoding="utf-8")

    # I6.2A itself remains dependency-light. I6.2B is allowed to consume the
    # contract from the controller, but UI code must continue delegating through
    # the existing controller Start entrypoint rather than importing preflight logic.
    assert "from app.reconstruction_one_start_preflight import" in controller
    assert "build_offline_one_start_preflight" in controller
    assert "reconstruction_one_start_preflight" not in page
    assert "build_offline_one_start_preflight" not in page
    assert "def start_selected_offline(" in controller
    assert "session_id = controller.start_selected_offline()" in page


def test_i6_2a_contract_contains_no_production_auto_or_provider_execution_imports() -> None:
    project = Path(__file__).resolve().parents[2]
    source = (project / "app/reconstruction_one_start_preflight.py").read_text(encoding="utf-8")

    for forbidden in (
        "i5_auto",
        "AutoPolicy",
        "dense_provider_implementation_contracts",
        "OfflineReconstructionController",
        "ReconstructionController",
        "PyQt6",
    ):
        assert forbidden not in source
