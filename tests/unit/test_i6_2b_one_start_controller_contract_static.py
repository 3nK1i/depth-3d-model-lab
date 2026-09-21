from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "app" / "reconstruction_controller.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
PREFLIGHT = ROOT / "app" / "reconstruction_one_start_preflight.py"


def _block(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


def test_i6_2b_controller_imports_and_wires_i6_2a_preflight() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")

    assert "from app.reconstruction_one_start_preflight import" in source
    assert "OfflineOneStartPreflight" in source
    assert "build_offline_one_start_preflight" in source
    assert "def preflight_selected_offline(" in source
    assert "def last_offline_one_start_preflight(" in source

    selected = _block(source, "    def start_selected_offline(", "    def start_selected_da3_canonical(")
    assert "preflight_selected_offline(" in selected
    assert "_evaluate_start_guard(" in selected
    assert "self.start_offline(" in selected
    assert "_offline_one_start_guard_committed" in selected
    assert "self._last_offline_one_start_preflight" in selected


def test_i6_2b_reuses_existing_ui_start_delegation_without_page_preflight_logic() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert "session_id = controller.start_selected_offline()" in page
    assert "reconstruction_one_start_preflight" not in page
    assert "build_offline_one_start_preflight" not in page


def test_i6_2b_keeps_preflight_contract_dependency_light_and_execution_scope_unchanged() -> None:
    source = PREFLIGHT.read_text(encoding="utf-8")
    assert 'ONE_START_EXECUTION_SCOPE = "sparse_sfm_only"' in source
    for forbidden in (
        "PyQt6",
        "i5_auto",
        "AutoPolicy",
        "dense_provider_implementation_contracts",
        "OfflineReconstructionController",
        "ReconstructionController",
    ):
        assert forbidden not in source


def test_i6_2b_refactors_direct_offline_start_through_one_private_commit_path() -> None:
    source = CONTROLLER.read_text(encoding="utf-8")
    assert "def _commit_offline_start(" in source
    direct = _block(source, "    def start_offline(", "    def cancel_offline(")
    assert "_run_start_guard(" in direct
    assert "_commit_offline_start(" in direct
