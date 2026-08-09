from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def test_f4_keeps_future_phase_boundaries() -> None:
    source=(ROOT/"core/reconstruction/pipeline_appearance.py").read_text()
    assert "PersistentReconstructionScene" not in source
    assert "reconstruction_ui_enabled" in source
    assert "APPEARANCE_ONLY" in source


def test_f4_keeps_heavy_neural_imports_out_of_core() -> None:
    source=(ROOT/"core/reconstruction/pipeline_appearance.py").read_text()
    for token in ("import torch", "import gsplat", "import nerfstudio", "import tinycudann"):
        assert token not in source


def test_f3_and_f2_runners_remain_present() -> None:
    assert (ROOT/"scripts/run_phase8_7f3_tests.sh").is_file()
    assert (ROOT/"scripts/run_phase8_7f2_tests.sh").is_file()
