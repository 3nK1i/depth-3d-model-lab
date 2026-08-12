from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_phase8_9e_endurance_resource_recovery.py"
UI_TEST = ROOT / "tests/ui/test_phase8_9e_endurance_resource_recovery.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase8_9e_endurance", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e_endurance_probe_is_bounded_and_does_not_import_qt_at_module_import() -> None:
    before = {name for name in sys.modules if name == "PyQt6" or name.startswith("PyQt6.")}
    module = _module()
    after = {name for name in sys.modules if name == "PyQt6" or name.startswith("PyQt6.")}
    assert callable(module._growth_within_limit)
    assert after == before
    source = SCRIPT.read_text(encoding="utf-8")
    assert "real_da3_gaussian_nerf_deferred_to" in source


def test_e_resource_growth_gate_allows_allocator_noise_but_rejects_unbounded_growth() -> None:
    module = _module()
    assert module._growth_within_limit(600.0, 700.0, 256.0)
    assert module._growth_within_limit(700.0, 690.0, 256.0)
    assert not module._growth_within_limit(600.0, 900.1, 256.0)


def test_e_heavy_runtime_list_preserves_external_isolation_boundaries() -> None:
    module = _module()
    assert set(module._HEAVY_MODULE_PREFIXES) >= {
        "pycolmap",
        "depth_anything_3",
        "gsplat",
        "nerfstudio",
        "stella_vslam",
    }


def test_e_fresh_recovery_fixture_name_cannot_retrigger_synthetic_failure() -> None:
    script_source = SCRIPT.read_text(encoding="utf-8")
    ui_source = UI_TEST.read_text(encoding="utf-8")
    assert 'offline_configuration(output, "recovered-success")' in script_source
    assert 'offline_configuration(tmp_path, "recovered-success")' in ui_source
    assert 'offline_configuration(output, "success-after-failure")' not in script_source
    assert 'offline_configuration(tmp_path, "success-after-fail")' not in ui_source
