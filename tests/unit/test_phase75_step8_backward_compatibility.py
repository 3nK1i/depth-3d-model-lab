from pathlib import Path
import subprocess
import sys

from core.models.registry import MODEL_REGISTRY
from core.version import __version__


def test_step8_keeps_version_depth_registry_and_reconstruction_contract():
    assert __version__ == "0.7.0"
    ids = MODEL_REGISTRY.ids()
    assert "development-mock" in ids
    source = Path("app/widgets/mode_tabs.py").read_text(encoding="utf-8")
    assert "AppMode.INSPECTION" in source
    enabled_section = source.split("_ENABLED_MODES", 1)[1].split("class ModeTabs", 1)[0]
    assert "AppMode.RECONSTRUCTION" not in enabled_section


def test_core_inspection_import_is_dependency_light_in_fresh_process():
    code = """
import sys
import core.inspection
blocked = [name for name in ('torch','ultralytics','PyQt6','vispy') if name in sys.modules]
if blocked:
    raise SystemExit(','.join(blocked))

"""
    completed = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr or completed.stdout
