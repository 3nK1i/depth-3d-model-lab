import subprocess,sys
from core.tracking import TrackingState

def test_existing_tracking_state_values_are_unchanged():
    assert [v.value for v in TrackingState]==['untracked','tentative','confirmed','lost','reacquired','terminated']

def test_step7_imports_do_not_load_optional_frameworks():
    code="import sys; import core.tracking.pipeline; print(','.join(x for x in ('torch','ultralytics','PyQt6','vispy') if x in sys.modules))"
    result=subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,text=True)
    assert result.stdout.strip()==''

def test_version_remains_phase7_stable():
    from core.version import __version__,RELEASE_NAME
    assert __version__=='0.7.0'
    assert RELEASE_NAME=='Phase 7 Stable'
