from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
from core.models.registry import MODEL_REGISTRY
from core.models.result import GeometryResult
from core.runtime.messages import InferenceResult
from core.contracts.common import FrameReference, ImageSize
from core.perception.result import ObjectObservation
from core.version import RELEASE_NAME, __version__
EXPECTED = ('moge-2', 'depth-pro', 'unidepth-v2', 'metric3d-v2', 'depth-anything-v2', 'video-depth-anything', 'marigold', 'development-mock')

def test_depth_registry_and_legacy_results_unchanged():
    assert MODEL_REGISTRY.ids() == EXPECTED
    g = GeometryResult(1, 2, 3.0, np.zeros((2, 3, 3), np.uint8), np.ones((2, 3), np.float32), None, None, None, None, None, None, 1, 4.0, 5.0, 6.0, 7.0, 8.0, True, 'legacy', 'Legacy', 'FP32', 0.1, 0.2, 0.3, 0.4, 0.5, 6.0)
    assert g.model_id == 'legacy'
    assert pickle.loads(pickle.dumps(InferenceResult(1, 2, 'development-mock', None, 3.0))).model_id == 'development-mock'

def test_ui_runtime_and_release_remain_unchanged():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'app/widgets/mode_tabs.py').read_text()
    assert '_ENABLED_MODES = {AppMode.LIVE_3D, AppMode.INSPECTION, AppMode.STABLE_VIDEO, AppMode.SNAPSHOT, AppMode.BENCHMARK}' in source
    assert 'AppMode.RECONSTRUCTION' not in source.split('_ENABLED_MODES =', 1)[1].split('\n', 1)[0]
    assert __version__ == '0.9.0'
    assert RELEASE_NAME == 'Phase 9 Digital Twin Stable'

def test_optional_ultralytics_imports_are_adapter_local():
    root = Path(__file__).resolve().parents[2]
    for path in (root / 'core/perception').glob('*.py'):
        text = path.read_text()
        assert 'from ultralytics' not in text and 'import ultralytics' not in text
    assert 'ultralytics-perception' in (root / 'pyproject.toml').read_text()

def test_generic_worker_contains_no_model_id_branches():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'core/perception/worker.py').read_text()
    assert 'ultralytics-yolo' not in source
    assert 'sam2' not in source.lower()
    assert 'development-perception-mock' not in source


def test_object_observation_step1_positional_constructor_is_append_compatible():
    frame = FrameReference(1, 2, 3.0, ImageSize(4, 3))
    observation = ObjectObservation(
        "object",
        frame,
        1,
        "part",
        None,
        (),
        "",
        "track",
        ("left",),
        0.25,
        ("legacy warning",),
    )
    assert observation.track_id == "track"
    assert observation.refinements == ()
