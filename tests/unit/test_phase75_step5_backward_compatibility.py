from core.version import __version__
from core.models.registry import MODEL_REGISTRY
from core.object_geometry import extract_object_point_clouds,fit_object_geometry

def test_phase75_version_preserved(): assert __version__ == "0.7.5"
def test_existing_entrypoint_still_exported(): assert callable(extract_object_point_clouds)
def test_depth_registry_unchanged_shape(): assert len(MODEL_REGISTRY.ids()) >= 1
def test_new_entrypoint_available(): assert callable(fit_object_geometry)
