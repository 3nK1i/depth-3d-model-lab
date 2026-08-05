from core.perception.availability import PerceptionAvailabilityStatus, check_perception_availability
from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def test_mock_is_available_without_optional_models():
    assert check_perception_availability(PERCEPTION_MODEL_REGISTRY.get('development-perception-mock')).available

def test_ultralytics_availability_is_dependency_driven(monkeypatch):
    import core.perception.availability as a
    monkeypatch.setattr(a, 'find_spec', lambda name: None)
    x = check_perception_availability(PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg'))
    assert x.status is PerceptionAvailabilityStatus.MISSING_DEPENDENCY
    assert x.missing_imports == ('ultralytics',)
