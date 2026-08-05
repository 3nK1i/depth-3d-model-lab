import pytest
from core.perception.availability import check_perception_availability
from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def test_yolo_adapter_optional_dependency_isolated():
    d = PERCEPTION_MODEL_REGISTRY.get('ultralytics-yolo-seg')
    a = check_perception_availability(d)
    if a.available:
        pytest.skip('Real Ultralytics smoke test belongs to scripts/run_perception_smoke.py on target hardware.')
    adapter = PERCEPTION_MODEL_REGISTRY.build(d.model_id)
    with pytest.raises(Exception, match='Ultralytics'):
        adapter.load({'checkpoint': 'yolo11n-seg.pt', 'device': 'cpu'})
