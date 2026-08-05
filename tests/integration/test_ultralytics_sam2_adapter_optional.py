import pytest
from core.perception.availability import check_perception_availability
from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def test_sam_adapter_optional_dependency_isolated():
    d = PERCEPTION_MODEL_REGISTRY.get('ultralytics-sam2-prompted')
    a = check_perception_availability(d)
    if a.available:
        pytest.skip('Real SAM smoke test belongs to target hardware validation.')
    adapter = PERCEPTION_MODEL_REGISTRY.build(d.model_id)
    with pytest.raises(Exception, match='Ultralytics'):
        adapter.load({'checkpoint': 'sam2.1_t.pt', 'device': 'cpu'})
