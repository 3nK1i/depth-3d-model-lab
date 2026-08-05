from core.perception.registry import PERCEPTION_MODEL_REGISTRY

def test_mock_registry_build_unload_stress():
    for _ in range(100):
        a = PERCEPTION_MODEL_REGISTRY.build('development-perception-mock')
        a.load({})
        a.warmup()
        a.unload()
