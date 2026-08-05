from __future__ import annotations

import pytest

from core.perception.base import BasePerceptionAdapter, FramePerceptionAdapter
from core.perception.metadata import (
    AdapterCapability,
    PerceptionCapabilities,
    PerceptionDescriptor,
    PerceptionTask,
)
from core.perception.registry import (
    PERCEPTION_MODEL_REGISTRY,
    PerceptionAdapterContractError,
    PerceptionRegistry,
)


class FrameOnlyAdapter(BasePerceptionAdapter):
    def load(self, config: dict[str, object]) -> None:
        self.config = dict(config)

    def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def get_metadata(self) -> dict[str, object]:
        return {"name": "frame-only"}

    def infer_frame(self, request):
        return request


class LifecycleOnlyAdapter(BasePerceptionAdapter):
    def load(self, config: dict[str, object]) -> None:
        return None

    def warmup(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def get_metadata(self) -> dict[str, object]:
        return {}


def _descriptor(adapter_name: str) -> PerceptionDescriptor:
    return PerceptionDescriptor(
        model_id="test-perception",
        display_name="Test Perception",
        adapter_path=f"tests.unit.test_perception_registry:{adapter_name}",
        capabilities=PerceptionCapabilities(
            tasks=(PerceptionTask.DETECTION,),
            adapter_capabilities=(AdapterCapability.FRAME_PERCEPTION,),
        ),
    )


def test_production_perception_registry_is_intentionally_empty() -> None:
    assert PERCEPTION_MODEL_REGISTRY.ids() == ()
    PERCEPTION_MODEL_REGISTRY.validate(allow_empty=True)


def test_registry_builds_only_declared_capability_protocols() -> None:
    registry = PerceptionRegistry()
    registry.register(_descriptor("FrameOnlyAdapter"))
    adapter = registry.build("test-perception")
    assert isinstance(adapter, BasePerceptionAdapter)
    assert isinstance(adapter, FramePerceptionAdapter)
    assert not hasattr(adapter, "refine_mask")


def test_registry_rejects_missing_declared_protocol() -> None:
    registry = PerceptionRegistry()
    registry.register(_descriptor("LifecycleOnlyAdapter"))
    with pytest.raises(PerceptionAdapterContractError, match="infer_frame"):
        registry.build("test-perception")
