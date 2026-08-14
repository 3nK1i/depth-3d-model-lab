"""Independent lazy perception-model registry."""
from __future__ import annotations
from collections import OrderedDict
from importlib import import_module
from typing import Any, Iterable
from core.perception.base import AutomaticSegmentationAdapter, BasePerceptionAdapter, FramePerceptionAdapter, PromptedSegmentationAdapter, SequencePerceptionAdapter
from core.perception.metadata import AdapterCapability, PerceptionDescriptor

class PerceptionRegistryError(ValueError):
    pass

class UnknownPerceptionModelError(PerceptionRegistryError):
    pass

class DuplicatePerceptionModelError(PerceptionRegistryError):
    pass

class PerceptionAdapterContractError(TypeError):
    pass

class PerceptionRegistry:

    def __init__(self) -> None:
        self._descriptors: OrderedDict[str, PerceptionDescriptor] = OrderedDict()

    def register(self, descriptor: PerceptionDescriptor) -> None:
        if descriptor.model_id in self._descriptors:
            raise DuplicatePerceptionModelError(f'Perception model id {descriptor.model_id!r} is already registered.')
        self._descriptors[descriptor.model_id] = descriptor

    def get(self, model_id: str) -> PerceptionDescriptor:
        try:
            return self._descriptors[str(model_id)]
        except KeyError:
            raise UnknownPerceptionModelError(f'Unknown perception model id: {model_id!r}') from None

    def descriptors(self, *, selectable_only: bool=False, enabled_only: bool=False) -> tuple[PerceptionDescriptor, ...]:
        values: Iterable[PerceptionDescriptor] = self._descriptors.values()
        if selectable_only:
            values = (v for v in values if v.selectable)
        if enabled_only:
            values = (v for v in values if v.enabled)
        return tuple(values)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def validate(self, *, allow_empty: bool=True) -> None:
        if not self._descriptors and (not allow_empty):
            raise PerceptionRegistryError('Perception registry is empty.')
        for d in self._descriptors.values():
            self._validate_descriptor(d)

    def build(self, model_id: str) -> BasePerceptionAdapter:
        d = self.get(model_id)
        cls = _import_symbol(d.adapter_path)
        if not isinstance(cls, type):
            raise PerceptionAdapterContractError(f'Adapter path {d.adapter_path!r} did not resolve to a class.')
        adapter = cls()
        if not isinstance(adapter, BasePerceptionAdapter):
            raise PerceptionAdapterContractError(f'Adapter {d.adapter_path!r} does not implement BasePerceptionAdapter.')
        self._validate_adapter_capabilities(adapter, d)
        return adapter

    @staticmethod
    def _validate_descriptor(d):
        if not isinstance(d, PerceptionDescriptor):
            raise PerceptionRegistryError('Registry entries must be PerceptionDescriptor values.')

    @staticmethod
    def _validate_adapter_capabilities(adapter, d):
        checks = ((AdapterCapability.FRAME_PERCEPTION, FramePerceptionAdapter, 'infer_frame'), (AdapterCapability.PROMPTED_SEGMENTATION, PromptedSegmentationAdapter, 'refine_mask'), (AdapterCapability.AUTOMATIC_SEGMENTATION, AutomaticSegmentationAdapter, 'propose_masks'), (AdapterCapability.SEQUENCE_PERCEPTION, SequencePerceptionAdapter, 'reset_sequence'))
        for capability, protocol, method in checks:
            if capability in d.capabilities.adapter_capabilities and (not isinstance(adapter, protocol)):
                raise PerceptionAdapterContractError(f'Adapter for {d.model_id!r} declares {capability.value!r} but does not implement {method}().')

def _import_symbol(path: str) -> Any:
    module, symbol = path.split(':', 1)
    return getattr(import_module(module), symbol)
PERCEPTION_MODEL_REGISTRY = PerceptionRegistry()
from core.perception.catalog import register_builtin_perception_models
register_builtin_perception_models(PERCEPTION_MODEL_REGISTRY)
