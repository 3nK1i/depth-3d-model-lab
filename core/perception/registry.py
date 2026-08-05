"""Independent, lazy perception-model registry with no production models in Step 1."""

from __future__ import annotations

from collections import OrderedDict
from importlib import import_module
from typing import Any, Iterable

from core.perception.base import (
    BasePerceptionAdapter,
    FramePerceptionAdapter,
    PromptedSegmentationAdapter,
    SequencePerceptionAdapter,
)
from core.perception.metadata import (
    AdapterCapability,
    PerceptionDescriptor,
)


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
            raise DuplicatePerceptionModelError(
                f"Perception model id {descriptor.model_id!r} is already registered."
            )
        self._descriptors[descriptor.model_id] = descriptor

    def get(self, model_id: str) -> PerceptionDescriptor:
        try:
            return self._descriptors[str(model_id)]
        except KeyError:
            raise UnknownPerceptionModelError(
                f"Unknown perception model id: {model_id!r}"
            ) from None

    def descriptors(
        self,
        *,
        selectable_only: bool = False,
        enabled_only: bool = False,
    ) -> tuple[PerceptionDescriptor, ...]:
        values: Iterable[PerceptionDescriptor] = self._descriptors.values()
        if selectable_only:
            values = (value for value in values if value.selectable)
        if enabled_only:
            values = (value for value in values if value.enabled)
        return tuple(values)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def validate(self, *, allow_empty: bool = True) -> None:
        if not self._descriptors and not allow_empty:
            raise PerceptionRegistryError("Perception registry is empty.")
        for descriptor in self._descriptors.values():
            self._validate_descriptor(descriptor)

    def build(self, model_id: str) -> BasePerceptionAdapter:
        descriptor = self.get(model_id)
        adapter_class = _import_symbol(descriptor.adapter_path)
        if not isinstance(adapter_class, type):
            raise PerceptionAdapterContractError(
                f"Adapter path {descriptor.adapter_path!r} did not resolve to a class."
            )
        adapter = adapter_class()
        if not isinstance(adapter, BasePerceptionAdapter):
            raise PerceptionAdapterContractError(
                f"Adapter {descriptor.adapter_path!r} does not implement BasePerceptionAdapter."
            )
        self._validate_adapter_capabilities(adapter, descriptor)
        return adapter

    @staticmethod
    def _validate_descriptor(descriptor: PerceptionDescriptor) -> None:
        if not isinstance(descriptor, PerceptionDescriptor):
            raise PerceptionRegistryError("Registry entries must be PerceptionDescriptor values.")

    @staticmethod
    def _validate_adapter_capabilities(
        adapter: BasePerceptionAdapter,
        descriptor: PerceptionDescriptor,
    ) -> None:
        checks: tuple[tuple[AdapterCapability, type, str], ...] = (
            (AdapterCapability.FRAME_PERCEPTION, FramePerceptionAdapter, "infer_frame"),
            (
                AdapterCapability.PROMPTED_SEGMENTATION,
                PromptedSegmentationAdapter,
                "refine_mask",
            ),
            (
                AdapterCapability.SEQUENCE_PERCEPTION,
                SequencePerceptionAdapter,
                "reset_sequence",
            ),
        )
        for capability, protocol, method_name in checks:
            declared = capability in descriptor.capabilities.adapter_capabilities
            implemented = isinstance(adapter, protocol)
            if declared and not implemented:
                raise PerceptionAdapterContractError(
                    f"Adapter for {descriptor.model_id!r} declares {capability.value!r} "
                    f"but does not implement {method_name}()."
                )


PERCEPTION_MODEL_REGISTRY = PerceptionRegistry()


def _import_symbol(path: str) -> Any:
    module_name, symbol_name = path.split(":", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)
