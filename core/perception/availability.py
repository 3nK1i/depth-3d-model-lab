"""Optional dependency availability without importing model runtimes."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from importlib.util import find_spec
from core.perception.metadata import PerceptionDescriptor

class PerceptionAvailabilityStatus(str, Enum):
    AVAILABLE = 'available'
    DISABLED = 'disabled'
    MISSING_DEPENDENCY = 'missing_dependency'
    ERROR = 'error'

@dataclass(frozen=True, slots=True)
class PerceptionAvailability:
    model_id: str
    status: PerceptionAvailabilityStatus
    message: str = ''
    missing_imports: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status is PerceptionAvailabilityStatus.AVAILABLE

def check_perception_availability(descriptor: PerceptionDescriptor) -> PerceptionAvailability:
    if not descriptor.enabled:
        return PerceptionAvailability(descriptor.model_id, PerceptionAvailabilityStatus.DISABLED, 'Descriptor is disabled.')
    missing = []
    for dep in descriptor.dependencies:
        try:
            if find_spec(dep.import_name) is None:
                missing.append(dep.import_name)
        except (ImportError, ValueError):
            missing.append(dep.import_name)
    if missing:
        return PerceptionAvailability(descriptor.model_id, PerceptionAvailabilityStatus.MISSING_DEPENDENCY, 'Optional perception dependency is not installed.', tuple(missing))
    return PerceptionAvailability(descriptor.model_id, PerceptionAvailabilityStatus.AVAILABLE)
