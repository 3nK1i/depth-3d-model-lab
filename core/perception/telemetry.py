"""Dependency-light perception telemetry records."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class PerceptionTelemetry:
    requests: int = 0
    successes: int = 0
    failures: int = 0
    dropped_requests: int = 0
    last_processing_ms: float = 0.0
