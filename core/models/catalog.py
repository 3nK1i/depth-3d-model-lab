"""Backward-compatible stable model identifiers.

Phase 6 makes :mod:`core.models.registry` authoritative for model metadata and
adapter construction. These constants remain here because Phase 5 callers and
third-party integrations may import them directly.
"""

from __future__ import annotations

MOGE2_MODEL_ID = "moge-2"
MOGE2_DISPLAY_LABEL = "MoGe-2"

DEPTH_PRO_MODEL_ID = "depth-pro"
DEPTH_PRO_DISPLAY_LABEL = "Depth Pro"

MOCK_MODEL_ID = "development-mock"
MOCK_DISPLAY_LABEL = "DEVELOPMENT MOCK — NOT AI OUTPUT"

# Kept as a Phase 5 compatibility alias. New code should query the registry.
# Existing identifiers and ordering remain stable; new production adapters are
# inserted before the development-only mock.
SUPPORTED_MODEL_IDS = (
    MOGE2_MODEL_ID,
    DEPTH_PRO_MODEL_ID,
    MOCK_MODEL_ID,
)
