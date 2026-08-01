"""Stable model identifiers and user-facing labels.

This module intentionally has no heavyweight imports. UI code, worker code and
model adapters can all import these constants without importing PyTorch or a
model package in the Qt process.
"""

from __future__ import annotations

MOGE2_MODEL_ID = "moge-2"
MOGE2_DISPLAY_LABEL = "MoGe-2"

MOCK_MODEL_ID = "development-mock"
MOCK_DISPLAY_LABEL = "DEVELOPMENT MOCK — NOT AI OUTPUT"

SUPPORTED_MODEL_IDS = (MOGE2_MODEL_ID, MOCK_MODEL_ID)
