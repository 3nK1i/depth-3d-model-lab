"""Typed errors for the dependency-light reconstruction foundation."""

from __future__ import annotations

from core.contracts.validation import ContractValidationError


class ReconstructionError(Exception):
    """Base class for reconstruction-specific runtime failures."""


class ReconstructionValidationError(ContractValidationError):
    """Raised when reconstruction metadata violates a public contract."""


class InvalidSessionTransitionError(ReconstructionError):
    """Raised when a reconstruction lifecycle transition is not allowed."""


class UnsupportedBackendOperationError(ReconstructionError):
    """Raised when a backend does not declare support for an operation."""


class IncompatibleCheckpointError(ReconstructionError):
    """Raised when checkpoint metadata cannot safely resume a session."""


class ReconstructionCancelledError(ReconstructionError):
    """Raised cooperatively when a future backend observes cancellation."""


class SfmBackendUnavailableError(ReconstructionError):
    """Raised when the optional external offline SfM backend is unavailable."""


class SfmWorkspaceError(ReconstructionError):
    """Raised when an SfM workspace cannot be materialized safely."""
