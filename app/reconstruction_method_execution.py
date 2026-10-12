"""I6.8.5C: fail-closed bridge from a verified method choice to real launches.

This module owns no worker, scene, camera, provider fallback, runtime probe, or
UI selection. The caller MUST provide an up-to-date, independently verified
MethodEligibilityContext; the existing launchers retain final preflight and
artifact-admission authority. See I6_8_5C_METHOD_EXECUTION_ROUTING.md.
"""
from __future__ import annotations

from collections.abc import Callable

from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_method_selection_contract import (
    MethodEligibilityContext,
    ReconstructionMethod,
    assess_reconstruction_method,
)


def start_eligible_reconstruction_method(
    method: ReconstructionMethod,
    *,
    context: MethodEligibilityContext,
    start_colmap_mvs: Callable[[], str],
    start_da3_quick: Callable[[], str],
    start_da3_full_scene: Callable[[], str],
) -> str:
    """Dispatch exactly one eligible existing action, never Auto or fallback.

    This is a *launch-time* gate, not a cached selection assessment. Ineligible
    or unwired methods are rejected before any launcher is invoked. Callbacks
    must refer to the existing, method-specific production actions.
    """
    eligibility = assess_reconstruction_method(method, context)
    if not eligibility.executable_via_existing_action:
        raise ReconstructionValidationError(
            f"Reconstruction method {method.value} is unavailable: "
            + ", ".join(eligibility.reason_codes)
        )
    actions = {
        ReconstructionMethod.COLMAP_MVS: start_colmap_mvs,
        ReconstructionMethod.DA3_QUICK: start_da3_quick,
        ReconstructionMethod.DA3_FULL_SCENE: start_da3_full_scene,
    }
    action = actions.get(method)
    if action is None:
        raise ReconstructionValidationError(
            f"Reconstruction method {method.value} has no wired execution action."
        )
    return action()
