"""I4.3H1 final provider-architecture qualification and audit contracts.

I4.3H1 is deliberately execution-free with respect to reconstruction backends.
It validates the formally closed I4.3G evidence chain, re-audits the current
canonical provider architecture/adaptive-planning contracts, and binds a full
I4.3 predecessor regression receipt.  It does *not* formally close I4.3; that
responsibility belongs exclusively to I4.3H2.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import re
from typing import Any, Mapping, Sequence

from core.reconstruction.adaptive_view_selection import provider_view_selection_requirements
from core.reconstruction.dense_provider_architecture import (
    DenseProviderFamily,
    DenseProviderImplementation,
    dense_provider_family_contracts,
    dense_provider_implementation_contracts,
)
from core.reconstruction.errors import ReconstructionValidationError

I43H1_SCHEMA_VERSION = 1
I43H1_PHASE = "I4.3H1"
I43H1_REGRESSION_PHASE = "I4.3H1_REGRESSION_AUDIT"
I43H1_EXPECTED_PREDECESSOR_UNIT_TESTS = 444
I43H1_EXPECTED_PREDECESSOR_INTEGRATION_TESTS = 86
I43H1_EXPECTED_PREDECESSOR_STRESS_TESTS = 138
I43H1_EXPECTED_PREDECESSOR_TOTAL_TESTS = (
    I43H1_EXPECTED_PREDECESSOR_UNIT_TESTS
    + I43H1_EXPECTED_PREDECESSOR_INTEGRATION_TESTS
    + I43H1_EXPECTED_PREDECESSOR_STRESS_TESTS
)
I43H1_EXPECTED_PREDECESSOR_TEST_FILES = 91

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_IMPLEMENTATIONS = (
    "colmap_mvs",
    "da3",
    "rolling_depth",
    "video_depth_anything",
    "metric3d_v2",
    "unidepth_v2",
    "moge_2",
    "depth_pro",
    "depth_anything_v2",
    "marigold",
)

_EXPECTED_IMPLEMENTATION_CONTRACT = {
    "colmap_mvs": ("mvs", "I4.3B"),
    "da3": ("multi_view_ai_depth", "I4.3C"),
    "rolling_depth": ("temporal_depth", "I4.3D"),
    "video_depth_anything": ("temporal_depth", "I4.3D"),
    "metric3d_v2": ("single_image_depth", "I4.3E"),
    "unidepth_v2": ("single_image_depth", "I4.3E"),
    "moge_2": ("single_image_depth", "I4.3E"),
    "depth_pro": ("single_image_depth", "I4.3E"),
    "depth_anything_v2": ("single_image_depth", "I4.3E"),
    "marigold": ("single_image_depth", "I4.3E"),
}

# Static import/symbol audit.  These are contract surfaces only; H1 never invokes
# runtime probes or backend execution.
_PROVIDER_SURFACES = {
    "colmap_mvs": (
        "core.reconstruction.mvs_provider",
        "MvsProductionProviderAdapter",
        "colmap_mvs_provider_descriptor",
        "validate_existing_i4_2_mvs_binding",
    ),
    "da3": (
        "core.reconstruction.da3_dense_provider",
        "DA3ProductionProviderAdapter",
        "da3_dense_provider_descriptor",
        "validate_i4_3c_binding",
    ),
    "rolling_depth": (
        "core.reconstruction.rolling_depth_dense_provider",
        "RollingDepthProductionProviderAdapter",
        "rolling_depth_dense_provider_descriptor",
        "validate_i4_3d2_binding",
    ),
    "video_depth_anything": (
        "core.reconstruction.video_depth_anything_dense_provider",
        "VideoDepthAnythingProductionProviderAdapter",
        "video_depth_anything_dense_provider_descriptor",
        "validate_i4_3d1_binding",
    ),
    "metric3d_v2": (
        "core.reconstruction.metric3d_v2_dense_provider",
        "Metric3DV2ProductionProviderAdapter",
        "metric3d_v2_dense_provider_descriptor",
        "validate_i4_3e1_binding",
    ),
    "unidepth_v2": (
        "core.reconstruction.unidepth_v2_dense_provider",
        "UniDepthV2ProductionProviderAdapter",
        "unidepth_v2_dense_provider_descriptor",
        "validate_i4_3e2_binding",
    ),
    "moge_2": (
        "core.reconstruction.moge2_dense_provider",
        "MoGe2ProductionProviderAdapter",
        "moge_2_dense_provider_descriptor",
        "validate_i4_3e3_binding",
    ),
    "depth_pro": (
        "core.reconstruction.depth_pro_dense_provider",
        "DepthProProductionProviderAdapter",
        "depth_pro_dense_provider_descriptor",
        "validate_i4_3e4_binding",
    ),
    "depth_anything_v2": (
        "core.reconstruction.depth_anything_v2_dense_provider",
        "DepthAnythingV2ProductionProviderAdapter",
        "depth_anything_v2_dense_provider_descriptor",
        "validate_i4_3e5_binding",
    ),
    "marigold": (
        "core.reconstruction.marigold_dense_provider",
        "MarigoldProductionProviderAdapter",
        "marigold_dense_provider_descriptor",
        "validate_i4_3e6_binding",
    ),
}

_EXPECTED_TERMINAL_QUALITY_EVENTS = {
    ("temporal_depth", "g3_temporal_static_sweep", "rolling_depth", "degraded"),
    ("temporal_depth", "g3_temporal_motion", "rolling_depth", "rejected"),
    ("temporal_depth", "g3_temporal_motion", "video_depth_anything", "degraded"),
    ("single_image_depth", "g4_single_image_low_texture", "depth_anything_v2", "rejected"),
    ("single_image_depth", "g4_single_image_low_texture", "marigold", "rejected"),
}

_AUTHORITY_FALSE = (
    "authoritative_sfm_modified",
    "provider_selected",
    "provider_promoted",
    "auto_policy_implemented",
    "hybrid_authorized",
    "metric_scale_claimed",
    "measurement_authoritative",
    "certified_metrology",
    "surface_mesh_generated",
)

_DECISION_FALSE = (
    "provider_selected",
    "provider_promoted",
    "provider_ranking_created",
    "universal_provider_winner_declared",
    "global_scalar_score_created",
    "auto_policy_implemented",
    "hybrid_authorized",
)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"I4.3H1 {label} must be an object.")
    return value


def _sequence(value: Any, *, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReconstructionValidationError(f"I4.3H1 {label} must be an array.")
    return value


def _require(mapping: Mapping[str, Any], key: str, expected: Any, *, label: str) -> None:
    if mapping.get(key) != expected:
        raise ReconstructionValidationError(
            f"I4.3H1 {label}.{key} contract drift: expected={expected!r} actual={mapping.get(key)!r}."
        )


def _validate_sha_binding(value: Any, *, label: str) -> dict[str, str]:
    binding = _mapping(value, label=label)
    path = str(binding.get("path", "")).strip()
    sha = str(binding.get("sha256", "")).strip().lower()
    if not path:
        raise ReconstructionValidationError(f"I4.3H1 {label} path is empty.")
    if not _SHA256_RE.fullmatch(sha):
        raise ReconstructionValidationError(f"I4.3H1 {label} SHA256 is malformed.")
    return {"path": path, "sha256": sha}


def _terminal_events(value: Any) -> tuple[tuple[str, str, str, str], ...]:
    events: list[tuple[str, str, str, str]] = []
    for raw in _sequence(value, label="terminal_quality_events"):
        item = _mapping(raw, label="terminal quality event")
        events.append((
            str(item.get("family_id", "")),
            str(item.get("scenario_id", "")),
            str(item.get("provider_implementation", "")),
            str(item.get("production_status", "")),
        ))
    return tuple(events)


def validate_i4_3g5b_closure(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Independently audit the frozen G5B formal closure before H work."""

    if (
        payload.get("phase") != "I4.3G5B"
        or payload.get("status") != "formally_closed"
        or payload.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3H1 requires the passing formal I4.3G5B closure.")

    scope = _mapping(payload.get("scope"), label="G5B scope")
    _require(scope, "evidence_only", True, label="G5B scope")
    for key in ("provider_execution_started", "sfm_rerun", "matched_view_rerun", "manual_review_rerun"):
        _require(scope, key, False, label="G5B scope")
    _require(scope, "formal_i4_3g_closure_performed", True, label="G5B scope")
    _require(scope, "i4_3h_work_started", False, label="G5B scope")

    chain = _mapping(payload.get("closure_chain"), label="G5B closure_chain")
    expected_chain = {
        "g1_framework_bound",
        "g2_formally_closed",
        "g3_formally_closed",
        "g4_formally_closed",
        "g5a_synthesis_complete",
        "g5a_deterministically_rebuilt_from_predecessors",
        "all_predecessor_sha256_bindings_verified",
    }
    if set(chain) != expected_chain or any(chain.get(key) is not True for key in expected_chain):
        raise ReconstructionValidationError("I4.3H1 G5B closure chain is incomplete or not fully verified.")

    coverage = _mapping(payload.get("coverage"), label="G5B coverage")
    expected_coverage = {
        "qualification_family_count": 3,
        "scenario_count": 11,
        "provider_implementation_count": 10,
        "g2_scenarios": 5,
        "g3_scenarios": 3,
        "g4_scenarios": 3,
    }
    for key, expected in expected_coverage.items():
        _require(coverage, key, expected, label="G5B coverage")

    summary = _mapping(payload.get("operating_envelope_summary"), label="G5B operating_envelope_summary")
    _require(summary, "scenario_specific_operating_envelopes_preserved", True, label="G5B operating_envelope_summary")
    _require(summary, "production_native_evidence_primary", True, label="G5B operating_envelope_summary")
    _require(summary, "matched_view_diagnostic_secondary", True, label="G5B operating_envelope_summary")
    _require(summary, "terminal_quality_event_count", 5, label="G5B operating_envelope_summary")
    _require(summary, "quality_event_counts_are_not_provider_scores", True, label="G5B operating_envelope_summary")
    _require(summary, "direct_cross_family_provider_ranking_valid", False, label="G5B operating_envelope_summary")
    events = _terminal_events(summary.get("terminal_quality_events"))
    if set(events) != _EXPECTED_TERMINAL_QUALITY_EVENTS or len(events) != 5:
        raise ReconstructionValidationError(f"I4.3H1 G5B terminal quality-event set drifted: {events!r}")

    roles = _mapping(summary.get("family_roles"), label="G5B family_roles")
    expected_roles = {
        "mvs_da3": ("colmap_mvs", "da3"),
        "temporal_depth": ("rolling_depth", "video_depth_anything"),
        "single_image_depth": (
            "metric3d_v2", "unidepth_v2", "moge_2", "depth_pro", "depth_anything_v2", "marigold"
        ),
    }
    if set(roles) != set(expected_roles):
        raise ReconstructionValidationError("I4.3H1 G5B family-role set drifted.")
    covered: list[str] = []
    for family_id, providers in expected_roles.items():
        role = _mapping(roles[family_id], label=f"G5B family role {family_id}")
        if tuple(role.get("provider_implementations", ())) != providers:
            raise ReconstructionValidationError(f"I4.3H1 G5B {family_id} provider coverage drifted.")
        _require(role, "formal_closure_valid", True, label=f"G5B family role {family_id}")
        _require(
            role,
            "direct_rank_comparison_with_other_families_valid",
            False,
            label=f"G5B family role {family_id}",
        )
        covered.extend(providers)
    if tuple(covered) != _EXPECTED_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3H1 G5B 10-provider coverage drifted.")

    decision = _mapping(payload.get("decision"), label="G5B decision")
    _require(decision, "g5a_cross_provider_synthesis_complete", True, label="G5B decision")
    _require(decision, "i4_3g_closed", True, label="G5B decision")
    _require(decision, "i4_3h_authorized", True, label="G5B decision")
    _require(decision, "production_native_evidence_primary", True, label="G5B decision")
    _require(decision, "matched_view_diagnostic_secondary", True, label="G5B decision")
    _require(decision, "scenario_specific_verdicts_preserved", True, label="G5B decision")
    _require(decision, "quality_rejections_preserved", True, label="G5B decision")
    for key in _DECISION_FALSE:
        _require(decision, key, False, label="G5B decision")

    authority = _mapping(payload.get("authority"), label="G5B authority")
    _require(authority, "camera_pose_authority_owner", "pycolmap_sfm", label="G5B authority")
    _require(authority, "permanent_global_view_cap", None, label="G5B authority")
    for key in _AUTHORITY_FALSE:
        _require(authority, key, False, label="G5B authority")

    bindings = _mapping(payload.get("evidence_bindings"), label="G5B evidence_bindings")
    expected_binding_names = {
        "i4_3g1_snapshot",
        "i4_3g2_closure",
        "i4_3g3c_closure",
        "i4_3g4c_closure",
        "i4_3g5a_synthesis",
    }
    if set(bindings) != expected_binding_names:
        raise ReconstructionValidationError("I4.3H1 G5B evidence-binding set drifted.")
    normalized_bindings = {
        key: _validate_sha_binding(bindings[key], label=f"G5B binding {key}")
        for key in sorted(expected_binding_names)
    }

    return {
        "coverage": deepcopy(dict(coverage)),
        "family_roles": deepcopy(dict(roles)),
        "terminal_quality_events": deepcopy(list(summary["terminal_quality_events"])),
        "predecessor_bindings": normalized_bindings,
    }


def validate_i4_3h1_regression_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the consolidated H1 + predecessor I4.3 automated-regression receipt."""

    if (
        payload.get("phase") != I43H1_REGRESSION_PHASE
        or payload.get("status") != "pass"
        or payload.get("pass") is not True
    ):
        raise ReconstructionValidationError("I4.3H1 requires a passing regression-audit receipt.")
    scope = _mapping(payload.get("scope"), label="regression receipt scope")
    for key in (
        "focused_h1_contract_tests",
        "predecessor_i4_3_unit_tests",
        "predecessor_i4_3_integration_tests",
        "predecessor_i4_3_stress_tests",
        "import_probe",
        "runner_help_probe",
    ):
        _require(scope, key, True, label="regression receipt scope")

    counts = _mapping(payload.get("counts"), label="regression receipt counts")
    _require(counts, "predecessor_test_file_count", I43H1_EXPECTED_PREDECESSOR_TEST_FILES, label="regression receipt counts")
    _require(counts, "predecessor_unit_tests", I43H1_EXPECTED_PREDECESSOR_UNIT_TESTS, label="regression receipt counts")
    _require(counts, "predecessor_integration_tests", I43H1_EXPECTED_PREDECESSOR_INTEGRATION_TESTS, label="regression receipt counts")
    _require(counts, "predecessor_stress_tests", I43H1_EXPECTED_PREDECESSOR_STRESS_TESTS, label="regression receipt counts")
    _require(counts, "predecessor_total_tests", I43H1_EXPECTED_PREDECESSOR_TOTAL_TESTS, label="regression receipt counts")
    focused = counts.get("focused_h1_tests")
    if isinstance(focused, bool) or not isinstance(focused, int) or focused < 15:
        raise ReconstructionValidationError("I4.3H1 focused regression coverage is unexpectedly small.")

    return deepcopy(dict(payload))


def audit_i4_3_provider_architecture() -> dict[str, Any]:
    """Re-audit current in-repository A-F provider/adaptive contracts without backend probes."""

    family_contracts = dense_provider_family_contracts()
    if len(family_contracts) != 5:
        raise ReconstructionValidationError("I4.3H1 expected exactly five canonical provider-family contracts.")
    by_family = {item.family.value: item for item in family_contracts}
    if set(by_family) != {item.value for item in DenseProviderFamily}:
        raise ReconstructionValidationError("I4.3H1 canonical provider-family enum/contract coverage drifted.")
    for family_id, contract in by_family.items():
        if not contract.requires_authoritative_sfm or not contract.preserves_authoritative_sfm_frame:
            raise ReconstructionValidationError(f"I4.3H1 {family_id} no longer preserves authoritative SfM.")
        if not contract.adaptive_view_selection_required or not contract.progressive_status_required:
            raise ReconstructionValidationError(f"I4.3H1 {family_id} lost adaptive/progressive contract requirements.")
        if contract.permanent_global_view_cap is not None:
            raise ReconstructionValidationError(f"I4.3H1 {family_id} introduced a permanent global view cap.")
        forbidden = (
            contract.establishes_metric_scale,
            contract.establishes_camera_pose_authority,
            contract.establishes_measurement_authority,
            contract.establishes_certified_metrology,
            contract.generates_surface_mesh,
        )
        if any(forbidden):
            raise ReconstructionValidationError(f"I4.3H1 {family_id} authority contract drifted.")
        if family_id == "hybrid":
            if contract.implementation_authorized:
                raise ReconstructionValidationError("I4.3H1 Hybrid must remain reserved and unauthorized.")
        elif not contract.implementation_authorized:
            raise ReconstructionValidationError(f"I4.3H1 executable family {family_id} became unauthorized.")

    implementation_contracts = dense_provider_implementation_contracts()
    if len(implementation_contracts) != 10:
        raise ReconstructionValidationError("I4.3H1 requires exactly ten canonical provider implementations.")
    by_impl = {item.implementation.value: item for item in implementation_contracts}
    if tuple(item.implementation.value for item in implementation_contracts) != _EXPECTED_IMPLEMENTATIONS:
        raise ReconstructionValidationError("I4.3H1 canonical provider implementation ordering/coverage drifted.")

    provider_inventory: dict[str, Any] = {}
    for implementation in _EXPECTED_IMPLEMENTATIONS:
        contract = by_impl[implementation]
        expected_family, expected_stage = _EXPECTED_IMPLEMENTATION_CONTRACT[implementation]
        if contract.family.value != expected_family or contract.integration_stage != expected_stage:
            raise ReconstructionValidationError(f"I4.3H1 {implementation} family/integration-stage contract drifted.")
        if contract.family_contract.native_output.value != by_family[expected_family].native_output.value:
            raise ReconstructionValidationError(f"I4.3H1 {implementation} native-output contract drifted.")
        if contract.family_contract.frame_binding.value != by_family[expected_family].frame_binding.value:
            raise ReconstructionValidationError(f"I4.3H1 {implementation} frame-binding contract drifted.")

        enum_value = DenseProviderImplementation(implementation)
        requirements = provider_view_selection_requirements(enum_value)
        if requirements.family.value != expected_family:
            raise ReconstructionValidationError(f"I4.3H1 {implementation} adaptive family contract drifted.")

        module_name, adapter_name, descriptor_name, validator_name = _PROVIDER_SURFACES[implementation]
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exact exception is environment-dependent
            raise ReconstructionValidationError(
                f"I4.3H1 provider contract module failed to import: {module_name}: {exc}"
            ) from exc
        for symbol in (adapter_name, descriptor_name, validator_name):
            value = getattr(module, symbol, None)
            if value is None or not callable(value):
                raise ReconstructionValidationError(
                    f"I4.3H1 provider contract symbol missing/non-callable: {module_name}.{symbol}"
                )

        provider_inventory[implementation] = {
            "family": expected_family,
            "integration_stage": expected_stage,
            "native_output": contract.family_contract.native_output.value,
            "frame_binding": contract.family_contract.frame_binding.value,
            "adapter_module": module_name,
            "adapter_class": adapter_name,
            "descriptor_function": descriptor_name,
            "binding_validator": validator_name,
            "module_contract_imported": True,
            "backend_runtime_probe_started": False,
            "adaptive_planning": {
                "minimum_views": requirements.minimum_views,
                "relative_cost_per_megapixel": requirements.relative_cost_per_megapixel,
                "relative_cost_is_empirical_runtime_claim": False,
                "source_order_required": requirements.source_order_required,
                "stateful_sequence": requirements.stateful_sequence,
            },
            "scenario_qualification_evidence_closed": True,
            "globally_promoted": False,
        }

    return {
        "provider_family_count": 5,
        "executable_provider_family_count": 4,
        "hybrid_family_reserved": True,
        "hybrid_implementation_authorized": False,
        "provider_implementation_count": 10,
        "integrated_provider_implementations": list(_EXPECTED_IMPLEMENTATIONS),
        "all_provider_contract_modules_importable": True,
        "all_provider_implementation_contracts_valid": True,
        "all_provider_adaptive_planning_contracts_valid": True,
        "adaptive_planning_coefficients_are_empirical_runtime_claims": False,
        "progressive_status_required": True,
        "permanent_global_view_cap": None,
        "provider_inventory": provider_inventory,
    }


@dataclass(frozen=True, slots=True)
class H1EvidenceBinding:
    path: str
    sha256: str

    def as_dict(self) -> dict[str, str]:
        path = str(self.path).strip()
        sha = str(self.sha256).strip().lower()
        if not path or not _SHA256_RE.fullmatch(sha):
            raise ReconstructionValidationError("I4.3H1 evidence binding is malformed.")
        return {"path": path, "sha256": sha}


def build_i4_3h1_final_qualification_audit(
    *,
    i4_3g5b_closure: Mapping[str, Any],
    regression_receipt: Mapping[str, Any],
    evidence_bindings: Mapping[str, H1EvidenceBinding],
    g5b_predecessor_hashes_reverified: bool,
) -> dict[str, Any]:
    """Build the H1 qualification artifact after evidence + regression gates pass."""

    validated_g = validate_i4_3g5b_closure(i4_3g5b_closure)
    validated_regression = validate_i4_3h1_regression_receipt(regression_receipt)
    if g5b_predecessor_hashes_reverified is not True:
        raise ReconstructionValidationError("I4.3H1 requires independent re-verification of G5B predecessor hashes.")
    if set(evidence_bindings) != {"i4_3g5b_closure", "i4_3h1_regression_receipt"}:
        raise ReconstructionValidationError("I4.3H1 final evidence-binding set is incomplete or unexpected.")

    architecture = audit_i4_3_provider_architecture()
    terminal_events = deepcopy(validated_g["terminal_quality_events"])
    events_by_provider: dict[str, list[dict[str, Any]]] = {name: [] for name in _EXPECTED_IMPLEMENTATIONS}
    for event in terminal_events:
        implementation = str(event.get("provider_implementation", ""))
        if implementation in events_by_provider:
            events_by_provider[implementation].append(deepcopy(dict(event)))
    for implementation, inventory in architecture["provider_inventory"].items():
        inventory["terminal_quality_events"] = events_by_provider[implementation]
        inventory["no_terminal_quality_event_implies_universal_robustness"] = False

    counts = validated_regression["counts"]
    return {
        "schema_version": I43H1_SCHEMA_VERSION,
        "phase": I43H1_PHASE,
        "status": "qualification_complete",
        "pass": True,
        "scope": {
            "evidence_and_repository_contract_audit": True,
            "provider_execution_started": False,
            "provider_runtime_probe_started": False,
            "sfm_rerun": False,
            "matched_view_rerun": False,
            "manual_review_rerun": False,
            "formal_i4_3_closure_performed": False,
            "i4_4_work_started": False,
        },
        "evidence_bindings": {
            key: value.as_dict()
            for key, value in evidence_bindings.items()
        },
        "g5b_predecessor_bindings": deepcopy(validated_g["predecessor_bindings"]),
        "closure_chain_audit": {
            "i4_3g_formally_closed": True,
            "g5a_deterministic_rebuild_verified_by_g5b": True,
            "g5b_sha256_verified": True,
            "g5b_predecessor_sha256_bindings_reverified": True,
            "qualification_family_count": 3,
            "scenario_count": 11,
            "provider_implementation_count": 10,
        },
        "architecture_audit": architecture,
        "qualification_audit": {
            "qualification_family_count": 3,
            "scenario_count": 11,
            "provider_implementation_count": 10,
            "production_native_evidence_primary": True,
            "matched_view_diagnostic_secondary": True,
            "scenario_specific_operating_envelopes_preserved": True,
            "terminal_quality_events": terminal_events,
            "terminal_quality_event_count": 5,
            "quality_event_counts_are_not_provider_scores": True,
            "direct_cross_family_provider_ranking_valid": False,
            "all_ten_integrated_providers_have_closed_family_qualification_evidence": True,
            "global_provider_promotion_performed": False,
        },
        "provenance_audit": {
            "frozen_g_closure_bound": True,
            "immediate_g_predecessor_artifacts_rehashed": True,
            "family_provenance_conclusions_preserved": True,
            "provider_contract_integration_stages_preserved": True,
            "provider_specific_semantics_collapsed": False,
            "raw_nested_artifact_tree_rehashed_by_h1": False,
        },
        "regression_audit": {
            "receipt_bound": True,
            "focused_h1_tests": counts["focused_h1_tests"],
            "predecessor_unit_tests": counts["predecessor_unit_tests"],
            "predecessor_integration_tests": counts["predecessor_integration_tests"],
            "predecessor_stress_tests": counts["predecessor_stress_tests"],
            "predecessor_total_tests": counts["predecessor_total_tests"],
            "all_required_suites_passed": True,
        },
        "experimental_surfaces": {
            "auto_selection_contract_scaffold_exists": True,
            "auto_policy_implemented": False,
            "hybrid_family_contract_reserved": True,
            "hybrid_authorized": False,
            "first_evidence_driven_auto_policy_deferred_to_i5": True,
        },
        "decision": {
            "i4_3h1_qualification_complete": True,
            "i4_3h2_authorized": True,
            "i4_3_closed": False,
            "i4_4_authorized": False,
            "provider_selected": False,
            "provider_promoted": False,
            "provider_ranking_created": False,
            "universal_provider_winner_declared": False,
            "global_scalar_score_created": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
        },
        "authority": {
            "camera_pose_authority_owner": "pycolmap_sfm",
            "authoritative_sfm_modified": False,
            "provider_selected": False,
            "provider_promoted": False,
            "auto_policy_implemented": False,
            "hybrid_authorized": False,
            "metric_scale_claimed": False,
            "measurement_authoritative": False,
            "certified_metrology": False,
            "surface_mesh_generated": False,
            "permanent_global_view_cap": None,
        },
        "next": "I4.3H2 formal I4.3 closure.",
    }


__all__ = [
    "I43H1_SCHEMA_VERSION",
    "I43H1_PHASE",
    "I43H1_REGRESSION_PHASE",
    "I43H1_EXPECTED_PREDECESSOR_TOTAL_TESTS",
    "H1EvidenceBinding",
    "validate_i4_3g5b_closure",
    "validate_i4_3h1_regression_receipt",
    "audit_i4_3_provider_architecture",
    "build_i4_3h1_final_qualification_audit",
]
