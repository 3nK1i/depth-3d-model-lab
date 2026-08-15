from __future__ import annotations

from copy import deepcopy

from scripts.run_phase9_5h_real_scene_qualification import validate_phase9_5_chain
from tests.phase9_5h_helpers import make_valid_phase9_5h_evidence


def test_5h_valid_chain_passes() -> None:
    passed, reasons = validate_phase9_5_chain(make_valid_phase9_5h_evidence())
    assert passed is True
    assert reasons == ()


def test_5h_rejects_nerf_measurement_authority() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["authority"]["nerf_measurement_authoritative"] = True
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "authority_violation:nerf_measurement_authoritative" in reasons


def test_5h_rejects_runtime_revision_drift() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["interactive"]["runtime_checkout_revision"] = "b" * 40
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "interactive_runtime_revision_mismatch" in reasons


def test_5h_rejects_dataset_fingerprint_drift() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["persistence"]["dataset_fingerprint_sha256"] = "c" * 64
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "persistent_dataset_fingerprint_mismatch" in reasons


def test_5h_rejects_failed_worker_release() -> None:
    evidence = deepcopy(make_valid_phase9_5h_evidence())
    evidence["resources"]["worker_release_count"] = 3
    evidence["resources"]["all_worker_pids_released_from_os_and_nvidia_smi"] = False
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "worker_lifecycle_counts_mismatch" in reasons
    assert "worker_release_failed" in reasons


def test_5h_quality_is_characterized_but_must_not_be_overclaimed() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["quality"]["production_quality_certified"] = True
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "undersized_dataset_quality_overclaimed" in reasons


def test_5h_rejects_arbitrary_scale_metric_exposure() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["measurement"]["state"] = "estimated"
    evidence["measurement"]["metric_value_exposed"] = True
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "arbitrary_scale_measurement_did_not_fail_closed" in reasons
    assert "arbitrary_scale_metric_value_exposed" in reasons


def test_5h_rejects_unbounded_resource_policy() -> None:
    evidence = make_valid_phase9_5h_evidence()
    evidence["resources"]["max_pending_renders"] = 2
    passed, reasons = validate_phase9_5_chain(evidence)
    assert passed is False
    assert "resource_bounds_changed" in reasons
