from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction import (
    GAUSSIAN_QUALIFIED_CUDA,
    GAUSSIAN_QUALIFIED_PYTHON,
    GAUSSIAN_QUALIFIED_TORCH,
    GSPLAT_QUALIFIED_VERSION,
    PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
    GaussianBackendKind,
    GaussianCheckoutEvidence,
    GaussianEnvironmentProbe,
    GaussianInteractiveProcessPolicy,
    GaussianInteractiveRuntimeSpec,
    GaussianInteractiveRuntimeState,
    GaussianRuntimeEvidence,
    assess_gaussian_interactive_runtime,
    current_process_gaussian_heavy_modules,
    default_phase9_gaussian_runtime_spec,
)
from core.reconstruction.errors import ReconstructionValidationError


def _spec(tmp_path: Path, *, backend=GaussianBackendKind.GSPLAT_EXTERNAL) -> GaussianInteractiveRuntimeSpec:
    python = tmp_path / "env" / "bin" / "python"
    checkout = tmp_path / "gsplat"
    python.parent.mkdir(parents=True, exist_ok=True)
    checkout.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    return GaussianInteractiveRuntimeSpec(
        python_executable=str(python),
        checkout_root=str(checkout),
        backend_kind=backend,
        expected_checkout_revision=PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
    )


def _environment(spec: GaussianInteractiveRuntimeSpec) -> GaussianEnvironmentProbe:
    return GaussianEnvironmentProbe(
        python_executable=spec.python_executable,
        available=True,
        backend_kind=spec.backend_kind,
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        gsplat_version=GSPLAT_QUALIFIED_VERSION if spec.backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL else "",
        graphdeco_root=(spec.checkout_root if spec.backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL else ""),
        graphdeco_license_present=spec.backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        graphdeco_runtime_ready=spec.backend_kind is GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",),
    )


def _runtime(spec: GaussianInteractiveRuntimeSpec) -> GaussianRuntimeEvidence:
    return GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version=GAUSSIAN_QUALIFIED_PYTHON,
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        torch_cuda_version=GAUSSIAN_QUALIFIED_CUDA,
        cuda_available=True,
        device_names=("NVIDIA GeForce RTX 2060",),
        device_capabilities=((7, 5),),
    )


def _checkout(spec: GaussianInteractiveRuntimeSpec, *, clean=True, revision=None) -> GaussianCheckoutEvidence:
    return GaussianCheckoutEvidence(
        root=spec.checkout_root,
        exists=True,
        revision=revision or PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
        clean=clean,
        status_lines=() if clean else (" M gsplat/__init__.py",),
    )


def test_4a_defaults_keep_gsplat_external_and_outside_depth(tmp_path: Path) -> None:
    spec = default_phase9_gaussian_runtime_spec(tmp_path)
    assert spec.backend_kind is GaussianBackendKind.GSPLAT_EXTERNAL
    assert "depth3d-gaussian-gsplat" in spec.python_executable
    assert spec.checkout_root.endswith("Desktop/depth3d-gaussian-backends/gsplat")
    assert spec.expected_checkout_revision == PHASE9_4A_GSPLAT_QUALIFIED_REVISION
    assert spec.process_policy.start_new_session is True
    assert spec.process_policy.shell is False


def test_4a_process_policy_is_fail_closed() -> None:
    with pytest.raises(ReconstructionValidationError):
        GaussianInteractiveProcessPolicy(start_new_session=False)
    with pytest.raises(ReconstructionValidationError):
        GaussianInteractiveProcessPolicy(shell=True)
    with pytest.raises(ReconstructionValidationError):
        GaussianInteractiveProcessPolicy(terminate_signal="SIGKILL")


def test_4a_ready_runtime_preserves_appearance_only_authority(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec),
    )
    assert report.state is GaussianInteractiveRuntimeState.READY
    assert report.ready is True
    assert report.reasons == ()
    assert report.gaussian_scene_role.value == "appearance_only"
    assert report.canonical_geometry_authoritative is False
    assert report.metric_scale_authoritative is False
    assert report.measurement_authoritative is False
    assert report.qc_authoritative is False
    assert report.certified_metrology is False
    assert report.training_performed is False
    assert report.rendering_performed is False
    assert report.cross_representation_fallback is False


def test_4a_graphdeco_is_rejected_for_interactive_production_path(tmp_path: Path) -> None:
    spec = _spec(tmp_path, backend=GaussianBackendKind.GRAPHDECO_REFERENCE_EXTERNAL)
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec),
    )
    assert report.state is GaussianInteractiveRuntimeState.REJECTED
    assert "interactive_backend_must_be_permissive_gsplat" in report.reasons
    assert report.backend_provenance.usage_scope.value == "research_evaluation_only"


def test_4a_production_heavy_gaussian_importability_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec),
        production_heavy_modules=("gsplat",),
    )
    assert report.state is GaussianInteractiveRuntimeState.REJECTED
    assert "heavy_gaussian_modules_present_in_production_runtime" in report.reasons


def test_4a_checkout_revision_or_dirty_state_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    wrong = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec, revision="0" * 40),
    )
    assert wrong.state is GaussianInteractiveRuntimeState.REJECTED
    assert "checkout_revision_mismatch" in wrong.reasons

    dirty = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec, clean=False),
    )
    assert dirty.state is GaussianInteractiveRuntimeState.REJECTED
    assert "checkout_not_clean" in dirty.reasons


def test_4a_missing_external_runtime_is_unavailable_not_fabricated(tmp_path: Path) -> None:
    missing_python = tmp_path / "missing" / "python"
    checkout = tmp_path / "missing-checkout"
    spec = GaussianInteractiveRuntimeSpec(
        python_executable=str(missing_python),
        checkout_root=str(checkout),
    )
    environment = GaussianEnvironmentProbe(
        python_executable=spec.python_executable,
        available=False,
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        error="python_executable_not_found",
    )
    runtime = GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version="",
        torch_version="",
        torch_cuda_version="",
        cuda_available=False,
        device_names=(),
        device_capabilities=(),
        error="python_executable_not_found",
    )
    checkout_evidence = GaussianCheckoutEvidence(
        root=spec.checkout_root,
        exists=False,
        revision="",
        clean=False,
        error="checkout_not_found",
    )
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=environment,
        runtime_evidence=runtime,
        checkout_evidence=checkout_evidence,
    )
    assert report.state is GaussianInteractiveRuntimeState.UNAVAILABLE
    assert report.ready is False
    assert report.training_performed is False
    assert report.rendering_performed is False


def test_4a_python_patch_release_matches_qualified_major_minor(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runtime = GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version="3.10.20",
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        torch_cuda_version=GAUSSIAN_QUALIFIED_CUDA,
        cuda_available=True,
        device_names=("NVIDIA GeForce RTX 2060",),
        device_capabilities=((7, 5),),
    )
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=runtime,
        checkout_evidence=_checkout(spec),
    )
    assert report.state is GaussianInteractiveRuntimeState.READY
    assert "python_version_mismatch" not in report.reasons


def test_4a_python_major_minor_drift_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runtime = GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version="3.11.9",
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        torch_cuda_version=GAUSSIAN_QUALIFIED_CUDA,
        cuda_available=True,
        device_names=("NVIDIA GeForce RTX 2060",),
        device_capabilities=((7, 5),),
    )
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=runtime,
        checkout_evidence=_checkout(spec),
    )
    assert report.state is GaussianInteractiveRuntimeState.REJECTED
    assert "python_version_mismatch" in report.reasons


def test_4a_runtime_version_drift_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    runtime = GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version=GAUSSIAN_QUALIFIED_PYTHON,
        torch_version="9.9.9",
        torch_cuda_version=GAUSSIAN_QUALIFIED_CUDA,
        cuda_available=True,
        device_names=("NVIDIA GeForce RTX 2060",),
        device_capabilities=((7, 5),),
    )
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=runtime,
        checkout_evidence=_checkout(spec),
    )
    assert report.state is GaussianInteractiveRuntimeState.REJECTED
    assert "torch_version_mismatch" in report.reasons


def test_4a_current_process_module_probe_does_not_import_modules() -> None:
    calls: list[str] = []

    def finder(name: str):
        calls.append(name)
        return object() if name == "gsplat" else None

    assert current_process_gaussian_heavy_modules(("gsplat", "simple_knn"), finder=finder) == ("gsplat",)
    assert calls == ["gsplat", "simple_knn"]


def test_4a_report_json_is_explicit_about_probe_only_state(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = assess_gaussian_interactive_runtime(
        spec,
        environment_probe=_environment(spec),
        runtime_evidence=_runtime(spec),
        checkout_evidence=_checkout(spec),
    )
    payload = report.as_dict()
    assert payload["state"] == "ready"
    assert payload["capability_probe_only"] is True
    assert payload["training_performed"] is False
    assert payload["rendering_performed"] is False
    assert payload["backend_provenance"]["license_name"] == "Apache-2.0"
