from __future__ import annotations

from pathlib import Path

import pytest

from core.reconstruction import (
    PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
    PHASE9_5A_TCNN_QUALIFIED_REVISION,
    NerfCheckoutEvidence,
    NerfEnvironmentProbe,
    NerfInteractiveProcessPolicy,
    NerfInteractiveRuntimeSpec,
    NerfInteractiveRuntimeState,
    NerfPythonRuntimeEvidence,
    assess_nerf_interactive_runtime,
    current_process_nerf_heavy_modules,
    default_phase9_nerf_runtime_spec,
)


def _spec(tmp_path: Path) -> NerfInteractiveRuntimeSpec:
    python = tmp_path / "depth3d-nerf-nerfstudio/bin/python"
    ns = tmp_path / "nerfstudio"
    tcnn = tmp_path / "tiny-cuda-nn"
    python.parent.mkdir(parents=True)
    ns.mkdir()
    tcnn.mkdir()
    python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    return NerfInteractiveRuntimeSpec(
        python_executable=str(python),
        nerfstudio_root=str(ns),
        tiny_cuda_nn_root=str(tcnn),
    )


def _environment(spec: NerfInteractiveRuntimeSpec, **changes) -> NerfEnvironmentProbe:
    kwargs = dict(
        python_executable=spec.python_executable,
        available=True,
        nerfstudio_version="1.1.5",
        torch_version="2.1.2+cu118",
        tinycudann_available=True,
        ns_train_available=True,
        ns_render_available=True,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",),
        cuda_compute_capabilities=((7, 5),),
        cuda_total_vram_mib=(5744.0,),
        torch_cuda_version="11.8",
    )
    kwargs.update(changes)
    return NerfEnvironmentProbe(**kwargs)


def _runtime(spec: NerfInteractiveRuntimeSpec, version: str = "3.10.20") -> NerfPythonRuntimeEvidence:
    return NerfPythonRuntimeEvidence(spec.python_executable, python_version=version)


def _checkout(root: str, revision: str) -> NerfCheckoutEvidence:
    return NerfCheckoutEvidence(root=root, exists=True, revision=revision, clean=True)


def _report(spec: NerfInteractiveRuntimeSpec, **kwargs):
    return assess_nerf_interactive_runtime(
        spec,
        environment_probe=kwargs.get("environment_probe", _environment(spec)),
        python_runtime=kwargs.get("python_runtime", _runtime(spec)),
        nerfstudio_checkout=kwargs.get(
            "nerfstudio_checkout", _checkout(spec.nerfstudio_root, PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION)
        ),
        tiny_cuda_nn_checkout=kwargs.get(
            "tiny_cuda_nn_checkout", _checkout(spec.tiny_cuda_nn_root, PHASE9_5A_TCNN_QUALIFIED_REVISION)
        ),
        production_heavy_modules=kwargs.get("production_heavy_modules", ()),
    )


def test_5a_default_runtime_paths_and_exact_pins_are_stable(tmp_path: Path) -> None:
    spec = default_phase9_nerf_runtime_spec(tmp_path)
    assert spec.python_executable.endswith("anaconda3/envs/depth3d-nerf-nerfstudio/bin/python")
    assert spec.nerfstudio_root.endswith("Desktop/depth3d-nerf-backends/nerfstudio")
    assert spec.tiny_cuda_nn_root.endswith("Desktop/depth3d-nerf-backends/tiny-cuda-nn")
    assert spec.expected_nerfstudio_revision == PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION
    assert spec.expected_tiny_cuda_nn_revision == PHASE9_5A_TCNN_QUALIFIED_REVISION


def test_5a_process_policy_requires_process_group_and_sigterm_sigkill() -> None:
    policy = NerfInteractiveProcessPolicy()
    assert policy.start_new_session is True
    assert policy.shell is False
    assert policy.as_dict()["termination_order"] == ["SIGTERM", "SIGKILL"]
    with pytest.raises(Exception):
        NerfInteractiveProcessPolicy(start_new_session=False)
    with pytest.raises(Exception):
        NerfInteractiveProcessPolicy(shell=True)


def test_5a_qualified_runtime_is_ready_and_appearance_only(tmp_path: Path) -> None:
    report = _report(_spec(tmp_path))
    assert report.state is NerfInteractiveRuntimeState.READY
    payload = report.as_dict()
    assert payload["ready"] is True
    assert payload["nerf_scene_role"] == "appearance_only"
    assert payload["canonical_geometry_authoritative"] is False
    assert payload["metric_scale_authoritative"] is False
    assert payload["measurement_authoritative"] is False
    assert payload["qc_authoritative"] is False
    assert payload["certified_metrology"] is False
    assert payload["cross_representation_fallback"] is False
    assert payload["training_performed"] is False
    assert payload["rendering_performed"] is False


def test_5a_python_patch_version_is_accepted_but_major_minor_drift_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    assert _report(spec, python_runtime=_runtime(spec, "3.10.20")).ready is True
    report = _report(spec, python_runtime=_runtime(spec, "3.11.1"))
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "python_version_mismatch" in report.reasons


def test_5a_runtime_version_drift_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = _report(spec, environment_probe=_environment(spec, torch_version="9.9.9"))
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "torch_version_mismatch" in report.reasons


def test_5a_checkout_revision_or_cleanliness_drift_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    bad_revision = _checkout(spec.nerfstudio_root, "0" * 40)
    report = _report(spec, nerfstudio_checkout=bad_revision)
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "nerfstudio_checkout_revision_mismatch" in report.reasons

    dirty = NerfCheckoutEvidence(
        root=spec.tiny_cuda_nn_root,
        exists=True,
        revision=PHASE9_5A_TCNN_QUALIFIED_REVISION,
        clean=False,
        status_lines=(" M bindings/torch/setup.py",),
    )
    report = _report(spec, tiny_cuda_nn_checkout=dirty)
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "tiny_cuda_nn_checkout_not_clean" in report.reasons


def test_5a_missing_runtime_is_unavailable_not_silently_fallback(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    missing = NerfCheckoutEvidence(root=spec.nerfstudio_root, exists=False, revision="", clean=False)
    report = _report(spec, nerfstudio_checkout=missing)
    assert report.state is NerfInteractiveRuntimeState.UNAVAILABLE
    assert "nerfstudio_checkout_missing" in report.reasons
    assert report.as_dict()["cross_representation_fallback"] is False


def test_5a_heavy_nerf_modules_in_production_runtime_are_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    report = _report(spec, production_heavy_modules=("nerfstudio",))
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "heavy_nerf_modules_present_in_production_runtime" in report.reasons


def test_5a_compute_capability_below_qualified_minimum_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    environment = _environment(
        spec,
        cuda_device_names=("Legacy GPU",),
        cuda_compute_capabilities=((6, 1),),
    )
    report = _report(spec, environment_probe=environment)
    assert report.state is NerfInteractiveRuntimeState.REJECTED
    assert "cuda_compute_capability_below_qualified_minimum" in report.reasons


def test_5a_current_process_module_probe_does_not_import_modules() -> None:
    calls: list[str] = []

    def finder(name: str):
        calls.append(name)
        return object() if name == "nerfstudio" else None

    assert current_process_nerf_heavy_modules(("nerfstudio", "tinycudann"), finder=finder) == ("nerfstudio",)
    assert calls == ["nerfstudio", "tinycudann"]
