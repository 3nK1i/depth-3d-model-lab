from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    GAUSSIAN_QUALIFIED_CUDA,
    GAUSSIAN_QUALIFIED_PYTHON,
    GAUSSIAN_QUALIFIED_TORCH,
    GSPLAT_QUALIFIED_VERSION,
    PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
    GaussianBackendKind,
    GaussianCheckoutEvidence,
    GaussianEnvironmentProbe,
    GaussianInteractiveRuntimeSpec,
    GaussianRuntimeEvidence,
    assess_gaussian_interactive_runtime,
)


def ready_gaussian_runtime(tmp_path: Path):
    python = (tmp_path / "depth3d-gaussian-gsplat" / "bin" / "python").resolve()
    checkout = (tmp_path / "gsplat").resolve()
    python.parent.mkdir(parents=True, exist_ok=True)
    checkout.mkdir(parents=True, exist_ok=True)
    python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    spec = GaussianInteractiveRuntimeSpec(
        python_executable=str(python),
        checkout_root=str(checkout),
        expected_checkout_revision=PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
    )
    environment = GaussianEnvironmentProbe(
        python_executable=spec.python_executable,
        available=True,
        backend_kind=GaussianBackendKind.GSPLAT_EXTERNAL,
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        gsplat_version=GSPLAT_QUALIFIED_VERSION,
        cuda_available=True,
        cuda_device_count=1,
        cuda_device_names=("NVIDIA GeForce RTX 2060",),
    )
    runtime = GaussianRuntimeEvidence(
        python_executable=spec.python_executable,
        python_version=GAUSSIAN_QUALIFIED_PYTHON,
        torch_version=GAUSSIAN_QUALIFIED_TORCH,
        torch_cuda_version=GAUSSIAN_QUALIFIED_CUDA,
        cuda_available=True,
        device_names=("NVIDIA GeForce RTX 2060",),
        device_capabilities=((7, 5),),
    )
    checkout_evidence = GaussianCheckoutEvidence(
        root=spec.checkout_root,
        exists=True,
        revision=PHASE9_4A_GSPLAT_QUALIFIED_REVISION,
        clean=True,
    )
    return assess_gaussian_interactive_runtime(
        spec,
        environment_probe=environment,
        runtime_evidence=runtime,
        checkout_evidence=checkout_evidence,
    )


def tetrahedral_canonical_arrays():
    import numpy as np

    vertices = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.15, 0.0, 1.0],
            [0.0, 0.15, 1.0],
            [0.0, 0.0, 1.15],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32
    )
    colors = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    return vertices, faces, colors
