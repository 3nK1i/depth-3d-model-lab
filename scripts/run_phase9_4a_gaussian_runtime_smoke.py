#!/usr/bin/env python3
"""Dependency-light Phase 9.4A Gaussian interactive-runtime contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_4a_") as tmp:
        root = Path(tmp)
        python = root / "depth3d-gaussian-gsplat" / "bin" / "python"
        checkout = root / "gsplat"
        python.parent.mkdir(parents=True)
        checkout.mkdir()
        python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        spec = GaussianInteractiveRuntimeSpec(
            python_executable=str(python),
            checkout_root=str(checkout),
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
        report = assess_gaussian_interactive_runtime(
            spec,
            environment_probe=environment,
            runtime_evidence=runtime,
            checkout_evidence=checkout_evidence,
        )
        payload = report.as_dict()
        payload.update(
            {
                "status": "PASS" if report.ready else "FAIL",
                "real_backend_started": False,
                "optimization_started": False,
                "interactive_renderer_started": False,
                "phase9_4a_only": True,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
