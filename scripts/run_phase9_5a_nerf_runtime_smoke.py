#!/usr/bin/env python3
"""Dependency-light Phase 9.5A interactive-NeRF runtime contract smoke."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (  # noqa: E402
    PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
    PHASE9_5A_TCNN_QUALIFIED_REVISION,
    NerfCheckoutEvidence,
    NerfEnvironmentProbe,
    NerfInteractiveRuntimeSpec,
    NerfPythonRuntimeEvidence,
    assess_nerf_interactive_runtime,
)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase9_5a_") as tmp:
        root = Path(tmp)
        python = root / "eia-nerf-nerfstudio" / "bin" / "python"
        ns_root = root / "nerfstudio"
        tcnn_root = root / "tiny-cuda-nn"
        python.parent.mkdir(parents=True)
        ns_root.mkdir()
        tcnn_root.mkdir()
        python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
        spec = NerfInteractiveRuntimeSpec(
            python_executable=str(python),
            nerfstudio_root=str(ns_root),
            tiny_cuda_nn_root=str(tcnn_root),
        )
        environment = NerfEnvironmentProbe(
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
        runtime = NerfPythonRuntimeEvidence(spec.python_executable, python_version="3.10.20")
        ns_checkout = NerfCheckoutEvidence(
            root=spec.nerfstudio_root,
            exists=True,
            revision=PHASE9_5A_NERFSTUDIO_QUALIFIED_REVISION,
            clean=True,
        )
        tcnn_checkout = NerfCheckoutEvidence(
            root=spec.tiny_cuda_nn_root,
            exists=True,
            revision=PHASE9_5A_TCNN_QUALIFIED_REVISION,
            clean=True,
        )
        report = assess_nerf_interactive_runtime(
            spec,
            environment_probe=environment,
            python_runtime=runtime,
            nerfstudio_checkout=ns_checkout,
            tiny_cuda_nn_checkout=tcnn_checkout,
        )
        payload = report.as_dict()
        payload.update(
            {
                "status": "PASS" if report.ready else "FAIL",
                "real_backend_started": False,
                "optimization_started": False,
                "training_started": False,
                "interactive_renderer_started": False,
                "phase9_5a_only": True,
            }
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
