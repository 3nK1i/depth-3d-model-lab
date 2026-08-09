from __future__ import annotations

from pathlib import Path

from core.reconstruction import (
    NerfBackendKind,
    NerfSceneRole,
    NerfUsageScope,
    nerf_backend_provenance,
    nerf_backend_runtime_pin,
)

ROOT = Path(__file__).resolve().parents[2]


def test_e1_backend_identity_and_policy_remain_unchanged() -> None:
    provenance = nerf_backend_provenance(NerfBackendKind.NERFSTUDIO_NERFACTO_EXTERNAL)
    assert provenance.method_name == "nerfacto"
    assert provenance.license_name == "Apache-2.0"
    assert provenance.usage_scope is NerfUsageScope.PERMISSIVE_GENERAL_USE
    assert provenance.runtime_dependency_name == "tiny-cuda-nn"
    assert nerf_backend_runtime_pin().release_tag == "v1.1.5"
    assert NerfSceneRole.APPEARANCE_ONLY.value == "appearance_only"


def test_production_dependency_manifest_does_not_add_nerf_runtime() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "nerfstudio" not in text
    assert "tiny-cuda-nn" not in text
    assert "tinycudann" not in text


def test_production_reconstruction_core_has_no_heavy_nerf_import() -> None:
    violations = []
    for path in (ROOT / "core/reconstruction").glob("*.py"):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("import nerfstudio") or stripped.startswith("from nerfstudio"):
                violations.append(f"{path.name}:{line_no}")
            if stripped.startswith("import tinycudann") or stripped.startswith("from tinycudann"):
                violations.append(f"{path.name}:{line_no}")
    assert violations == []


def test_e2_setup_is_explicitly_isolated_and_pinned() -> None:
    text = (ROOT / "scripts/setup_nerfstudio_backend.sh").read_text(encoding="utf-8")
    assert 'ENV_NAME="eia-nerf-nerfstudio"' in text
    assert 'NERFSTUDIO_TAG="v1.1.5"' in text
    assert 'TCNN_REVISION="b3473c81396fe927293bdfd5a6be32df8769927c"' in text
    assert "production `depth` environment" in text
    assert "TCNN_CUDA_ARCHITECTURES" in text


def test_e2_worker_does_not_import_nerfstudio_at_module_import_time() -> None:
    lines = (ROOT / "scripts/nerf_workers/nerfstudio_worker.py").read_text(encoding="utf-8").splitlines()
    top_level = [line for line in lines if line.startswith("import nerfstudio") or line.startswith("from nerfstudio")]
    assert top_level == []
