from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _module():
    repo = Path(__file__).resolve().parents[2]
    path = repo / "scripts/run_phase8_9f2_real_geometry_campaign.py"
    spec = importlib.util.spec_from_file_location("phase8_9f2_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return repo, module


def test_f2_real_chain_reuses_existing_contracts_and_keeps_heavy_backends_external() -> None:
    repo, module = _module()
    assert (repo / "scripts/run_reconstruction_pycolmap_smoke.py").is_file()
    assert (repo / "scripts/run_da3_real_qualification.py").is_file()
    assert (repo / "scripts/run_da3_canonical_geometry_integration.py").is_file()

    da3_provider = (repo / "core/reconstruction/da3_provider.py").read_text(encoding="utf-8")
    da3_worker = (repo / "scripts/da3_workers/da3_worker.py").read_text(encoding="utf-8")
    canonical = (repo / "scripts/run_da3_canonical_geometry_integration.py").read_text(encoding="utf-8")
    assert "SUPPORTING_GEOMETRY" in da3_provider
    assert "prediction.extrinsics" in da3_worker
    assert '"canonical_geometry_authoritative": True' in canonical
    assert '"da3_supporting_evidence_authoritative": False' in canonical
    assert '"certified_metrology": False' in canonical

    forbidden = set(module.HEAVY_MODULES)
    findings = []
    for path in (repo / "core/reconstruction").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.split(".", 1)[0] in forbidden:
                    findings.append((path.name, name))
    assert findings == []


def test_f2_validation_script_does_not_modify_production_modules() -> None:
    _, module = _module()
    assert module.F2_REPO_FILES == {
        "docs/phase_history/PHASE8_9F2_VALIDATION.md",
        "scripts/run_phase8_9f2_real_geometry_campaign.py",
        "scripts/run_phase8_9f2_tests.sh",
        "tests/integration/test_phase8_9f2_real_geometry_contract.py",
        "tests/unit/test_phase8_9f2_real_geometry_campaign.py",
    }
